import os
import sys
import math
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from transformers import ViTModel
from peft import LoraConfig, get_peft_model
from scipy.ndimage import sobel

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_ensemble_80plus_output"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
LR = 5e-4
WEIGHT_DECAY = 0.01
SEEDS = [42, 123]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1 & 5. Physics-Informed Preprocessing + Shadow-Ray Gradient
# -------------------------------------------------------------
def compute_shadow_gradient(img_array, azimuth_deg):
    rad = math.radians(azimuth_deg)
    sun_x = math.sin(rad)
    sun_y = -math.cos(rad)

    gx = sobel(img_array, axis=1) / 4.0
    gy = sobel(img_array, axis=0) / 4.0

    dir_grad = gx * sun_x + gy * sun_y
    norm_grad = (dir_grad - dir_grad.min()) / (dir_grad.max() - dir_grad.min() + 1e-8)
    return norm_grad.astype(np.float32)

class LunarDataset(Dataset):
    def __init__(self, metadata_df, img_dir, is_train=True, is_test=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')
        azimuth = float(row['sun_azimuth_angle'])

        if self.is_train:
            p = random.random()
            if p < 0.40:
                # 360° Uniform synthetic rotation
                delta_angle = random.uniform(0.0, 360.0)
                image = image.rotate(delta_angle, resample=Image.BILINEAR)
                azimuth = (azimuth - delta_angle) % 360.0
            elif p < 0.70:
                # Quadrant-Targeted Rotation: specifically shift into adversarial 90°-180° quadrant
                target_az = random.uniform(90.0, 180.0)
                delta_angle = (azimuth - target_az) % 360.0
                image = image.rotate(delta_angle, resample=Image.BILINEAR)
                azimuth = target_az

        # Coordinate-aligned rotation: Sun to North (Top)
        rotated_img = image.rotate(-azimuth, resample=Image.BILINEAR)
        rotated_arr = np.array(rotated_img, dtype=np.float32) / 255.0

        # Directional shadow-ray gradient channel
        shadow_channel = compute_shadow_gradient(np.array(image, dtype=np.float32), azimuth)
        shadow_rotated = Image.fromarray((shadow_channel * 255).astype(np.uint8)).rotate(-azimuth, resample=Image.BILINEAR)
        shadow_arr = np.array(shadow_rotated, dtype=np.float32) / 255.0

        raw_arr = np.array(image, dtype=np.float32) / 255.0

        r_img = Image.fromarray((rotated_arr * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        s_img = Image.fromarray((shadow_arr * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        o_img = Image.fromarray((raw_arr * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)

        ch1 = (np.array(r_img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        ch2 = (np.array(s_img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        ch3 = (np.array(o_img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        img_tensor = torch.tensor(np.stack([ch1, ch2, ch3], axis=0), dtype=torch.float32)

        rad = math.radians(azimuth)
        angle_feat = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)

        if self.is_test:
            return img_tensor, angle_feat, img_name
        return img_tensor, angle_feat, int(row['label'])

# -------------------------------------------------------------
# 2. Multimodal ViT + LoRA Architecture
# -------------------------------------------------------------
class MultimodalLunarViT(nn.Module):
    def __init__(self):
        super().__init__()
        base_vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value", "dense"],
            lora_dropout=0.1,
            bias="none"
        )
        self.vit = get_peft_model(base_vit, lora_config)
        hidden_size = self.vit.config.hidden_size # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 2, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )

    def forward(self, pixel_values, angle_feats):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        fused = torch.cat([cls_token, angle_feats], dim=1)
        logits = self.classifier(fused)
        return logits

# -------------------------------------------------------------
# 2-Seed Training Function
# -------------------------------------------------------------
def train_seed(seed, raw_df):
    print(f"\n{'='*70}")
    print(f"  TRAINING MULTIMODAL MODEL FOR SEED {seed} (50/50 BALANCED, 20 EPOCHS)")
    print(f"{'='*70}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    df_0 = raw_df[raw_df['label'] == 0].sample(n=3500, replace=True, random_state=seed)
    df_1 = raw_df[raw_df['label'] == 1].sample(n=3500, replace=False, random_state=seed)
    balanced_df = pd.concat([df_0, df_1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df, val_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=seed)
    print(f"Seed {seed} | Train: {len(train_df)} | Val: {len(val_df)}")

    train_loader = DataLoader(LunarDataset(train_df, TRAIN_IMG_DIR, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(LunarDataset(val_df, TRAIN_IMG_DIR, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MultimodalLunarViT().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_bal_acc = 0.0
    weights_path = os.path.join(OUTPUT_DIR, f"seed_{seed}_best.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, angles, labels in train_loader:
            imgs, angles, labels = imgs.to(DEVICE), angles.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs, angles)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, angles, labels in val_loader:
                imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
                logits = model(imgs, angles)
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(p)
                targets.extend(labels.numpy())

        bal_acc = balanced_accuracy_score(targets, preds)
        raw_acc = accuracy_score(targets, preds)
        print(f"Seed {seed} | Epoch [{epoch:02d}/{EPOCHS:02d}] Loss: {avg_loss:.4f} | Val Bal Acc: {bal_acc*100:.2f}% | Val Acc: {raw_acc*100:.2f}%")

        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            torch.save(model.state_dict(), weights_path)
            print(f"  --> Saved Seed {seed} best checkpoint: {bal_acc*100:.2f}%")

    print(f"Seed {seed} Training Finished! Peak Balanced Acc: {best_val_bal_acc*100:.2f}%")
    return weights_path

# -------------------------------------------------------------
# Main Ensemble Pipeline
# -------------------------------------------------------------
def main():
    raw_df = pd.read_csv(TRAIN_META_CSV)
    print(f"Loaded training metadata: {len(raw_df)} samples")

    weight_paths = []
    for s in SEEDS:
        wp = train_seed(s, raw_df)
        weight_paths.append(wp)

    # Verification on 1,400 Holdout Validation Split
    print("\n" + "=" * 70)
    print("  EVALUATING 2-SEED ENSEMBLE ON HOLDOUT VALIDATION DATA")
    print("=" * 70)
    
    # Stratified holdout set
    df_0 = raw_df[raw_df['label'] == 0].sample(n=700, random_state=999)
    df_1 = raw_df[raw_df['label'] == 1].sample(n=700, random_state=999)
    eval_df = pd.concat([df_0, df_1]).sample(frac=1.0, random_state=999).reset_index(drop=True)
    eval_loader = DataLoader(LunarDataset(eval_df, TRAIN_IMG_DIR, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_seed_probs = []
    for wp in weight_paths:
        m = MultimodalLunarViT().to(DEVICE)
        m.load_state_dict(torch.load(wp))
        m.eval()
        probs = []
        with torch.no_grad():
            for imgs, angles, _ in eval_loader:
                imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
                l = m(imgs, angles)
                probs.extend(torch.softmax(l, dim=1).cpu().numpy())
        all_seed_probs.append(np.array(probs))

    ens_val_probs = np.mean(all_seed_probs, axis=0)
    ens_val_preds = np.argmax(ens_val_probs, axis=1)
    true_labels = eval_df['label'].values

    ens_bal_acc = balanced_accuracy_score(true_labels, ens_val_preds)
    ens_acc = accuracy_score(true_labels, ens_val_preds)
    cm = confusion_matrix(true_labels, ens_val_preds)

    print(f"\n>>> 2-SEED ENSEMBLE BALANCED ACCURACY: {ens_bal_acc*100:.2f}% <<<")
    print(f"Raw Accuracy: {ens_acc*100:.2f}%")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Crater (0) Recall: {cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%")
    print(f"Hill   (1) Recall: {cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%")

    # -------------------------------------------------------------
    # Test Inference with Dynamic Median Thresholding
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  GENERATING FINAL 2,000 TEST PREDICTIONS VIA MEDIAN THRESHOLDING")
    print("=" * 70)
    test_df = pd.read_csv(TEST_META_CSV)
    test_loader = DataLoader(LunarDataset(test_df, TEST_IMG_DIR, is_train=False, is_test=True), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    test_seed_probs = []
    img_names = None
    for wp in weight_paths:
        m = MultimodalLunarViT().to(DEVICE)
        m.load_state_dict(torch.load(wp))
        m.eval()
        probs = []
        names = []
        with torch.no_grad():
            for imgs, angles, n in test_loader:
                imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
                l = m(imgs, angles)
                probs.extend(torch.softmax(l, dim=1).cpu().numpy())
                names.extend(n)
        test_seed_probs.append(np.array(probs))
        if img_names is None:
            img_names = names

    ens_test_probs = np.mean(test_seed_probs, axis=0)
    p_rise = ens_test_probs[:, 1] # Probability of Rise (Class 1)

    # 4. Dynamic Quantile / Median Threshold
    median_thresh = float(np.median(p_rise))
    final_preds = (p_rise >= median_thresh).astype(int)

    print(f"Optimal Median Threshold: {median_thresh:.4f}")
    print(f"Final Class Distribution: {pd.Series(final_preds).value_counts().to_dict()}")

    sub_path = os.path.join(OUTPUT_DIR, "submission_ensemble_80plus.csv")
    pd.DataFrame({"image_id": img_names, "label": final_preds}).to_csv(sub_path, index=False)
    print(f"Saved ensemble submission to: {sub_path}")
    print("ALL RUNS COMPLETE!")

if __name__ == "__main__":
    main()
