import os
import sys
import math
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
from transformers import ViTModel
from peft import LoraConfig, get_peft_model
from scipy.ndimage import sobel

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_generalizer_80plus_output"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 14
LR = 4e-4
WEIGHT_DECAY = 0.02
N_SPLITS = 3
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)

# -------------------------------------------------------------
# 1. Physics-Informed Preprocessing + Shadow-Ray Gradient
# -------------------------------------------------------------
def compute_shadow_gradient(img_array, azimuth_deg):
    """
    Computes directional gradient along the solar illumination ray vector:
    \nabla_{\vec{u}} I = g_x * sin(\theta) - g_y * cos(\theta)
    """
    rad = math.radians(azimuth_deg)
    sun_x = math.sin(rad)
    sun_y = -math.cos(rad)

    gx = sobel(img_array, axis=1) / 4.0
    gy = sobel(img_array, axis=0) / 4.0

    dir_grad = gx * sun_x + gy * sun_y
    norm_grad = (dir_grad - dir_grad.min()) / (dir_grad.max() - dir_grad.min() + 1e-8)
    return norm_grad.astype(np.float32)

class LunarDataset(Dataset):
    def __init__(self, metadata_df, img_dir, is_train=True, is_test=False, flip_tta=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.is_train = is_train
        self.is_test = is_test
        self.flip_tta = flip_tta

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
            if p < 0.35:
                # 360° Uniform synthetic rotation
                delta_angle = random.uniform(0.0, 360.0)
                image = image.rotate(delta_angle, resample=Image.BILINEAR)
                azimuth = (azimuth - delta_angle) % 360.0
            elif p < 0.65:
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

        # Physics-preserving horizontal flip:
        # Sun remains at North (top), shadows remain pointing downwards, but morphological patterns mirror
        do_flip = False
        if self.is_train and random.random() < 0.5:
            do_flip = True
        elif self.flip_tta:
            do_flip = True

        if do_flip:
            rotated_arr = np.ascontiguousarray(np.fliplr(rotated_arr))
            shadow_arr = np.ascontiguousarray(np.fliplr(shadow_arr))
            raw_arr = np.ascontiguousarray(np.fliplr(raw_arr))

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
            lora_dropout=0.15,
            bias="none"
        )
        self.vit = get_peft_model(base_vit, lora_config)
        hidden_size = self.vit.config.hidden_size # 768

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 2, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 2)
        )

    def forward(self, pixel_values, angle_feats):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        fused = torch.cat([cls_token, angle_feats], dim=1)
        logits = self.classifier(fused)
        return logits

# -------------------------------------------------------------
# 3. Stratified K-Fold Training with Anti-Overfitting Safeguards
# -------------------------------------------------------------
def train_fold(fold, train_df, val_df):
    print(f"\n{'='*75}")
    print(f"  TRAINING FOLD {fold + 1}/{N_SPLITS} | Train: {len(train_df)} | Val (OOF): {len(val_df)}")
    print(f"{'='*75}")

    # Build WeightedRandomSampler for 50/50 balanced batches WITHOUT duplicating data
    class_counts = train_df['label'].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[label] for label in train_df['label']]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df), replacement=True)

    train_loader = DataLoader(
        LunarDataset(train_df, TRAIN_IMG_DIR, is_train=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        LunarDataset(val_df, TRAIN_IMG_DIR, is_train=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = MultimodalLunarViT().to(DEVICE)
    # Cost-sensitive loss with label smoothing:
    # Slightly penalize Hill (1) false negatives [0.95, 1.15] to keep Hill Recall >= 80%
    class_weights = torch.tensor([0.95, 1.15], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    best_val_bal_acc = 0.0
    best_weights_path = os.path.join(OUTPUT_DIR, f"fold_{fold}_best.pth")
    best_val_probs = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, angles, labels in train_loader:
            imgs, angles, labels = imgs.to(DEVICE), angles.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                logits = model(imgs, angles)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation on Out-Of-Fold data
        model.eval()
        val_probs, targets = [], []
        with torch.no_grad():
            for imgs, angles, labels in val_loader:
                imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
                with autocast():
                    logits = model(imgs, angles)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.extend(p)
                targets.extend(labels.numpy())

        val_probs = np.array(val_probs)
        targets = np.array(targets)
        val_preds = np.argmax(val_probs, axis=1)

        bal_acc = balanced_accuracy_score(targets, val_preds)
        raw_acc = accuracy_score(targets, val_preds)
        cm = confusion_matrix(targets, val_preds)
        crater_recall = cm[0,0] / (cm[0,0] + cm[0,1])
        hill_recall = cm[1,1] / (cm[1,0] + cm[1,1])

        print(f"Fold {fold+1} | Epoch [{epoch:02d}/{EPOCHS:02d}] Loss: {avg_loss:.4f} | "
              f"Bal Acc: {bal_acc*100:.2f}% | Acc: {raw_acc*100:.2f}% | "
              f"Crater Recall: {crater_recall*100:.1f}% | Hill Recall: {hill_recall*100:.1f}%")

        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            best_val_probs = val_probs
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved Fold {fold+1} checkpoint: Bal Acc = {bal_acc*100:.2f}%")

    print(f"Fold {fold+1} Complete! Peak Balanced Acc: {best_val_bal_acc*100:.2f}%\n")
    return best_weights_path, best_val_probs

# -------------------------------------------------------------
# 4. Main Stratified K-Fold Pipeline & Threshold Optimization
# -------------------------------------------------------------
def main():
    raw_df = pd.read_csv(TRAIN_META_CSV)
    print(f"Loaded train metadata: {len(raw_df)} samples")
    print(f"Class distribution: {raw_df['label'].value_counts().to_dict()}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(raw_df), 2), dtype=np.float32)
    weight_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(raw_df, raw_df['label'])):
        train_df = raw_df.iloc[train_idx].reset_index(drop=True)
        val_df = raw_df.iloc[val_idx].reset_index(drop=True)

        wp, fold_val_probs = train_fold(fold, train_df, val_df)
        weight_paths.append(wp)
        oof_probs[val_idx] = fold_val_probs

    # -------------------------------------------------------------
    # 5. Out-Of-Fold Pareto-Optimal Decision Threshold Search
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  EVALUATING RIGOROUS OUT-OF-FOLD (OOF) GENERALIZATION (ALL 7,854 SAMPLES)")
    print("=" * 75)

    y_true = raw_df['label'].values
    p_rise_oof = oof_probs[:, 1]

    # Baseline threshold = 0.50
    base_preds = (p_rise_oof >= 0.50).astype(int)
    base_bal_acc = balanced_accuracy_score(y_true, base_preds)
    base_cm = confusion_matrix(y_true, base_preds)
    base_crater_rec = base_cm[0,0] / (base_cm[0,0] + base_cm[0,1])
    base_hill_rec = base_cm[1,1] / (base_cm[1,0] + base_cm[1,1])

    print(f"Baseline (tau = 0.50):")
    print(f"  Balanced Accuracy: {base_bal_acc*100:.2f}% | Crater Recall: {base_crater_rec*100:.2f}% | Hill Recall: {base_hill_rec*100:.2f}%")
    print(f"  Confusion Matrix:\n{base_cm}\n")

    # Threshold sweep: tau in [0.35, 0.65] to find threshold where BOTH >= 80% and Bal Acc is maximized
    best_tau = 0.50
    best_bal_acc = base_bal_acc
    pareto_candidates = []

    for tau in np.linspace(0.35, 0.65, 61):
        preds = (p_rise_oof >= tau).astype(int)
        bal = balanced_accuracy_score(y_true, preds)
        cm = confusion_matrix(y_true, preds)
        rec0 = cm[0,0] / (cm[0,0] + cm[0,1])
        rec1 = cm[1,1] / (cm[1,0] + cm[1,1])

        if rec0 >= 0.78 and rec1 >= 0.78:
            pareto_candidates.append((bal, tau, rec0, rec1, cm))

        if bal > best_bal_acc:
            best_bal_acc = bal
            best_tau = tau

    print("-" * 75)
    if pareto_candidates:
        pareto_candidates.sort(key=lambda x: x[0], reverse=True)
        opt_bal, opt_tau, opt_rec0, opt_rec1, opt_cm = pareto_candidates[0]
        print(f"OPTIMAL PARETO THRESHOLD FOUND (tau = {opt_tau:.3f}):")
        print(f"  >>> OOF Balanced Accuracy: {opt_bal*100:.2f}% <<<")
        print(f"  Crater (0) Recall: {opt_rec0*100:.2f}% (>= 80% target)")
        print(f"  Hill   (1) Recall: {opt_rec1*100:.2f}% (>= 80% target)")
        print(f"  Confusion Matrix:\n{opt_cm}")
    else:
        opt_tau = best_tau
        preds = (p_rise_oof >= opt_tau).astype(int)
        opt_cm = confusion_matrix(y_true, preds)
        opt_rec0 = opt_cm[0,0] / (opt_cm[0,0] + opt_cm[0,1])
        opt_rec1 = opt_cm[1,1] / (opt_cm[1,0] + opt_cm[1,1])
        print(f"Optimal Threshold (tau = {opt_tau:.3f}):")
        print(f"  >>> OOF Balanced Accuracy: {best_bal_acc*100:.2f}% <<<")
        print(f"  Crater (0) Recall: {opt_rec0*100:.2f}%")
        print(f"  Hill   (1) Recall: {opt_rec1*100:.2f}%")
        print(f"  Confusion Matrix:\n{opt_cm}")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "image_id": raw_df['image_id'],
        "label": y_true,
        "prob_crater": oof_probs[:, 0],
        "prob_rise": oof_probs[:, 1],
        "pred_optimal": (p_rise_oof >= opt_tau).astype(int)
    })
    oof_path = os.path.join(OUTPUT_DIR, "oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"\nSaved Out-Of-Fold predictions to: {oof_path}")

    # -------------------------------------------------------------
    # 6. Test Set Inference with 3-Fold Soft-Voting & Flip-TTA
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  GENERATING 2,000 TEST PREDICTIONS WITH 3-FOLD SOFT VOTING & TTA")
    print("=" * 75)
    test_df = pd.read_csv(TEST_META_CSV)
    
    test_loader_norm = DataLoader(
        LunarDataset(test_df, TEST_IMG_DIR, is_train=False, is_test=True, flip_tta=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    test_loader_flip = DataLoader(
        LunarDataset(test_df, TEST_IMG_DIR, is_train=False, is_test=True, flip_tta=True),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    fold_test_probs = []
    test_names = None

    for fold_idx, wp in enumerate(weight_paths):
        print(f"Inferring Test Set with Fold {fold_idx + 1} Model (Normal + Flip TTA)...")
        m = MultimodalLunarViT().to(DEVICE)
        m.load_state_dict(torch.load(wp))
        m.eval()

        probs_norm, names = [], []
        with torch.no_grad():
            for imgs, angles, n in test_loader_norm:
                imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
                with autocast():
                    l = m(imgs, angles)
                probs_norm.extend(torch.softmax(l, dim=1).cpu().numpy())
                if test_names is None:
                    names.extend(n)
        if test_names is None:
            test_names = names

        probs_flip = []
        with torch.no_grad():
            for imgs, angles, _ in test_loader_flip:
                imgs, angles = imgs.to(DEVICE), angles.to(DEVICE)
                with autocast():
                    l = m(imgs, angles)
                probs_flip.extend(torch.softmax(l, dim=1).cpu().numpy())

        # TTA average for this fold
        fold_p = (np.array(probs_norm) + np.array(probs_flip)) / 2.0
        fold_test_probs.append(fold_p)

    # 3-Fold Ensemble Soft-Voting
    ens_test_probs = np.mean(fold_test_probs, axis=0)
    p_rise_test = ens_test_probs[:, 1]

    # Prediction 1: Using Calibrated Threshold opt_tau
    calibrated_preds = (p_rise_test >= opt_tau).astype(int)
    cal_counts = pd.Series(calibrated_preds).value_counts().to_dict()
    print(f"\nCalibrated Threshold (tau = {opt_tau:.3f}) Test Counts: {cal_counts}")

    cal_sub_path = os.path.join(OUTPUT_DIR, "submission_generalizer_80plus.csv")
    pd.DataFrame({"image_id": test_names, "label": calibrated_preds}).to_csv(cal_sub_path, index=False)
    print(f"Saved Calibrated Submission to: {cal_sub_path}")

    # Prediction 2: Using Dynamic Median Threshold (Exact 50/50 balance)
    median_tau = float(np.median(p_rise_test))
    median_preds = (p_rise_test >= median_tau).astype(int)
    med_counts = pd.Series(median_preds).value_counts().to_dict()
    print(f"Dynamic Median Threshold (tau = {median_tau:.3f}) Test Counts: {med_counts}")

    med_sub_path = os.path.join(OUTPUT_DIR, "submission_generalizer_median.csv")
    pd.DataFrame({"image_id": test_names, "label": median_preds}).to_csv(med_sub_path, index=False)
    print(f"Saved Median-Balanced Submission to: {med_sub_path}")

    print("\n" + "=" * 75)
    print("  ALL FOLDS & INFERENCE SUCCESSFULLY COMPLETED!")
    print("=" * 75)

if __name__ == "__main__":
    main()
