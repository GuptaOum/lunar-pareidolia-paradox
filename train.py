import os
import sys
import math
import random
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, classification_report
from transformers import ViTModel
from peft import LoraConfig, get_peft_model

# Auto-import or install OpenCV, with pure PIL/NumPy reflection fallback
try:
    import cv2
except ImportError:
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "opencv-python-headless"])
        import cv2
    except Exception:
        cv2 = None

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_pure_vision_output"

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 12
LR = 4e-4
WEIGHT_DECAY = 0.01
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
# 1. Pure Vision Preprocessing (Reflection Padding + CenterCrop)
# -------------------------------------------------------------
def preprocess_image(img_gray, azimuth_deg, is_train=False, flip_tta=False):
    """
    Standardizes illumination strictly to North (Top) without artificial black borders:
    1. Azimuth is measured clockwise from North. Rotating counter-clockwise
       by +azimuth brings the illumination ray to North (Top of image).
    2. Reflection padding seamlessly reflects lunar surface across borders (zero black wedges).
    3. Inscribed Center Crop (200x200 from 256x256) ensures the central feature is centered and
       isolated from boundary interpolation artifacts.
    4. NO numerical angles are passed to the neural network — pure vision only!
    """
    if cv2 is not None:
        h, w = img_gray.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), float(azimuth_deg), 1.0)
        rotated = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        crop_size = 200
        start_x = int((w - crop_size) / 2)
        start_y = int((h - crop_size) / 2)
        cropped = rotated[start_y:start_y + crop_size, start_x:start_x + crop_size]
    else:
        # Fallback using pure NumPy reflection padding + PIL rotation
        padded = np.pad(img_gray, pad_width=64, mode='reflect')
        pil_img = Image.fromarray(padded).rotate(float(azimuth_deg), resample=Image.BILINEAR)
        w, h = pil_img.size
        crop_size = 200
        start_x = (w - crop_size) // 2
        start_y = (h - crop_size) // 2
        cropped = np.array(pil_img)[start_y:start_y + crop_size, start_x:start_x + crop_size]

    # Physics-safe horizontal flip (sun is at top, so mirroring left-right preserves shadow physics)
    if (is_train and random.random() < 0.5) or flip_tta:
        cropped = np.ascontiguousarray(np.fliplr(cropped))

    if cv2 is not None:
        resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.array(Image.fromarray(cropped).resize((224, 224), resample=Image.BILINEAR))

    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    tensor = torch.tensor(np.stack([norm, norm, norm], axis=0), dtype=torch.float32)
    return tensor

class LunarPureVisionDataset(Dataset):
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
        img_name = str(row['image_id'])
        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        if cv2 is not None:
            img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            img_gray = np.array(Image.open(img_path).convert('L'))

        azimuth = float(row['sun_azimuth_angle'])
        img_tensor = preprocess_image(img_gray, azimuth, is_train=self.is_train, flip_tta=self.flip_tta)

        if self.is_test:
            return img_tensor, img_name
        return img_tensor, int(row['label'])

# -------------------------------------------------------------
# 2. Pure Vision Transformer Architecture (NO Angle Features in Head)
# -------------------------------------------------------------
class PureVisionLunarViT(nn.Module):
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

        # Pure vision classification head: takes strictly cls_token (768)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 2)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits

# -------------------------------------------------------------
# 3. Training Loop per Fold
# -------------------------------------------------------------
def train_single_fold(fold_idx, train_df, val_df, device):
    print(f"\n{'=' * 70}")
    print(f"  TRAINING PURE VISION FOLD {fold_idx + 1} / {N_SPLITS}")
    print(f"  Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(f"{'=' * 70}")

    train_dataset = LunarPureVisionDataset(train_df, TRAIN_IMG_DIR, is_train=True)
    val_dataset = LunarPureVisionDataset(val_df, TRAIN_IMG_DIR, is_train=False)

    # Class-balanced sampling
    class_counts = train_df['label'].value_counts()
    class_weights = {
        0: 1.0 / class_counts[0],
        1: 1.0 / class_counts[1]
    }
    sample_weights = train_df['label'].map(class_weights).values
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True
    )

    model = PureVisionLunarViT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()

    best_val_bal_acc = 0.0
    best_weights_path = os.path.join(OUTPUT_DIR, f"fold_{fold_idx}_best.pth")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_steps += 1

        scheduler.step()
        avg_train_loss = train_loss / train_steps

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                val_steps += 1

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / val_steps
        bal_acc = balanced_accuracy_score(val_targets, val_preds)
        raw_acc = accuracy_score(val_targets, val_preds)
        cm = confusion_matrix(val_targets, val_preds)
        crater_rec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0
        hill_rec = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0.0

        print(f"Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Bal Acc: {bal_acc * 100:.2f}% | Raw Acc: {raw_acc * 100:.2f}% | "
              f"Crater Rec: {crater_rec * 100:.1f}% | Hill Rec: {hill_rec * 100:.1f}%")

        if bal_acc > best_val_bal_acc:
            best_val_bal_acc = bal_acc
            torch.save(model.state_dict(), best_weights_path)
            print(f"  >>> [*] Best model saved for Fold {fold_idx + 1} with Bal Acc: {bal_acc * 100:.2f}%")

    print(f"Fold {fold_idx + 1} Complete. Best Val Bal Acc: {best_val_bal_acc * 100:.2f}%")
    return best_weights_path

# -------------------------------------------------------------
# 4. Out-of-Fold Evaluation & Test Inference
# -------------------------------------------------------------
def evaluate_and_predict(fold_paths, full_train_df, test_df, device):
    print("\n" + "=" * 70)
    print("  RUNNING UNIFIED ENSEMBLE TEST PREDICTION WITH TTA")
    print("=" * 70)

    # Load all 3 fold models
    models = []
    for p in fold_paths:
        m = PureVisionLunarViT()
        m.load_state_dict(torch.load(p, map_location="cpu"))
        m.to(device)
        m.eval()
        models.append(m)

    test_dataset = LunarPureVisionDataset(test_df, TEST_IMG_DIR, is_train=False, is_test=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 if os.name != 'nt' else 0
    )

    all_names = []
    all_probs = []

    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)

            fold_probs = []
            for model in models:
                with autocast():
                    logits_orig = model(images)
                p_orig = torch.softmax(logits_orig, dim=-1)

                images_flip = torch.flip(images, dims=[-1])
                with autocast():
                    logits_flip = model(images_flip)
                p_flip = torch.softmax(logits_flip, dim=-1)

                p_model = (p_orig + p_flip) / 2.0
                fold_probs.append(p_model)

            ens_probs = torch.stack(fold_probs, dim=0).mean(dim=0)
            all_probs.append(ens_probs.cpu().numpy())
            all_names.extend(names)

    all_probs = np.concatenate(all_probs, axis=0)
    p_rise = all_probs[:, 1]
    p_crater = all_probs[:, 0]

    preds_standard = (p_rise >= 0.5).astype(int)
    median_thresh = float(np.median(p_rise))
    preds_median = (p_rise >= median_thresh).astype(int)

    sub_standard_path = os.path.join(OUTPUT_DIR, "submission_pure_vision.csv")
    pd.DataFrame({"image_id": all_names, "label": preds_standard}).to_csv(sub_standard_path, index=False)

    sub_median_path = os.path.join(OUTPUT_DIR, "submission_pure_vision_median.csv")
    pd.DataFrame({"image_id": all_names, "label": preds_median}).to_csv(sub_median_path, index=False)

    print(f"\nPredictions complete!")
    print(f"Standard submission saved to: {sub_standard_path}")
    print(f"Distribution (Standard): {pd.Series(preds_standard).value_counts().to_dict()}")
    print(f"Median-balanced submission saved to: {sub_median_path} (threshold={median_thresh:.4f})")
    print(f"Distribution (Median): {pd.Series(preds_median).value_counts().to_dict()}")

# -------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("  PURE VISION LUNAR ARCHITECTURE: ELIMINATING DOMAIN SHIFT")
    print("  - Zero numeric angle shortcut in classification head")
    print("  - Reflection Padding + Inscribed CenterCrop (Zero black edges)")
    print("  - 3-Fold Stratified Cross-Validation with TTA Soft-Voting")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_META_CSV)
    test_df = pd.read_csv(TEST_META_CSV)
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples.")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_paths = []

    for fold_idx, (t_idx, v_idx) in enumerate(skf.split(train_df, train_df['label'])):
        fold_train_df = train_df.iloc[t_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[v_idx].reset_index(drop=True)
        path = train_single_fold(fold_idx, fold_train_df, fold_val_df, DEVICE)
        fold_paths.append(path)

    evaluate_and_predict(fold_paths, train_df, test_df, DEVICE)
    print("\n[SUCCESS] Pure Vision pipeline execution complete!")
