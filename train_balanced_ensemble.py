import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, PeftModel

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# --- Configuration ---
TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_model_output"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 25
EARLY_STOP_PATIENCE = 6
BALANCED_SEEDS = [42, 123]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Transforms (proven best)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class LunarDataset(Dataset):
    def __init__(self, metadata_df, img_dir, transform=None, is_test=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        sun_azimuth = row['sun_azimuth_angle']
        image = image.rotate(-sun_azimuth, resample=Image.BILINEAR)
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_name
        return image, int(row['label'])

def build_model():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    for param in model.parameters():
        param.requires_grad = False

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    peft_model = get_peft_model(model, config)
    return peft_model

def create_balanced_dataset(df, seed, target_count=3500):
    df_0 = df[df['label'] == 0]
    df_1 = df[df['label'] == 1]

    # Oversample Class 0 to target_count with replacement
    df_0_balanced = df_0.sample(n=target_count, replace=True, random_state=seed)

    # Downsample Class 1 to target_count without replacement
    df_1_balanced = df_1.sample(n=target_count, replace=False, random_state=seed)

    balanced_df = pd.concat([df_0_balanced, df_1_balanced]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return balanced_df

def train_balanced_seed(seed, raw_df):
    save_path = os.path.join(OUTPUT_DIR, f"seed_{seed}_balanced_best")
    print(f"\n{'='*65}")
    print(f"  CREATING BALANCED 50/50 DATASET FOR SEED {seed} (3500 vs 3500)")
    print(f"{'='*65}")

    balanced_df = create_balanced_dataset(raw_df, seed=seed, target_count=3500)
    print(f"Total balanced dataset size: {len(balanced_df)}")
    print(f"Class counts: {balanced_df['label'].value_counts().to_dict()}")

    train_df, val_df = train_test_split(
        balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=seed
    )
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")

    train_dataset = LunarDataset(train_df, TRAIN_IMG_DIR, transform=transform)
    val_dataset = LunarDataset(val_df, TRAIN_IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_bal_acc = 0.0
    patience_counter = 0

    print(f"\nTraining Seed {seed} Balanced Model...")
    for epoch in range(MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Seed {seed} Balanced Epoch {epoch+1}/{MAX_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Bal Acc: {bal_acc:.4f}, LR: {current_lr:.6f}")
        scheduler.step(bal_acc)

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            patience_counter = 0
            model.save_pretrained(save_path)
            print(f"  >>> New Best Balanced for Seed {seed}: {bal_acc:.4f} (Saved to {save_path})")
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (patience {EARLY_STOP_PATIENCE} reached)")
            break

    print(f"Seed {seed} Balanced complete! Peak Bal Acc: {best_bal_acc:.4f}")
    return save_path

def predict_probs_for_model(model_path, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    base_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, model_path).to(DEVICE)
    model.eval()

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)

if __name__ == "__main__":
    print("=" * 70)
    print("  BALANCED 50/50 RETRAINING PIPELINE (3500 CRATERS vs 3500 HILLS)")
    print("=" * 70)

    raw_df = pd.read_csv(TRAIN_META_CSV)

    # Train the 2 Balanced Models
    balanced_paths = []
    for s in BALANCED_SEEDS:
        path = train_balanced_seed(s, raw_df)
        balanced_paths.append(path)

    # Verification on 800 random images from the original unaltered data
    print("\n" + "=" * 70)
    print("  VERIFYING BALANCED ENSEMBLE ON 800 UNBIASED RANDOM TEST IMAGES")
    print("=" * 70)
    eval_df = raw_df.sample(n=800, random_state=555).reset_index(drop=True)
    true_labels = eval_df['label'].values
    eval_dataset = LunarDataset(eval_df, TRAIN_IMG_DIR, transform=transform)

    # 1. Balanced Ensemble (Seed 42 Balanced + Seed 123 Balanced)
    b_probs = []
    for p in balanced_paths:
        print(f"Predicting with {os.path.basename(p)}...")
        probs = predict_probs_for_model(p, eval_dataset)
        b_probs.append(probs)

    ens_b_probs = np.mean(b_probs, axis=0)
    ens_b_preds = np.argmax(ens_b_probs, axis=1)
    b_bal_acc = balanced_accuracy_score(true_labels, ens_b_preds)
    b_acc = accuracy_score(true_labels, ens_b_preds)
    b_cm = confusion_matrix(true_labels, ens_b_preds)

    print("\n" + "=" * 70)
    print("  RESULTS: 2-SEED BALANCED ENSEMBLE (ON 800 RANDOM IMAGES)")
    print("=" * 70)
    print(f"Balanced Accuracy: {b_bal_acc * 100:.2f}%")
    print(f"Raw Accuracy:      {b_acc * 100:.2f}%")
    print(f"Confusion Matrix:\n{b_cm}")
    print(f"Crater (0) Recall: {b_cm[0,0]/(b_cm[0,0]+b_cm[0,1])*100:.1f}%")
    print(f"Hill   (1) Recall: {b_cm[1,1]/(b_cm[1,0]+b_cm[1,1])*100:.1f}%")

    # Generate test submission
    print("\nGenerating submission on 2,000 test images using Balanced Ensemble...")
    test_df = pd.read_csv(TEST_META_CSV)
    test_dataset = LunarDataset(test_df, TEST_IMG_DIR, transform=transform, is_test=True)

    test_b_probs = []
    for p in balanced_paths:
        test_b_probs.append(predict_probs_for_model(p, test_dataset))

    final_b_probs = np.mean(test_b_probs, axis=0)
    final_b_preds = np.argmax(final_b_probs, axis=1)

    sub_b_path = os.path.join(OUTPUT_DIR, "submission_balanced_ensemble.csv")
    pd.DataFrame({
        "image_id": test_df["image_id"],
        "label": final_b_preds
    }).to_csv(sub_b_path, index=False)

    print(f"Saved balanced ensemble submission to: {sub_b_path}")
    print(f"Prediction distribution: {pd.Series(final_b_preds).value_counts().to_dict()}")
    print("BALANCED TRAINING PIPELINE COMPLETE!")
