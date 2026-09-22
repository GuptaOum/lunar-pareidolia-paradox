import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tqdm import tqdm
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, PeftModel

# --- Configuration ---
TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_model_output"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 7
SEEDS = [42, 123, 456]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Transforms (NO augmentations - proven best) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Dataset ---
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

        # MANDATORY: Rotate by -sun_azimuth_angle to normalize sun to North
        sun_azimuth = row['sun_azimuth_angle']
        image = image.rotate(-sun_azimuth, resample=Image.BILINEAR)

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, img_name
        else:
            return image, int(row['label'])

# --- Model ---
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
    peft_model.print_trainable_parameters()
    return peft_model

# --- Train one seed ---
def train_one_seed(seed, df):
    print(f"\n{'='*60}")
    print(f"  SEED {seed} - Training ViT + LoRA (r=16)")
    print(f"{'='*60}")

    # Set all random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['label'], random_state=seed
    )
    print(f"Train: {len(train_df)}, Val: {len(val_df)}")

    train_dataset = LunarDataset(train_df, TRAIN_IMG_DIR, transform=transform)
    val_dataset = LunarDataset(val_df, TRAIN_IMG_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = build_model()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # ReduceLROnPlateau: drop LR by half when val accuracy stalls for 3 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_bal_acc = 0.0
    patience_counter = 0
    save_path = os.path.join(OUTPUT_DIR, f"seed_{seed}_best")

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}/{MAX_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()

            # GRADIENT CLIPPING: Prevents the epoch-10 collapse!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Seed {seed} Epoch {epoch+1}/{MAX_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nSeed {seed} Epoch {epoch+1} - Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Balanced Acc: {bal_acc:.4f}, LR: {current_lr:.6f}")
        print(f"Confusion Matrix:\n{cm}\n")

        # Step scheduler
        scheduler.step(bal_acc)

        # Save best
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            patience_counter = 0
            model.save_pretrained(save_path)
            print(f"*** NEW BEST for seed {seed}: {bal_acc:.4f} ***\n")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {EARLY_STOP_PATIENCE} epochs)")
            break

    print(f"\nSeed {seed} FINAL BEST Balanced Accuracy: {best_bal_acc:.4f}")
    return save_path, best_bal_acc

# --- Inference for one model -> returns probabilities ---
def predict_probs(model_path):
    test_df = pd.read_csv(TEST_META_CSV)
    test_dataset = LunarDataset(test_df, TEST_IMG_DIR, transform=transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    base_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.to(DEVICE)
    model.eval()

    all_probs = []
    all_names = []
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc=f"Inference [{model_path}]"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_names.extend(img_names)

    all_probs = np.concatenate(all_probs, axis=0)
    return all_names, all_probs

# --- Main ---
if __name__ == "__main__":
    print("=" * 60)
    print("  MULTI-SEED SOFT-VOTING ENSEMBLE")
    print(f"  Seeds: {SEEDS}")
    print(f"  Max Epochs: {MAX_EPOCHS}, Early Stop: {EARLY_STOP_PATIENCE}")
    print(f"  Gradient Clipping: max_norm=1.0")
    print(f"  Scheduler: ReduceLROnPlateau(patience=3, factor=0.5)")
    print("=" * 60)

    df = pd.read_csv(TRAIN_META_CSV)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")

    # Phase 1: Train all seeds
    model_paths = []
    seed_scores = []
    for seed in SEEDS:
        path, score = train_one_seed(seed, df)
        model_paths.append(path)
        seed_scores.append(score)

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE - SUMMARY")
    print("=" * 60)
    for seed, score in zip(SEEDS, seed_scores):
        print(f"  Seed {seed}: {score:.4f}")
    print(f"  Average: {np.mean(seed_scores):.4f}")

    # Phase 2: Soft-voting ensemble inference
    print("\n" + "=" * 60)
    print("  SOFT-VOTING ENSEMBLE INFERENCE")
    print("=" * 60)

    all_probs_list = []
    img_names = None
    for path in model_paths:
        names, probs = predict_probs(path)
        all_probs_list.append(probs)
        if img_names is None:
            img_names = names

    # Average probabilities across all seeds
    ensemble_probs = np.mean(all_probs_list, axis=0)
    final_preds = np.argmax(ensemble_probs, axis=1)

    # Save submission
    sub_df = pd.DataFrame({"image_id": img_names, "label": final_preds})
    sub_csv_path = os.path.join(OUTPUT_DIR, "submission_ensemble.csv")
    sub_df.to_csv(sub_csv_path, index=False)
    print(f"\nENSEMBLE Submission saved to {sub_csv_path}")
    print(f"Prediction distribution: {pd.Series(final_preds).value_counts().to_dict()}")
    print("\nDONE!")
