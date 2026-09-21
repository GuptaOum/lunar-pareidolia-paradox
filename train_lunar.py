import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from torchvision import transforms
from transformers import ViTForImageClassification, ViTConfig
from peft import LoraConfig, get_peft_model

# --- Configuration ---
TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_model_output"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Transforms ---
# PHYSICS CLUE: No destructive augmentations!
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Dataset Definition ---
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
        
        # PHYSICS CLUE
        sun_azimuth = row['sun_azimuth_angle']
        image = image.rotate(-sun_azimuth, resample=Image.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            label = int(row['label'])
            return image, label

# --- Model Building ---
def get_model():
    print("Using Google ViT + LoRA (all-linear) for Classification...")
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Freeze the base model
    for param in model.parameters():
        param.requires_grad = False
        
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear", # all-linear style
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"],
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model

def train():
    print("Loading data...")
    df = pd.read_csv(TRAIN_META_CSV)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_dataset = LunarDataset(train_df, TRAIN_IMG_DIR, transform=train_transform)
    val_dataset = LunarDataset(val_df, TRAIN_IMG_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = get_model()
    model.to(DEVICE)
    
    class_counts = train_df['label'].value_counts().sort_index()
    total_samples = len(train_df)
    count_0 = class_counts.get(0, 1)
    count_1 = class_counts.get(1, 1)
    class_weights = torch.tensor([total_samples / (2.0 * count_0), 
                                  total_samples / (2.0 * count_1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    best_bal_acc = 0.0
    
    TOTAL_EPOCHS = 20
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS)
    
    for epoch in range(TOTAL_EPOCHS):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                
        val_loss /= len(val_loader.dataset)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Balanced Acc: {bal_acc:.4f}")
        
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            save_path = os.path.join(OUTPUT_DIR, "best_model")
            model.save_pretrained(save_path)
            print(f"Saved best model with Balanced Acc: {bal_acc:.4f}")
            
    print("Training complete.")

def inference():
    print("Starting inference on test set...")
    test_df = pd.read_csv(TEST_META_CSV)
    test_dataset = LunarDataset(test_df, TEST_IMG_DIR, transform=val_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    base_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, os.path.join(OUTPUT_DIR, "best_model"))
    model.to(DEVICE)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs.logits, dim=1)
            
            for name, pred in zip(img_names, preds.cpu().numpy().flatten()):
                results.append({"image_id": name, "label": pred})
                
    sub_df = pd.DataFrame(results)
    sub_csv_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub_df.to_csv(sub_csv_path, index=False)
    print(f"Submission saved to {sub_csv_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "predict":
        inference()
    else:
        train()
