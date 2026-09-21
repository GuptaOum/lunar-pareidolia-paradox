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

# Try importing PEFT and Transformers for LoRA + Google ViT. 

# If not available, instructions will guide the user to pip install them.
try:
    from transformers import ViTForImageClassification, ViTConfig
    from peft import LoraConfig, get_peft_model
    USE_HUGGINGFACE_PEFT = True
except ImportError:
    USE_HUGGINGFACE_PEFT = False
    import torchvision.models as models

# --- Configuration ---
TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_model_output"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
        
        # Load image (convert to RGB as pre-trained models expect 3 channels)
        image = Image.open(img_path).convert('RGB')
        
        # --- PHYSICS CLUE: Rotate image based on sun azimuth ---
        # Rotating by -sun_azimuth_angle standardizes the illumination direction 
        # for all images (e.g., sun always coming from angle 0).
        sun_azimuth = row['sun_azimuth_angle']
        image = image.rotate(-sun_azimuth, resample=Image.BILINEAR)
        
        if self.transform:
            image = self.transform(image)
            
        if self.is_test:
            return image, img_name
        else:
            label = int(row['label'])
            return image, label

# --- Transforms ---
# Standard resize and normalization for ViT or ResNet (224x224)
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Model Building ---
def get_model():
    """Load standard ResNet-50 for classical transfer learning."""
    print("Using Torchvision ResNet-50 for classical transfer learning...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze the entire backbone initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the final fully connected layer (this will be unfrozen by default)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    return model, False

def train():
    # Load Data
    print("Loading data...")
    df = pd.read_csv(TRAIN_META_CSV)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_dataset = LunarDataset(train_df, TRAIN_IMG_DIR, transform=data_transform)
    val_dataset = LunarDataset(val_df, TRAIN_IMG_DIR, transform=data_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Model, optimizer, loss
    model, is_huggingface = get_model()
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    best_bal_acc = 0.0
    
    FREEZE_EPOCHS = 5
    UNFREEZE_EPOCHS = 15
    TOTAL_EPOCHS = FREEZE_EPOCHS + UNFREEZE_EPOCHS
    
    # Training Loop
    for epoch in range(TOTAL_EPOCHS):
        
        # --- PHASE MANAGEMENT ---
        if epoch == 0:
            print("--- PHASE 1: FROZEN BACKBONE ---")
            # Only train the FC layer
            optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
        
        elif epoch == FREEZE_EPOCHS:
            print("--- PHASE 2: UNFROZEN BACKBONE (FINE-TUNING) ---")
            # Unfreeze everything
            for param in model.parameters():
                param.requires_grad = True
            # Train the whole model with a much smaller learning rate
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Train]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            if is_huggingface:
                outputs = model(images).logits
            else:
                outputs = model(images)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                if is_huggingface:
                    outputs = model(images).logits
                else:
                    outputs = model(images)
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Balanced Acc: {bal_acc:.4f}")
        
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            save_path = os.path.join(OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Balanced Acc: {bal_acc:.4f}")
            
    print("Training complete.")


def inference():
    print("Starting inference on test set...")
    test_df = pd.read_csv(TEST_META_CSV)
    test_dataset = LunarDataset(test_df, TEST_IMG_DIR, transform=data_transform, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model, is_huggingface = get_model()
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
    model.to(DEVICE)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Inference"):
            images = images.to(DEVICE)
            if is_huggingface:
                outputs = model(images).logits
            else:
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            for name, pred in zip(img_names, preds.cpu().numpy()):
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
