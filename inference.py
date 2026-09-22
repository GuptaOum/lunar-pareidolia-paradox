import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification
from peft import PeftModel

TEST_IMG_DIR = "./eval_images"
TEST_META_CSV = "./test_metadata.csv"
OUTPUT_DIR = "./lunar_model_output"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class LunarDataset(Dataset):
    def __init__(self, metadata_df, img_dir, transform=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

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
        return image, img_name

def predict_probs(model_path):
    test_df = pd.read_csv(TEST_META_CSV)
    test_dataset = LunarDataset(test_df, TEST_IMG_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    base_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k", num_labels=2, ignore_mismatched_sizes=True
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
    return all_names, np.concatenate(all_probs, axis=0)

if __name__ == "__main__":
    paths = [os.path.join(OUTPUT_DIR, "seed_42_best"), os.path.join(OUTPUT_DIR, "seed_123_best")]
    all_probs = []
    img_names = None
    for p in paths:
        names, probs = predict_probs(p)
        all_probs.append(probs)
        if img_names is None: img_names = names
        
    ensemble_probs = np.mean(all_probs, axis=0)
    final_preds = np.argmax(ensemble_probs, axis=1)
    sub_df = pd.DataFrame({"image_id": img_names, "label": final_preds})
    sub_csv_path = os.path.join(OUTPUT_DIR, "submission_2seed_ensemble.csv")
    sub_df.to_csv(sub_csv_path, index=False)
    print(f"Saved to {sub_csv_path}")
