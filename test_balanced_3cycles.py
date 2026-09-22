import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from transformers import ViTForImageClassification
from peft import PeftModel

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

TRAIN_IMG_DIR = "./train_images"
TRAIN_META_CSV = "./train_metadata.csv"
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
        return image, int(row['label'])

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
        for images, _ in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs.logits, dim=1)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)

if __name__ == "__main__":
    print("=" * 75)
    print("  EVALUATING 2-SEED BALANCED ENSEMBLE ACROSS 3 CYCLES OF 600 RANDOM IMAGES")
    print("=" * 75)

    train_df = pd.read_csv(TRAIN_META_CSV)

    cycle_seeds = [101, 202, 303]
    cycle_samples = []
    for i, s in enumerate(cycle_seeds):
        sample = train_df.sample(n=600, random_state=s).reset_index(drop=True)
        cycle_samples.append(sample)
        dist = sample['label'].value_counts().to_dict()
        print(f"Cycle {i+1} (seed={s}): 600 images | Class distribution: {dist} (Hills: {dist.get(1, 0)}, Craters: {dist.get(0, 0)})")

    models = {
        "Seed 42 Balanced": os.path.join(OUTPUT_DIR, "seed_42_balanced_best"),
        "Seed 123 Balanced": os.path.join(OUTPUT_DIR, "seed_123_balanced_best")
    }

    # Precompute probabilities for each model on all 3 cycles
    model_cycle_probs = {name: [] for name in models}
    for name, path in models.items():
        print(f"\nRunning predictions for {name} across all 3 cycles...")
        for c_idx, s_df in enumerate(cycle_samples):
            ds = LunarDataset(s_df, TRAIN_IMG_DIR, transform=transform)
            probs = predict_probs_for_model(path, ds)
            model_cycle_probs[name].append(probs)

    # Evaluate Ensemble for each cycle
    print("\n" + "=" * 75)
    print("                      CYCLE-BY-CYCLE RESULTS")
    print("=" * 75)

    cycle_bal_accs = []
    cycle_raw_accs = []
    cycle_crater_recs = []
    cycle_hill_recs = []

    for c_idx in range(3):
        true_labels = cycle_samples[c_idx]['label'].values
        probs_42 = model_cycle_probs["Seed 42 Balanced"][c_idx]
        probs_123 = model_cycle_probs["Seed 123 Balanced"][c_idx]

        # Soft-voting average
        ens_probs = (probs_42 + probs_123) / 2.0
        preds = np.argmax(ens_probs, axis=1)

        bal_acc = balanced_accuracy_score(true_labels, preds)
        raw_acc = accuracy_score(true_labels, preds)
        cm = confusion_matrix(true_labels, preds)

        crater_rec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) * 100
        hill_rec = cm[1, 1] / (cm[1, 0] + cm[1, 1]) * 100

        cycle_bal_accs.append(bal_acc)
        cycle_raw_accs.append(raw_acc)
        cycle_crater_recs.append(crater_rec)
        cycle_hill_recs.append(hill_rec)

        print(f"Cycle {c_idx+1} (600 Images):")
        print(f"  Balanced Accuracy: {bal_acc * 100:.2f}% | Raw Accuracy: {raw_acc * 100:.2f}%")
        print(f"  Crater (0) Recall: {crater_rec:.1f}% ({cm[0,0]}/{cm[0,0]+cm[0,1]})")
        print(f"  Hill   (1) Recall: {hill_rec:.1f}% ({cm[1,1]}/{cm[1,0]+cm[1,1]})")
        print(f"  Confusion Matrix:\n{cm}\n")

    mean_bal_acc = np.mean(cycle_bal_accs) * 100
    std_bal_acc = np.std(cycle_bal_accs) * 100
    mean_raw_acc = np.mean(cycle_raw_accs) * 100
    mean_crater_rec = np.mean(cycle_crater_recs)
    mean_hill_rec = np.mean(cycle_hill_recs)

    print("=" * 75)
    print("             OVERALL 3-CYCLE CONSOLIDATED SUMMARY (1,800 IMAGES)")
    print("=" * 75)
    print(f"Mean Balanced Accuracy: {mean_bal_acc:.2f}% (+/- {std_bal_acc:.2f}%)")
    print(f"Mean Raw Accuracy:      {mean_raw_acc:.2f}%")
    print(f"Average Crater Recall:  {mean_crater_rec:.1f}%")
    print(f"Average Hill Recall:    {mean_hill_rec:.1f}%")
    print(f"Cycle Breakdown:        Round 1: {cycle_bal_accs[0]*100:.2f}% | Round 2: {cycle_bal_accs[1]*100:.2f}% | Round 3: {cycle_bal_accs[2]*100:.2f}%")
    print("=" * 75)
