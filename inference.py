import os
import math
import argparse
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from transformers import ViTModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 🎯 EVALUATOR CONFIGURATION: PLACEHOLDER FOR TEST DATA
# ==============================================================================
# Evaluator: Set the path to your test data below, OR pass them via CLI:
#   python inference.py --img_dir <path_to_images> --meta_csv <path_to_csv>
#
# If you drop your test images into './eval_images' or './test_images' and your
# metadata into './test_metadata.csv', simply run:
#   python inference.py
# ==============================================================================
DEFAULT_TEST_IMG_DIR = "./eval_images"        # <-- [EVALUATOR PLACEHOLDER: Path to test images folder]
DEFAULT_TEST_META_CSV = "./test_metadata.csv" # <-- [EVALUATOR PLACEHOLDER: Path to test metadata CSV]
DEFAULT_OUTPUT_CSV = "./submission.csv"       # <-- Path where output predictions will be saved
# ==============================================================================

# -------------------------------------------------------------
# Pure Vision Preprocessing: Reflection Padding + Inscribed Center Crop
# -------------------------------------------------------------
def preprocess_lunar_image(image_input, azimuth_deg=0.0):
    """
    Standardizes illumination strictly to North (Top) without artificial black borders:
    1. In OpenCV, azimuth is measured clockwise from North. Rotating counter-clockwise
       by +azimuth brings the illumination ray to North (Top of image).
    2. cv2.BORDER_REFLECT_101 seamlessly reflects lunar surface across borders (zero black wedges).
    3. Inscribed Center Crop (200x200 from 256x256) ensures the central feature is centered and
       isolated from boundary interpolation artifacts.
    4. NO numerical angles are passed to the classifier head — pure vision only!
    """
    if isinstance(image_input, str):
        img_gray = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Failed to load image at: {image_input}")
    elif isinstance(image_input, Image.Image):
        img_gray = np.array(image_input.convert('L'))
    elif isinstance(image_input, np.ndarray):
        img_gray = image_input if len(image_input.shape) == 2 else cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("image_input must be a file path, PIL Image, or numpy array")

    h, w = img_gray.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(azimuth_deg), 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    crop_size = 200
    start_x = int((w - crop_size) / 2)
    start_y = int((h - crop_size) / 2)
    cropped = rotated[start_y:start_y + crop_size, start_x:start_x + crop_size]

    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    img_tensor = torch.tensor(np.stack([norm, norm, norm], axis=0), dtype=torch.float32)
    return img_tensor

# -------------------------------------------------------------
# Pure Vision Transformer Architecture (NO Angle Shortcut in Head)
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

        # Pure vision head: strictly 768 image features, zero numeric angle metadata
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
# Adaptive Dataset for Batch Inference
# -------------------------------------------------------------
class LunarInferenceDataset(Dataset):
    def __init__(self, metadata_df, img_dir):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.img_col, self.az_col = self._detect_columns(self.metadata)

    def _detect_columns(self, df):
        # Auto-detect image identifier column
        img_col = None
        for candidate in ['image_id', 'id', 'image', 'filename', 'file_name', 'name', 'img']:
            for c in df.columns:
                if str(c).strip().lower() == candidate:
                    img_col = c
                    break
            if img_col:
                break
        if img_col is None:
            img_col = df.columns[0]

        # Auto-detect sun azimuth angle column
        az_col = None
        for candidate in ['sun_azimuth_angle', 'sun_azimuth', 'azimuth', 'sun_angle', 'angle', 'azimuth_angle']:
            for c in df.columns:
                if str(c).strip().lower() == candidate:
                    az_col = c
                    break
            if az_col:
                break

        return img_col, az_col

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = str(row[self.img_col])
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            for ext in ['.png', '.jpg', '.jpeg']:
                if os.path.exists(img_path + ext):
                    img_path = img_path + ext
                    break

        azimuth = 0.0
        if self.az_col and self.az_col in row and not pd.isna(row[self.az_col]):
            try:
                azimuth = float(row[self.az_col])
            except (ValueError, TypeError):
                azimuth = 0.0

        img_tensor = preprocess_lunar_image(img_path, azimuth)
        return img_tensor, img_name

# -------------------------------------------------------------
# Unified Soft-Voting Ensemble Engine
# -------------------------------------------------------------
class LunarEnsemble(nn.Module):
    DEFAULT_WEIGHT_PATHS = [
        "./lunar_pure_vision_output/fold_0_best.pth",
        "./lunar_pure_vision_output/fold_1_best.pth",
        "./lunar_pure_vision_output/fold_2_best.pth"
    ]

    def __init__(self, weight_paths=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = weight_paths or self.DEFAULT_WEIGHT_PATHS

        self.models = nn.ModuleList([PureVisionLunarViT() for _ in range(len(paths))])
        for idx, (m, path) in enumerate(zip(self.models, paths)):
            if not os.path.exists(path):
                print(f"Checkpoint not found locally at {path}.")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                url = f"https://github.com/GuptaOum/lunar-pareidolia-paradox/releases/download/v3.0.0/pure_vision_fold_{idx}.pth"
                print(f"Auto-downloading Fold {idx + 1} weights from GitHub Releases: {url} ...")
                import urllib.request
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(path, 'wb') as f_out:
                    f_out.write(resp.read())
                print(f"Fold {idx + 1} weights successfully downloaded!")

            print(f"Loading Fold {idx + 1} weights from: {path}")
            m.load_state_dict(torch.load(path, map_location="cpu"))

        self.to(self.device)
        self.eval()

    def forward(self, pixel_values, apply_tta=True):
        pixel_values = pixel_values.to(self.device)
        all_probs = []

        with torch.no_grad():
            for model in self.models:
                logits = model(pixel_values)
                probs = torch.softmax(logits, dim=-1)

                if apply_tta:
                    flipped_pixels = torch.flip(pixel_values, dims=[-1])
                    logits_flip = model(flipped_pixels)
                    probs_flip = torch.softmax(logits_flip, dim=-1)
                    probs = (probs + probs_flip) / 2.0

                all_probs.append(probs)

            ens_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            return ens_probs

    def predict_dataset(self, metadata_df, img_dir, batch_size=32, apply_tta=True):
        dataset = LunarInferenceDataset(metadata_df, img_dir)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4 if os.name != 'nt' else 0
        )

        all_names = []
        all_probs = []

        with torch.no_grad():
            for batch_imgs, batch_names in loader:
                probs = self.forward(batch_imgs, apply_tta=apply_tta)
                all_probs.append(probs.cpu().numpy())
                all_names.extend(batch_names)

        all_probs = np.concatenate(all_probs, axis=0)
        p_rise = all_probs[:, 1]
        p_depth = all_probs[:, 0]

        # Median threshold ensures 50/50 balance on arbitrary domain distributions
        median_thresh = float(np.median(p_rise))
        preds_median = (p_rise >= median_thresh).astype(int)
        preds_standard = (p_rise >= 0.5).astype(int)

        return pd.DataFrame({
            "image_id": all_names,
            "label": preds_median,
            "label_standard": preds_standard,
            "prob_rise": p_rise,
            "prob_depth": p_depth
        })

# -------------------------------------------------------------
# Smart Evaluator-Adaptive Path Resolver
# -------------------------------------------------------------
def resolve_evaluator_inputs(img_dir, meta_csv):
    if not os.path.exists(img_dir):
        candidates = ["./eval_images", "./test_images", "./Test", "./test", "./images", "./data/eval_images", "./data/test_images"]
        for c in candidates:
            if os.path.exists(c) and os.path.isdir(c):
                print(f"Auto-detected test image directory: {c}")
                img_dir = c
                break

    resolved_meta = meta_csv
    if meta_csv and not os.path.exists(meta_csv):
        candidates = ["./test_metadata.csv", "./eval_metadata.csv", "./test.csv", "./eval.csv", "./metadata.csv"]
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                print(f"Auto-detected test metadata CSV: {c}")
                resolved_meta = c
                break

    if (not resolved_meta or not os.path.exists(resolved_meta)) and os.path.exists(img_dir):
        valid_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)])
        if img_files:
            print(f"No metadata CSV provided. Found {len(img_files)} test images in {img_dir}. Auto-generating metadata...")
            df = pd.DataFrame({"image_id": img_files, "sun_azimuth_angle": 0.0})
            return img_dir, df

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: '{img_dir}'. Please provide path via --img_dir <path>")
    if not os.path.exists(resolved_meta):
        raise FileNotFoundError(f"Metadata file not found: '{resolved_meta}'. Please provide path via --meta_csv <path>")

    df = pd.read_csv(resolved_meta)
    return img_dir, df

def run_ensemble_inference(img_dir=DEFAULT_TEST_IMG_DIR, meta_csv=DEFAULT_TEST_META_CSV, output_csv=DEFAULT_OUTPUT_CSV):
    print("=" * 70)
    print("  LUNAR PURE VISION ENSEMBLE INFERENCE ENGINE (v3.0)")
    print("  - Zero numerical angle shortcut in classification head")
    print("  - OpenCV BORDER_REFLECT_101 + Inscribed Center Crop (Zero black edges)")
    print("  - 3-Fold Stratified Soft-Voting with Horizontal-Flip TTA")
    print("=" * 70)

    img_dir, test_df = resolve_evaluator_inputs(img_dir, meta_csv)
    print(f"Evaluating on {len(test_df)} images from '{img_dir}'...")

    ensemble = LunarEnsemble()
    results_df = ensemble.predict_dataset(test_df, img_dir, batch_size=32, apply_tta=True)

    sub_df = results_df[["image_id", "label"]]
    sub_df.to_csv(output_csv, index=False)

    print(f"\n[SUCCESS] Saved predictions for {len(sub_df)} images to: '{output_csv}'")
    counts = sub_df["label"].value_counts().to_dict()
    print(f"Prediction Breakdown: Depth (0): {counts.get(0, 0)} | Rise (1): {counts.get(1, 0)}")
    return sub_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar Pure Vision Ensemble Inference")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_TEST_IMG_DIR, help="Path to test images folder")
    parser.add_argument("--meta_csv", type=str, default=DEFAULT_TEST_META_CSV, help="Path to test metadata CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to save output submission CSV")
    args = parser.parse_args()

    run_ensemble_inference(img_dir=args.img_dir, meta_csv=args.meta_csv, output_csv=args.output)
