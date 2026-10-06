import os
import math
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
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
# Physics-Informed Solar Ray & Shadow Gradient Helpers
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

def preprocess_lunar_image(image_input, azimuth_deg=0.0):
    """
    Converts a lunar image into the normalized 3-channel physics representation:
    Channel 1: Sun-to-North aligned image
    Channel 2: Directional shadow-ray gradient along illumination vector
    Channel 3: Raw grayscale image
    Returns: (img_tensor [3, 224, 224], angle_feat [2])
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('L')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('L')
    else:
        raise ValueError("image_input must be a file path or PIL Image")

    azimuth = float(azimuth_deg)

    # 1. Coordinate-aligned rotation: Sun to North (Top)
    rotated_img = image.rotate(-azimuth, resample=Image.BILINEAR)
    rotated_arr = np.array(rotated_img, dtype=np.float32) / 255.0

    # 2. Directional shadow-ray gradient channel
    shadow_channel = compute_shadow_gradient(np.array(image, dtype=np.float32), azimuth)
    shadow_rotated = Image.fromarray((shadow_channel * 255).astype(np.uint8)).rotate(-azimuth, resample=Image.BILINEAR)
    shadow_arr = np.array(shadow_rotated, dtype=np.float32) / 255.0

    # 3. Raw image channel
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

    return img_tensor, angle_feat

# -------------------------------------------------------------
# Base Multimodal ViT + LoRA Network
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
# Adaptive Dataset for Batch Inference
# -------------------------------------------------------------
class LunarInferenceDataset(Dataset):
    def __init__(self, metadata_df, img_dir):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir

        # Smart column auto-detection
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
            # Check with extension fallbacks if omitted in CSV
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

        img_tensor, angle_feat = preprocess_lunar_image(img_path, azimuth)
        return img_tensor, angle_feat, img_name

# -------------------------------------------------------------
# Unified 3-Fold Ensemble Model with Auto-Downloader
# -------------------------------------------------------------
class LunarEnsemble(nn.Module):
    """
    Unified Ensemble Model that encapsulates all 3 fold models into a single container.
    Executes all 3 model predictions in one go, aggregates soft probabilities with TTA,
    and returns final calibrated predictions.
    """
    DEFAULT_WEIGHT_PATHS = [
        os.path.join(os.path.dirname(__file__), "lunar_generalizer_80plus_output", f"fold_{i}_best.pth")
        for i in range(3)
    ]

    def __init__(self, weight_paths=None, threshold=0.768, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        paths = weight_paths or self.DEFAULT_WEIGHT_PATHS

        self.models = nn.ModuleList([MultimodalLunarViT() for _ in range(len(paths))])
        for idx, (m, path) in enumerate(zip(self.models, paths)):
            if not os.path.exists(path):
                print(f"Checkpoint not found locally at {path}.")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                url = f"https://github.com/GuptaOum/lunar-pareidolia-paradox/releases/download/v2.0.0/fold_{idx}_best.pth"
                print(f"Auto-downloading Fold {idx + 1} weights from GitHub Releases v2.0.0: {url} ...")
                import urllib.request
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(path, 'wb') as f_out:
                    f_out.write(resp.read())
                print(f"Fold {idx + 1} weights successfully downloaded!")

            print(f"Loading Fold {idx + 1} weights from: {path}")
            m.load_state_dict(torch.load(path, map_location="cpu"))

        self.to(self.device)
        self.eval()

    def forward(self, pixel_values, angle_feats, apply_tta=True):
        """
        Executes all 3 models in ONE pass and averages soft-voting probabilities.
        If apply_tta=True, also performs horizontal-flip TTA and averages across passes.
        """
        pixel_values = pixel_values.to(self.device)
        angle_feats = angle_feats.to(self.device)

        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(pixel_values, angle_feats)
                probs = torch.softmax(logits, dim=-1)

                if apply_tta:
                    flipped_pixels = torch.flip(pixel_values, dims=[-1])
                    logits_flip = model(flipped_pixels, angle_feats)
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

        print(f"Executing Unified Ensemble on {len(metadata_df)} images...")
        with torch.no_grad():
            for imgs, angles, names in loader:
                probs = self.forward(imgs, angles, apply_tta=apply_tta).cpu().numpy()
                all_probs.extend(probs)
                all_names.extend(names)

        all_probs = np.array(all_probs)
        p_rise = all_probs[:, 1]
        preds = (p_rise >= self.threshold).astype(int)

        return pd.DataFrame({
            "image_id": all_names,
            "label": preds,
            "prob_rise": p_rise,
            "prob_depth": all_probs[:, 0]
        })

# -------------------------------------------------------------
# Smart Evaluator-Adaptive Path Resolver
# -------------------------------------------------------------
def resolve_evaluator_inputs(img_dir, meta_csv):
    """
    Intelligently resolves image directory and metadata file:
    - Auto-detects custom folder names (e.g. test_images, eval_images, Test, test)
    - Auto-detects custom CSV names (e.g. test_metadata.csv, eval_metadata.csv, test.csv)
    - Auto-generates metadata if evaluator only provides an image folder
    """
    # 1. Resolve image directory
    if not os.path.exists(img_dir):
        candidates = ["./eval_images", "./test_images", "./Test", "./test", "./images", "./data/eval_images", "./data/test_images"]
        for c in candidates:
            if os.path.exists(c) and os.path.isdir(c):
                print(f"Auto-detected test image directory: {c}")
                img_dir = c
                break

    # 2. Resolve metadata CSV
    resolved_meta = meta_csv
    if meta_csv and not os.path.exists(meta_csv):
        candidates = ["./test_metadata.csv", "./eval_metadata.csv", "./test.csv", "./eval.csv", "./metadata.csv"]
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                print(f"Auto-detected test metadata CSV: {c}")
                resolved_meta = c
                break

    # 3. If no metadata CSV found, scan the image folder directly
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
    print("  LUNAR 3-FOLD UNIFIED ENSEMBLE INFERENCE ENGINE (v2.0)")
    print("=" * 70)

    # Smart auto-resolve for whatever files the evaluator provides
    img_dir, test_df = resolve_evaluator_inputs(img_dir, meta_csv)
    print(f"Evaluating on {len(test_df)} images from '{img_dir}'...")

    ensemble = LunarEnsemble(threshold=0.768)
    results_df = ensemble.predict_dataset(test_df, img_dir, batch_size=32, apply_tta=True)

    sub_df = results_df[["image_id", "label"]]
    sub_df.to_csv(output_csv, index=False)

    print("\n" + "-" * 70)
    print(f"Saved final predictions to: {output_csv}")
    print(f"Prediction Class Distribution:\n{sub_df['label'].value_counts().to_dict()}")
    print("Inference successfully complete in one unified execution!")
    print("-" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lunar Unified Ensemble Inference")
    parser.add_argument("--img_dir", type=str, default=DEFAULT_TEST_IMG_DIR, help="Path to evaluation images directory")
    parser.add_argument("--meta_csv", type=str, default=DEFAULT_TEST_META_CSV, help="Path to test metadata CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to save output submission CSV")
    args = parser.parse_args()

    run_ensemble_inference(args.img_dir, args.meta_csv, args.output)
