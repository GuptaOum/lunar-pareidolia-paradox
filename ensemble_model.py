import os
import math
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from scipy.ndimage import sobel
from transformers import ViTModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

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

def preprocess_lunar_image(image_input, azimuth_deg):
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
# Dataset for Batch Inference
# -------------------------------------------------------------
class LunarInferenceDataset(Dataset):
    def __init__(self, metadata_df, img_dir):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = row['image_id']
        img_path = os.path.join(self.img_dir, img_name)
        azimuth = float(row['sun_azimuth_angle'])
        img_tensor, angle_feat = preprocess_lunar_image(img_path, azimuth)
        return img_tensor, angle_feat, img_name

# -------------------------------------------------------------
# Unified 3-Fold Ensemble Model
# -------------------------------------------------------------
class LunarEnsemble(nn.Module):
    """
    Unified Ensemble Model that encapsulates all 3 fold models into a single container.
    Executes all 3 model predictions in one go, aggregates soft probabilities with TTA,
    and returns final predictions.
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
            if os.path.exists(path):
                print(f"Loading Fold {idx + 1} weights from: {path}")
                m.load_state_dict(torch.load(path, map_location="cpu"))
            else:
                print(f"Warning: Checkpoint not found at: {path}")

        self.to(self.device)
        self.eval()

    def forward(self, pixel_values, angle_feats, apply_tta=True):
        """
        Executes all 3 models in ONE pass and averages soft-voting probabilities.
        If apply_tta=True, also performs horizontal-flip TTA and averages across 6 passes.
        Returns: ens_probs [B, 2] where col 0 is Depth/Crater and col 1 is Rise/Elevation.
        """
        pixel_values = pixel_values.to(self.device)
        angle_feats = angle_feats.to(self.device)

        all_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(pixel_values, angle_feats)
                probs = torch.softmax(logits, dim=-1)

                if apply_tta:
                    # Physics-preserving horizontal flip TTA (width is dim -1)
                    flipped_pixels = torch.flip(pixel_values, dims=[-1])
                    logits_flip = model(flipped_pixels, angle_feats)
                    probs_flip = torch.softmax(logits_flip, dim=-1)
                    probs = (probs + probs_flip) / 2.0

                all_probs.append(probs)

            # Average across all 3 models in the ensemble
            ens_probs = torch.stack(all_probs, dim=0).mean(dim=0)
            return ens_probs

    def predict_image(self, image_input, azimuth_deg, apply_tta=True):
        """
        Runs full inference on a single image in one go.
        Returns: dict with label (0 or 1), class_name ('Depth' or 'Rise'), and probabilities.
        """
        img_tensor, angle_feat = preprocess_lunar_image(image_input, azimuth_deg)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        angle_feat = angle_feat.unsqueeze(0).to(self.device)

        probs = self.forward(img_tensor, angle_feat, apply_tta=apply_tta).squeeze(0)
        p_depth = probs[0].item()
        p_rise = probs[1].item()

        label = 1 if p_rise >= self.threshold else 0
        class_name = "Rise / Elevation" if label == 1 else "Depth / Crater"

        return {
            "label": label,
            "class_name": class_name,
            "prob_depth": p_depth,
            "prob_rise": p_rise
        }

    def predict_dataset(self, metadata_df, img_dir, batch_size=32, apply_tta=True):
        """
        Predicts an entire dataset using the ensemble in one go.
        Returns: pd.DataFrame with ['image_id', 'label', 'prob_rise', 'prob_depth']
        """
        loader = DataLoader(
            LunarInferenceDataset(metadata_df, img_dir),
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

    def save_unified_bundle(self, save_path="lunar_ensemble_unified.pth"):
        """
        Saves the entire 3-fold ensemble into ONE single weight file!
        """
        torch.save({
            "ensemble_state_dict": self.state_dict(),
            "threshold": self.threshold,
            "num_models": len(self.models),
            "architecture": "MultimodalLunarViT (3-Fold Stratified Generalizer)"
        }, save_path)
        print(f"Saved unified ensemble checkpoint ({len(self.models)} models in 1 file) to: {save_path}")

    @classmethod
    def load_unified_bundle(cls, bundle_path, device=None):
        """
        Loads the entire ensemble from a single unified bundle file in one line!
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(bundle_path, map_location=device)
        num_models = checkpoint.get("num_models", 3)
        threshold = checkpoint.get("threshold", 0.768)

        ensemble = cls(weight_paths=[], threshold=threshold, device=device)
        ensemble.models = nn.ModuleList([MultimodalLunarViT() for _ in range(num_models)])
        ensemble.load_state_dict(checkpoint["ensemble_state_dict"])
        ensemble.to(device)
        ensemble.eval()
        print(f"Successfully loaded Unified Ensemble from: {bundle_path}")
        return ensemble

if __name__ == "__main__":
    print("Initializing Lunar Unified Ensemble...")
    ensemble = LunarEnsemble()
    print("Ensemble successfully loaded!")
