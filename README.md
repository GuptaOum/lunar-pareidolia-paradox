# The Pareidolia Paradox: Lunar Surface Feature Classification

An end-to-end Computer Vision pipeline designed to solve **The Pareidolia Paradox** (IEEE SIES GST) by classifying 256 × 256 grayscale lunar surface images into two distinct physical geological categories:
* **Class 0 (Depression / Depth):** Craters, holes, and surface depressions.
* **Class 1 (Rise / Elevation):** Mounds, hills, rocks, and boulders.

---

## 🚀 Performance Highlights

| Metric | Score | Validation Context |
| :--- | :--- | :--- |
| **Ensemble Balanced Accuracy** | **89.93%** | Final multi-seed consensus evaluation |
| **Raw Accuracy** | **90.00%** | Unseen 800-sample test set |
| **Crater Recall (Class 0)** | **89.7%** | Solves majority-class bias completely |
| **Hill Recall (Class 1)** | **90.2%** | Preserves high precision on elevated features |
| **Single-Model Balanced Val Acc** | **82.36%** | Best standalone ViT checkpoint (Val Loss: 0.4838) |
| **Stability Across Cycles** | **87.62% ± 0.91%** | Verified over 3 independent 600-sample cycles (1,800 images) |

---

## 📈 Ablation & Benchmark Progression

How we systematically engineered the pipeline to surpass the 80% and 90% accuracy barriers:

| Stage | Strategy | Balanced Val Accuracy | Impact / Observations |
| :--- | :--- | :---: | :--- |
| 1 | Baseline ViT (Standard sampling) | 77.12% | Heavy bias toward majority class (Hills 63.7%). Crater recall was low. |
| 2 | ViT + LoRA (r=16, alpha=32) + Class Weights | 79.40% | Improved crater recall, but loss landscape was noisy with high variance. |
| 3 | ViT + LoRA + **50/50 Balanced Resampling** | **82.36%** | Balanced gradient backpropagation; single-model breakthrough. |
| 4 | **2-Seed Soft-Voting Ensemble (Champion)** | **89.93%** | Independent weight trajectories cancel out fringe edge-case noise. |

---

## 🧠 Methodology & Architectural Innovation

### 1. Physics-Informed Solar Lighting Normalization
Lunar shadows invert depending on the illumination angle, creating optical illusions where craters appear convex (hills) and hills appear concave (craters).

* **Rotation Angle Correction:** Each image is dynamically rotated counter-clockwise by `-sun_azimuth_angle` using bilinear interpolation:

$$\theta_{\text{corrected}} = -\theta_{\text{azimuth}}$$

* **Physical Invariance:** This rotation mathematically fixes the sunlight direction directly to the **North (Top)** across every single image in the dataset.
* **Physics Preservation:** Uncontrolled spatial flips (e.g., standard horizontal/vertical flips) were strictly omitted because they invert the shadow-casting geometry and violate physical illumination laws.

### 2. Deep Transformer Architecture with LoRA
* **Backbone:** Google Vision Transformer (`google/vit-base-patch16-224-in21k`) pre-trained on ImageNet-21k.
* **Parameter-Efficient Fine-Tuning (PEFT):** Low-Rank Adaptation (LoRA) applied across all linear attention projections (`target_modules="all-linear"`):
  * **Rank (`r`):** 16
  * **Scaling Factor (`α`):** 32
  * **LoRA Dropout:** 0.10
* **Optimization & Regularization:**
  * **Optimizer:** AdamW (`learning_rate = 1e-3`, `weight_decay = 0.01`)
  * **Scheduler:** `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`
  * **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(max_norm=1.0)` to safeguard against gradient instability during deep layer backpropagation.

### 3. Class Imbalance Resolution (50/50 Resampling)
The training dataset exhibits natural physical imbalance (**5,000 Hills / 63.7% vs. 2,854 Craters / 36.3%**). Standard cross-entropy loss causes models to converge towards hill-biased local minima.

* **Symmetric Resampling:**
  * Oversampled Class 0 (Craters) to exactly **3,500 samples**.
  * Subsampled Class 1 (Hills) to exactly **3,500 samples**.
  * Equalized gradient signals across epochs, elevating crater detection sensitivity to **89.7%**.

### 4. Multi-Seed Soft-Voting Ensemble
* Multiple independent models were trained with distinct initializations (Seed 42, Seed 123).
* Prediction probabilities are aggregated through soft-voting consensus:

$$P_{\text{ensemble}}(y=c) = \frac{1}{M} \sum_{m=1}^{M} P_{m}(y=c)$$

* The ensemble filters out ambiguous border-condition topography, pushing final balanced accuracy to **89.93%**.

---

## 📁 Repository Structure

```
├── train.py                       # Main competition entrypoint for balanced ViT+LoRA training
├── train_balanced_ensemble.py     # Multi-seed balanced ensemble training pipeline
├── train_lunar.py                 # Core ViT + LoRA model definitions
├── inference.py                   # High-throughput ensemble inference engine
├── test_balanced_3cycles.py       # 3-cycle cross-validation validation script
├── submission.csv                 # Final competition submission (2,000 predictions)
├── requirements.txt               # Complete Python dependencies
└── README.md                      # Methodology, ablation benchmarks, and replication guide
```

---

## 📦 Model Weights & Checkpoints

The trained LoRA adapter weights for the ensemble models are available for direct public download:
* **Release Download:** [v1.0.0 Model Weights (model_weights.zip)](https://github.com/GuptaOum/lunar-pareidolia-paradox/releases/download/v1.0.0/model_weights.zip)
* **Direct Checkpoint (.pth):** [best_model.pth](https://github.com/GuptaOum/lunar-pareidolia-paradox/releases/download/v1.0.0/best_model.pth)
* **Adapter Format:** HuggingFace PEFT / SafeTensors (`adapter_model.safetensors`, `adapter_config.json`)
* **Base Architecture:** `google/vit-base-patch16-224-in21k`

---

## 🛠️ Reproduction & Training

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Train the Balanced Ensemble
```bash
python train.py
```

### 3. Generate Competition Predictions
```bash
python inference.py
```
This produces `submission.csv` containing the final predictions formatted strictly according to competition requirements (`image_id,label`).

---

## 👥 Authors
* **Team:** Oum Gupta
* **Competition:** The Pareidolia Paradox | IEEE SIES GST
