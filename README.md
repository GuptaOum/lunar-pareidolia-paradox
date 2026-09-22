# The Pareidolia Paradox: Lunar Surface Feature Classification

An end-to-end Computer Vision pipeline designed to solve **The Pareidolia Paradox** by classifying $256 \times 256$ grayscale lunar surface images into two distinct physical geological categories:
- **Class 0 (Depth):** Craters, holes, and surface depressions.
- **Class 1 (Rise):** Mounds, hills, rocks, and boulders.

---

## 🚀 Performance Highlights
* **Peak Single-Model Validation Accuracy:** **82.36% Balanced Accuracy** (Val Loss: 0.4838)
* **Final Ensemble Evaluation Score:** **89.93% Balanced Accuracy** (90.00% Raw Accuracy)
* **Crater Recall (Class 0):** **89.7%**
* **Hill Recall (Class 1):** **90.2%**
* **Evaluated Across:** Over 2,400 multi-sample validation images with low variance ($\pm 0.91\%$).

---

## 🧠 Methodology & Architectural Innovation

### 1. Physics-Informed Lighting Normalization
Lunar shadows invert based on the Sun's position relative to the camera, creating the optical illusion known as **pareidolia** (craters appearing as hills and vice versa).
* To eliminate lighting ambiguity, each image is dynamically rotated counter-clockwise by **`-sun_azimuth_angle`** using high-precision bilinear interpolation:
  $$\theta_{\text{corrected}} = -\theta_{\text{sun\_azimuth}}$$
* This mathematically fixes the illumination source directly to the **North (Top)** across the entire dataset.
* Destructive spatial augmentations (e.g., random vertical/horizontal flips) were avoided because they violate solar shadow physics.

### 2. Deep Transformer Architecture with LoRA
* **Backbone:** Google Vision Transformer (`google/vit-base-patch16-224-in21k`) pre-trained on ImageNet-21k.
* **Parameter-Efficient Fine-Tuning (PEFT):** Low-Rank Adaptation (**LoRA**) applied across all linear attention projections:
  - LoRA Rank: $r = 16$
  - Scaling Factor: $\alpha = 32$
  - Dropout: $p = 0.10$
* **Optimization & Regularization:**
  - Optimizer: AdamW ($\text{LR} = 1\times 10^{-3}$, $\text{weight\_decay} = 0.01$)
  - Scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`
  - Gradient Clipping: `max_norm=1.0` to prevent catastrophic gradient explosions in deep transformer blocks.

### 3. Class Imbalance Resolution (50/50 Resampling)
The raw dataset is naturally imbalanced (**63.7% Hills vs. 36.3% Craters**), which causes standard classifiers to bias toward hills and perform poorly on the competition's **Balanced Accuracy** metric.
* **Balanced 50/50 Resampling:**
  - Class 0 (Craters) was oversampled to exactly **3,500 images**.
  - Class 1 (Hills) was downsampled to exactly **3,500 images**.
  - Equalized gradient pressure during backpropagation, boosting crater sensitivity to **89.7%**.

### 4. Multi-Seed Soft-Voting Ensemble
* Trained multiple independent models initialized with distinct random seeds (Seed 42, Seed 123).
* Combined prediction probabilities via **Soft-Voting**:
  $$P_{\text{ensemble}}(y=c) = \frac{1}{M} \sum_{m=1}^M P_m(y=c)$$
* Multi-model consensus eliminates individual outliers and pushes overall balanced accuracy to **89.93%**.

---

## 📁 Repository Structure
```
├── train_balanced_ensemble.py     # Main 50/50 balanced training & ensemble script
├── train_lunar.py                 # Multi-seed ViT+LoRA training pipeline
├── inference.py                   # Fast multi-model ensemble inference generator
├── test_balanced_3cycles.py       # 3-cycle cross-validation verification script
├── submission.csv                 # Final competition submission (2,000 predictions)
└── README.md                      # Comprehensive methodology documentation
```

---

## 🛠️ Reproduction & Training

### Environment Setup
```bash
pip install torch torchvision transformers peft pillow pandas numpy scikit-learn
```

### Train Balanced Ensemble
```bash
python train_balanced_ensemble.py
```

### Generate Predictions
```bash
python inference.py
```

---

## 👥 Authors
* **Team:** Oum Gupta
* **Event:** The Pareidolia Paradox | IEEE SIES GST
