# The Pareidolia Paradox 🌕

A machine learning pipeline for classifying lunar surface crops (256x256 grayscale) into **Depth** (craters, holes) and **Rise** (mounds, hills, rocks). 

Built to solve the paradox where lighting geometry tricks computer vision models: shadows flip based on the sun's position!

## 💡 The Physics Clue
The dataset includes the `sun_azimuth_angle` for each image. To prevent the model from being tricked by shadow directions, we dynamically normalize the illumination geometry. Before being fed into the neural network, every image is rotated by `-sun_azimuth_angle`, aligning the sun's rays to a consistent 0-degree angle.

## 🚀 Model & Approach
- **Base Architecture**: Google Vision Transformer (`google/vit-base-patch16-224-in21k`).
- **Fine-Tuning Strategy**: LoRA (Low-Rank Adaptation) via Hugging Face `peft`, allowing fast, parameter-efficient training on a single GPU.
- **Environment**: Fully automated training pipeline for AWS EC2 `g4dn.xlarge` instances (NVIDIA T4 GPU).

## 📂 Repository Structure
```
lunar-pareidolia-paradox/
├── train_lunar.py         # Main PyTorch training and inference script
├── aws_train_lunar.ps1    # PowerShell script for automated EC2 execution
├── README.md              # You are here
└── .gitignore             
```

## 🛠 Usage
### 1. Training on AWS EC2
Use the `aws_train_lunar.ps1` script to automate everything. It will:
1. Spin up a `g4dn.xlarge` GPU instance using your AWS profile.
2. Upload the local training datasets (`train_images.zip`, `eval_images.zip`, and metadata CSVs).
3. Execute the `train_lunar.py` script.
4. Download the `submission.csv` and trained model weights locally.
5. Terminate the EC2 instance to prevent idle billing.

```powershell
powershell -ExecutionPolicy Bypass -File .\aws_train_lunar.ps1
```

### 2. Running Locally
If you have a local GPU, extract your dataset to `./train_images/` and `./eval_images/` and run:
```bash
pip install transformers peft timm scikit-learn pandas tqdm
python train_lunar.py           # Trains the model
python train_lunar.py predict   # Runs inference
```

## 📊 Experiments & Results

| Architecture | Approach | Backbone Status | Best Val Balanced Acc | Epoch Achieved |
|--------------|----------|-----------------|-----------------------|----------------|
| `google/vit-base-patch16-224-in21k` | ViT + LoRA (`all-linear`) | Frozen | **74.53%** | 1 |
| `microsoft/resnet-50` | ResNet50 + LoRA (`convolution`) | Frozen | **72.47%** | 4 |

The model optimizes for **Balanced Accuracy** across both the "Rise" and "Depth" classes to ensure minority features aren't overpowered. Output predictions are saved to `submission.csv`.
