import os
import argparse
import pandas as pd
import torch
from ensemble_model import LunarEnsemble

DEFAULT_TEST_IMG_DIR = "./eval_images"
DEFAULT_TEST_META_CSV = "./test_metadata.csv"
DEFAULT_OUTPUT_CSV = "./submission.csv"

def run_ensemble_inference(img_dir=DEFAULT_TEST_IMG_DIR, meta_csv=DEFAULT_TEST_META_CSV, output_csv=DEFAULT_OUTPUT_CSV):
    print("=" * 70)
    print("  LUNAR 3-FOLD UNIFIED ENSEMBLE INFERENCE ENGINE (v2.0)")
    print("=" * 70)

    if not os.path.exists(meta_csv):
        raise FileNotFoundError(f"Metadata file not found: {meta_csv}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    test_df = pd.read_csv(meta_csv)
    print(f"Loaded test metadata: {len(test_df)} images to predict.")

    # Initialize the unified ensemble (loads all 3 folds into 1 container)
    ensemble = LunarEnsemble(threshold=0.768)

    # Run all 3 model predictions in one go with TTA
    results_df = ensemble.predict_dataset(test_df, img_dir, batch_size=32, apply_tta=True)

    # Save output
    sub_df = results_df[["image_id", "label"]]
    sub_df.to_csv(output_csv, index=False)

    print("\n" + "-" * 70)
    print(f"Saved final unified ensemble predictions to: {output_csv}")
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
