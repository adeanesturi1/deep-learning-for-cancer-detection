import os
import nibabel as nib
import numpy as np
import pandas as pd
from medpy.metric import binary
from tqdm import tqdm

print("Script started")

# ====== PATH CONFIGURATION ======
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/gt_flat"
output_csv = "bcbm_fold3_val_dice.csv"

# ====== SCAN FOR FILES ======
pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".nii.gz"))
print(f"Found {len(pred_files)} predicted files in: {pred_dir}")

results = []

# ====== LOOP OVER PREDICTIONS ======
for pred_file in tqdm(pred_files):
    pred_path = os.path.join(pred_dir, pred_file)
    gt_filename = pred_file.replace(".nii.gz", "_0000.nii.gz")
    gt_path = os.path.join(gt_dir, gt_filename)

    if not os.path.exists(gt_path):
        print(f"  🚫 Missing ground truth: {gt_path}")
        continue

    # Load NIfTI images
    pred_img = nib.load(pred_path).get_fdata() > 0  # binary mask
    gt_img = nib.load(gt_path).get_fdata() > 0      # binary mask

    # Skip empty cases if needed
    if not pred_img.any() and not gt_img.any():
        dice_score = 1.0
    else:
        dice_score = binary.dc(pred_img, gt_img)

    results.append({
        "Patient": pred_file.replace(".nii.gz", ""),
        "Dice": round(dice_score, 4)
    })

# ====== SAVE RESULTS ======
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to {output_csv}")
