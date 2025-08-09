import os
import numpy as np
import nibabel as nib
from medpy.metric.binary import dc
import pandas as pd

print("Script started")

pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/gt_flat"
results = []

# Gather predictions
pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
print(f"Found {len(pred_files)} predicted files in: {pred_dir}")

for pred_file in pred_files:
    case_id = pred_file.replace(".nii.gz", "")
    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(gt_dir, pred_file)  # same name assumed

    if not os.path.isfile(gt_path):
        print(f"  🚫 Missing ground truth: {gt_path}")
        continue

    pred = nib.load(pred_path).get_fdata() > 0
    gt = nib.load(gt_path).get_fdata() > 0

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        dice = 1.0
    elif np.sum(gt) == 0 or np.sum(pred) == 0:
        dice = 0.0
    else:
        dice = dc(pred, gt)

    print(f"→ {case_id} | Dice: {dice:.4f}")
    results.append({'patient': case_id, 'dice': dice})

# Save results
df = pd.DataFrame(results)
out_csv = "bcbm_fold3_val_dice.csv"
df.to_csv(out_csv, index=False)
print(f"\n✅ Results saved to {out_csv}")
