import os
import glob
import nibabel as nib
import numpy as np

print("Script started")

pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/gt_flat/"

pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
print(f"Found {len(pred_files)} predicted files in: {pred_dir}")

for pred_path in pred_files[:3]:
    base = os.path.basename(pred_path).replace(".nii.gz", "")
    gt_file = os.path.join(gt_dir, base + "_0000.nii.gz")

    print(f"→ {base} | GT: {os.path.basename(gt_file)}")

    if not os.path.exists(gt_file):
        print(f"  🚫 Missing ground truth: {gt_file}")
        continue

    pred = nib.load(pred_path).get_fdata() > 0
    gt = nib.load(gt_file).get_fdata() > 0

    print(f"  ✅ Pred unique: {np.unique(pred)} | GT unique: {np.unique(gt)}")
