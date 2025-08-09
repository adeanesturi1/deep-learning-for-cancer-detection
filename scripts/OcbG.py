import numpy as np, nibabel as nib, os, glob

def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    return 2. * intersection / (np.sum(pred) + np.sum(gt) + 1e-8)

gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation"

for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
    case_id = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, case_id.replace('.nii.gz', '_0000.nii.gz'))

    if not os.path.exists(gt_path):
        print(f"[MISSING GT] {gt_path}")
        continue

    pred = nib.load(pred_path).get_fdata() > 0
    gt = nib.load(gt_path).get_fdata() > 0

    print(f"[INFO] {case_id}: pred unique = {np.unique(pred)}, gt unique = {np.unique(gt)}")

    if np.sum(gt) == 0 and np.sum(pred) == 0:
        print(f"{case_id}: Dice = N/A (both empty)")
    else:
        dice = dice_score(pred, gt)
        print(f"{case_id}: Dice = {dice:.4f}")
