import numpy as np, nibabel as nib, glob, os

def dice_score(pred, gt):
    intersection = np.sum(pred * gt)
    return 2.0 * intersection / (np.sum(pred) + np.sum(gt) + 1e-8)

gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation"

for pred_path in sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz"))):
    case_id = os.path.basename(pred_path)
    gt_path = os.path.join(gt_dir, case_id)
    
    if os.path.exists(gt_path):
        pred = nib.load(pred_path).get_fdata() > 0
        gt = nib.load(gt_path).get_fdata() > 0
        dice = dice_score(pred, gt)
        print(f"{case_id}: Dice = {dice:.4f}")
    else:
        print(f"Missing GT for {case_id}")
