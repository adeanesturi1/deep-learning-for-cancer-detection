import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm


gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/labelsTr"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats"
output_csv_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dice_per_class.csv"


pred_files = glob(os.path.join(pred_dir, "fold_*", "*.nii.gz"))
if not pred_files:
    pred_files = glob(os.path.join(pred_dir, "*.nii.gz"))
print(f"Found {len(pred_files)} prediction files.")

def dice_score(gt, pred):
    """
    Calculates the Dice similarity coefficient between two binary masks.
    Returns np.nan if both masks are empty.
    """
    # Ensure inputs are boolean
    gt = gt > 0
    pred = pred > 0
    
    intersection = np.sum(gt & pred)
    total = np.sum(gt) + np.sum(pred)
    
    if total == 0:
        # Both masks are empty, perfect agreement
        return 1.0
    
    return (2. * intersection) / total

# --- Main Processing Loop ---
records = []

# Use tqdm for a progress bar
for pred_path in tqdm(pred_files, desc="Calculating Dice Scores"):
    # Extract patient ID from the filename
    pid = os.path.basename(pred_path).replace(".nii.gz", "")
    gt_path = os.path.join(gt_dir, f"{pid}.nii.gz")
    
    # Skip if the corresponding ground truth file doesn't exist
    if not os.path.exists(gt_path):
        print(f"Warning: Ground truth not found for {pid}, skipping.")
        continue

    # Load the NIfTI files
    pred_nii = nib.load(pred_path)
    pred = pred_nii.get_fdata().astype(np.uint8)
    
    gt_nii = nib.load(gt_path)
    gt = gt_nii.get_fdata().astype(np.uint8)

    # --- Define Tumor Regions based on BraTS labels ---
    # Label 1: Necrotic and non-enhancing tumor core (NCR)
    # Label 2: Peritumoral edema (ED)
    # Label 4: GD-enhancing tumor (ET)

    # Whole Tumor (WT): Includes all tumor regions (Labels 1, 2, 4)
    wt_gt = gt > 0
    wt_pred = pred > 0

    # Tumor Core (TC): Includes enhancing and non-enhancing core (Labels 1, 4)
    tc_gt = np.isin(gt, [1, 4])
    tc_pred = np.isin(pred, [1, 4])

    # Enhancing Tumor (ET): Only the enhancing core (Label 4)
    et_gt = gt == 4
    et_pred = pred == 4
    
    # --- FIX: Add individual sub-regions ---
    # Necrotic & Non-Enhancing Tumor (NCR): (Label 1)
    ncr_gt = gt == 1
    ncr_pred = pred == 1
    
    # Edema (ED): (Label 2)
    ed_gt = gt == 2
    ed_pred = pred == 2

    # --- Calculate Dice Scores for all regions ---
    # The function returns 1.0 if both GT and Pred are empty for a class.
    records.append({
        "patient_id": pid,
        "WT_Dice": dice_score(wt_gt, wt_pred),
        "TC_Dice": dice_score(tc_gt, tc_pred),
        "ET_Dice": dice_score(et_gt, et_pred),
        "NCR_Dice": dice_score(ncr_gt, ncr_pred),
        "ED_Dice": dice_score(ed_gt, ed_pred)
    })

if records:
    df = pd.DataFrame(records)
    df.set_index("patient_id", inplace=True)
    df.to_csv(output_csv_path)
    print(f"\nDice scores for {len(df)} patients saved to {output_csv_path}")
    print("\nMean Dice Scores:")
    print(df.mean())
else:
    print("\nNo records were generated. Please check your file paths and data.")

