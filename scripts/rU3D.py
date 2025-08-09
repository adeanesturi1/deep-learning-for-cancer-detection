import os
import pandas as pd

# --- 1. SETTINGS (Please double-check these paths) ---
# Training Data Paths
gt_tr_dir       = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
mri_tr_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"

# Validation Data Paths
gt_val_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/labelsVal"
mri_val_dir     = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesVal"

# Prediction and CSV Paths
pred_dir_base   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres"
dice_scores_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dicescores_val_brats.csv"

# --- 2. SCRIPT TO CHECK FILE EXISTENCE ---
try:
    df = pd.read_csv(dice_scores_csv)
except FileNotFoundError:
    print(f"❌ ERROR: Dice scores CSV not found at: {dice_scores_csv}")
    exit()

print("--- Checking file availability for all patients in CSV ---\n")

all_files_found = 0
missing_files = 0

for index, row in df.iterrows():
    patient_id = row['patient_id']
    fold = row['fold']
    
    # Construct all possible paths
    pred_path = os.path.join(pred_dir_base, fold, "validation", f"{patient_id}.nii.gz")
    
    gt_path_tr = os.path.join(gt_tr_dir, f"{patient_id}.nii.gz")
    gt_path_val = os.path.join(gt_val_dir, f"{patient_id}.nii.gz")
    
    mri_path_tr = os.path.join(mri_tr_dir, f"{patient_id}_0001.nii.gz")
    mri_path_val = os.path.join(mri_val_dir, patient_id, f"{patient_id}_0001.nii.gz") # For BraTS structure

    # Check existence
    pred_ok = os.path.exists(pred_path)
    gt_ok = os.path.exists(gt_path_tr) or os.path.exists(gt_path_val)
    mri_ok = os.path.exists(mri_path_tr) or os.path.exists(mri_path_val)

    if pred_ok and gt_ok and mri_ok:
        status = "OK"
        all_files_found += 1
    else:
        status = f" MISSING (Pred: {pred_ok}, GT: {gt_ok}, MRI: {mri_ok})"
        missing_files += 1
        
    print(f"Patient: {patient_id.ljust(20)} (Fold: {fold}) - Status: {status}")

print("\n--- End of Report ---")
print(f"Found all necessary files for {all_files_found} patients.")
print(f"Could not find all files for {missing_files} patients.")
print("\nPlease check the 'MISSING' entries and ensure the corresponding MRI/GT files exist in the correct Tr/Val directories.")