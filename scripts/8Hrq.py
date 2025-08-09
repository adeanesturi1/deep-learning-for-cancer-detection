import os
import pandas as pd

# --- 1. SETTINGS (These are the paths we need to verify) ---
dice_csv    = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
image_dir   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/nnUNetPlans_3d_fullres"
gt_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
# ---

# --- 2. SCRIPT TO CHECK FILE EXISTENCE ---
try:
    df = pd.read_csv(dice_csv)
except FileNotFoundError:
    print(f"❌ ERROR: Dice scores CSV not found at: {dice_csv}")
    exit()

print("--- Checking file availability for patients in CSV ---\n")

for index, row in df.iterrows():
    # Use the corrected column names 'Patient' and 'Dice'
    patient_id = str(row['Patient']).strip()
    
    # Construct paths
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(image_dir, f"{patient_id}.npz") # Preprocessed files are .npz

    # Check existence of each file
    gt_ok = os.path.exists(gt_path)
    pred_ok = os.path.exists(pred_path)
    mri_ok = os.path.exists(mri_path)

    if pred_ok and gt_ok and mri_ok:
        status = "✅ OK"
    else:
        status = f"❌ MISSING (GT: {gt_ok}, Pred: {pred_ok}, MRI: {mri_ok})"
        
    print(f"Patient: {patient_id.ljust(20)} - Status: {status}")

print("\n--- End of Report ---")
print("This report shows which file type is consistently missing (True = Found, False = Missing).")
print("Please share this output.")