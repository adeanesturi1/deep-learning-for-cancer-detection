import os
import time
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

def dice_score(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    intersection = np.sum(pred * true)
    return 2. * intersection / (np.sum(pred) + np.sum(true) + 1e-6)

pred_folder = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats"
label_root = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/labelsVal"
output_csv = os.path.join(pred_folder, "dicescores_val_brats.csv")

dice_records = []

fold_dirs = sorted([f for f in os.listdir(pred_folder) if f.startswith("fold_") and os.path.isdir(os.path.join(pred_folder, f))])

for fold in fold_dirs:
    fold_path = os.path.join(pred_folder, fold)
    pred_files = [f for f in os.listdir(fold_path) if f.endswith(".nii.gz")]

    for file in pred_files:
        patient_id = file.replace(".nii.gz", "")
        pred_path = os.path.join(fold_path, file)
        label_path = os.path.join(label_root, file)

        try:
            start = time.time()
            pred_img = nib.load(pred_path).get_fdata()
            label_img = nib.load(label_path).get_fdata()

            # Convert to binary
            pred_bin = (pred_img > 0).astype(np.uint8)
            label_bin = (label_img > 0).astype(np.uint8)

            dice = dice_score(pred_bin, label_bin)
            end = time.time()

            print(f"Processing {file} in {fold}... Done in {end - start:.2f}s")
            dice_records.append({"patient_id": patient_id, "fold": fold, "dice": round(dice, 4)})

        except Exception as e:
            print(f"Skipped {file} in {fold} due to error: {e}")

df = pd.DataFrame(dice_records)
df.to_csv(output_csv, index=False)
print(f"\n Saved: {output_csv} with {len(df)} rows")
