import os
import json
import shutil

# Paths
split_json = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/splits_final.json"
predictions_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions"
output_base = predictions_dir  # we will create val_brats/fold_0 etc

# Load splits
with open(split_json, "r") as f:
    splits = json.load(f)

# Create folders and move predictions
for fold_idx, split in enumerate(splits):
    fold_dir = os.path.join(output_base, f"fold_{fold_idx}")
    os.makedirs(fold_dir, exist_ok=True)

    for case_id in split["val"]:
        src = os.path.join(predictions_dir, f"{case_id}.nii.gz")
        dst = os.path.join(fold_dir, f"{case_id}.nii.gz")
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"[WARNING] Missing prediction: {src}")
