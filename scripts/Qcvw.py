import os
import numpy as np
import pandas as pd
import nibabel as nib
from medpy.metric import binary
import time

base_pred_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats"
base_gt_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/labelsVal"
output_csv = os.path.join(base_pred_path, "hausdorff_sample.csv")

results = []

# folds 0 to 4
for fold in range(5):
    fold_path = os.path.join(base_pred_path, f"fold_{fold}")
    if not os.path.isdir(fold_path):
        continue

    count = 0
    for fname in sorted(os.listdir(fold_path)):
        if not fname.endswith(".nii.gz"):
            continue

        patient_id = fname.replace(".nii.gz", "")
        pred_file = os.path.join(fold_path, fname)
        gt_file = os.path.join(base_gt_path, f"{patient_id}.nii.gz")

        if not os.path.exists(gt_file):
            print(f"Missing GT for {patient_id}")
            continue

        try:
            pred_img = nib.load(pred_file).get_fdata()
            gt_img = nib.load(gt_file).get_fdata()

            pred_bin = pred_img > 0
            gt_bin = gt_img > 0

            if np.sum(gt_bin) == 0 or np.sum(pred_bin) == 0:
                continue  #  empty segmentations

            start = time.time()
            hd = binary.hd(pred_bin, gt_bin)
            duration = time.time() - start

            print(f"Processed {patient_id} in fold {fold} — HD: {hd:.2f} in {duration:.2f}s")
            results.append({
                "patient_id": patient_id,
                "fold": fold,
                "hausdorff_distance": hd
            })
            count += 1
            if count >= 10:
                break

        except Exception as e:
            print(f"Error with {patient_id}: {e}")
            continue

# csv
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved: {output_csv} with {len(df)} rows")
