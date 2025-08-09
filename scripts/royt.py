import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from matplotlib.lines import Line2D

# === Settings ===
patients = [
    ("BraTS2021_00000", 3),
    ("BraTS2021_00005", 3),
    ("BraTS2021_00006", 3),
]

# Paths
base_pred = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats"
gt_path_base = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
t1ce_path_base = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesVal"

outdir = os.path.join(base_pred, "hausdorff_visuals_w_mri")
os.makedirs(outdir, exist_ok=True)

class_labels = {1: "NCR", 2: "ED", 3: "ET"}

def get_surface(mask):
    eroded = binary_erosion(mask)
    return np.argwhere(mask & ~eroded)

def hausdorff_2d(gt_slice, pred_slice):
    A = get_surface(gt_slice)
    B = get_surface(pred_slice)
    if A.size == 0 or B.size == 0:
        return np.nan
    d1 = directed_hausdorff(A, B)[0]
    d2 = directed_hausdorff(B, A)[0]
    return max(d1, d2)

def dice_score(gt, pred):
    intersection = np.sum((gt == 1) & (pred == 1))
    return (2. * intersection) / (np.sum(gt == 1) + np.sum(pred == 1) + 1e-8)

def get_bbox_union(gt_slice, pred_slice):
    union_mask = (gt_slice > 0) | (pred_slice > 0)
    coords = np.argwhere(union_mask)
    if coords.size == 0:
        return 0, gt_slice.shape[0], 0, gt_slice.shape[1]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad = 5
    return max(y_min - pad, 0), min(y_max + pad, gt_slice.shape[0]), max(x_min - pad, 0), min(x_max + pad, gt_slice.shape[1])

# === Main Loop ===
for pid, fold in patients:
    pred_path = os.path.join(base_pred, f"fold_{fold}", f"{pid}.nii.gz")
    gt_path = os.path.join(gt_path_base, f"{pid}.nii.gz")
    t1ce_path = os.path.join(t1ce_path_base, f"{pid}_0001.nii.gz")

    if not all(os.path.exists(p) for p in [pred_path, gt_path, t1ce_path]):
        print(f"Skipping {pid} — missing file")
        continue

    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    t1ce = nib.load(t1ce_path).get_fdata()

    slice_range = range(pred.shape[2]//2 - 2, pred.shape[2]//2 + 3)

    for class_id, class_name in class_labels.items():
        fig, axes = plt.subplots(1, len(slice_range), figsize=(18, 4))
        fig.suptitle(f"{pid} — Fold {fold} | Class {class_name} (Label {class_id})", fontsize=13)

        for i, z in enumerate(slice_range):
            gt_slice = (gt[:, :, z] == class_id)
            pred_slice = (pred[:, :, z] == class_id)
            mri_slice = t1ce[:, :, z]

            hd = hausdorff_2d(gt_slice, pred_slice)
            dsc = dice_score(gt_slice, pred_slice)

            ymin, ymax, xmin, xmax = get_bbox_union(gt_slice, pred_slice)
            ax = axes[i]
            ax.imshow(mri_slice, cmap='gray')

            ax.contour(gt_slice, colors='green', linewidths=1.5, linestyles='solid')
            ax.contour(pred_slice, colors='red', linewidths=1.5, linestyles='dashed')

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
            ax.set_title(f"Slice {z}\nHD={hd:.2f} | Dice={dsc:.2f}", fontsize=9)
            ax.axis('off')

            if i == 0:
                custom_lines = [
                    Line2D([0], [0], color='green', lw=2, linestyle='solid', label='GT'),
                    Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Prediction')
                ]
                ax.legend(handles=custom_lines, loc='lower right', fontsize=8)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"{pid}_fold{fold}_class{class_id}_{class_name}.png")
        plt.savefig(outfile, dpi=150)
        plt.close()

print(f"Saved to: {outdir}")
