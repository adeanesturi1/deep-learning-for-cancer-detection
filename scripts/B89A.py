import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import directed_hausdorff
from matplotlib.lines import Line2D
from sklearn.metrics import f1_score

# === Settings ===
patients = [
    "BraTS2021_00554",
    "BraTS2021_00231",
    "BraTS2021_01220"
]
fold = 3

pred_base = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/fold_3"
gt_base = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
img_base = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesVal"
outdir = os.path.join(pred_base, "hausdorff_slice_visuals_zoomed_by_class")
os.makedirs(outdir, exist_ok=True)

class_labels = {1: "NCR", 2: "ED", 3: "ET"}

def get_surface(mask):
    eroded = binary_erosion(mask)
    return np.argwhere(mask & ~eroded)

def hausdorff_2d(gt_slice, pred_slice):
    A = get_surface(gt_slice > 0)
    B = get_surface(pred_slice > 0)
    if len(A) == 0 or len(B) == 0:
        return np.nan
    d1 = directed_hausdorff(A, B)[0]
    d2 = directed_hausdorff(B, A)[0]
    return max(d1, d2)

def dice_coef(gt, pred):
    gt_flat = gt.flatten()
    pred_flat = pred.flatten()
    if gt_flat.sum() == 0 and pred_flat.sum() == 0:
        return 1.0
    return f1_score(gt_flat, pred_flat)

def get_bbox_union(gt_slice, pred_slice):
    union_mask = (gt_slice > 0) | (pred_slice > 0)
    coords = np.argwhere(union_mask)
    if coords.size == 0:
        return 0, gt_slice.shape[0], 0, gt_slice.shape[1]
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    pad = 5
    return max(y_min - pad, 0), min(y_max + pad, gt_slice.shape[0]), max(x_min - pad, 0), min(x_max + pad, gt_slice.shape[1])

# === Main loop ===
for pid in patients:
    pred_path = os.path.join(pred_base, f"{pid}.nii.gz")
    gt_path = os.path.join(gt_base, f"{pid}.nii.gz")
    img_path = os.path.join(img_base, f"{pid}_0000.nii.gz")

    if not os.path.exists(pred_path) or not os.path.exists(gt_path) or not os.path.exists(img_path):
        print(f"Skipping {pid} — missing file")
        continue

    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    img = nib.load(img_path).get_fdata()

    slice_range = range(pred.shape[2]//2 - 2, pred.shape[2]//2 + 3)

    for class_id, class_name in class_labels.items():
        fig, axes = plt.subplots(1, len(slice_range), figsize=(18, 4))
        fig.suptitle(f"{pid} | Fold {fold} | Class {class_name} (Label {class_id})", fontsize=13)

        for i, z in enumerate(slice_range):
            gt_slice = (gt[:, :, z] == class_id)
            pred_slice = (pred[:, :, z] == class_id)
            img_slice = img[:, :, z]

            hd = hausdorff_2d(gt_slice, pred_slice)
            dice = dice_coef(gt_slice, pred_slice)

            ymin, ymax, xmin, xmax = get_bbox_union(gt_slice, pred_slice)
            ax = axes[i]
            ax.imshow(img_slice, cmap='gray')

            ax.contour(gt_slice, colors='green', linewidths=1.5, linestyles='solid')
            ax.contour(pred_slice, colors='red', linewidths=1.5, linestyles='dashed')

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymax, ymin)
            ax.set_title(f"Slice {z}\nDice={dice:.2f}, HD={hd:.2f}", fontsize=9)
            ax.axis('off')

            if i == 0:
                legend_lines = [
                    Line2D([0], [0], color='green', lw=2, linestyle='solid', label='GT'),
                    Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Prediction')
                ]
                ax.legend(handles=legend_lines, loc='lower right', fontsize=6)

        plt.tight_layout()
        outfile = os.path.join(outdir, f"{pid}_fold{fold}_class{class_id}_{class_name}.png")
        plt.savefig(outfile)
        plt.close()

print(f"Saved zoomed MRI + class contours (with Dice + HD) to: {outdir}")
