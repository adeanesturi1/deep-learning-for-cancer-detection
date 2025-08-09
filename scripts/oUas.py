import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from matplotlib.colors import ListedColormap

# === Config ===
dice_csv = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/per_patient_analysis"
os.makedirs(out_dir, exist_ok=True)

# === Load Data ===
dice_df = pd.read_csv(dice_csv)
metadata_df = pd.read_csv(metadata_csv)
metadata_df = metadata_df.rename(columns={"nnUNet_ID": "Patient"})
merged_df = pd.merge(dice_df, metadata_df[["Patient", "HER2_Status"]], on="Patient", how="left")

# === Select Top/Bottom 3 Patients ===
top3 = merged_df.sort_values("Dice", ascending=False).head(3)
bottom3 = merged_df.sort_values("Dice", ascending=True).head(3)

# === Hausdorff Distance ===
def hausdorff(pred, gt):
    pred_pts = np.argwhere(pred)
    gt_pts = np.argwhere(gt)
    if pred_pts.size == 0 or gt_pts.size == 0:
        return np.nan
    d1 = directed_hausdorff(pred_pts, gt_pts)[0]
    d2 = directed_hausdorff(gt_pts, pred_pts)[0]
    return max(d1, d2)

hausdorff_list = []
for _, row in merged_df.iterrows():
    pid = row["Patient"]
    try:
        pred = nib.load(os.path.join(pred_dir, f"{pid}.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{pid}.nii.gz")).get_fdata()
        hd = hausdorff(pred > 0.5, gt > 0.5)
        hausdorff_list.append(hd)
    except:
        hausdorff_list.append(np.nan)

merged_df["Hausdorff"] = hausdorff_list
merged_df.to_csv(os.path.join(out_dir, "dice_with_hausdorff.csv"), index=False)

# === Visualisation Function ===
def plot_fp_fn_overlay(pid, out_path):
    try:
        img = nib.load(os.path.join(image_dir, f"{pid}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{pid}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{pid}.nii.gz")).get_fdata()

        mid = img.shape[2] // 2
        img_slice = img[:, :, mid]
        gt_slice = gt[:, :, mid]
        pred_slice = pred[:, :, mid]

        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

        fp = (pred_slice > 0.5) & (gt_slice == 0)
        fn = (pred_slice <= 0.5) & (gt_slice > 0)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img_slice, cmap="gray")
        ax.imshow(fn, cmap=ListedColormap(["none", "blue"]), alpha=0.5)
        ax.imshow(fp, cmap=ListedColormap(["none", "red"]), alpha=0.5)
        ax.set_title(f"{pid} FP=Red, FN=Blue")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Failed for {pid}: {e}")

# === Generate Overlays ===
def plot_fp_fn_triplet(pid, out_path):
    try:
        img = nib.load(os.path.join(image_dir, f"{pid}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{pid}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{pid}.nii.gz")).get_fdata()

        z_slices = [img.shape[2] // 2 - 5, img.shape[2] // 2, img.shape[2] // 2 + 5]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for i, z in enumerate(z_slices):
            img_slice = img[:, :, z]
            gt_slice = gt[:, :, z]
            pred_slice = pred[:, :, z]

            img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

            fp = (pred_slice > 0.5) & (gt_slice == 0)
            fn = (pred_slice <= 0.5) & (gt_slice > 0)

            axs[i].imshow(img_slice, cmap="gray")
            axs[i].imshow(fn, cmap=ListedColormap(["none", "blue"]), alpha=0.5)
            axs[i].imshow(fp, cmap=ListedColormap(["none", "red"]), alpha=0.5)
            axs[i].set_title(f"Slice {z}")
            axs[i].axis('off')

        fig.suptitle(f"{pid} FP=Red, FN=Blue", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"Failed for {pid}: {e}")

# === HER2 Boxplot ===
plt.figure(figsize=(7, 5))
sns.boxplot(data=merged_df, x="HER2_Status", y="Dice", palette="Set2")
plt.title("Dice Score by HER2 Status")
plt.xlabel("HER2 Status")
plt.ylabel("Dice Score")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "boxplot_dice_by_HER2.png"))
plt.close()

# === Dice vs HER2 Scatter ===
plt.figure(figsize=(6, 5))
sns.stripplot(data=merged_df, x="HER2_Status", y="Dice", jitter=True, alpha=0.7)
plt.title("Dice Scores vs HER2 Status")
plt.ylabel("Dice Score")
plt.xlabel("HER2 Status")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "scatter_dice_vs_her2.png"))
plt.close()

print(f"Per-patient visualisations and metrics saved to: {out_dir}")
