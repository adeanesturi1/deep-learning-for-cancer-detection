import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure
from matplotlib.colors import ListedColormap

# === Config ===
dice_csv = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/per_patient_analysis"
os.makedirs(out_dir, exist_ok=True)

# === Load data and select top 3 patients ===
df = pd.read_csv(os.path.join(out_dir, "dice_with_hausdorff.csv"))
metadata_df = pd.read_csv(metadata_csv)
metadata_df = metadata_df.rename(columns={"nnUNet_ID": "Patient"})
df = pd.merge(df, metadata_df[["Patient", "HER2_Status"]], on="Patient", how="left")
top3 = df.sort_values("Dice", ascending=False).head(3)


top3 = df.sort_values("Dice", ascending=False).head(3)

def plot_patient_contour(pid, dice, hd, her2, out_path):
    try:
        img = nib.load(os.path.join(image_dir, f"{pid}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{pid}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{pid}.nii.gz")).get_fdata()

        z_slices = [img.shape[2] // 2 - 5, img.shape[2] // 2, img.shape[2] // 2 + 5]
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 1, 0.7]})

        for i, z in enumerate(z_slices):
            img_slice = img[:, :, z]
            gt_slice = gt[:, :, z]
            pred_slice = pred[:, :, z]

            img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

            fp = (pred_slice > 0.5) & (gt_slice == 0)
            fn = (pred_slice <= 0.5) & (gt_slice > 0)

            axs[i].imshow(img_slice, cmap="gray")
            axs[i].imshow(fp, cmap=ListedColormap(["none", "red"]), alpha=0.4)
            axs[i].imshow(fn, cmap=ListedColormap(["none", "blue"]), alpha=0.4)

            for c in measure.find_contours(gt_slice, 0.5):
                axs[i].plot(c[:, 1], c[:, 0], color='red', linewidth=1.5, label="GT" if i == 0 else "")
            for c in measure.find_contours(pred_slice, 0.5):
                axs[i].plot(c[:, 1], c[:, 0], color='lime', linewidth=1.5, label="Prediction" if i == 0 else "")

            axs[i].set_title(f"Slice {z}")
            axs[i].axis('off')

        # Side panel with metrics
        axs[3].axis('off')
        axs[3].text(0, 0.8, f"Patient: {pid}", fontsize=12)
        axs[3].text(0, 0.6, f"Dice: {dice:.2f}", fontsize=12)
        axs[3].text(0, 0.4, f"Hausdorff: {hd:.2f}", fontsize=12)
        axs[3].text(0, 0.2, f"HER2: {her2}", fontsize=12)

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"{pid} | Dice = {dice:.2f} | Hausdorff = {hd:.2f} | HER2 = {her2}", fontsize=14)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Error for {pid}: {e}")

# === Generate visualisations for top 3
for _, row in top3.iterrows():
    pid = row["Patient"]
    dice = row["Dice"]
    hd = row["Hausdorff"]
    her2 = row["HER2_Status"]
    out_path = os.path.join(out_dir, f"contour_triplet_{pid}.png")
    plot_patient_contour(pid, dice, hd, her2, out_path)
