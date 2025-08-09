import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib

from matplotlib.colors import ListedColormap
from skimage import measure

# === Config ===
dice_csv = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures"
os.makedirs(out_dir, exist_ok=True)

# === Load Data ===
dice_df = pd.read_csv(dice_csv)
metadata_df = pd.read_csv(metadata_csv)
metadata_df = metadata_df.rename(columns={"nnUNet_ID": "Patient"})
merged_df = pd.merge(dice_df, metadata_df[["Patient", "HER2_Status"]], on="Patient", how="left")
sorted_df = merged_df.sort_values(by="Dice", ascending=False).reset_index(drop=True)

# === Plot 1: Histogram ===
plt.figure(figsize=(10, 5))
sns.histplot(merged_df["Dice"], bins=20, kde=True, color='skyblue', edgecolor='black')
plt.title("Histogram of Dice Scores (BCBM Fold 3)")
plt.xlabel("Dice Score")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "histogram_dice_kde.png"))
plt.close()

# === Plot 2: Sorted Bar Plot ===
plt.figure(figsize=(14, 6))
sns.barplot(x=sorted_df.index, y=sorted_df["Dice"], palette="coolwarm")
plt.xticks([])
plt.title("Per-Patient Dice Scores (Sorted)")
plt.ylabel("Dice Score")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sorted_barplot_dice.png"))
plt.close()

# === Plot 3: Boxplot ===
plt.figure(figsize=(6, 5))
sns.boxplot(y=merged_df["Dice"], color='lightgreen')
plt.title("Boxplot of Dice Scores (BCBM Fold 3)")
plt.ylabel("Dice Score")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "boxplot_dice.png"))
plt.close()

# === Plot 4: Violin Plot by HER2 Status ===
if "HER2_Status" in merged_df.columns and merged_df["HER2_Status"].notna().any():
    plt.figure(figsize=(7, 5))
    sns.violinplot(x="HER2_Status", y="Dice", data=merged_df, hue="HER2_Status", palette="pastel", legend=False)
    plt.title("Dice Score Distribution by HER2 Status")
    plt.xlabel("HER2 Status")
    plt.ylabel("Dice Score")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "violin_dice_by_her2.png"))
    plt.close()

# === Utility: Mid-slice overlay with optional contours ===
def plot_overlay(patient_id, axarr, dice_score=None, title_prefix="", use_contours=True):
    try:
        img = nib.load(os.path.join(image_dir, f"{patient_id}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{patient_id}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{patient_id}.nii.gz")).get_fdata()

        mid = img.shape[2] // 2
        img_slice = img[:, :, mid]
        gt_slice = gt[:, :, mid]
        pred_slice = pred[:, :, mid]
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

        axarr[0].imshow(img_slice, cmap='gray')
        axarr[1].imshow(img_slice, cmap='gray')

        if use_contours:
            axarr[0].contour(gt_slice, levels=[0.5], colors='red')
            axarr[1].contour(pred_slice, levels=[0.5], colors='blue')
        else:
            axarr[0].imshow(gt_slice, cmap=ListedColormap(['none', 'red']), alpha=0.5)
            axarr[1].imshow(pred_slice, cmap=ListedColormap(['none', 'blue']), alpha=0.5)

        title_suffix = f"\nDice={dice_score:.2f}" if dice_score is not None else ""
        axarr[0].set_title(f"{title_prefix}GT{title_suffix}")
        axarr[1].set_title(f"{title_prefix}Prediction")

        for ax in axarr:
            ax.axis('off')
    except Exception as e:
        print(f"Failed to plot {patient_id}: {e}")

# === Plot 5: Top & Bottom 3 with Dice and Legend ===
extreme_df = pd.concat([sorted_df.head(3), sorted_df.tail(3)])
for _, row in extreme_df.iterrows():
    patient_id = row["Patient"]
    dice_score = row["Dice"]
    fig, axarr = plt.subplots(1, 3, figsize=(14, 5))
    plot_overlay(patient_id, axarr[:2], dice_score=dice_score, title_prefix=f"{patient_id} ")

    axarr[2].axis('off')
    axarr[2].plot([], [], color='red', label='Ground Truth')
    axarr[2].plot([], [], color='blue', label='Prediction')
    axarr[2].legend(loc='center')
    axarr[2].set_title("Legend")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"overlay_{patient_id}_with_dice.png"))
    plt.close()

# === Plot 6: Montage of All Patients (max 54) ===
fig, axs = plt.subplots(9, 6, figsize=(18, 28))
axs = axs.flatten()

for i, patient_id in enumerate(sorted_df["Patient"][:54]):
    try:
        img = nib.load(os.path.join(image_dir, f"{patient_id}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{patient_id}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{patient_id}.nii.gz")).get_fdata()

        mid = img.shape[2] // 2
        img_slice = img[:, :, mid]
        gt_slice = gt[:, :, mid]
        pred_slice = pred[:, :, mid]

        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        axs[i].imshow(img_slice, cmap='gray')
        axs[i].contour(pred_slice, levels=[0.5], colors='blue')
        axs[i].contour(gt_slice, levels=[0.5], colors='red')
        dice = sorted_df.loc[sorted_df["Patient"] == patient_id, "Dice"].values[0]
        axs[i].set_title(f"{patient_id}\nDice={dice:.2f}")
        axs[i].axis('off')
    except Exception as e:
        axs[i].axis('off')
        print(f"Could not load: {patient_id} | Error: {e}")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "montage_mid_slice_overlays.png"))
plt.close()

# === Legend as separate figure ===
fig_leg = plt.figure(figsize=(2, 2))
plt.plot([], [], color='red', label='Ground Truth')
plt.plot([], [], color='blue', label='Prediction')
plt.legend(loc='center')
plt.axis('off')
plt.savefig(os.path.join(out_dir, "legend_overlay.png"))
plt.close()

print(f"Visualisations saved in: {out_dir}")

# === Plot 7: Individual 3-panel comparison for selected patients ===
selected_ids = ["BCBM_0265", "BCBM_0063", "BCBM_0179"]
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, patient_id in enumerate(selected_ids):
    try:
        img = nib.load(os.path.join(image_dir, f"{patient_id}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{patient_id}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{patient_id}.nii.gz")).get_fdata()

        mid = img.shape[2] // 2
        img_slice = img[:, :, mid]
        gt_slice = gt[:, :, mid]
        pred_slice = pred[:, :, mid]
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

        axs[i].imshow(img_slice, cmap='gray')
        axs[i].contour(gt_slice, levels=[0.5], colors='red', linewidths=1)
        axs[i].contour(pred_slice, levels=[0.5], colors='blue', linewidths=1)

        dice = sorted_df.loc[sorted_df["Patient"] == patient_id, "Dice"].values[0]
        axs[i].set_title(f"{patient_id}\nDice={dice:.2f}")
        axs[i].axis('off')

    except Exception as e:
        axs[i].axis('off')
        print(f"Error loading {patient_id}: {e}")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "selected_overlay_comparison.png"))
plt.close()
