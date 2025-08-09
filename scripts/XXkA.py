import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib

from matplotlib.colors import ListedColormap

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
sorted_df = merged_df.sort_values(by="Dice", ascending=False).reset_index(drop=True)
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

# === Utility: Plot mid-slice overlay ===
def plot_overlay(patient_id, axarr, title_prefix=""):
    img_path = os.path.join(image_dir, f"{patient_id}_0000.nii.gz")
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")

    try:
        img = nib.load(img_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()

        mid_slice = img.shape[2] // 2
        img_slice = img[:, :, mid_slice]
        gt_slice = gt[:, :, mid_slice]
        pred_slice = pred[:, :, mid_slice]

        # Normalize image
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

        axarr[0].imshow(img_slice, cmap='gray')
        axarr[0].imshow(gt_slice, cmap=ListedColormap(['none', 'red']), alpha=0.5)
        axarr[0].set_title(f"{title_prefix}GT")

        axarr[1].imshow(img_slice, cmap='gray')
        axarr[1].imshow(pred_slice, cmap=ListedColormap(['none', 'blue']), alpha=0.5)
        axarr[1].set_title(f"{title_prefix}Prediction")

        for ax in axarr:
            ax.axis('off')
    except Exception as e:
        print(f"Failed to plot {patient_id}: {e}")

# === Plot 5: Overlays for Top and Bottom 3 ===
top3 = sorted_df.head(3)["Patient"].tolist()
bottom3 = sorted_df.tail(3)["Patient"].tolist()

for patient_id in top3 + bottom3:
    fig, axarr = plt.subplots(1, 2, figsize=(10, 5))
    plot_overlay(patient_id, axarr, title_prefix=f"{patient_id} ")
    plt.tight_layout()
    fname = f"overlay_{patient_id}.png"
    plt.savefig(os.path.join(out_dir, fname))
    plt.close()

# === Plot 6: Montage of All Patients ===
fig, axs = plt.subplots(9, 6, figsize=(18, 28))
axs = axs.flatten()

for i, patient_id in enumerate(sorted_df["Patient"][:54]):
    img_path = os.path.join(image_dir, f"{patient_id}_0000.nii.gz")
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    try:
        img = nib.load(img_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        pred = nib.load(pred_path).get_fdata()
        mid = img.shape[2] // 2
        img_slice = img[:, :, mid]
        gt_slice = gt[:, :, mid]
        pred_slice = pred[:, :, mid]

        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        axs[i].imshow(img_slice, cmap='gray')
        axs[i].imshow(pred_slice, cmap=ListedColormap(['none', 'blue']), alpha=0.5)
        axs[i].imshow(gt_slice, cmap=ListedColormap(['none', 'red']), alpha=0.5)
        dice = sorted_df.loc[sorted_df["Patient"] == patient_id, "Dice"].values[0]
        axs[i].set_title(f"{patient_id}\nDice={dice:.2f}")
        axs[i].axis('off')
    except Exception as e:
        axs[i].axis('off')
        print(f"Could not load: {patient_id} | Error: {e}")

plt.tight_layout()
plt.savefig(os.path.join(out_dir, "montage_mid_slice_overlays.png"))
plt.close()

print(f"Visualisations saved in: {out_dir}")
