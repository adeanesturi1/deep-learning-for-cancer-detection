import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
from skimage import measure

# === Config ===
dice_csv = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/per_patient_analysis"
os.makedirs(out_dir, exist_ok=True)

# === Load and merge data ===
dice_df = pd.read_csv(dice_csv)
metadata_df = pd.read_csv(metadata_csv)
metadata_df = metadata_df.rename(columns={"nnUNet_ID": "Patient"})

merged_df = pd.merge(dice_df, metadata_df[["Patient", "HER2_Status"]], on="Patient", how="left")

hausdorff_csv = os.path.join(out_dir, "dice_with_hausdorff.csv")
if os.path.exists(hausdorff_csv):
    merged_df = pd.read_csv(hausdorff_csv)
else:
    from scipy.spatial.distance import directed_hausdorff

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
    merged_df.to_csv(hausdorff_csv, index=False)

# === Select top 3 patients ===
merged_df.columns = merged_df.columns.str.strip()
top3 = merged_df.sort_values("Dice", ascending=False).head(3)

# === Plotting Function ===
def plot_patient_contour(pid, dice, hd, her2, out_path):
    try:
        img = nib.load(os.path.join(image_dir, f"{pid}_0000.nii.gz")).get_fdata()
        gt = nib.load(os.path.join(gt_dir, f"{pid}.nii.gz")).get_fdata()
        pred = nib.load(os.path.join(pred_dir, f"{pid}.nii.gz")).get_fdata()

        z_slices = [img.shape[2] // 2 - 5, img.shape[2] // 2, img.shape[2] // 2 + 5]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        for i, z in enumerate(z_slices):
            img_slice = (img[:, :, z] - np.min(img)) / (np.max(img) - np.min(img))
            gt_slice = gt[:, :, z]
            pred_slice = pred[:, :, z]

            axs[i].imshow(img_slice, cmap="gray")

            for c in measure.find_contours(gt_slice, 0.5):
                axs[i].plot(c[:, 1], c[:, 0], color='red', linewidth=1.5, label="Ground Truth" if i == 0 else "")
            for c in measure.find_contours(pred_slice, 0.5):
                axs[i].plot(c[:, 1], c[:, 0], color='lime', linewidth=1.5, label="Prediction" if i == 0 else "")

            axs[i].set_title(f"Slice {z}")
            axs[i].axis('off')

        handles, labels = axs[0].get_legend_handles_labels()
        fig.suptitle(f"{pid} | Dice = {dice:.3f} | Hausdorff = {hd:.2f} | HER2: {her2}", fontsize=14)
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(out_path)
        plt.close()
    except Exception as e:
        print(f"[ERROR] Failed for {pid}: {e}")

# === Generate Visualisations ===
for _, row in top3.iterrows():
    pid = row["Patient"]
    dice = row["Dice"]
    hd = row["Hausdorff"]
    her2 = row["HER2_Status"]
    out_path = os.path.join(out_dir, f"panel_top3_{pid}.png")
    plot_patient_contour(pid, dice, hd, her2, out_path)

print(f"Saved side-by-side visualisations with HER2 info to: {out_dir}")
