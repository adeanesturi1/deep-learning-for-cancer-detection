import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib


dice_csv = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures"

os.makedirs(out_dir, exist_ok=True)

try:
    dice_df = pd.read_csv(dice_csv)
    metadata_df = pd.read_csv(metadata_csv)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure your file paths in the 'User Configuration' section are correct.")
    exit()

metadata_df = metadata_df.rename(columns={"nnUNet_ID": "Patient"})
merged_df = pd.merge(dice_df, metadata_df, on="Patient", how="left")
sorted_df = merged_df.sort_values(by="Dice", ascending=False).reset_index()

top_3_with_tumors = []
print("Searching for top 3 performing patients with actual tumors...")
for index, row in sorted_df.iterrows():
    patient_id = row["Patient"]
    try:
        gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
        gt_mask = nib.load(gt_path).get_fdata()
        if np.sum(gt_mask) > 0:
            top_3_with_tumors.append(row)
            print(f"  -> Found patient: {patient_id} (Dice: {row['Dice']:.4f})")
        if len(top_3_with_tumors) == 3:
            break
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found for {patient_id}. Skipping.")

top_3_df = pd.DataFrame(top_3_with_tumors)

bottom_3_df = sorted_df.tail(3)
patient_grid_df = pd.concat([top_3_df, bottom_3_df])

print("\nPatients selected for visualization:")
print(patient_grid_df[['Patient', 'Dice']])

fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
axs = axs.flatten()

for i, row in enumerate(patient_grid_df.iterrows()):
    _, data = row
    patient_id = data["Patient"]
    dice_score = data["Dice"]
    ax = axs[i]

    try:
        img_path = os.path.join(image_dir, f"{patient_id}_0000.nii.gz")
        gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
        img_data = nib.load(img_path).get_fdata()
        gt_data = nib.load(gt_path).get_fdata()
        pred_data = nib.load(pred_path).get_fdata()
        mid_slice_idx = img_data.shape[2] // 2
        img_slice = img_data[:, :, mid_slice_idx]
        gt_slice = gt_data[:, :, mid_slice_idx]
        pred_slice = pred_data[:, :, mid_slice_idx]
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))
        ax.imshow(np.rot90(img_slice), cmap='gray')
        ax.contour(np.rot90(gt_slice), levels=[0.5], colors='red', linewidths=1.5)
        ax.contour(np.rot90(pred_slice), levels=[0.5], colors='blue', linewidths=1.5)
        title_prefix = "Top Performer" if i < 3 else "Bottom Performer"
        ax.set_title(f"{title_prefix}: {patient_id}\nDice: {dice_score:.4f}")
        ax.axis('off')

    except Exception as e:
        print(f"Could not process patient {patient_id}. Error: {e}")
        ax.set_title(f"{patient_id}\n(Error)")
        ax.axis('off')

plt.tight_layout(pad=3.0)
grid_image_path = os.path.join(out_dir, "Top_Bottom_3_Visual_Comparison.png")
plt.savefig(grid_image_path)
plt.close()

fig_legend = plt.figure(figsize=(3, 1))
red_patch = plt.Line2D([0], [0], color='red', lw=2, label='Ground Truth')
blue_patch = plt.Line2D([0], [0], color='blue', lw=2, label='Prediction')
plt.legend(handles=[red_patch, blue_patch], loc='center')
plt.axis('off')
legend_path = os.path.join(out_dir, "comparison_legend.png")
plt.savefig(legend_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n Visualization complete!")
print(f"Grid comparison image saved to: {grid_image_path}")
print(f"Legend image saved to: {legend_path}")