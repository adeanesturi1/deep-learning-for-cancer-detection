import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dice_csv = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures"

# Create the output directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# === Load and Prepare Data ===
try:
    dice_df = pd.read_csv(dice_csv)
    metadata_df = pd.read_csv(metadata_csv)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure your file paths in the 'User Configuration' section are correct.")
    # Exit if essential data is missing
    exit()

# Merge and sort the data to find best and worst cases
metadata_df = metadata_df.rename(columns={"nnUNet_ID": "Patient"})
merged_df = pd.merge(dice_df, metadata_df[["Patient", "HER2_Status"]], on="Patient", how="left")
sorted_df = merged_df.sort_values(by="Dice", ascending=False).reset_index(drop=True)

# === Select Top and Bottom 3 Patients ===
# Get the 3 best (highest Dice) and 3 worst (lowest Dice) patients
top_3_patients = sorted_df.head(3)
bottom_3_patients = sorted_df.tail(3)
patient_grid_df = pd.concat([top_3_patients, bottom_3_patients])

# === Create the 2x3 Grid Visualization ===
fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
axs = axs.flatten()

for i, row in enumerate(patient_grid_df.iterrows()):
    _, data = row
    patient_id = data["Patient"]
    dice_score = data["Dice"]
    ax = axs[i]

    try:
        # Define file paths for the current patient
        img_path = os.path.join(image_dir, f"{patient_id}_0000.nii.gz")
        gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
        pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")

        # Load NIfTI files
        img_nii = nib.load(img_path)
        gt_nii = nib.load(gt_path)
        pred_nii = nib.load(pred_path)

        # Get the image data as numpy arrays
        img_data = img_nii.get_fdata()
        gt_data = gt_nii.get_fdata()
        pred_data = pred_nii.get_fdata()

        # Find the middle slice index for visualization
        mid_slice_idx = img_data.shape[2] // 2
        
        # Extract the 2D slices
        img_slice = img_data[:, :, mid_slice_idx]
        gt_slice = gt_data[:, :, mid_slice_idx]
        pred_slice = pred_data[:, :, mid_slice_idx]

        # Normalize the image slice for better contrast
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

        # Display the base image (rotated and with correct origin for medical images)
        ax.imshow(np.rot90(img_slice), cmap='gray')

        # Overlay contours for ground truth (red) and prediction (blue)
        ax.contour(np.rot90(gt_slice), levels=[0.5], colors='red', linewidths=1.5)
        ax.contour(np.rot90(pred_slice), levels=[0.5], colors='blue', linewidths=1.5)

        # Set title with patient ID and its Dice score
        ax.set_title(f"Patient: {patient_id}\nDice: {dice_score:.4f}")
        ax.axis('off')

    except FileNotFoundError:
        print(f"Data for patient {patient_id} not found. Skipping.")
        ax.set_title(f"{patient_id}\n(File Not Found)")
        ax.axis('off')
    except Exception as e:
        print(f"Could not process patient {patient_id}. Error: {e}")
        ax.set_title(f"{patient_id}\n(Error)")
        ax.axis('off')

plt.tight_layout(pad=3.0)
grid_image_path = os.path.join(out_dir, "Top_Bottom_3_Dice_Comparison.png")
plt.savefig(grid_image_path)
plt.close()

# === Generate a Separate Legend File ===
fig_legend = plt.figure(figsize=(3, 1))
ax_legend = fig_legend.add_subplot(111)
# Create dummy plots for legend handles
red_patch = plt.Line2D([0], [0], color='red', lw=2, label='Ground Truth')
blue_patch = plt.Line2D([0], [0], color='blue', lw=2, label='Prediction')
ax_legend.legend(handles=[red_patch, blue_patch], loc='center')
ax_legend.axis('off')

legend_path = os.path.join(out_dir, "comparison_legend.png")
plt.savefig(legend_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Visualization complete!")
print(f"Grid comparison image saved to: {grid_image_path}")
print(f"Legend image saved to: {legend_path}")