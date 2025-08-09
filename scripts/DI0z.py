import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def find_best_slice(mask_data):
    """Finds the slice index with the largest tumor area for each view."""
    if np.sum(mask_data) == 0:
        shape = mask_data.shape
        return shape[2] // 2, shape[1] // 2, shape[0] // 2
    axial_slice = np.sum(mask_data, axis=(0, 1)).argmax()
    sagittal_slice = np.sum(mask_data, axis=(0, 2)).argmax()
    coronal_slice = np.sum(mask_data, axis=(1, 2)).argmax()
    return axial_slice, sagittal_slice, coronal_slice

def plot_slice_with_border(ax, mri_slice, seg_slice, colormap, norm):
    """Plots a single slice with overlay and a visible border."""
    ax.imshow(np.rot90(mri_slice), cmap='gray')
    ax.imshow(np.rot90(seg_slice), cmap=colormap, norm=norm, alpha=0.5, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])

gt_dir          = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
pred_dir_base   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres"
mri_dir         = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
dice_scores_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dicescores_val_brats.csv"
output_dir      = "visualizations_brats_best_worst"
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(dice_scores_csv)
except FileNotFoundError:
    print(f"ERROR: Dice scores CSV not found at: {dice_scores_csv}")
    exit()

# sort by the 'dice' column to find best and worst
df_sorted = df.sort_values(by='dice', ascending=True) 

worst_5 = df_sorted.head(5)
best_5 = df_sorted.tail(5)

patients_to_plot = pd.concat([worst_5, best_5])
print("Selected Patients for Visualization:")
print(patients_to_plot[['patient_id', 'fold', 'dice']])

colors = [[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]]
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=len(colors)-1)

for index, row in patients_to_plot.iterrows():
    patient_id = row['patient_id']
    fold = row['fold']
    dice_score = row['dice'] 
    
    print(f"\nProcessing patient: {patient_id} (Fold {fold}, Dice {dice_score:.4f})...")

    pred_dir = os.path.join(pred_dir_base, f"fold_{fold}", "validation")
    
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(mri_dir, f"{patient_id}_0001.nii.gz")

    if not all(os.path.exists(p) for p in [gt_path, pred_path, mri_path]):
        print(f"  -> Skipping {patient_id} - a required file was not found.")
        continue

    try:
        gt_data = nib.load(gt_path).get_fdata().astype(np.uint8)
        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)
        mri_data = nib.load(mri_path).get_fdata()

        z_idx, y_idx, x_idx = find_best_slice(gt_data)
        slices_to_plot = [(z_idx, 2), (y_idx, 1), (x_idx, 0)]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
        fig.suptitle(f'Patient: {patient_id} (Dice: {dice_score:.4f})', fontsize=20)
        
        for i, (slice_idx, axis) in enumerate(slices_to_plot):
            gt_slice = np.take(gt_data, slice_idx, axis=axis)
            pred_slice = np.take(pred_data, slice_idx, axis=axis)
            mri_slice = np.take(mri_data, slice_idx, axis=axis)
            
            plot_slice_with_border(axes[0, i], mri_slice, gt_slice, cmap, norm)
            plot_slice_with_border(axes[1, i], mri_slice, pred_slice, cmap, norm)

        axes[0, 0].set_title('Axial', fontsize=16)
        axes[0, 1].set_title('Sagittal', fontsize=16)
        axes[0, 2].set_title('Coronal', fontsize=16)
        axes[0, 0].set_ylabel('Ground Truth', fontsize=16, labelpad=20)
        axes[1, 0].set_ylabel('Prediction', fontsize=16, labelpad=20)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        output_path = os.path.join(output_dir, f"{patient_id}_comparison.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"  -> Failed to process {patient_id}. Error: {e}")
        if 'fig' in locals():
            plt.close(fig)

print("\n Processing complete.")