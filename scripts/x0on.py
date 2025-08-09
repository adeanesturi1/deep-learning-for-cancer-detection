import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def find_best_slice(mask_data):
    """Finds the slice index with the largest tumor area for each view."""
    if np.sum(mask_data) == 0:  # Handle cases with no tumor
        shape = mask_data.shape
        return shape[2] // 2, shape[1] // 2, shape[0] // 2
    axial_slice = np.sum(mask_data, axis=(0, 1)).argmax()
    sagittal_slice = np.sum(mask_data, axis=(0, 2)).argmax()
    coronal_slice = np.sum(mask_data, axis=(1, 2)).argmax()
    return axial_slice, sagittal_slice, coronal_slice

def plot_slice(ax, mri_slice, seg_slice, colormap, norm):
    """Plots a single slice with its segmentation overlay."""
    ax.imshow(np.rot90(mri_slice), cmap='gray')
    ax.imshow(np.rot90(seg_slice), cmap=colormap, norm=norm, alpha=0.5, interpolation='none')
    ax.axis('off')

# --- 1. SETTINGS ---
# Define the base directories for your data
gt_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
pred_dir    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/validation" # <-- UPDATED PATH
mri_dir     = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
output_dir  = "visualizations_brats"
os.makedirs(output_dir, exist_ok=True)

# --- 2. PREPARE PLOT STYLES ---
colors = [[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]]
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=len(colors)-1)

# --- 3. MAIN LOOP ---
# Iterate over all ground truth label files
for gt_filename in os.listdir(gt_dir):
    if not gt_filename.endswith(".nii.gz"):
        continue

    patient_id = gt_filename.replace('.nii.gz', '')
    print(f"Processing patient: {patient_id}...")

    # Construct the full paths for all required files
    gt_path = os.path.join(gt_dir, gt_filename)
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(mri_dir, f"{patient_id}_0001.nii.gz") # T1c contrast

    if not all(os.path.exists(p) for p in [pred_path, mri_path]):
        print(f"  -> Skipping {patient_id} - missing prediction or MRI file.")
        continue

    try:
        gt_data = nib.load(gt_path).get_fdata().astype(np.uint8)
        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)
        mri_data = nib.load(mri_path).get_fdata()

        z_idx, y_idx, x_idx = find_best_slice(gt_data)
        slices_to_plot = [(z_idx, 2), (y_idx, 1), (x_idx, 0)]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor='black')
        plt.subplots_adjust(wspace=0.05, hspace=0.1)

        for i, (slice_idx, axis) in enumerate(slices_to_plot):
            gt_slice = np.take(gt_data, slice_idx, axis=axis)
            pred_slice = np.take(pred_data, slice_idx, axis=axis)
            mri_slice = np.take(mri_data, slice_idx, axis=axis)
            
            plot_slice(axes[0, i], mri_slice, gt_slice, cmap, norm)
            plot_slice(axes[1, i], mri_slice, pred_slice, cmap, norm)

        axes[0, 0].set_title('Axial', color='white', fontsize=16)
        axes[0, 1].set_title('Sagittal', color='white', fontsize=16)
        axes[0, 2].set_title('Coronal', color='white', fontsize=16)
        axes[0, 0].set_ylabel('Ground Truth', color='white', fontsize=16, labelpad=20)
        axes[1, 0].set_ylabel('Prediction', color='white', fontsize=16, labelpad=20)

        output_path = os.path.join(output_dir, f"{patient_id}_segmentation_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')
        plt.close(fig)

    except Exception as e:
        print(f"  -> Failed to process {patient_id}. Error: {e}")
        if 'fig' in locals():
            plt.close(fig)

print("\nProcessing complete.")