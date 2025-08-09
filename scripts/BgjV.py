import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

def plot_slice_pretty(ax, mri_slice, seg_slice, colormap, norm):
    """Plots a single slice with refined aesthetics."""
    ax.imshow(np.rot90(mri_slice), cmap='gray')
    ax.imshow(np.rot90(seg_slice), cmap=colormap, norm=norm, alpha=0.6, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#a0a0a0')
        spine.set_linewidth(0.8)

dice_csv    = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
image_dir   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/imagesTr"
gt_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
output_dir  = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/per_patient_analysis"
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(dice_csv)
except FileNotFoundError:
    print(f"ERROR: Dice scores CSV not found at: {dice_csv}")
    exit()

df_sorted = df.sort_values(by='dice', ascending=True)

worst_5 = df_sorted.head(5)
best_5 = df_sorted.tail(5)
patients_to_plot = pd.concat([worst_5, best_5])

print("Selected Patients for Visualization:")
print(patients_to_plot)

# --- 3. PREPARE PLOT STYLES ---
# Simplified color map for BCBM (0: background, 1: tumor)
colors = [[0, 0, 0, 0],    # 0=background (transparent)
          [1, 0, 0, 1]]    # 1=Tumor (red)
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=len(colors)-1)

# --- 4. MAIN PLOTTING LOOP ---
for index, row in patients_to_plot.iterrows():
    patient_id = str(row['patient_id']).strip()
    dice_score = row['dice']
    
    print(f"\nProcessing patient: {patient_id} (Dice: {dice_score:.4f})...")
    
    # Construct paths
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(image_dir, f"{patient_id}_0000.nii.gz")

    if not all(os.path.exists(p) for p in [gt_path, pred_path, mri_path]):
        print(f"  -> Skipping {patient_id} - a required file was not found.")
        continue

    try:
        gt_data = nib.load(gt_path).get_fdata().astype(np.uint8)
        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)
        mri_data = nib.load(mri_path).get_fdata()

        z_idx, y_idx, x_idx = find_best_slice(gt_data)
        slices_to_plot = [(z_idx, 2), (y_idx, 1), (x_idx, 0)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
        fig.suptitle(f'Segmentation Results for Patient {patient_id} (Dice: {dice_score:.4f})', fontsize=20, y=0.98)
        
        panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
        
        for i, (slice_idx, axis) in enumerate(slices_to_plot):
            mri_slice = np.take(mri_data, slice_idx, axis=axis)
            
            # Plot Ground Truth Row
            ax_gt = axes[0, i]
            gt_slice = np.take(gt_data, slice_idx, axis=axis)
            plot_slice_pretty(ax_gt, mri_slice, gt_slice, cmap, norm)
            ax_gt.text(0.03, 0.97, panel_labels[i], transform=ax_gt.transAxes, fontsize=16, 
                       fontweight='bold', va='top', ha='left', color='white')

            # Plot Prediction Row
            ax_pred = axes[1, i]
            pred_slice = np.take(pred_data, slice_idx, axis=axis)
            plot_slice_pretty(ax_pred, mri_slice, pred_slice, cmap, norm)
            ax_pred.text(0.03, 0.97, panel_labels[i+3], transform=ax_pred.transAxes, fontsize=16, 
                         fontweight='bold', va='top', ha='left', color='white')

        axes[0, 0].set_title('Axial', fontsize=18)
        axes[0, 1].set_title('Sagittal', fontsize=18)
        axes[0, 2].set_title('Coronal', fontsize=18)
        axes[0, 0].set_ylabel('Ground Truth', fontsize=18, labelpad=20)
        axes[1, 0].set_ylabel('Prediction', fontsize=18, labelpad=20)

        legend_handles = [mpatches.Patch(color='red', label='Tumor')]
        fig.legend(handles=legend_handles, loc='lower center', ncol=1, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0))
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.1, wspace=0.1, hspace=0.1)
        
        output_path = os.path.join(output_dir, f"{patient_id}_comparison.png")
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Publication-quality visualization saved to: {output_path}")

    except Exception as e:
        print(f"  -> Failed to process {patient_id}. Error: {e}")
        if 'fig' in locals():
            plt.close(fig)

print("\nProcessing complete.")