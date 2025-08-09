import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

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

gt_tr_dir       = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations"
mri_tr_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_val_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/labelsVal"
mri_val_dir     = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesVal"
pred_dir_base   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres"
dice_scores_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/val_brats/dicescores_val_brats.csv"
output_dir      = "visualizations_brats_publication"
os.makedirs(output_dir, exist_ok=True)

try:
    df = pd.read_csv(dice_scores_csv)
except FileNotFoundError:
    print(f"ERROR: Dice scores CSV not found at: {dice_scores_csv}")
    exit()

valid_patients = []
for index, row in df.iterrows():
    patient_id = str(row['patient_id']).strip()
    fold = str(row['fold']).strip()
    
    pred_path = os.path.join(pred_dir_base, fold, "validation", f"{patient_id}.nii.gz")
    gt_path = os.path.join(gt_tr_dir, f"{patient_id}.nii.gz")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(gt_val_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(mri_tr_dir, f"{patient_id}_0001.nii.gz")
    if not os.path.exists(mri_path):
        mri_path = os.path.join(mri_val_dir, patient_id, f"{patient_id}_0001.nii.gz")

    if all(os.path.exists(p) for p in [gt_path, pred_path, mri_path]):
        valid_patients.append(row)

if not valid_patients:
    print("ERROR: Could not find any patients with a complete set of files.")
    exit()

df_valid = pd.DataFrame(valid_patients)
df_sorted = df_valid.sort_values(by='dice', ascending=True)

worst_5 = df_sorted.head(5)
best_5 = df_sorted.tail(5)
patients_to_plot = pd.concat([worst_5, best_5])

print("Selected Patients for Visualization:")
print(patients_to_plot[['patient_id', 'fold', 'dice']])

colors = [[0, 0, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]]
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=len(colors)-1)

for index, row in patients_to_plot.iterrows():
    patient_id = str(row['patient_id']).strip()
    fold = str(row['fold']).strip()
    dice_score = row['dice']
    
    print(f"\nProcessing patient: {patient_id} (Fold {fold}, Dice {dice_score:.4f})...")
    
    pred_dir = os.path.join(pred_dir_base, fold, "validation")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    gt_path = os.path.join(gt_tr_dir, f"{patient_id}.nii.gz")
    if not os.path.exists(gt_path):
        gt_path = os.path.join(gt_val_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(mri_tr_dir, f"{patient_id}_0001.nii.gz")
    if not os.path.exists(mri_path):
        mri_path = os.path.join(mri_val_dir, patient_id, f"{patient_id}_0001.nii.gz")

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
            # Ground Truth Row
            ax_gt = axes[0, i]
            gt_slice = np.take(gt_data, slice_idx, axis=axis)
            mri_slice = np.take(mri_data, slice_idx, axis=axis)
            plot_slice_pretty(ax_gt, mri_slice, gt_slice, cmap, norm)
            ax_gt.text(0.03, 0.97, panel_labels[i], transform=ax_gt.transAxes, fontsize=16, 
                       fontweight='bold', va='top', ha='left', color='white')

            # PPrediction Row
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

        legend_handles = [
            mpatches.Patch(color='yellow', label='Enhancing Tumor (ET)'),
            mpatches.Patch(color='red', label='Tumor Core (TC)'),
            mpatches.Patch(color='green', label='Edema (ED)')
        ]
        fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0))
        
        plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.1, wspace=0.1, hspace=0.1)
        
        output_path = os.path.join(output_dir, f"{patient_id}_pub_comparison.png")
        plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        print(f"  ->  Visualization saved to: {output_path}")

    except Exception as e:
        print(f"  -> Failed to process {patient_id}. Error: {e}")
        if 'fig' in locals():
            plt.close(fig)

print("\nProcessing complete.")