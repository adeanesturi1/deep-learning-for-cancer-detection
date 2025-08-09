import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# --- Matplotlib Configuration for Dissertation Quality ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

def find_best_slice(mask_data):
    """
    Finds the slice index with the largest segmentation area for each view.
    This ensures the visualization focuses on the most relevant part of the image.
    """
    if np.sum(mask_data) == 0:
        shape = mask_data.shape
        return shape[2] // 2, shape[0] // 2, shape[1] // 2 # Z, X, Y for Axial, Sagittal, Coronal
    
    axial_slice = np.sum(mask_data, axis=(0, 1)).argmax()    # Slice along Z
    sagittal_slice = np.sum(mask_data, axis=(1, 2)).argmax()  # Slice along X
    coronal_slice = np.sum(mask_data, axis=(0, 2)).argmax()   # Slice along Y
    return axial_slice, sagittal_slice, coronal_slice

def plot_slice_on_grid(ax, mri_slice, seg_slice, colormap, norm):
    """
    Plots a single MRI slice with its segmentation overlay on a given axis.
    """
    ax.imshow(np.rot90(mri_slice), cmap='gray')
    ax.imshow(np.rot90(seg_slice), cmap=colormap, norm=norm, alpha=0.6, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#a0a0a0')
        spine.set_linewidth(1.0)

# --- Configuration: File and Directory Paths ---
dice_csv    = "/home/an252/deep-learning-for-cancer-detection/bcbm_fold3_val_dice.csv"
gt_dir      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/gt_segmentations"
pred_dir    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/validation"
output_dir  = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_predictions/bcbm_results/figures_dissertation"
image_dir   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/nnUNetPlans_3d_fullres"
os.makedirs(output_dir, exist_ok=True)

# --- Data Loading and Validation ---
try:
    df = pd.read_csv(dice_csv)
except FileNotFoundError:
    print(f"ERROR: Dice scores CSV not found at: {dice_csv}")
    exit()

print("Scanning CSV and checking for all required files...")
valid_patients = []
for index, row in df.iterrows():
    patient_id = str(row['Patient']).strip()
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(image_dir, f"{patient_id}.npz")
    if all(os.path.exists(p) for p in [gt_path, pred_path, mri_path]):
        valid_patients.append(row)

if not valid_patients:
    print("ERROR: Could not find any patients with a complete set of files.")
    exit()

# --- Patient Selection: 3 Best and 3 Worst Cases ---
df_valid = pd.DataFrame(valid_patients)
df_sorted = df_valid.sort_values(by='Dice', ascending=True)

worst_3 = df_sorted.head(3).copy()
worst_3['CaseType'] = 'Worst Case'
best_3 = df_sorted.tail(3).copy()
best_3['CaseType'] = 'Best Case'

# Concatenate with worst cases first, then best cases.
patients_to_plot = pd.concat([worst_3, best_3.iloc[::-1]]) # Show best case with highest Dice first

print("\nFound valid files for the following best/worst patients:")
print(patients_to_plot[['Patient', 'Dice', 'CaseType']])

# --- Visualization Color and Style Setup ---
colors = [[0, 0, 0, 0], [1, 0, 0, 1]] 
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=len(colors) - 1)

# --- Create a Single, Combined Plot ---
# 6 rows for 6 patients, 6 columns for GT/Pred of 3 views
fig, axes = plt.subplots(6, 6, figsize=(18, 24))
fig.suptitle('Comparison of Best and Worst Segmentation Cases', fontsize=28, y=0.96)

# --- Main Plotting Loop for the Grid ---
for i, (index, row) in enumerate(patients_to_plot.iterrows()):
    patient_id = str(row['Patient']).strip()
    dice_score = row['Dice']
    case_type = row['CaseType']
    
    print(f"\nProcessing {case_type}: {patient_id} (Dice: {dice_score:.4f})...")
    
    # Load data
    gt_path = os.path.join(gt_dir, f"{patient_id}.nii.gz")
    pred_path = os.path.join(pred_dir, f"{patient_id}.nii.gz")
    mri_path = os.path.join(image_dir, f"{patient_id}.npz")
    
    try:
        gt_data = nib.load(gt_path).get_fdata().astype(np.uint8)
        pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)
        mri_data = np.load(mri_path)['data'][0]
        
        axial_idx, sagittal_idx, coronal_idx = find_best_slice(gt_data)
        
        # Slices to plot: (index, axis_to_slice)
        # Axis 2: Axial, Axis 0: Sagittal, Axis 1: Coronal
        slices_info = [
            (axial_idx, 2),
            (sagittal_idx, 0),
            (coronal_idx, 1)
        ]
        
        # Set the y-label for the entire row to identify the patient
        row_label = f"{case_type}\nPatient: {patient_id}\nDice: {dice_score:.3f}"
        axes[i, 0].set_ylabel(row_label, fontsize=14, rotation=90, labelpad=20, va='center')

        for j, (slice_idx, axis) in enumerate(slices_info):
            col_gt = j * 2
            col_pred = col_gt + 1
            
            mri_slice = np.take(mri_data, slice_idx, axis=axis)
            gt_slice = np.take(gt_data, slice_idx, axis=axis)
            pred_slice = np.take(pred_data, slice_idx, axis=axis)
            
            # Plot Ground Truth
            plot_slice_on_grid(axes[i, col_gt], mri_slice, gt_slice, cmap, norm)
            
            # Plot Prediction
            plot_slice_on_grid(axes[i, col_pred], mri_slice, pred_slice, cmap, norm)

    except Exception as e:
        print(f" -> Failed to process {patient_id}. Error: {e}")
        # If one patient fails, mark the row as failed.
        for ax in axes[i, :]:
            ax.text(0.5, 0.5, 'Error loading data', ha='center', va='center')


# --- Set Column Titles for the Top Row ---
axes[0, 0].set_title("Axial View\n(Ground Truth)", fontsize=16)
axes[0, 1].set_title("Axial View\n(Prediction)", fontsize=16)
axes[0, 2].set_title("Sagittal View\n(Ground Truth)", fontsize=16)
axes[0, 3].set_title("Sagittal View\n(Prediction)", fontsize=16)
axes[0, 4].set_title("Coronal View\n(Ground Truth)", fontsize=16)
axes[0, 5].set_title("Coronal View\n(Prediction)", fontsize=16)

# --- Final Touches: Legend and Layout ---
legend_handles = [mpatches.Patch(color='red', label='Tumor Segmentation')]
fig.legend(handles=legend_handles, loc='lower center', ncol=1, fontsize=20,
           frameon=False, bbox_to_anchor=(0.5, 0.02))

fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95], h_pad=3, w_pad=1.5) # Adjust main rect and padding

output_path = os.path.join(output_dir, "best_worst_segmentation_comparison_ALL_IN_ONE.png")
plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
plt.close(fig)

print(f"\n-> Combined publication-quality visualization saved to: {output_path}")
print("\nProcessing complete.")