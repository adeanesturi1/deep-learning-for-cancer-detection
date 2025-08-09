import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from batchgenerators.utilities.file_and_folder_operations import load_json

print("Starting FGSM attack...")

# === Settings ===
patient_id = "BraTS2021_00005"
slice_idx = 75
epsilon = 0.03

# === Paths ===
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_path = f"/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations/{patient_id}.nii.gz"
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks/fgsm_attack"

# === Set the correct paths ===
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"

# --- Device Setup ---
device = torch.device('cpu')

# === Load nnUNetTrainer ===
try:
    dataset_json_content = load_json(dataset_json_path)
except FileNotFoundError as e:
    print(f"Error: dataset.json not found at {dataset_json_path}. Please verify the path.")
    exit()

# It's important that trainer.initialize() sets up the preprocessor correctly.
trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
trainer.initialize() # This sets up self.load_dataset, self.load_dataloader, etc.

# Load checkpoint AFTER initialize()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval()

# === Load ALL 4 image modalities and Ground Truth ===
# Load raw image data and GT
raw_image_data = []
for i in range(4):
    mod_path = os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz")
    try:
        raw_image_data.append(nib.load(mod_path).get_fdata())
    except FileNotFoundError:
        print(f"Error: Modality {i} not found for {patient_id} at {mod_path}. Exiting.")
        exit()

# nnUNet's preprocessing expects data as (C, D, H, W) for 3D or (C, H, W) for 2D.
# It also needs the original affine matrix for resampling.
# For simplicity, let's assume all modalities have the same affine.
# In a real nnUNet prediction, it would use the original image file directly.
# Here we mimic the input format expected by trainer.preprocess_predict_data.

# Stack raw image data to (C, D, H, W)
full_volume_np = np.stack(raw_image_data, axis=0)
original_affine = nib.load(os.path.join(image_dir, f"{patient_id}_0000.nii.gz")).affine # Get affine from one image

# Load raw ground truth
try:
    gt_full_volume_raw = nib.load(gt_path).get_fdata()
except FileNotFoundError as e:
    print(f"Error loading GT file: {e}. Please check your paths.")
    exit()

# --- PREPROCESSING using trainer's capabilities ---
# nnUNet's trainer.preprocess_predict_data method expects a dictionary with 'image' and 'original_properties'
# 'image' should be a numpy array of shape (C, D, H, W)
# 'original_properties' is a dict containing affine, original_shape_for_return, etc.
# We need to replicate what a nnUNet 'inference job' would feed into the preprocessor.

# The data comes out of trainer.load_dataset(patient_id) in a similar format.
# Let's try to mock the structure for trainer.preprocess_predict_data.
properties = {
    'original_shape_for_return': full_volume_np.shape[1:], # (D, H, W)
    'original_spacing': nib.load(os.path.join(image_dir, f"{patient_id}_0000.nii.gz")).header.get_zooms(),
    'list_of_data_files': [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)],
    'seg_file': gt_path,
    'patient_id': patient_id,
    'affine': original_affine
}

# Preprocess the data for the model
# The output is (C, D, H, W) np array, and preprocessed properties
# The preprocessor will handle resampling, normalization, and padding/cropping
# Note: This method actually returns (preprocessed_data, preprocessed_properties).
# We only need the preprocessed_data.
preprocessed_data, _ = trainer.data_preprocessing_class.run_preprocessing(full_volume_np[None], properties) # [None] adds batch dim temporarily for run_preprocessing
# Remove the batch dimension added by run_preprocessing
preprocessed_data = preprocessed_data[0] # Shape (C, D, H, W)

# Convert to tensor and add batch dimension for model input
img_tensor_normalized = torch.tensor(preprocessed_data, dtype=torch.float32).unsqueeze(0).to(device) # [1, C, D, H, W]
img_tensor_normalized.requires_grad = True

print(f"Shape of img_tensor_normalized after nnUNet preprocessing: {img_tensor_normalized.shape}")

# Preprocess ground truth as well (nnUNet also resamples/pads GT)
# The GT should be preprocessed using the same transformations as the image data.
# The `run_preprocessing` method returns data, properties. For GT, the data is the segmentation.
# We will use the same properties as the image preprocessing to get consistent GT shape.
# However, `run_preprocessing` expects (1, D, H, W) for seg.
# For simplicity, let's just resample/pad the GT manually to match the image shape if necessary,
# or ideally use nnUNet's internal mechanisms which `trainer.predict_single_example` would handle.

# For attack loss, we need the GT in the preprocessed shape.
# This part is a bit tricky if we don't have a direct 'preprocess_segmentation' function.
# A common way in nnUNet is to load a preprocessed GT from the preprocessed directory
# or ensure the resampling/padding matches.
# Given your GT path is in 'nnUNet_preprocessed', it might already be in a compatible shape.
# Let's assume it's roughly compatible after the image preprocessing.
# If not, you might need to adapt the GT volume to match the preprocessed image shape.
# For now, let's just resize the GT to match the preprocessed image data's spatial dimensions.

# Get target shape from preprocessed image
target_D, target_H, target_W = img_tensor_normalized.shape[2:]

# Manual resampling of GT to target shape if needed (simplistic, nnUNet does more complex resampling)
# This part is a potential point of failure if nnUNet's resampling is complex.
# A more robust solution might involve mimicking nnUNet's `GenericPreprocessor.resample_and_normalize`.
# For now, we'll try a simple resize if needed.
# If original GT is (D,H,W), and new is (target_D, target_H, target_W)
gt_full_volume_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device)

if gt_full_volume_tensor.shape[0] != target_D or \
   gt_full_volume_tensor.shape[1] != target_H or \
   gt_full_volume_tensor.shape[2] != target_W:
    print(f"Warning: GT shape {gt_full_volume_tensor.shape} does not match preprocessed image shape {img_tensor_normalized.shape[2:]}. Resampling GT.")
    # This is a basic nearest-neighbor resize; nnUNet's resampling is more sophisticated.
    # It might be necessary to use scipy.ndimage.zoom or torch.nn.functional.interpolate if this fails.
    # For now, let's assume the preprocessed GT exists, or this simple resize is enough.
    # If the GT is not in the same preprocessed resolution, your loss will be computed on mismatched shapes.
    # The most reliable way is to load the preprocessed GT (if it exists) or use nnUNet's GT preprocessing.

    # Option 1: Try to load preprocessed GT (if nnUNet saved it)
    # This path is often /nnUNet_preprocessed/DatasetXXX_YYY/segmentations/patient_id.npz (or similar)
    # Not used here to keep the example simpler for direct data processing.

    # Option 2: Use torch.nn.functional.interpolate for resampling GT
    # Need to add batch and channel dims for interpolate
    gt_full_volume_tensor = gt_full_volume_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, D, H, W]
    gt_full_volume_tensor = torch.nn.functional.interpolate(gt_full_volume_tensor.float(),
                                                            size=(target_D, target_H, target_W),
                                                            mode='nearest-exact') # For segmentation, use nearest-neighbor
    gt_full_volume_tensor = gt_full_volume_tensor.squeeze(0).squeeze(0).long() # Remove added dims, convert back to long
    print(f"Resampled GT shape: {gt_full_volume_tensor.shape}")

# Ensure GT tensor is [1, D, H, W] for cross_entropy target
gt_full_volume_tensor_for_loss = gt_full_volume_tensor.unsqueeze(0).to(device)


# Extract the specific ground truth slice for visualization (from the potentially resampled GT)
gt_slice_tensor = gt_full_volume_tensor[slice_idx, :, :].clone()


# === Original prediction ===
with torch.no_grad():
    output = model(img_tensor_normalized) # Output is (B, NumClasses, D, H, W)
    pred_full_volume = output.argmax(dim=1).squeeze(0) # Remove batch dimension, shape (D, H, W)
    pred_slice = pred_full_volume[slice_idx, :, :] # Extract slice for visualization


# === FGSM Attack ===
img_tensor_attack = img_tensor_normalized.clone().detach().requires_grad_(True)
output_attack_full = model(img_tensor_attack) # This output is (B, Num_Classes, D, H, W)

loss = cross_entropy(output_attack_full, gt_full_volume_tensor_for_loss)

model.zero_grad()
img_tensor_attack.grad = None
loss.backward()
data_grad = img_tensor_attack.grad.data

adv_image = img_tensor_attack + epsilon * data_grad.sign()
adv_image = torch.clamp(adv_image, -5, 5)

# --- Adversarial Prediction ---
with torch.no_grad():
    adv_output_full = model(adv_image)
    adv_pred_full_volume = adv_output_full.argmax(dim=1).squeeze(0) # Shape (D, H, W)
    adv_pred_slice = adv_pred_full_volume[slice_idx, :, :] # Extract slice for visualization


# === Dice Calculation ===
def dice(gt_seg, pred_seg, label):
    gt_bin = (gt_seg == label)
    pred_bin = (pred_seg == label)
    union_sum = gt_bin.sum() + pred_bin.sum()
    if union_sum == 0:
        return 1.0
    return 2.0 * (gt_bin & pred_bin).sum() / (union_sum + 1e-6)

d_orig = dice(gt_slice_tensor, pred_slice, 3)  # ET class
d_adv = dice(gt_slice_tensor, adv_pred_slice, 3)

# --- Save Outputs ---
os.makedirs(out_dir, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original Slice: show one channel (e.g., T1ce, which was modality_idx=1)
# Note: img_tensor_normalized is [1, C, D, H, W]. Need to extract one slice and one channel for imshow.
axs[0].imshow(img_tensor_normalized[0, 1, slice_idx, :, :].cpu().numpy(), cmap='gray')
axs[0].set_title("Original Slice (Modality 1)")

axs[1].imshow(pred_slice.cpu().numpy(), cmap='viridis')
axs[1].set_title(f"Original Prediction\nDice={d_orig:.3f}")

axs[2].imshow(adv_pred_slice.cpu().numpy(), cmap='viridis')
axs[2].set_title(f"Adversarial Prediction\nDice={d_adv:.3f}")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(out_dir, f"{patient_id}_slice{slice_idx}_fgsm_mod_all.png")
plt.savefig(save_path, dpi=150)
plt.close()

print(f"FGSM attack completed successfully for {patient_id}, slice {slice_idx}.")
print(f"Results saved to: {save_path}")
print(f"Original Dice (ET): {d_orig:.3f}")
print(f"Adversarial Dice (ET): {d_adv:.3f}")