import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO # For reading raw data with proper affine/spacing

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
    # Also load plans JSON content if needed for direct access to properties
    plans_json_content = load_json(plans_path)
except FileNotFoundError as e:
    print(f"Error: {e}. Please verify paths: {dataset_json_path} and {plans_path}")
    exit()

trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
trainer.initialize() # This sets up self.configuration_manager and its preprocessor

# Load checkpoint AFTER initialize()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval()

# === Load ALL 4 image modalities and Ground Truth ===
# Use SimpleITKIO to load image and properties as nnUNet does
# It returns (data, properties) where data is (C, D, H, W) numpy array
# and properties contains affine, original_spacing etc.
image_loader = SimpleITKIO()
full_image_path = [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)]
try:
    # This reads all modalities and stacks them correctly
    loaded_image_data, loaded_image_properties = image_loader.read_images(full_image_path)
except Exception as e: # Catch a broader exception for reading
    print(f"Error loading images: {e}. Ensure all 4 modalities exist and paths are correct.")
    exit()

# Loaded image data is already (C, D, H, W) numpy array
# loaded_image_properties contains 'spacing_after_transposition' which is what nnunet preprocessor uses as target_spacing
# and 'original_affine' etc.

# Load raw ground truth
try:
    gt_nifti = nib.load(gt_path)
    gt_full_volume_raw = gt_nifti.get_fdata()
    gt_affine = gt_nifti.affine
except FileNotFoundError as e:
    print(f"Error loading GT file: {e}. Please check your paths.")
    exit()


# --- PREPROCESSING using the trainer's preprocessor ---
# Get the preprocessor instance from the trainer's configuration manager
preprocessor = trainer.configuration_manager.preprocessor

# The preprocessor.run_preprocessor expects raw_image_data (C,D,H,W) and properties dict
# The output will be (C, D_preprocessed, H_preprocessed, W_preprocessed) numpy array
# And updated properties
preprocessed_data, preprocessed_properties = preprocessor.run_preprocessor(loaded_image_data, loaded_image_properties)

# Convert to tensor and add batch dimension for model input
# preprocessed_data is already (C, D, H, W) numpy
img_tensor_normalized = torch.tensor(preprocessed_data, dtype=torch.float32).unsqueeze(0).to(device) # [1, C, D, H, W]
img_tensor_normalized.requires_grad = True

print(f"Shape of img_tensor_normalized after nnUNet preprocessing: {img_tensor_normalized.shape}")

# --- Preprocess Ground Truth to match image dimensions ---
# The GT must also be resampled and padded/cropped to match the preprocessed image data.
# We will use the same preprocessor's logic for the GT.
# The preprocessor's `resample_and_normalize` is typically called.
# For GT, we need to pass it through a similar resampling process using the *preprocessed_properties*
# to get it to the exact same spatial dimensions as img_tensor_normalized.

# nnUNet's preprocessor has 'resample_seg_to_spacing' (or similar methods)
# Let's try to mimic `GenericPreprocessor.resample_and_normalize` for segmentation.
# A simpler way is to just resample the GT to the final preprocessed image shape using interpolate.
# However, nnUNet's resampling is more exact and uses affine matrices.

# Get target shape from preprocessed image tensor
target_D, target_H, target_W = img_tensor_normalized.shape[2:]

# Convert raw GT to tensor and ensure it's on device
gt_full_volume_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device)

# Resample GT to match preprocessed image dimensions using nn.functional.interpolate
# Add batch and channel dims for interpolate: [1, 1, D, H, W]
gt_full_volume_tensor_for_loss = gt_full_volume_tensor.unsqueeze(0).unsqueeze(0)
gt_full_volume_tensor_for_loss = torch.nn.functional.interpolate(gt_full_volume_tensor_for_loss.float(),
                                                                size=(target_D, target_H, target_W),
                                                                mode='nearest-exact', # For segmentation, use nearest-neighbor
                                                                align_corners=None) # True/False might matter, but for nearest-exact usually None is fine
gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.squeeze(0).squeeze(0).long() # Remove added dims, convert back to long
# Now gt_full_volume_tensor_for_loss is (D_preprocessed, H_preprocessed, W_preprocessed)
gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.unsqueeze(0) # Add batch dim for cross_entropy: [1, D, H, W]
print(f"Shape of gt_full_volume_tensor_for_loss after resampling: {gt_full_volume_tensor_for_loss.shape}")


# Extract the specific ground truth slice for visualization (from the potentially resampled GT)
# Ensure gt_full_volume_tensor is correctly indexed as it's now (D,H,W) after previous squeeze(0).squeeze(0)
gt_slice_tensor = gt_full_volume_tensor_for_loss.squeeze(0)[slice_idx, :, :].clone()


# === Original prediction ===
with torch.no_grad():
    output = model(img_tensor_normalized) # Output is (B, NumClasses, D, H, W)
    # Get the prediction for the specific slice for visualization
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