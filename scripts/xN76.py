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
slice_idx = 75 # Still useful for visualization of a specific slice later
# modality_idx is no longer needed for loading, as we load all 4
epsilon = 0.03 # Perturbation strength for FGSM

# === Paths ===
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_path = f"/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations/{patient_id}.nii.gz"
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks/fgsm_attack"

# === Set the correct path to your nnUNetPlans.json file ===
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"

# === Set the correct path to your dataset.json file ===
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"

# --- Device Setup ---
device = torch.device('cpu')

# === Load nnUNetTrainer
try:
    dataset_json_content = load_json(dataset_json_path)
except FileNotFoundError as e:
    print(f"Error: dataset.json not found at {dataset_json_path}. Please verify the path.")
    exit()

trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
trainer.initialize()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval()

# === Load ALL 4 image modalities and Ground Truth ===
# The model expects 4 channels for the 3D_fullres configuration
modalities_data = []
for i in range(4): # Iterate through 4 modalities (0, 1, 2, 3)
    mod_path = os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz")
    try:
        modality_image = nib.load(mod_path).get_fdata()
        modalities_data.append(modality_image)
    except FileNotFoundError:
        print(f"Warning: Modality {i} not found for {patient_id} at {mod_path}. Skipping.")
        # Handle missing modalities, perhaps by padding with zeros or raising an error
        # For this example, we'll assume all 4 are present and stop if not
        print("Error: Could not load all 4 modalities. Please check data integrity.")
        exit()

# Stack modalities along a new channel dimension
# The shape will be (C, D, H, W) -> (4, Depth, Height, Width)
full_volume_np = np.stack(modalities_data, axis=0) # [C, D, H, W]

try:
    gt_full_volume = nib.load(gt_path).get_fdata()
except FileNotFoundError as e:
    print(f"Error loading GT file: {e}. Please check your paths.")
    exit()

# Convert to tensor. nnUNet's 3D models expect (B, C, D, H, W)
img_tensor = torch.tensor(full_volume_np, dtype=torch.float32).unsqueeze(0).to(device) # [1, C, D, H, W]

# Normalization: Apply per-channel normalization (as nnUNet usually does)
# Get mean/std from the plans or use global for simplicity if not available directly
# For simplicity, let's use global normalization for now, but per-channel is more accurate for nnUNet
# Based on your plans, it uses ZScoreNormalization per channel, so we should apply that.
# You might need to load the actual normalization parameters from the plans or compute them
# For a quick fix, let's normalize each channel independently
normalized_channels = []
for c in range(img_tensor.shape[1]):
    channel_data = img_tensor[:, c, :, :, :]
    mean = channel_data.mean()
    std = channel_data.std() + 1e-6
    normalized_channels.append((channel_data - mean) / std)
img_tensor_normalized = torch.cat(normalized_channels, dim=1)


# We need to ensure img_tensor_normalized requires gradients for the attack
img_tensor_normalized.requires_grad = True


# Extract the specific ground truth slice for loss calculation and visualization
gt_slice_tensor = torch.tensor(gt_full_volume[:, :, slice_idx].astype(np.int64)).to(device)


# === Original prediction ===
# For 3D models, nnUNet typically expects a full volume and will output a full volume prediction.
# The output will be (B, Num_Classes, D, H, W)
with torch.no_grad():
    output = model(img_tensor_normalized) # Output is (B, NumClasses, D, H, W)
    # Get the prediction for the specific slice for visualization
    pred_full_volume = output.argmax(dim=1).squeeze() # Shape (D, H, W)
    pred_slice = pred_full_volume[slice_idx, :, :] # Extract slice for visualization


# === FGSM Attack ===
# Create a clone for gradient computation to avoid modifying the original img_tensor
img_tensor_attack = img_tensor_normalized.clone().detach().requires_grad_(True)
output_attack_full = model(img_tensor_attack) # This output is (B, Num_Classes, D, H, W)

# We need to calculate loss on the full 3D output and full 3D ground truth if we are doing a 3D attack
# The gt_slice is only 2D. We need the full 3D ground truth volume for loss calculation.
# Ensure gt_full_volume is also a tensor and on the correct device.
gt_full_volume_tensor = torch.tensor(gt_full_volume.astype(np.int64)).unsqueeze(0).to(device) # [1, D, H, W]

# Cross-entropy expects (N, C, ...) and target (N, ...)
# For 3D segmentation, it's (N, Num_Classes, D, H, W) and (N, D, H, W)
loss = cross_entropy(output_attack_full, gt_full_volume_tensor)

model.zero_grad()
img_tensor_attack.grad = None
loss.backward()
data_grad = img_tensor_attack.grad.data

adv_image = img_tensor_attack + epsilon * data_grad.sign()
adv_image = torch.clamp(adv_image, -5, 5)

# --- Adversarial Prediction ---
with torch.no_grad():
    adv_output_full = model(adv_image)
    adv_pred_full_volume = adv_output_full.argmax(dim=1).squeeze() # Shape (D, H, W)
    adv_pred_slice = adv_pred_full_volume[slice_idx, :, :] # Extract slice for visualization


# === Dice Calculation ===
# Dice is calculated on the *specific slice* for visualization purposes,
# but note that the attack was on the full 3D volume.
d_orig = dice(gt_slice_tensor, pred_slice, 3)  # ET class
d_adv = dice(gt_slice_tensor, adv_pred_slice, 3)

# --- Save Outputs ---
os.makedirs(out_dir, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Original Slice: show one channel (e.g., T1ce, which was modality_idx=1)
# Note: img_tensor_normalized is [1, C, D, H, W]. Need to extract one slice and one channel for imshow.
axs[0].imshow(img_tensor_normalized.squeeze(0)[1, slice_idx, :, :].cpu().numpy(), cmap='gray')
axs[0].set_title("Original Slice (Modality 1)")

axs[1].imshow(pred_slice.cpu().numpy(), cmap='viridis') # Already extracted and moved to CPU
axs[1].set_title(f"Original Prediction\nDice={d_orig:.3f}")

axs[2].imshow(adv_pred_slice.cpu().numpy(), cmap='viridis') # Already extracted and moved to CPU
axs[2].set_title(f"Adversarial Prediction\nDice={d_adv:.3f}")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(out_dir, f"{patient_id}_slice{slice_idx}_fgsm_mod_all.png") # Changed filename
plt.savefig(save_path, dpi=150)
plt.close()

print(f"FGSM attack completed successfully for {patient_id}, slice {slice_idx}.")
print(f"Results saved to: {save_path}")
print(f"Original Dice (ET): {d_orig:.3f}")
print(f"Adversarial Dice (ET): {d_adv:.3f}")