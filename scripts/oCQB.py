import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy

print("Starting FGSM attack...")

# === Settings ===
patient_id = "BraTS2021_00005"
slice_idx = 75
modality_idx = 1  # 0=T1, 1=T1ce, 2=T2, 3=FLAIR
epsilon = 0.03 # Perturbation strength for FGSM

# === Paths ===
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_path = f"/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations/{patient_id}.nii.gz"
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks/fgsm_attack"

# === Set the correct path to your nnUNetPlans.json file ===
# This path is confirmed from your previous message.
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"

# --- Device Setup ---
device = torch.device('cpu')

# === Load nnUNetTrainer
# Pass the correct plans_path to the 'plans' argument
trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=None, device=device)
trainer.initialize() # Correctly separated call
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # map_location should match device
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval() # Set model to evaluation mode

# === Load Image Slice and Ground Truth ===
mod_path = os.path.join(image_dir, f"{patient_id}_{modality_idx:04d}.nii.gz")

try:
    image = nib.load(mod_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
except FileNotFoundError as e:
    print(f"Error loading file: {e}. Please check your paths.")
    exit()

gt_slice = torch.tensor(gt[:, :, slice_idx].astype(np.int64)).to(device) # Ensure gt_slice is on the correct device

img_slice = image[:, :, slice_idx]
# Add batch and channel dimensions: [1, 1, H, W]
img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
# Normalize the image slice
img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)
img_tensor.requires_grad = True # Enable gradient computation for the input

# === Original Prediction ===
with torch.no_grad(): # No need to compute gradients for the original prediction
    output = model(img_tensor)
    pred = output.argmax(dim=1).squeeze()

# === FGSM Attack ===
# Make a copy of the input tensor for which gradients will be computed
# This ensures that the original img_tensor (used for visualization later) is not altered by gradient calculations
img_tensor_attack = img_tensor.clone().detach().requires_grad_(True)
output_attack = model(img_tensor_attack)

# Calculate loss with respect to the ground truth
loss = cross_entropy(output_attack, gt_slice.unsqueeze(0))

# Zero existing gradients before backpropagation
model.zero_grad() # Clear gradients of model parameters
img_tensor_attack.grad = None # Clear gradients of input tensor if any exist from previous ops

# Compute gradients of the loss with respect to the input image
loss.backward()

# Get the data_grad (gradient of loss w.r.t. input)
data_grad = img_tensor_attack.grad.data

# Generate adversarial image
adv_image = img_tensor_attack + epsilon * data_grad.sign()
# Clamp to maintain valid pixel ranges after perturbation
adv_image = torch.clamp(adv_image, -5, 5) # Assuming the normalized range is roughly -5 to 5

# --- Adversarial Prediction ---
with torch.no_grad(): # No need to compute gradients for adversarial prediction
    adv_output = model(adv_image)
    adv_pred = adv_output.argmax(dim=1).squeeze()

# === Dice Calculation ===
def dice(gt_seg, pred_seg, label):
    """
    Calculates the Dice similarity coefficient for a specific label.
    """
    gt_bin = (gt_seg == label)
    pred_bin = (pred_seg == label)
    # Handle potential division by zero
    union_sum = gt_bin.sum() + pred_bin.sum()
    if union_sum == 0:
        return 1.0 # Or 0.0, depending on desired behavior for empty ground truth/prediction
    return 2.0 * (gt_bin & pred_bin).sum() / (union_sum + 1e-6)

d_orig = dice(gt_slice, pred, 3)  # Dice for Enhancing Tumor (ET) class (label 3)
d_adv = dice(gt_slice, adv_pred, 3)

# --- Save Outputs ---
os.makedirs(out_dir, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 5)) # Increased figure size for better visualization

# Original Slice
axs[0].imshow(img_slice, cmap='gray') # Use original non-normalized slice for display
axs[0].set_title("Original Slice")

# Original Prediction
axs[1].imshow(pred.cpu().numpy(), cmap='viridis') # Move tensor to CPU for plotting
axs[1].set_title(f"Original Prediction\nDice (ET) = {d_orig:.3f}")

# Adversarial Prediction
axs[2].imshow(adv_pred.cpu().numpy(), cmap='viridis') # Move tensor to CPU for plotting
axs[2].set_title(f"Adversarial Prediction\nDice (ET) = {d_adv:.3f}")

for ax in axs:
    ax.axis('off') # Hide axes ticks and labels

plt.tight_layout()
save_path = os.path.join(out_dir, f"{patient_id}_slice{slice_idx}_fgsm_mod{modality_idx}.png")
plt.savefig(save_path, dpi=150)
plt.close()

print(f"FGSM attack completed successfully for {patient_id}, slice {slice_idx}, modality {modality_idx}.")
print(f"Results saved to: {save_path}")
print(f"Original Dice (ET): {d_orig:.3f}")
print(f"Adversarial Dice (ET): {d_adv:.3f}")