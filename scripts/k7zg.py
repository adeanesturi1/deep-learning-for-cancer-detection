import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO # For reading raw data with proper affine/spacing
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor # Import nnUNetPredictor

print("Starting FGSM attack...")
print("---")

# --- Settings ---
patient_id = "BraTS2021_00005"
slice_idx = 75
epsilon = 0.03

# --- Paths ---
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_path = f"/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations/{patient_id}.nii.gz"
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
out_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks/fgsm_attack"

# === Set the correct paths ===
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"

# --- Device Setup ---
device = torch.device('cpu')
print(f"Using device: {device}")

# --- Load nnUNetTrainer (to get the model and relevant configurations) ---
try:
    dataset_json_content = load_json(dataset_json_path)
    plans_json_content = load_json(plans_path) # Useful for direct checks if needed
except FileNotFoundError as e:
    print(f"Error: {e}. Please verify paths: {dataset_json_path} and {plans_path}")
    exit()

trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
trainer.initialize()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval() # Set model to evaluation mode

# --- Initialize nnUNetPredictor ---
# This object handles all the preprocessing, sliding window prediction, and postprocessing.
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True, # Keep on device for speed if GPU, or false if CPU-only pipeline
    device=device,
    verbose=False,
    verbose_preprocessing=False, # Set to True for more debug prints from nnUNet's preprocessor
    allow_tqdm=False
)
# Manually initialize the predictor with components from the trainer
predictor.manual_initialization(
    model,
    trainer.plans_manager,
    trainer.configuration_manager,
    None, # dataset_directory_from_raw_data is not needed when passing data directly
    trainer.dataset_json,
    trainer.__class__.__name__,
    trainer.inference_allowed_mirroring_axes
)

# --- Load ALL 4 image modalities and Ground Truth (raw data) ---
# Use SimpleITKIO to load raw image data and properties
image_loader = SimpleITKIO()
full_image_paths = [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)]
try:
    # loaded_image_data: (C, D, H, W) numpy array
    # loaded_image_properties: dict with affine, spacing, etc.
    loaded_image_data, loaded_image_properties = image_loader.read_images(full_image_paths)
except Exception as e:
    print(f"Error loading images: {e}. Ensure all 4 modalities exist and paths are correct.")
    exit()

# Load raw ground truth
try:
    gt_nifti = nib.load(gt_path)
    gt_full_volume_raw = gt_nifti.get_fdata() # NumPy array (D, H, W)
except FileNotFoundError as e:
    print(f"Error loading GT file: {e}. Please check your paths.")
    exit()

# --- Preprocessing for the Attack ---
# We need to get the *preprocessed* image tensor that the model would actually see.
# The `predict_from_data_array_only` method returns preprocessed data along with logits.
# Run a dummy prediction to get the preprocessed data and its properties
with torch.no_grad(): # Don't track gradients during this initial dummy prediction
    # This call performs preprocessing, sliding window inference, and postprocessing (if enabled)
    # The second return value `preprocessed_data_for_debug` is the exact input tensor to the model.
    # We provide a dummy properties_dict here, as the preprocessor expects it even if empty for this direct call.
    # The internal preprocessor of `predictor` will use `loaded_image_properties` from `manual_initialization`.
    # Let's use an empty dict for properties_dict, as the real properties are stored within `predictor`.
    
    # Temporarily disable requires_grad for the original prediction pass to ensure it's clean
    img_tensor_initial = torch.tensor(loaded_image_data, dtype=torch.float32).unsqueeze(0).to(device)
    # The predict_from_data_array_only method will handle normalization and reshaping to optimal patch size
    # So we give it the raw loaded_image_data (C,D,H,W) np array, and let it handle the rest.
    
    # Use predictor to get the preprocessed input that the model would see.
    # This involves calling an internal preprocessor function on the loaded data.
    # This is slightly tricky, as `predict_from_data_array_only` typically does it all.
    # Let's use a function that returns the preprocessed data directly.
    
    # The `nnUNetPredictor` takes care of preprocessing when you call its `predict_from_data_array_only`.
    # To get the *exact* preprocessed tensor for the attack, we need to extract it *before* the forward pass.
    # The `predictor` class stores the preprocessor object at `predictor.preprocessor`.
    
    preprocessed_image_np, _ = predictor.preprocessor.run_preprocessor(loaded_image_data, loaded_image_properties)
    # Now, preprocessed_image_np is (C, D_preprocessed, H_preprocessed, W_preprocessed) numpy array
    
    img_tensor_normalized = torch.tensor(preprocessed_image_np, dtype=torch.float32).unsqueeze(0).to(device)
    img_tensor_normalized.requires_grad = True # Enable gradients for the adversarial attack
    
print(f"Shape of img_tensor_normalized after nnUNet's Preprocessor: {img_tensor_normalized.shape}")


# --- Preprocess Ground Truth to match image dimensions ---
# Get target shape from preprocessed image tensor
target_C, target_D, target_H, target_W = img_tensor_normalized.shape

gt_full_volume_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device)

# Resample GT to match preprocessed image dimensions using nn.functional.interpolate
# Add batch and channel dims for interpolate: [1, 1, D, H, W]
gt_full_volume_tensor_for_loss = gt_full_volume_tensor.unsqueeze(0).unsqueeze(0)
gt_full_volume_tensor_for_loss = torch.nn.functional.interpolate(gt_full_volume_tensor_for_loss.float(),
                                                                size=(target_D, target_H, target_W),
                                                                mode='nearest-exact',
                                                                align_corners=None)
gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.squeeze(0).squeeze(0).long()
gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.unsqueeze(0) # Add batch dim: [1, D, H, W]
print(f"Shape of gt_full_volume_tensor_for_loss after resampling: {gt_full_volume_tensor_for_loss.shape}")

# Extract the specific ground truth slice for visualization
gt_slice_tensor = gt_full_volume_tensor_for_loss.squeeze(0)[slice_idx, :, :].clone()


# === Original prediction ===
# Now use the preprocessed tensor directly with your model
with torch.no_grad():
    output = model(img_tensor_normalized) # Output is (B, NumClasses, D, H, W)
    pred_full_volume = output.argmax(dim=1).squeeze(0) # Remove batch dimension, shape (D, H, W)
    pred_slice = pred_full_volume[slice_idx, :, :] # Extract slice for visualization


# === FGSM Attack ===
# The attack itself remains the same as it operates on the already preprocessed tensor
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
        return 1.0 # Perfect agreement on absence
    return 2.0 * (gt_bin & pred_bin).sum() / (union_sum + 1e-6)

d_orig = dice(gt_slice_tensor, pred_slice, 3)
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
axs[2].set_title(f"Adversarial\nDice={d_adv:.3f}")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(out_dir, f"{patient_id}_slice{slice_idx}_fgsm_mod_all.png")
plt.savefig(save_path, dpi=150)
plt.close()

print(f"---")
print(f"FGSM attack completed successfully for {patient_id}, slice {slice_idx}.")
print(f"Results saved to: {save_path}")
print(f"Original Dice (ET): {d_orig:.3f}")
print(f"Adversarial Dice (ET): {d_adv:.3f}")