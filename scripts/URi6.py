import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk # Explicitly import SimpleITK for resampling
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
# We no longer need nnUNetPredictor for preprocessing the input tensor for the attack
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

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
device = torch.device('cpu') # Keep on CPU for compatibility as per your error logs
print(f"Using device: {device}")

# --- Load JSON contents ---
try:
    dataset_json_content = load_json(dataset_json_path)
    plans_json_content = load_json(plans_path)
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}. Please verify paths.")
    exit()

# --- Custom Preprocessing Function (mimicking nnUNet's behavior) ---
def preprocess_single_image_for_nnunet(raw_image_data_np, raw_image_properties, plans_dict, dataset_dict, device):
    """
    Manually preprocesses a single multi-channel 3D image to match nnUNet's expected input format
    (resampling, normalization, padding).

    Args:
        raw_image_data_np (np.ndarray): Original image data, shape (C, D, H, W).
        raw_image_properties (dict): Properties dict from SimpleITKIO.read_images.
        plans_dict (dict): Loaded nnUNetPlans.json content.
        dataset_dict (dict): Loaded dataset.json content.
        device (torch.device): Target device for the output tensor.

    Returns:
        torch.Tensor: Preprocessed image tensor, shape (1, C, D_preproc, H_preproc, W_preproc), on device.
        torch.Tensor: Corresponding preprocessed GT tensor (if available), shape (1, D_preproc, H_preproc, W_preproc).
                      Returns None if GT not provided or error in loading.
    """
    configuration = "3d_fullres" # Assuming this is the configuration you're using
    
    # 1. Extract relevant plans/dataset info
    current_config = plans_dict['configurations'][configuration]
    target_spacing = np.array(current_config['spacing']) # e.g., [1.0, 1.0, 1.0]
    patch_size = np.array(current_config['patch_size']) # e.g., [128, 128, 128]
    
    # Get normalization stats from plans (per channel Z-score)
    # This structure can vary, inspect your plans_json_content carefully
    intensity_properties = plans_dict['foreground_intensity_properties_per_channel']

    # 2. Resampling (using SimpleITK for consistency with nnUNet)
    # Create SimpleITK image objects
    # Note: SimpleITK expects (W, H, D) order for dimensions. NumPy is (D, H, W).
    # Also, SimpleITK.Image expects image data to be (X, Y, Z) and applies spacing/origin to it.
    # The loaded_image_data is (C, D, H, W)
    
    # For each channel, create an SITK image, resample, then stack.
    resampled_channels_np = []
    
    # Get original spacing and affine from loaded_image_properties
    original_spacing = raw_image_properties['original_spacing'] # (z,y,x) in ITK, (d,h,w) in numpy
    original_size = raw_image_properties['original_shape_for_return'] # (d,h,w)
    original_affine = raw_image_properties['affine']
    
    # Calculate desired new size after resampling to target_spacing
    # new_size_D = round(original_size[0] * original_spacing[0] / target_spacing[0])
    # new_size_H = round(original_size[1] * original_spacing[1] / target_spacing[1])
    # new_size_W = round(original_size[2] * original_spacing[2] / target_spacing[2])
    # nnUNet does a slightly more complex resampling. It calculates the target shape based on current spacing and target spacing.
    # A robust way is to use the `plans.json` output for `current_config['median_image_size_in_voxels']`
    # or to use the `resampling_fn_data` and its kwargs to simulate the exact resampling.
    # For now, let's assume direct scaling to target spacing is okay for demo.
    
    # More accurate way to get target size: (copied from nnUNet's GenericPreprocessor logic)
    # This involves finding the target size such that after padding, the image can be efficiently processed.
    # For simplicity, let's just directly resample to match `patch_size` for now, or if full image is larger, to a multiple.
    # nnUNet actually resamples to an optimal spacing first, then pads to a power of 2 for each conv stage.
    
    # Let's use a simpler resampling that just targets the *approximate* expected size based on scaling.
    # This is often the source of 'size mismatch' if not done perfectly as nnUNet does.
    
    # A more robust approach might be to try and infer the exact preprocessed size from the loaded plans and
    # the original image's dimensions, then resample to that specific size.
    # This is what nnUNet's preprocessor does.
    
    # Let's try to infer from the `patch_size` and a common nnUNet padding strategy (padding to be divisible by powers of 2).
    # Since it's a 3D fullres model, the expected input size could vary, but the patch size must be compatible.
    
    # For this example, let's simplify and use SimpleITK's ResampleImageFilter directly,
    # targeting the `median_image_size_in_voxels` from the configuration, if available, or just keeping original size.
    
    # Determine output image size after resampling to target_spacing
    # ITK is XYZ, numpy is DZYX (or ZYX in SimpleITKImage which is more like D,H,W)
    # The `original_spacing` is usually (x,y,z). `raw_image_properties['original_spacing']` is in (z,y,x) order, so (D,H,W) order in numpy.
    
    # Let's use the actual resampling method from nnUNet: `resample_data_or_seg_to_shape`
    # This means we need `batchgenerators.transforms.spatial.resampling.resample_data_or_seg_to_shape`
    # which is not directly exposed as a class.
    
    # Ok, let's fall back to manual SimpleITK resampling, which is usually accurate enough for inference.
    # It requires: `input_img`, `new_spacing`, `new_size`, `input_origin`, `input_direction`, `interpolator`, `default_pixel_value`
    
    # Convert numpy array to SimpleITK image for each channel
    sitk_images = []
    for c in range(raw_image_data_np.shape[0]):
        # SimpleITK expects (X, Y, Z) (reversed from (D,H,W) numpy convention if image is loaded (D,H,W))
        # and has its own concept of spacing and origin.
        # SimpleITKIO.read_images gives numpy as (C, D, H, W).
        # We need to transpose to (W, H, D) for SimpleITK.
        img_channel = sitk.GetImageFromArray(raw_image_data_np[c].transpose(2, 1, 0)) # (W, H, D)
        img_channel.SetSpacing(raw_image_properties['original_spacing'][::-1]) # Reverse for ITK (X,Y,Z)
        img_channel.SetOrigin(raw_image_properties['itk_origin']) # Use ITK origin directly from properties
        img_channel.SetDirection(raw_image_properties['itk_direction']) # Use ITK direction
        sitk_images.append(img_channel)

    # Calculate new size based on target spacing
    # This is a critical step where mismatch often occurs.
    # nnUNet's internal resampling ensures powers of 2 for downsampling.
    # A more robust way is to infer the expected output size from `plans_dict` based on `median_image_size_in_voxels`
    # and the specific network configuration.
    
    # For simplicity, let's use the median image size as a guide.
    # However, nnUNet's actual preprocessing pads to nearest multiple of 32 or similar.
    # The actual output size after nnUNet's preprocessing is something like:
    # `np.array(current_config['median_image_size_in_voxels']) * np.array(current_config['spacing']) / target_spacing`
    
    # Let's just use the `patch_size` from plans for inference as the *target* size for simplicity for now,
    # as nnUNet predicts on patches during inference.
    # If the image is larger than patch_size, `predict_sliding_window` handles it.
    
    # A better approach: Figure out the actual target size after nnUNet's preprocessor.
    # This involves: initial resampling (based on target spacing), then padding to a multiple of 2^n_stages.
    # For `3d_fullres`, n_stages is 6, so final dim should be divisible by 32.
    
    # Given the previous error `expected input[1, 1, 960, 240, 155]` and now `expected size 16 but got size 15`,
    # this implies the dimensions after initial loading were (240, 240, 155).
    #
    # Let's calculate the size if resampled to target spacing [1.0, 1.0, 1.0]
    # Assuming original_spacing (D,H,W) and original_size (D,H,W)
    # ITK is X,Y,Z so it's (W,H,D)
    
    # Original image data is (C, D, H, W) in numpy.
    original_shape_xyz = np.array([raw_image_data_np.shape[3], raw_image_data_np.shape[2], raw_image_data_np.shape[1]]) # (W, H, D)
    original_spacing_xyz = np.array([raw_image_properties['original_spacing'][2], raw_image_properties['original_spacing'][1], raw_image_properties['original_spacing'][0]]) # (W_sp, H_sp, D_sp)
    
    target_spacing_xyz = np.array(current_config['spacing'][::-1]) # Assuming plans stores (D,H,W) or (Z,Y,X)
    
    new_size_xyz = np.round(original_shape_xyz * original_spacing_xyz / target_spacing_xyz).astype(int)
    
    # Padding to make it divisible by 32 (for a 6-stage UNet)
    divisible_by = 2**(len(current_config['architecture']['arch_kwargs']['strides']) - 1) # Assuming first stride is 1
    # Check the strides in your provided plans JSON:
    # "strides": [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
    # This means 5 pooling layers, so 2^5 = 32.
    
    padded_size_xyz = np.ceil(new_size_xyz / divisible_by).astype(int) * divisible_by
    
    # Resample all channels
    resampled_data_channels = []
    for sitk_img in sitk_images:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img) # Use original image as reference to set properties
        resampler.SetOutputSpacing(target_spacing_xyz)
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetSize(padded_size_xyz.tolist()) # Set new size (XYZ)
        resampler.SetInterpolator(sitk.sitkBSpline) # For images, use linear or B-spline
        
        resampled_img_sitk = resampler.Execute(sitk_img)
        # Convert back to numpy (D, H, W)
        resampled_channels_np.append(sitk.GetArrayFromImage(resampled_img_sitk).transpose(2, 1, 0)) # (D, H, W)

    preprocessed_image_np = np.stack(resampled_channels_np, axis=0) # (C, D_preproc, H_preproc, W_preproc)
    
    print(f"Shape after resampling and padding: {preprocessed_image_np.shape}")
    
    # 3. Normalization (Z-score per channel)
    normalized_image_channels = []
    for c in range(preprocessed_image_np.shape[0]):
        channel_data = preprocessed_image_np[c] # (D_preproc, H_preproc, W_preproc)
        
        # Get mean and std from plans_dict for this channel
        # This assumes your plans_dict is structured as provided earlier
        mean_val = intensity_properties[str(c)]['mean']
        std_val = intensity_properties[str(c)]['std'] + 1e-6 # Add epsilon to prevent div by zero
        
        normalized_channel = (channel_data - mean_val) / std_val
        normalized_image_channels.append(normalized_channel)
        
    preprocessed_image_np = np.stack(normalized_image_channels, axis=0) # (C, D_preproc, H_preproc, W_preproc)

    # 4. Convert to PyTorch tensor
    img_tensor_normalized = torch.tensor(preprocessed_image_np, dtype=torch.float32).unsqueeze(0).to(device) # [1, C, D, H, W]
    
    # --- Preprocess Ground Truth ---
    # Need to apply the same resampling/padding to GT using nearest-neighbor
    # Load raw GT as SimpleITK image
    gt_sitk = sitk.GetImageFromArray(gt_full_volume_raw.transpose(2, 1, 0))
    gt_sitk.SetSpacing(raw_image_properties['original_spacing'][::-1])
    gt_sitk.SetOrigin(raw_image_properties['itk_origin'])
    gt_sitk.SetDirection(raw_image_properties['itk_direction'])

    resampler_gt = sitk.ResampleImageFilter()
    resampler_gt.SetReferenceImage(gt_sitk)
    resampler_gt.SetOutputSpacing(target_spacing_xyz)
    resampler_gt.SetOutputOrigin(gt_sitk.GetOrigin())
    resampler_gt.SetOutputDirection(gt_sitk.GetDirection())
    resampler_gt.SetSize(padded_size_xyz.tolist()) # Use same padded size as image
    resampler_gt.SetInterpolator(sitk.sitkNearestNeighbor) # Nearest neighbor for segmentation

    resampled_gt_sitk = resampler_gt.Execute(gt_sitk)
    gt_preprocessed_np = sitk.GetArrayFromImage(resampled_gt_sitk).transpose(2, 1, 0) # (D, H, W)
    
    gt_full_volume_tensor_for_loss = torch.tensor(gt_preprocessed_np.astype(np.int64)).unsqueeze(0).to(device) # [1, D, H, W]

    return img_tensor_normalized, gt_full_volume_tensor_for_loss


# --- Main Script Execution ---

# === Load nnUNetTrainer (to get the model) ===
trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
trainer.initialize()
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval() # Set model to evaluation mode

# --- Load ALL 4 image modalities and Ground Truth (raw data) ---
image_loader = SimpleITKIO()
full_image_paths = [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)]
try:
    loaded_image_data, loaded_image_properties = image_loader.read_images(full_image_paths) # (C, D, H, W) numpy
except Exception as e:
    print(f"Error loading images: {e}. Ensure all 4 modalities exist and paths are correct.")
    exit()

# --- Preprocess data using our custom function ---
# This function will handle all resampling, normalization, and padding.
# It returns a PyTorch tensor ready for the model.
img_tensor_normalized, gt_full_volume_tensor_for_loss = preprocess_single_image_for_nnunet(
    loaded_image_data, loaded_image_properties, plans_json_content, dataset_json_content, device
)

# Enable gradients for the adversarial attack AFTER preprocessing
img_tensor_normalized.requires_grad = True

print(f"Shape of img_tensor_normalized after custom preprocessing: {img_tensor_normalized.shape}")
print(f"Shape of gt_full_volume_tensor_for_loss after custom preprocessing: {gt_full_volume_tensor_for_loss.shape}")


# Extract the specific ground truth slice for visualization (from the preprocessed GT)
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
img_tensor_attack.grad = None # Clear gradients from previous steps if any
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
# img_tensor_normalized is (B, C, D, H, W)
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