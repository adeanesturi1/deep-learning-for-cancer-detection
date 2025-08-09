import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.nn.functional import cross_entropy
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import pandas as pd # For saving results to CSV

print("Starting FGSM attack script...")
print("---")

# ===========================================================================
# IMPORTANT: nnUNet Environment Variables
# These variables MUST be set in your shell environment *before* running this script,
# or in your SLURM batch script *before* the 'python' command.
# Example:
# export nnUNet_raw="/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw"
# export nnUNet_preprocessed="/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed"
# export nnUNet_results="/sharedscratch/an252/cancerdetectiondataset/nnUNet_results"
# ===========================================================================

# --- Global Settings & Paths ---
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_base_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations" # Base dir for GT
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
output_base_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks_multi_run" # New base output directory for multiple runs

plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"

device = torch.device('cuda') # Changed to 'cuda' for GPU usage as discussed. Change to 'cpu' if no GPU.
print(f"Using device: {device}", flush=True)

# --- Load JSON contents (once for all attacks) ---
try:
    dataset_json_content = load_json(dataset_json_path)
    plans_json_content = load_json(plans_path)
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}. Please verify paths: {dataset_json_path} and {plans_path}", flush=True)
    exit()

# === Load nnUNetTrainer (model) once ===
# This part is done once outside the attack function for efficiency
print("Initializing nnUNetTrainer and loading model weights...", flush=True)
try:
    trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3, dataset_json=dataset_json_content, device=device)
    trainer.initialize()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    trainer.network.load_state_dict(checkpoint['network_weights'])
    model = trainer.network.eval() # Set model to evaluation mode
    print("Model loaded successfully.", flush=True)
except Exception as e:
    print(f"FATAL ERROR: Could not initialize nnUNetTrainer or load model: {e}", flush=True)
    exit()

# --- Helper function to extract logits from nnUNet output ---
def _get_logits_from_nnunet_output(output):
    """
    Extracts the highest-resolution logits tensor from nnUNet model output,
    handling deep supervision (tuple/list output) and potential nested lists.
    """
    if isinstance(output, (list, tuple)):
        return _get_logits_from_nnunet_output(output[0])
    elif isinstance(output, torch.Tensor):
        return output
    else:
        raise TypeError(f"Unexpected output type from nnUNet model: {type(output)}. "
                        "Expected a torch.Tensor, list, or tuple containing tensors.")

# --- Custom Preprocessing Function (mimicking nnUNet's behavior) ---
def preprocess_single_image_for_nnunet(raw_image_data_np, raw_image_properties, plans_dict, device):
    configuration = "3d_fullres" # Assuming this is the configuration you're using
    
    current_config = plans_dict['configurations'][configuration]
    target_spacing_plans = np.array(current_config['spacing'])
    intensity_properties = plans_dict['foreground_intensity_properties_per_channel']

    original_spacing_xyz = np.array(raw_image_properties['spacing'])
    sitk_stuff = raw_image_properties['sitk_stuff']
    original_origin_xyz = sitk_stuff['origin']
    original_direction_matrix = sitk_stuff['direction']

    sitk_images = []
    for c in range(raw_image_data_np.shape[0]):
        img_channel_np_xyz = raw_image_data_np[c].transpose(2, 1, 0)
        sitk_img = sitk.GetImageFromArray(img_channel_np_xyz)
        sitk_img.SetSpacing(original_spacing_xyz)
        sitk_img.SetOrigin(original_origin_xyz)
        sitk_img.SetDirection(original_direction_matrix)
        sitk_images.append(sitk_img)

    current_shape_xyz = np.array(sitk_images[0].GetSize())
    target_spacing_xyz = target_spacing_plans[::-1]

    new_size_xyz = np.round(current_shape_xyz * original_spacing_xyz / target_spacing_xyz).astype(int)

    divisible_by = 2**(len(current_config['architecture']['arch_kwargs']['strides']) - 1)
    padded_size_xyz = np.ceil(new_size_xyz / divisible_by).astype(int) * divisible_by
    
    # print(f"Original image shape (D,H,W): {raw_image_data_np.shape[1:]}") # Keep as debug if needed
    # print(f"Resampled/padded target shape (W,H,D): {padded_size_xyz}") # Keep as debug if needed

    resampled_data_channels_np_dhw = []
    for sitk_img in sitk_images:
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitk_img)
        resampler.SetOutputSpacing(target_spacing_xyz)
        resampler.SetOutputOrigin(sitk_img.GetOrigin())
        resampler.SetOutputDirection(sitk_img.GetDirection())
        resampler.SetSize(padded_size_xyz.tolist())
        resampler.SetInterpolator(sitk.sitkBSpline)
        resampler.SetOutputPixelType(sitk.sitkFloat32)

        resampled_img_sitk = resampler.Execute(sitk_img)
        resampled_data_channels_np_dhw.append(sitk.GetArrayFromImage(resampled_img_sitk).transpose(2, 1, 0))

    preprocessed_image_np = np.stack(resampled_data_channels_np_dhw, axis=0)
    
    normalized_image_channels = []
    for c in range(preprocessed_image_np.shape[0]):
        channel_data = preprocessed_image_np[c]
        
        mean_val = intensity_properties[str(c)]['mean']
        std_val = intensity_properties[str(c)]['std'] + 1e-6
        
        normalized_channel = (channel_data - mean_val) / std_val
        normalized_image_channels.append(normalized_channel)
        
    preprocessed_image_np = np.stack(normalized_image_channels, axis=0)

    img_tensor_normalized = torch.tensor(preprocessed_image_np, dtype=torch.float32).unsqueeze(0).to(device)

    return img_tensor_normalized

# --- Dice Calculation (Enhanced for multiple labels) ---
def dice(gt_seg, pred_seg, labels_to_include):
    if not isinstance(labels_to_include, (list, tuple)):
        labels_to_include = [labels_to_include] # Ensure it's iterable for single label too

    gt_bin = torch.zeros_like(gt_seg, dtype=torch.bool)
    pred_bin = torch.zeros_like(pred_seg, dtype=torch.bool)

    for label in labels_to_include:
        gt_bin = gt_bin | (gt_seg == label) # OR operation for combined regions
        pred_bin = pred_bin | (pred_seg == label)
    
    gt_bin_np = gt_bin.cpu().numpy()
    pred_bin_np = pred_bin.cpu().numpy()

    intersection = (gt_bin_np & pred_bin_np).sum()
    union = gt_bin_np.sum() + pred_bin_np.sum()
    
    if union == 0:
        return 1.0 # Perfect agreement on absence
    return 2.0 * intersection / (union + 1e-6)

# --- Define Tumor Regions for Evaluation ---
TUMOR_REGIONS = {
    "ET": [3],        # Enhancing Tumor
    "ED": [2],        # Peritumoral Edema
    "NCR_NET": [1],   # Necrotic and Non-Enhancing Tumor Core
    "WT": [1, 2, 3],  # Whole Tumor (NCR/NET + ED + ET)
    "TC": [1, 3]      # Tumor Core (NCR/NET + ET)
}


# --- Main Attack Function ---
def run_attack_and_save_results(patient_id, slice_idx, epsilon, model, device, output_base_dir,
                                image_dir, gt_base_dir, plans_json_content, tumor_regions_dict):
    
    print(f"\n--- Running attack for {patient_id}, slice {slice_idx}, epsilon {epsilon:.3f} ---", flush=True)
    
    # Define output directory for this specific attack
    attack_output_dir = os.path.join(output_base_dir, f"{patient_id}_epsilon{epsilon:.3f}")
    os.makedirs(attack_output_dir, exist_ok=True)

    result_data = {
        'patient_id': patient_id,
        'slice_idx': slice_idx,
        'epsilon': epsilon,
        'status': 'Success'
    }

    try:
        # --- Load ALL 4 image modalities and Ground Truth (raw data) ---
        image_loader = SimpleITKIO()
        full_image_paths = [os.path.join(image_dir, f"{patient_id}_{i:04d}.nii.gz") for i in range(4)]
        
        loaded_image_data, loaded_image_properties = image_loader.read_images(full_image_paths)
        
        gt_path = os.path.join(gt_base_dir, f"{patient_id}.nii.gz")
        gt_nifti = nib.load(gt_path)
        gt_full_volume_raw = gt_nifti.get_fdata()
        gt_original_affine = gt_nifti.affine # Store original affine for saving later

        # --- Preprocess Image data using custom function ---
        img_tensor_normalized = preprocess_single_image_for_nnunet(
            loaded_image_data, loaded_image_properties, plans_json_content, device
        )
        img_tensor_normalized.requires_grad = True # Enable gradients for the adversarial attack

        # --- Preprocess Ground Truth to match the preprocessed image dimensions ---
        target_D, target_H, target_W = img_tensor_normalized.shape[2:]
        gt_full_volume_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device)

        gt_full_volume_tensor_for_loss = gt_full_volume_tensor.unsqueeze(0).unsqueeze(0)
        gt_full_volume_tensor_for_loss = torch.nn.functional.interpolate(gt_full_volume_tensor_for_loss.float(),
                                                                        size=(target_D, target_H, target_W),
                                                                        mode='nearest-exact',
                                                                        align_corners=None)
        gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.squeeze(0).squeeze(0).long()
        gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.unsqueeze(0)
        
        gt_slice_tensor = gt_full_volume_tensor_for_loss.squeeze(0)[slice_idx, :, :].clone()

        # --- Original prediction ---
        with torch.no_grad():
            output_logits = _get_logits_from_nnunet_output(model(img_tensor_normalized))
            pred_full_volume = output_logits.argmax(dim=1).squeeze(0)
            pred_slice = pred_full_volume[slice_idx, :, :]

        # --- FGSM Attack ---
        img_tensor_attack = img_tensor_normalized.clone().detach().requires_grad_(True)
        output_attack_logits = _get_logits_from_nnunet_output(model(img_tensor_attack))

        loss = cross_entropy(output_attack_logits, gt_full_volume_tensor_for_loss)

        model.zero_grad()
        img_tensor_attack.grad = None
        loss.backward()
        data_grad = img_tensor_attack.grad.data

        adv_image_tensor = img_tensor_attack + epsilon * data_grad.sign()
        adv_image_tensor = torch.clamp(adv_image_tensor, -5, 5) # Clamped normalized values

        # --- Adversarial Prediction ---
        with torch.no_grad():
            adv_output_logits = _get_logits_from_nnunet_output(model(adv_image_tensor))
            adv_pred_full_volume = adv_output_logits.argmax(dim=1).squeeze(0)
            adv_pred_slice = adv_pred_full_volume[slice_idx, :, :]

        # --- Dice Calculation for All Regions ---
        for region_name, labels in tumor_regions_dict.items():
            d_orig_region = dice(gt_slice_tensor, pred_slice, labels)
            d_adv_region = dice(gt_slice_tensor, adv_pred_slice, labels)
            result_data[f'dice_orig_{region_name}'] = d_orig_region
            result_data[f'dice_adv_{region_name}'] = d_adv_region
            print(f"    Dice ({region_name}): Original={d_orig_region:.3f}, Adversarial={d_adv_region:.3f}", flush=True)

        # --- Save Outputs ---
        # 1. Prediction Overlays (PNG)
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(img_tensor_normalized[0, 1, slice_idx, :, :].cpu().detach().numpy(), cmap='gray')
        axs[0].set_title("Original Slice (Modality 1)")
        axs[1].imshow(pred_slice.cpu().numpy(), cmap='viridis')
        axs[1].set_title(f"Original Prediction\nDice={result_data['dice_orig_ET']:.3f}") # Display ET Dice on plot
        axs[2].imshow(adv_pred_slice.cpu().numpy(), cmap='viridis')
        axs[2].set_title(f"Adversarial\nDice={result_data['dice_adv_ET']:.3f}") # Display ET Dice on plot
        for ax in axs: ax.axis('off')
        plt.tight_layout()
        overlay_save_path = os.path.join(attack_output_dir, f"{patient_id}_slice{slice_idx}_epsilon{epsilon:.3f}_overlay.png")
        plt.savefig(overlay_save_path, dpi=150)
        plt.close()
        print(f"    Prediction overlay saved to:\n    {overlay_save_path}", flush=True)
        result_data['overlay_path'] = overlay_save_path

        # 2. Perturbed Image (NIfTI)
        denormalized_channels = []
        for c in range(adv_image_tensor.shape[1]):
            channel_data = adv_image_tensor[0, c, :, :, :] # remove batch dim
            mean_val = plans_json_content['foreground_intensity_properties_per_channel'][str(c)]['mean']
            std_val = plans_json_content['foreground_intensity_properties_per_channel'][str(c)]['std'] + 1e-6
            denormalized_channels.append((channel_data * std_val + mean_val).cpu().numpy())
        
        denormalized_adv_image_np = np.stack(denormalized_channels, axis=0) # (C, D_preproc, H_preproc, W_preproc)
        adv_nifti_np = denormalized_adv_image_np.transpose(1, 2, 3, 0) # (D, H, W, C) for NIfTI

        # Use a placeholder affine that corresponds to the preprocessed space (1mm isotropic for example)
        # For precise visualization in external viewers that respect affine, consider how the preprocessor handles it.
        # This one is a simple identity transform if spacing is 1,1,1 and origin is 0,0,0
        preprocessed_affine = np.array([[1., 0., 0., 0.],
                                        [0., 1., 0., 0.],
                                        [0., 0., 1., 0.],
                                        [0., 0., 0., 1.]])
        
        # If your preprocessed images typically use 1mm isotropic spacing
        # you might want to use that for the affine for clarity in viewers.
        # This assumes the preprocessed data's origin starts at 0,0,0 and pixel spacing is 1,1,1
        
        adv_nifti = nib.Nifti1Image(adv_nifti_np, preprocessed_affine)
        adv_nifti_save_path = os.path.join(attack_output_dir, f"{patient_id}_slice{slice_idx}_epsilon{epsilon:.3f}_adv_image.nii.gz")
        nib.save(adv_nifti, adv_nifti_save_path)
        print(f"    Perturbed NIfTI image saved to:\n    {adv_nifti_save_path}", flush=True)
        result_data['adv_nifti_path'] = adv_nifti_save_path

    except Exception as e:
        print(f"ERROR: Attack failed for {patient_id}, slice {slice_idx}, epsilon {epsilon:.3f}: {e}", flush=True)
        result_data['status'] = f'Failed: {e}'
        # Fill Dice with NaN for failed cases
        for region_name in tumor_regions_dict.keys():
            result_data[f'dice_orig_{region_name}'] = np.nan
            result_data[f'dice_adv_{region_name}'] = np.nan
        result_data['overlay_path'] = None
        result_data['adv_nifti_path'] = None

    return result_data

# --- Main Execution Loop ---
if __name__ == "__main__":
    
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Define cases to attack ---
    patient_cases = ["BraTS2021_00005", "BraTS2021_00006"] # Add more patient IDs as needed
    slices_to_attack = [75, 80] # Example slices to process for each patient

    # Epsilon values to test (optional, can be a single value)
    epsilon_values = [0.03, 0.05] # Example epsilon values

    all_results = []

    print(f"\nStarting multi-case FGSM attack. Total attack configurations: {len(patient_cases) * len(slices_to_attack) * len(epsilon_values)}", flush=True)

    for patient_id in patient_cases:
        for slice_idx in slices_to_attack:
            for epsilon in epsilon_values:
                result = run_attack_and_save_results(
                    patient_id, slice_idx, epsilon, model, device, output_base_dir,
                    image_dir, gt_base_dir, plans_json_content, TUMOR_REGIONS
                )
                all_results.append(result)

    print("\n--- All attacks completed. Saving summary. ---", flush=True)

    # Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(output_base_dir, "fgsm_attack_summary.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Attack summary saved to:\n{results_csv_path}", flush=True)

    print("\nFGSM attack script finished.", flush=True)