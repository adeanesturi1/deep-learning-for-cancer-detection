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
device = torch.device('cpu') # Or 'cuda' if you want to use GPU
print(f"Using device: {device}")

# --- Load JSON contents ---
try:
    dataset_json_content = load_json(dataset_json_path)
    plans_json_content = load_json(plans_path)
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}. Please verify paths.")
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
    configuration = "3d_fullres"
    
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
    
    print(f"Original image shape (D,H,W): {raw_image_data_np.shape[1:]}")
    print(f"Resampled/padded target shape (W,H,D): {padded_size_xyz}")

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
    loaded_image_data, loaded_image_properties = image_loader.read_images(full_image_paths)
    print("Keys in loaded_image_properties:", loaded_image_properties.keys())
    print("Keys in loaded_image_properties['sitk_stuff']:", loaded_image_properties['sitk_stuff'].keys())
except Exception as e:
    print(f"Error loading images: {e}. Ensure all 4 modalities exist and paths are correct.")
    exit()

# Load raw ground truth
try:
    gt_nifti = nib.load(gt_path)
    gt_full_volume_raw = gt_nifti.get_fdata()
except FileNotFoundError as e:
    print(f"Error loading GT file: {e}. Please check your paths.")
    exit()

# --- Preprocess Image data using our custom function ---
img_tensor_normalized = preprocess_single_image_for_nnunet(
    loaded_image_data, loaded_image_properties, plans_json_content, device
)

# Enable gradients for the adversarial attack AFTER preprocessing
# This tensor needs gradients for the attack, but not for visualization before the attack.
# We will create a detached copy for visualization.
img_tensor_normalized.requires_grad = True

print(f"Shape of img_tensor_normalized after custom preprocessing: {img_tensor_normalized.shape}")


# --- Preprocess Ground Truth to match the preprocessed image dimensions ---
target_D, target_H, target_W = img_tensor_normalized.shape[2:]

gt_full_volume_tensor = torch.tensor(gt_full_volume_raw.astype(np.int64)).to(device)

# Resample GT to match preprocessed image dimensions using nn.functional.interpolate
gt_full_volume_tensor_for_loss = gt_full_volume_tensor.unsqueeze(0).unsqueeze(0)
gt_full_volume_tensor_for_loss = torch.nn.functional.interpolate(gt_full_volume_tensor_for_loss.float(),
                                                                size=(target_D, target_H, target_W),
                                                                mode='nearest-exact',
                                                                align_corners=None)
gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.squeeze(0).squeeze(0).long()
gt_full_volume_tensor_for_loss = gt_full_volume_tensor_for_loss.unsqueeze(0)
print(f"Shape of gt_full_volume_tensor_for_loss after resampling: {gt_full_volume_tensor_for_loss.shape}")

gt_slice_tensor = gt_full_volume_tensor_for_loss.squeeze(0)[slice_idx, :, :].clone()


# === Original prediction ===
# Use the helper function to extract the logits tensor
output_logits = _get_logits_from_nnunet_output(model(img_tensor_normalized))

pred_full_volume = output_logits.argmax(dim=1).squeeze(0)
pred_slice = pred_full_volume[slice_idx, :, :]


# === FGSM Attack ===
# Create a clone for gradient computation that will be perturbed
img_tensor_attack = img_tensor_normalized.clone().detach().requires_grad_(True)
# Use the helper function for adversarial attack output
output_attack_logits = _get_logits_from_nnunet_output(model(img_tensor_attack))

loss = cross_entropy(output_attack_logits, gt_full_volume_tensor_for_loss)

model.zero_grad() # Zero gradients for the model parameters
img_tensor_attack.grad = None # Clear gradients for the input tensor explicitly
loss.backward()
data_grad = img_tensor_attack.grad.data

adv_image = img_tensor_attack + epsilon * data_grad.sign()
adv_image = torch.clamp(adv_image, -5, 5)

# --- Adversarial Prediction ---
with torch.no_grad():
    adv_output_logits = _get_logits_from_nnunet_output(model(adv_image))
    adv_pred_full_volume = adv_output_logits.argmax(dim=1).squeeze(0)
    adv_pred_slice = adv_pred_full_volume[slice_idx, :, :]


# === Dice Calculation ===
def dice(gt_seg, pred_seg, label):
    gt_bin = (gt_seg == label)
    pred_bin = (pred_seg == label)
    union_sum = gt_bin.sum() + pred_bin.sum()
    if union_sum == 0:
        return 1.0
    return 2.0 * (gt_bin & pred_bin).sum() / (union_sum + 1e-6)

d_orig = dice(gt_slice_tensor, pred_slice, 3)
d_adv = dice(gt_slice_tensor, adv_pred_slice, 3)

# --- Save Outputs ---
os.makedirs(out_dir, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# FIX: Detach the tensor before converting to numpy for plotting
axs[0].imshow(img_tensor_normalized[0, 1, slice_idx, :, :].cpu().detach().numpy(), cmap='gray')
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
print(f"Results saved to:\n{save_path}")
print(f"Original Dice (ET): {d_orig:.3f}")
print(f"Adversarial Dice (ET): {d_adv:.3f}")