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
epsilon = 0.03

# === Paths ===
image_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/imagesTr"
gt_path = f"/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/gt_segmentations/{patient_id}.nii.gz"
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"

# === Load nnUNetTrainer
trainer = nnUNetTrainer(plans=None, configuration="3d_fullres", fold=3, dataset_json=None, device='cpu')
trainer.initialize()
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
trainer.network.load_state_dict(checkpoint['network_weights'])
model = trainer.network.eval()

# === Load image slice
mod_path = os.path.join(image_dir, f"{patient_id}_{modality_idx:04d}.nii.gz")
image = nib.load(mod_path).get_fdata()
gt = nib.load(gt_path).get_fdata()
gt_slice = torch.tensor(gt[:, :, slice_idx].astype(np.int64))

img_slice = image[:, :, slice_idx]
img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
img_tensor = (img_tensor - img_tensor.mean()) / (img_tensor.std() + 1e-6)
img_tensor.requires_grad = True

# === Original prediction
output = model(img_tensor)
pred = output.argmax(dim=1).squeeze()

# === FGSM attack
loss = cross_entropy(output, gt_slice.unsqueeze(0))
model.zero_grad()
loss.backward()
data_grad = img_tensor.grad.data

adv_image = img_tensor + epsilon * data_grad.sign()
adv_image = torch.clamp(adv_image, -5, 5)
adv_output = model(adv_image)
adv_pred = adv_output.argmax(dim=1).squeeze()

# === Dice calculation
def dice(gt, pred, label):
    gt_bin = (gt == label)
    pred_bin = (pred == label)
    return 2.0 * (gt_bin & pred_bin).sum() / (gt_bin.sum() + pred_bin.sum() + 1e-6)

d_orig = dice(gt_slice, pred, 3)  # ET class
d_adv = dice(gt_slice, adv_pred, 3)

# === Save outputs
out_dir = "/sharedscratch/an252/cancerdetectiondataset/brats_attacks/fgsm_attack"
os.makedirs(out_dir, exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(img_slice, cmap='gray')
axs[0].set_title("Original Slice")

axs[1].imshow(pred, cmap='viridis')
axs[1].set_title(f"Prediction\nDice={d_orig:.3f}")

axs[2].imshow(adv_pred, cmap='viridis')
axs[2].set_title(f"Adversarial\nDice={d_adv:.3f}")

for ax in axs:
    ax.axis('off')

plt.tight_layout()
save_path = os.path.join(out_dir, f"{patient_id}_slice{slice_idx}_fgsm.png")
plt.savefig(save_path, dpi=150)
plt.close()

print(f"FGSM result saved to:\n{save_path}")
