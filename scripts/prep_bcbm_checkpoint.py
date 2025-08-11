import torch

orig_ckpt = (
    "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/"
    "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "fold_3/checkpoint_final.pth"
)
new_ckpt  = (
    "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/"
    "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/"
    "fold_3/checkpoint_BCBM_ready.pth"
)

print(f"Loading original checkpoint from {orig_ckpt}")
ckpt = torch.load(orig_ckpt, map_location="cpu", weights_only=False)

# BraTS checkpoint stores everything under 'network_weights'
orig_sd = ckpt.get("network_weights", ckpt)
print("Original first‐conv shape:", orig_sd["encoder.stages.0.0.convs.0.conv.weight"].shape)

# Drop the 4->1 channel conv weights
for k in [
    "encoder.stages.0.0.convs.0.conv.weight",
    "encoder.stages.0.0.convs.0.conv.bias"
]:
    if k in orig_sd:
        orig_sd.pop(k)
        print(f"Dropped {k}")

# Wrap into the dict nnU-Net expects:
out_dict = {"network_weights": orig_sd}

torch.save(out_dict, new_ckpt)
print(f"Saved wrapped state_dict to {new_ckpt}")
