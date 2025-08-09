#!/usr/bin/env python
import torch
import torch.serialization

# path to your BraTS nnU-Net checkpoint
CHECKPOINT = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/" \
             "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/" \
             "fold_3/checkpoint_final.pth"
OUTPUT = "braTS_encoder.pth"

# 1) allow the old numpy scalar global (safe if you trust this checkpoint):
torch.serialization.add_safe_globals([ "numpy._core.multiarray.scalar" ])

# 2) load the full checkpoint (turn off weights_only)
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

# 3) grab its state_dict
sd = ck.get("state_dict", ck)

# 4) filter for encoder params (they live under network.encoder.*)
encoder_sd = {}
prefix = "network.encoder."
for k, v in sd.items():
    if k.startswith(prefix):
        encoder_sd[k[len(prefix):]] = v

# 5) dump out just the encoder weights
torch.save(encoder_sd, OUTPUT)
print(f"✔ Extracted {len(encoder_sd)} tensors → {OUTPUT}")
