#!/usr/bin/env python
import os
import torch

# 1) point these at your installation
CHECKPOINT = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/" \
             "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/" \
             "fold_3/checkpoint_final.pth"
OUTPUT = "braTS_encoder.pth"

# 2) load the checkpoint
ck = torch.load(CHECKPOINT, map_location="cpu")
# nnUNetv2 saves the model under either 'state_dict' or directly at top level
sd = ck.get("state_dict", ck)

# 3) filter out only the encoder parameters
encoder_sd = {}
prefix = "network.encoder."
for k, v in sd.items():
    if k.startswith(prefix):
        # strip the prefix so you can load cleanly into your own U-Net
        newk = k[len(prefix):]
        encoder_sd[newk] = v

# 4) save
torch.save(encoder_sd, OUTPUT)
print(f"Saved encoder weights ({len(encoder_sd)} tensors) to {OUTPUT}")
