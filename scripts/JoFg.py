#!/usr/bin/env python
import os

# 1) point nnU-Net at your folders using the v2 var names *before* any nnUNet import
os.environ["nnUNet_raw"]         = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw"
os.environ["nnUNet_preprocessed"]= "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed"
os.environ["nnUNet_results"]     = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results"

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# 2) your paths
PLANS_DIR    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans_3d_fullres"
DATASET_JSON = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/dataset.json"
CHECKPOINT   = (
    "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/"
    "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth"
)
FOLD   = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3) instantiate trainer (positional args!)
trainer = nnUNetTrainer(
    PLANS_DIR,
    "3d_fullres",
    FOLD,
    DATASET_JSON,
    DEVICE
)

# 4) build the network (no actual training) and load your BraTS checkpoint
trainer.initialize(training=False)
trainer.load_checkpoint(CHECKPOINT)

# 5) extract just the encoder weights
enc_sd = {
    k[len("encoder."):]: v
    for k, v in trainer.network.state_dict().items()
    if k.startswith("encoder.")
}

# 6) save out
out_fp = "braTS_encoder.pth"
torch.save(enc_sd, out_fp)
print(f"✓ extracted {len(enc_sd)} encoder keys → {out_fp}")
