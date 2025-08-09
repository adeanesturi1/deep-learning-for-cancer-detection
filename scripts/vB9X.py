#!/usr/bin/env python
import os

# 1) point nnU-Net at your data/checkpoints BEFORE importing anything from nnunetv2
os.environ["nnUNet_raw_data_base"]   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw"
os.environ["nnUNet_preprocessed"]    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed"
os.environ["nnUNet_results"]         = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results"

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# 2) paths to your BraTS plans, dataset.json, and checkpoint
PLANS_DIR    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans_3d_fullres"
DATASET_JSON = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/dataset.json"
CHECKPOINT   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth"
FOLD         = 3
DEVICE       = "cuda"  # or "cpu"

# 3) instantiate trainer (positional args: plans, configuration, fold, dataset_json, device)
trainer = nnUNetTrainer(
    PLANS_DIR,
    "3d_fullres",
    FOLD,
    DATASET_JSON,
    DEVICE
)

# 4) create the network & load weights
trainer.initialize(training=False)
trainer.load_checkpoint(CHECKPOINT)

# 5) pull out just the encoder weights
encoder_state = {
    k[len("encoder."):]: v
    for k, v in trainer.network.state_dict().items()
    if k.startswith("encoder.")
}

# 6) save
out_fp = "braTS_encoder.pth"
torch.save(encoder_state, out_fp)
print(f"✓ saved encoder state dict ({len(encoder_state)} keys) → {out_fp}")
