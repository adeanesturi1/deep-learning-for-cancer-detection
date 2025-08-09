import torch
from nnunetv2.run.run_training import load_pretrained_trainer  # utility to get trainer
import json

# 1. point to your checkpoint and dataset.json
CHECKPOINT = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/" \
             "nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth"
DATASET_JSON = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"
PLANS = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans_3d_fullres"

# 2. load the trainer (this builds the network according to the same plans/config)
trainer = load_pretrained_trainer(
    plans_file=PLANS,
    dataset_json=DATASET_JSON,
    trainer_class_name="nnUNetTrainer",
    configuration="3d_fullres",
    fold=3,
    checkpoint=CHECKPOINT,
    device="cpu"  # extract on CPU
)

# 3. grab only the encoder sub-module state_dict
full_sd = trainer.network.state_dict()
enc_sd = {k.replace("encoder.", ""): v
          for k, v in full_sd.items()
          if k.startswith("encoder.")}

# 4. save just the encoder
torch.save(enc_sd, "braTS3d_fullres_encoder.pth")
print("Saved encoder weights to braTS3d_fullres_encoder.pth")
