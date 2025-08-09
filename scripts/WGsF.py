import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# 1) nnUNet paths must be set in the ENV (see above) or you can set them here:
os.environ["nnUNet_raw_data_base"]    = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw"
os.environ["nnUNet_preprocessed"]     = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed"
os.environ["nnUNet_results"]          = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results"

# 2) point to the 3d_fullres plans and your dataset.json
PLANS_DIR      = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans_3d_fullres"
DATASET_JSON   = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/dataset.json"
CHECKPOINT     = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth"
FOLD           = 3
DEVICE         = "cuda"

# 3) instantiate the trainer
trainer = nnUNetTrainer(
    plans_file=PLANS_DIR,
    configuration="3d_fullres",
    fold=FOLD,
    dataset_json=DATASET_JSON,
    device=DEVICE
)
trainer.initialize(training=False)  # only need weights

# 4) load your BraTS checkpoint
trainer.load_checkpoint(CHECKPOINT)

# 5) extract just the encoder weights
encoder_state = {
    k[len("encoder."):]: v
    for k, v in trainer.network.state_dict().items()
    if k.startswith("encoder.")
}

torch.save(encoder_state, "braTS_encoder.pth")
print("Saved BraTS encoder to braTS_encoder.pth")
