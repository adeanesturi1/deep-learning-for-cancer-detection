import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# === Paths ===
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
save_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/nnUNetTrainer_FrozenEncoderBCBM__nnUNetPlans__3d_fullres/fold_3/nnunet_model_traced.pt"

# === Make sure the output directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# === Load model from checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu')
trainer = nnUNetTrainer(plans=None, configuration="3d_fullres", fold=3, dataset_json=None, device='cpu')
trainer.initialize()
trainer.network.load_state_dict(checkpoint['network_weights'])

# === Dummy input
example_input = torch.randn(1, 4, 128, 128, 128)

# === Trace and save
traced_model = torch.jit.trace(trainer.network, example_input)
torch.jit.save(traced_model, save_path)

print(f"✅ TorchScript model saved to:\n{save_path}")
