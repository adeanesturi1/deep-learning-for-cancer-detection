# Place these os.environ assignments AT THE VERY TOP OF YOUR SCRIPT
# before any other imports, especially nnunetv2 imports.
import os

# === Forcefully set nnUNet environment variables at the earliest point ===
# This ensures nnUNetTrainer can find its base paths regardless of shell exports
# or nnunetv2's early module loading.
# Make sure these paths are absolutely correct for your system.
os.environ['nnUNet_raw'] = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/sharedscratch/an252/cancercdetectiondataset/nnUNet_preprocessed" # Corrected typo "cancercdetectiondataset" -> "cancerdetectiondataset"
os.environ['nnUNet_results'] = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results"
# =======================================================================


import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import load_json

print("Starting TorchScript tracing...")
print("---")
print(f"Checking environment variables (as seen by Python):")
print(f"nnUNet_raw: {os.environ.get('nnUNet_raw')}")
print(f"nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed')}")
print(f"nnUNet_results: {os.environ.get('nnUNet_results')}")
print("---")


# === Paths ===
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
# Corrected typo in save_path if it was intended to be in 'cancerdetectiondataset'
save_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/nnunet_model_traced.pt"

# Define paths to nnUNet plans and dataset JSON files
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"


# === Make sure the output directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# === Load JSON contents needed for trainer initialization ===
try:
    dataset_json_content = load_json(dataset_json_path)
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}. Please verify paths: {dataset_json_path} and {plans_path}")
    exit()

# === Load model from checkpoint
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# === Initialize nnUNetTrainer ===
device = torch.device('cpu')
trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3,
                        dataset_json=dataset_json_content, device=device)
trainer.initialize()

# Load network weights from the checkpoint
trainer.network.load_state_dict(checkpoint['network_weights'])
trainer.network.eval()

# === Dummy input ===
example_input = torch.randn(1, 4, 128, 128, 128).to(device)

# === Trace and save ===
try:
    traced_model = torch.jit.trace(trainer.network, example_input)
    torch.jit.save(traced_model, save_path)
    print(f"TorchScript model saved to:\n{save_path}")
except Exception as e:
    print(f"Error during TorchScript tracing or saving: {e}")
    print("Ensure that the model's forward method returns Tensors or tuples/lists of Tensors directly.")
    print("If deep supervision returns nested lists, tracing might fail. Consider adapting the model's forward method for tracing.")