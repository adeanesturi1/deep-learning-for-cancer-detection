import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import load_json # Required to load JSON files

print("Starting TorchScript tracing...")

# === Paths ===
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"
save_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/nnunet_model_traced.pt"

# Define paths to nnUNet plans and dataset JSON files
# These are crucial for nnUNetTrainer initialization
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"


# === Make sure the output directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# === Load JSON contents needed for trainer initialization ===
try:
    dataset_json_content = load_json(dataset_json_path)
    # plans_json_content = load_json(plans_path) # Not strictly needed if `plans_path` is passed directly
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}. Please verify paths: {dataset_json_path} and {plans_path}")
    exit() # FIX: Removed 'ld_3' here

# === Load model from checkpoint
# FIX: Add weights_only=False to address UnpicklingError
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# === Initialize nnUNetTrainer ===
# FIX: Provide correct plans_path, dataset_json_content, and device object
device = torch.device('cpu') # Ensure it's a torch.device object
trainer = nnUNetTrainer(plans=plans_path, configuration="3d_fullres", fold=3,
                        dataset_json=dataset_json_content, device=device)
trainer.initialize() # This sets up network architecture, etc.

# Load network weights from the checkpoint
# The checkpoint contains 'network_weights' as a key
trainer.network.load_state_dict(checkpoint['network_weights'])
trainer.network.eval() # Set the model to evaluation mode for tracing

# === Dummy input ===
# The nnUNet 3d_fullres model expects (B, C, D, H, W) input.
# C=4 for BraTS, D, H, W should be compatible with the model's patch_size or padded size.
# 128x128x128 is a common patch size for 3d_fullres, so it's a reasonable dummy input.
example_input = torch.randn(1, 4, 128, 128, 128).to(device) # Ensure dummy input is on the correct device

# === Trace and save ===
# nnUNet's models with deep supervision return a tuple of outputs.
# torch.jit.trace will trace all outputs of the forward pass.
# If the output is a tuple of tensors, it should work fine.
# If it's a tuple of lists of tensors (as sometimes inferred from previous errors),
# tracing might be problematic or require adjusting the network's forward for tracing.
# Assuming standard nnUNet output (tuple of tensors), this should proceed.
try:
    traced_model = torch.jit.trace(trainer.network, example_input)
    torch.jit.save(traced_model, save_path)
    print(f"TorchScript model saved to:\n{save_path}")
except Exception as e:
    print(f"Error during TorchScript tracing or saving: {e}")
    print("Ensure that the model's forward method returns Tensors or tuples/lists of Tensors directly.")
    print("If deep supervision returns nested lists, tracing might fail. Consider adapting the model's forward method for tracing.")
    # For complex outputs not directly supported by trace, torch.jit.script might be an alternative
    # but it requires the model to be written in a TorchScript-compatible way.
    # Another approach: trace only the main output:
    # class MyTraceableModel(torch.nn.Module):
    #     def __init__(self, original_model):
    #         super().__init__()
    #         self.original_model = original_model
    #     def forward(self, x):
    #         output_tuple = self.original_model(x)
    #         return output_tuple[0] # Return only the highest resolution output for tracing
    # traced_model = torch.jit.trace(MyTraceableModel(trainer.network), example_input)