import os
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from batchgenerators.utilities.file_and_folder_operations import load_json

# ===========================================================================
# IMPORTANT: nnUNet Environment Variables
# These variables MUST be set in your shell environment *before* running this script,
# or in your SLURM batch script *before* the 'python' command.
# DO NOT uncomment and use os.environ here, as it may be too late for nnUNet's internal path resolution.
# Example:
# export nnUNet_raw="/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw"
# export nnUNet_preprocessed="/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed"
# export nnUNet_results="/sharedscratch/an252/cancerdetectiondataset/nnUNet_results"
# ===========================================================================

print("Starting TorchScript tracing...")
print("---")
print(f"Checking environment variables (as seen by Python):")
print(f"nnUNet_raw: {os.environ.get('nnUNet_raw')}")
print(f"nnUNet_preprocessed: {os.environ.get('nnUNet_preprocessed')}")
print(f"nnUNet_results: {os.environ.get('nnUNet_results')}")
print("---")


# === Paths ===
# Path to the best checkpoint of your trained nnUNet model
checkpoint_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"

# Path where the TorchScript traced model will be saved
# Ensure this path is correct and has write permissions.
# This example saves it within the nnUNet_raw structure, which might be unusual
# but matches your provided path. Consider saving it in nnUNet_results or a dedicated model output folder.
save_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/nnunet_model_traced.pt"

# Define paths to nnUNet plans.json and dataset.json files
# These are crucial for nnUNetTrainer to correctly load model configuration.
plans_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset001_BraTS/nnUNetPlans.json"
dataset_json_path = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset001_BraTS/dataset.json"


# === Make sure the output directory for the traced model exists ===
# This creates parent directories if they don't already exist.
os.makedirs(os.path.dirname(save_path), exist_ok=True)


# === Load JSON contents needed for nnUNetTrainer initialization ===
try:
    dataset_json_content = load_json(dataset_json_path)
    # plans_json_content is implicitly used by nnUNetTrainer when plans_path is provided
except FileNotFoundError as e:
    print(f"Error loading JSON files: {e}.")
    print(f"Please verify paths: {dataset_json_path} and {plans_path}")
    exit()


# === Load model from checkpoint ===
# Using weights_only=False to address _pickle.UnpicklingError for older checkpoints
# or those containing non-weight data.
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# === Initialize nnUNetTrainer ===
# Provide necessary configuration parameters from your nnUNet setup.
# 'device' must be a torch.device object.
device = torch.device('cpu')
trainer = nnUNetTrainer(
    plans=plans_path,                # Path to nnUNetPlans.json
    configuration="3d_fullres",      # Name of the configuration used for training
    fold=3,                          # Fold number used for training (important for fold-specific models)
    dataset_json=dataset_json_content, # Loaded content of dataset.json
    device=device                    # The device (CPU or CUDA)
)
trainer.initialize() # This sets up the network architecture based on the plans

# Load the network weights into the initialized model
trainer.network.load_state_dict(checkpoint['network_weights'])

# Set the model to evaluation mode. This is crucial for inference and tracing.
# It disables dropout, batch normalization updates, etc.
trainer.network.eval()


# === Prepare a dummy input for tracing ===
# The dummy input shape must match the expected input shape of your 3D_fullres model:
# (Batch_size, Num_channels, Depth, Height, Width)
# For BraTS, Num_channels is 4. The spatial dimensions (128, 128, 128) are common patch sizes.
example_input = torch.randn(1, 4, 128, 128, 128).to(device) # Ensure dummy input is on the correct device


# === Perform TorchScript tracing and save the model ===
try:
    # torch.jit.trace records the operations performed on the 'example_input'
    # nnUNet models with deep supervision typically return a tuple of tensors.
    # torch.jit.trace will capture all elements of this tuple.
    traced_model = torch.jit.trace(trainer.network, example_input)

    # Save the traced model to the specified path
    torch.jit.save(traced_model, save_path)
    
    print(f"✅ TorchScript model successfully saved to:\n{save_path}")

except Exception as e:
    print(f"❌ Error during TorchScript tracing or saving: {e}")
    print("\n--- Troubleshooting Tips ---")
    print("1. Ensure the model's forward method returns Tensors or tuples/lists of Tensors directly.")
    print("   If deep supervision returns deeply nested lists (e.g., list of list of tensors),")
    print("   `torch.jit.trace` might struggle. You might need to temporarily modify the model's")
    print("   forward method to return only the primary tensor for tracing purposes, or consider `torch.jit.script`.")
    print("   Example for returning only the primary output for tracing (add this inside a wrapper):")
    print("   `output_tuple = self.original_model(x)`")
    print("   `return output_tuple[0]`")
    print("2. Verify the `example_input` shape (1, 4, 128, 128, 128) matches what the model expects.")
    print("3. Check for any non-torch operations or control flow that `torch.jit.trace` cannot handle (e.g., if/else on tensor values).")