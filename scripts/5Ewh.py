#!/usr/bin/env python
import torch
import sys

CHECKPOINT = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/" \
             "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/" \
             "fold_3/checkpoint_final.pth"
OUTPUT = "braTS_encoder.pth"

# Allow the old numpy scalar global so torch.load works
import torch.serialization
torch.serialization.add_safe_globals([ "numpy._core.multiarray.scalar" ])

# Load the checkpoint (force full unpickling)
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

# Grab the actual state dict
if isinstance(ck, dict) and "state_dict" in ck:
    sd = ck["state_dict"]
else:
    sd = ck

# Show a few keys so you can see what prefix to use
print(">>> example keys:")
for k in list(sd.keys())[:20]:
    print("   ", k)
print()

# Determine what prefix your encoder keys actually use
if any(k.startswith("network.encoder.") for k in sd):
    prefix = "network.encoder."
elif any(k.startswith("encoder.") for k in sd):
    prefix = "encoder."
else:
    # fallback: list top-level module names
    roots = sorted({k.split(".", 1)[0] for k in sd})
    print("ERROR: no 'encoder.' or 'network.encoder.' keys found.")
    print("       Top-level modules in this state_dict:", roots)
    sys.exit(1)

# Extract only those weights
encoder_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

if not encoder_sd:
    print(f"ERROR: no keys matched prefix '{prefix}' (extracted 0 tensors)")
    sys.exit(1)

# Save out
torch.save(encoder_sd, OUTPUT)
print(f"✔ Extracted {len(encoder_sd)} tensors (prefix '{prefix}') → {OUTPUT}")
