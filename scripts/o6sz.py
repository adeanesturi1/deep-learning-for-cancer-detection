import torch
import sys

CHECKPOINT = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/" \
             "Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/" \
             "fold_3/checkpoint_final.pth"
OUTPUT = "braTS_encoder.pth"

import torch.serialization
torch.serialization.add_safe_globals([ "numpy._core.multiarray.scalar" ])
ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
if "network_weights" in ck:
    sd = ck["network_weights"]
else:
    print("ERROR: no 'network_weights' field found in checkpoint")
    sys.exit(1)

print(">>> example keys inside network_weights:")
for k in list(sd.keys())[:20]:
    print("   ", k)
print()
if any(k.startswith("encoder.") for k in sd):
    prefix = "encoder."
elif any(k.startswith("network.encoder.") for k in sd):
    prefix = "network.encoder."
elif any(k.startswith("network.encoder.stages") for k in sd):
    prefix = "network.encoder."
else:
    roots = sorted({k.split(".", 1)[0] for k in sd})
    print("ERROR: no encoder.* keys found.")
    print("       Top-level modules in state_dict:", roots)
    sys.exit(1)

# build new dict
encoder_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
if not encoder_sd:
    print(f"ERROR: matched prefix '{prefix}' but extracted 0 tensors")
    sys.exit(1)

torch.save(encoder_sd, OUTPUT)
print(f"Extracted {len(encoder_sd)} encoder tensors (prefix '{prefix}') → {OUTPUT}")
