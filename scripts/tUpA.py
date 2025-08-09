import os
import json
import importlib
import argparse
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader  
from torch import nn
from torch.optim import AdamW

parser = argparse.ArgumentParser(description="2D slice‐wise fine-tuning of nnU-Net on BCBM")
parser.add_argument("--batch-size",   type=int,   default=16)
parser.add_argument("--lr",           type=float, default=1e-4)
parser.add_argument("--epochs",       type=int,   default=50)
parser.add_argument("--img-size",     type=int, nargs=2, default=[320,384],
                    help="H W for 2D slices")
parser.add_argument("--raw-data-dir", type=str,   required=True,
                    help="nnUNet_raw/Dataset002_BCBM root")
parser.add_argument("--prep-data-dir",type=str,   required=True,
                    help="nnUNet_preprocessed/Dataset002_BCBM root")
parser.add_argument("--pretrained",   type=str,   required=True,
                    help="Path to BraTS pretrained checkpoint_final.pth")
parser.add_argument("--save-path",    type=str,   required=True,
                    help="Where to dump finetuned weights")
args = parser.parse_args()
print("Arguments:", args, flush=True)

BATCH_SIZE = args.batch_size
LR         = args.lr
EPOCHS     = args.epochs
IMG_SIZE   = tuple(args.img_size)
RAW_DATA   = args.raw_data_dir
PREP_DATA  = args.prep_data_dir
PRETRAINED = args.pretrained
SAVE_PATH  = args.save_path

IMG_DIR    = os.path.join(RAW_DATA,  "imagesTr")
GT_DIR     = os.path.join(PREP_DATA, "gt_segmentations")
PLANS_PATH = os.path.join(PREP_DATA, "nnUNetPlans.json")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) 2D Slice Dataset
class Slice2DDataset(Dataset):
    def __init__(self, img_dir, gt_dir, target_size):
        self.slices = []
        self.target_size = target_size
        img_files = sorted(f for f in os.listdir(img_dir) if f.endswith(".nii.gz"))
        for fn in img_files:
            vol_path = os.path.join(img_dir, fn)
            lbl_path = os.path.join(gt_dir, fn.replace("_0000.nii.gz", ".nii.gz"))
            if not os.path.isfile(lbl_path):
                print(f"  Warning: no label for {fn}, skipping", flush=True)
                continue
            data = nib.load(vol_path).get_fdata()
            D, H, W = data.shape
            for z in range(D):
                self.slices.append((vol_path, lbl_path, z))
        if not self.slices:
            raise RuntimeError("No 2D slices found! Check your folders.")
        print(f"Loaded {len(self.slices)} slices from {len(img_files)} volumes", flush=True)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, lbl_path, z = self.slices[idx]
        image = nib.load(img_path).get_fdata()[z]
        label = nib.load(lbl_path).get_fdata()[z]
        image = (image - image.mean()) / (image.std() + 1e-8)
        if image.shape != self.target_size:
            image = np.array(nn.functional.interpolate(
                torch.tensor(image[None, None]), size=self.target_size,
                mode="bilinear", align_corners=False
            )[0,0])
            label = np.array(nn.functional.interpolate(
                torch.tensor(label[None, None].astype(np.float32)),
                size=self.target_size, mode="nearest"
            )[0,0])
        return (
            torch.tensor(image, dtype=torch.float32).unsqueeze(0),  # [1,H,W]
            torch.tensor(label, dtype=torch.float32)               # [H,W]
        )

# 3) Build 2D nnU-Net model
with open(PLANS_PATH, "r") as f:
    plans = json.load(f)
arch_info   = plans["configurations"]["2d"]["architecture"]
arch_kwargs = arch_info["arch_kwargs"]
for key in arch_info.get("_kw_requires_import", []):
    if isinstance(arch_kwargs[key], str):
        mp, cn = arch_kwargs[key].rsplit(".",1)
        mod = importlib.import_module(mp)
        arch_kwargs[key] = getattr(mod, cn)
mp, cn      = arch_info["network_class_name"].rsplit(".",1)
ModelClass  = getattr(importlib.import_module(mp), cn)
model       = ModelClass(input_channels=1, num_classes=1, **arch_kwargs).to(DEVICE)

# 4) Load pretrained (partial)
ckpt = torch.load(PRETRAINED, map_location=DEVICE, weights_only=False)
pre_w = ckpt["network_weights"]
cur_w = model.state_dict()
compatible = {k:v for k,v in pre_w.items() if k in cur_w and cur_w[k].shape==v.shape}
print(f"Loading {len(compatible)}/{len(cur_w)} layers from pretrained", flush=True)
cur_w.update(compatible)
model.load_state_dict(cur_w)

# freeze first 3 encoder stages
for n,p in model.named_parameters():
    if any(n.startswith(f"encoder.stages.{i}.") for i in (0,1,2)):
        p.requires_grad = False

# 5) Prepare training
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
loss_fn   = nn.BCEWithLogitsLoss()
dataset   = Slice2DDataset(IMG_DIR, GT_DIR, IMG_SIZE)
loader    = DataLoader(dataset, batch_size=BATCH_SIZE,
                       shuffle=True, num_workers=4, pin_memory=True)

# 6) Training loop
model.train()
for epoch in range(1, EPOCHS+1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===", flush=True)
    epoch_loss = 0.0
    for i, (imgs, lbls) in enumerate(loader, 1):
        imgs  = imgs.to(DEVICE, non_blocking=True)
        lbls  = lbls.to(DEVICE, non_blocking=True)
        logits = model(imgs)                        # [B,1,H,W]
        loss   = loss_fn(logits, lbls.unsqueeze(1)) # match dims

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if i % 100 == 0:
            print(f"  batch {i}/{len(loader)}  loss {loss.item():.4f}", flush=True)

    avg = epoch_loss / len(loader)
    print(f"[Epoch {epoch}] average loss: {avg:.4f}", flush=True)

# 7) Save
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model saved to {SAVE_PATH}", flush=True)
