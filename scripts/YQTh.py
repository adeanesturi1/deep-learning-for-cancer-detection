#!/usr/bin/env python
# scripts/trainher2unettransfer.py

import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
# NOTE ON MEMORY:
# • We default base_feats=16 and target_z=128 to keep you under ~24 GB.
# • Mixed-precision (amp) will cut your peak by ~50%.
# • If you still OOM: drop base_feats to 8, target_z to 64, or batch_size to 1.
# -----------------------------------------------------------------------------

class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=16):
        super().__init__()
        self.enc1 = self._block(in_ch,     base_feats)
        self.enc2 = self._block(base_feats, base_feats*2)
        self.enc3 = self._block(base_feats*2, base_feats*4)
        self.enc4 = self._block(base_feats*4, base_feats*8)
        self.bot  = self._block(base_feats*8, base_feats*16)

        self.up4  = nn.ConvTranspose3d(base_feats*16, base_feats*8, 2, 2)
        self.dec4 = self._block(base_feats*16, base_feats*8)
        self.up3  = nn.ConvTranspose3d(base_feats*8,  base_feats*4, 2, 2)
        self.dec3 = self._block(base_feats*8,  base_feats*4)
        self.up2  = nn.ConvTranspose3d(base_feats*4,  base_feats*2, 2, 2)
        self.dec2 = self._block(base_feats*4,  base_feats*2)
        self.up1  = nn.ConvTranspose3d(base_feats*2,  base_feats,   2, 2)
        self.dec1 = self._block(base_feats*2,  base_feats)

        self.seg_head   = nn.Conv3d(base_feats, 1, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_feats*16, 1)
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool3d(e1,2))
        e3 = self.enc3(nn.functional.max_pool3d(e2,2))
        e4 = self.enc4(nn.functional.max_pool3d(e3,2))
        b  = self.bot(nn.functional.max_pool3d(e4,2))

        d4 = self.dec4(torch.cat([self.up4(b),  e4],1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3],1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],1))

        seg = self.seg_head(d1)
        cls = self.classifier(b).squeeze(1)
        return seg, cls


class BCBM3DDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, metadata_csv, target_z):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"] == "+").astype(np.float32)
        self.cases  = df[["nnUNet_ID","HER2"]].to_records(index=False)
        self.raw    = raw_dir
        self.gt     = gt_dir
        self.target_z = target_z

    def __len__(self): return len(self.cases)

    def __getitem__(self, i):
        case_id, her2 = self.cases[i]
        img_path = os.path.join(self.raw, "imagesTr", f"{case_id}_0000.nii.gz")
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        # look in gt/gt_segmentations or gt/labelsTr
        p1 = os.path.join(self.gt, "gt_segmentations", f"{case_id}.nii.gz")
        p2 = os.path.join(self.gt, "labelsTr",          f"{case_id}.nii.gz")
        if   os.path.isfile(p1): seg_path = p1
        elif os.path.isfile(p2): seg_path = p2
        else: raise FileNotFoundError(f"Missing seg: {p1} or {p2}")

        img = nib.load(img_path).get_fdata().astype(np.float32)
        seg = nib.load(seg_path).get_fdata().astype(np.float32)

        # center‐crop or pad Z to target_z
        H,W,Z0  = img.shape; Zt = self.target_z
        if Z0 > Zt:
            s = (Z0-Zt)//2
            img = img[:,:,s:s+Zt]; seg = seg[:,:,s:s+Zt]
        elif Z0 < Zt:
            p = Zt-Z0; f,pad_b = p//2, p-p//2
            img = np.pad(img,((0,0),(0,0),(f,pad_b)),"constant")
            seg = np.pad(seg,((0,0),(0,0),(f,pad_b)),"constant")

        return (
            torch.from_numpy(img)[None],
            torch.from_numpy(seg)[None],
            torch.tensor(her2)
        )

def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

import re
def remap_braTS_encoder(raw_sd):
    mapped = {}
    # handle keys with or without the "encoder." prefix
    pattern = re.compile(
      r'^(?:encoder\.)?stages\.(\d)\.(?:\d)\.convs\.(\d)\.(conv|norm)\.(weight|bias)$'
    )
    for k,v in raw_sd.items():
        m = pattern.match(k)
        if not m: continue
        stage, conv_i, kind, ptype = map(int if idx<2 else str,
                                         [m.group(1), m.group(3), m.group(4), m.group(5)])
        stage = int(stage); conv_i = int(conv_i)
        prefix = f"enc{stage+1}" if stage < 4 else "bot"
        idx    = conv_i*3 + (0 if kind=="conv" else 1)
        new_k  = f"{prefix}.{idx}.{ptype}"
        mapped[new_k] = v
    return mapped

def train_epoch(model, loader, opt, device, scaler):
    model.train()
    seg_loss = nn.BCEWithLogitsLoss()
    cls_loss = nn.BCEWithLogitsLoss()
    total = 0.0
    for img, seg_gt, her2 in loader:
        img, seg_gt, her2 = img.to(device), seg_gt.to(device), her2.to(device)
        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            seg_logits, cls_logits = model(img)
            loss = seg_loss(seg_logits, seg_gt) + 0.5*cls_loss(cls_logits, her2)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        total += loss.item()
    return total/len(loader)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",       required=True,
                   help="nnUNet_raw/Dataset002_BCBM")
    p.add_argument("--gt_dir",        required=True,
                   help="nnUNet_preprocessed/... (gt_segmentations or labelsTr)")
    p.add_argument("--metadata_csv",  required=True)
    p.add_argument("--pretrained_unet",required=True)
    p.add_argument("--target_z",      type=int,   default=128)
    p.add_argument("--base_feats",    type=int,   default=16)
    p.add_argument("--freeze",        action="store_true")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--lr",            type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(in_ch=1, base_feats=args.base_feats).to(device)

    # load & remap BraTS encoder
    raw_ckpt = torch.load(args.pretrained_unet, map_location="cpu")
    raw_sd   = raw_ckpt if not isinstance(raw_ckpt, dict) else (
                raw_ckpt.get("network_weights", raw_ckpt))
    mapped   = remap_braTS_encoder(raw_sd)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("Loaded encoder ✓  missing:", missing, "unexpected:", unexpected)

    if args.freeze:
        for n,p in model.named_parameters():
            if n.startswith("enc") or n.startswith("bot"):
                p.requires_grad=False

    ds     = BCBM3DDataset(args.raw_dir, args.gt_dir, args.metadata_csv, args.target_z)
    dl     = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, collate_fn=collate_fn)
    opt    = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    for epoch in range(1, args.epochs+1):
        try:
            avg = train_epoch(model, dl, opt, device, scaler)
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                print(f"OOM at epoch {epoch} — try lowering base_feats/target_z or batch_size")
                torch.cuda.empty_cache()
                break
            else:
                raise
        print(f"Epoch {epoch:03d}  Loss {avg:.4f}")
        if device.type=="cuda":
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Saved → her2_bcbm_unet_transfer.pth")

if __name__=="__main__":
    main()
