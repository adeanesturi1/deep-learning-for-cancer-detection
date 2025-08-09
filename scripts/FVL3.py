#!/usr/bin/env python
# scripts/trainher2unettransfer.py

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------
# 3D U‑Net with segmentation + classification heads
# --------------------------------------------
class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=32):
        super().__init__()
        # encoder blocks
        self.enc1 = self._block(in_ch,    base_feats)
        self.enc2 = self._block(base_feats, base_feats*2)
        self.enc3 = self._block(base_feats*2, base_feats*4)
        self.enc4 = self._block(base_feats*4, base_feats*8)
        # bottleneck
        self.bot  = self._block(base_feats*8, base_feats*16)
        # decoder blocks
        self.up4  = nn.ConvTranspose3d(base_feats*16, base_feats*8, 2, 2)
        self.dec4 = self._block(base_feats*16, base_feats*8)
        self.up3  = nn.ConvTranspose3d(base_feats*8,  base_feats*4, 2, 2)
        self.dec3 = self._block(base_feats*8,  base_feats*4)
        self.up2  = nn.ConvTranspose3d(base_feats*4,  base_feats*2, 2, 2)
        self.dec2 = self._block(base_feats*4,  base_feats*2)
        self.up1  = nn.ConvTranspose3d(base_feats*2,  base_feats,   2, 2)
        self.dec1 = self._block(base_feats*2,  base_feats)
        # heads
        self.seg_head   = nn.Conv3d(base_feats, 1, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_feats*16, 1)
        )

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool3d(e1,2))
        e3 = self.enc3(nn.functional.max_pool3d(e2,2))
        e4 = self.enc4(nn.functional.max_pool3d(e3,2))
        b  = self.bot(nn.functional.max_pool3d(e4,2))
        d4 = self.dec4(torch.cat([self.up4(b), e4],1))
        d3 = self.dec3(torch.cat([self.up3(d4),e3],1))
        d2 = self.dec2(torch.cat([self.up2(d3),e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2),e1],1))
        seg = self.seg_head(d1)
        cls = self.classifier(b).squeeze(1)
        return seg, cls

# --------------------------------------------
# Dataset: load single-channel, replicate to pretrained in_ch
# --------------------------------------------
class BCBM3DDataset(Dataset):
    def __init__(self, preproc_dir, metadata_csv, target_z, in_ch):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"]=="+").astype(np.float32)
        self.cases = df[["nnUNet_ID","HER2"]].to_records(index=False)
        self.preproc = preproc_dir
        self.target_z = target_z
        self.in_ch = in_ch

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        case_id, her2 = self.cases[i]
        img_path = os.path.join(self.preproc, "imagesTr", f"{case_id}_0000.nii.gz")
        seg_path = os.path.join(self.preproc, "labelsTr", f"{case_id}.nii.gz")
        img = nib.load(img_path).get_fdata().astype(np.float32)
        seg = nib.load(seg_path).get_fdata().astype(np.float32)
        H,W,Z = img.shape
        tz = self.target_z
        # center-crop or pad Z
        if Z>tz:
            s=(Z-tz)//2; img=img[:,:,s:s+tz]; seg=seg[:,:,s:s+tz]
        elif Z<tz:
            p=tz-Z; f=p//2; b=p-f
            img=np.pad(img,((0,0),(0,0),(f,b)),mode='constant')
            seg=np.pad(seg,((0,0),(0,0),(f,b)),mode='constant')
        # to torch + replicate channels
        timg = torch.from_numpy(img)[None].repeat(self.in_ch,1,1,1)
        tseg = torch.from_numpy(seg)[None]
        return timg, tseg, torch.tensor(her2)

# remap BraTS keys → UNet3D state_dict
# (handles conv0, norm0, conv1, norm1 ordering)
def remap_braTS_encoder(raw_sd):
    mapped={}
    for k,v in raw_sd.items():
        if not k.startswith("encoder.stages."): continue
        parts=k.split('.')
        stage=int(parts[2])        # 0..4
        ci   =int(parts[5])        # conv idx
        kind =parts[6]             # 'conv' or 'norm'
        param=parts[7]             # 'weight'/'bias'/...
        prefix = f"enc{stage+1}" if stage<4 else "bot"
        idx = ci*3 + (0 if kind=="conv" else 1)
        new_k = f"{prefix}.{idx}.{param}"
        mapped[new_k]=v
    return mapped

# one training epoch
def train_epoch(model, loader, opt, device):
    model.train()
    seg_loss=nn.BCEWithLogitsLoss()
    cls_loss=nn.BCEWithLogitsLoss()
    tot=0.0
    for img, seg_gt, her2 in loader:
        img,seg_gt,her2 = img.to(device), seg_gt.to(device), her2.to(device)
        opt.zero_grad()
        seg_logits, cls_logits = model(img)
        loss = seg_loss(seg_logits, seg_gt) + 0.5*cls_loss(cls_logits, her2)
        loss.backward()
        opt.step()
        tot += loss.item()
    return tot/len(loader)

# ----------------------
# Main
# ----------------------
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--preproc_dir",    required=True)
    p.add_argument("--metadata_csv",   required=True)
    p.add_argument("--pretrained_unet",required=True)
    p.add_argument("--target_z",       type=int, default=176)
    p.add_argument("--batch_size",     type=int, default=1)
    p.add_argument("--epochs",         type=int, default=100)
    p.add_argument("--lr",             type=float, default=1e-4)
    args=p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ck = torch.load(args.pretrained_unet, map_location="cpu")
    raw_sd = ck.get("network_weights", ck)
    # infer pretrained dims
    # pick first conv weight key
    ck_key = next(k for k in raw_sd if k.endswith("conv.weight"))
    w0 = raw_sd[ck_key]
    in_ch, base_feats = w0.shape[1], w0.shape[0]
    print(f"⇒ pretrained in_ch={in_ch}, base_feats={base_feats}")

    # build model to match
    model = UNet3D(in_ch=in_ch, base_feats=base_feats).to(device)
    # remap + load
    mapped = remap_braTS_encoder(raw_sd)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("Loaded encoder ✓ missing:", missing, "unexpected:", unexpected)

    ds = BCBM3DDataset(args.preproc_dir, args.metadata_csv,
                       args.target_z, in_ch=in_ch)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=True, num_workers=4, collate_fn=lambda b: tuple(torch.stack(x) for x in zip(*b)))
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    for ep in range(1, args.epochs+1):
        try:
            avg = train_epoch(model, dl, opt, device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at epoch {ep} — reduce batch_size/target_z/base_feats")
                torch.cuda.empty_cache()
                break
            else:
                raise
        print(f"Epoch {ep:03d}  Loss {avg:.4f}")

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Saved → her2_bcbm_unet_transfer.pth")

if __name__=="__main__":
    main()
