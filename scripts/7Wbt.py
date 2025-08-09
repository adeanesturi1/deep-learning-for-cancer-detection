#!/usr/bin/env python
# scripts/trainher2unettransfer.py

import argparse
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader

# -----------------------------------------------------------------------------
#  NOTE ON OOM: 
#  If you hit CUDA OOM again, try one or more of:
#    • Lower --batch_size (e.g. 1)
#    • Lower --base_feats (e.g. 16)
#    • Reduce --target_z (center‐crop smaller Z)
#    • Enable torch.cuda.empty_cache() between epochs
# -----------------------------------------------------------------------------

class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=32):
        super().__init__()
        self.enc1 = self._block(in_ch,    base_feats)
        self.enc2 = self._block(base_feats, base_feats*2)
        self.enc3 = self._block(base_feats*2, base_feats*4)
        self.enc4 = self._block(base_feats*4, base_feats*8)
        self.bot  = self._block(base_feats*8, base_feats*16)

        self.up4  = nn.ConvTranspose3d(base_feats*16, base_feats*8, 2, 2)
        self.dec4 = self._block(base_feats*16, base_feats*8)
        self.up3  = nn.ConvTranspose3d(base_feats*8, base_feats*4, 2, 2)
        self.dec3 = self._block(base_feats*8, base_feats*4)
        self.up2  = nn.ConvTranspose3d(base_feats*4, base_feats*2, 2, 2)
        self.dec2 = self._block(base_feats*4, base_feats*2)
        self.up1  = nn.ConvTranspose3d(base_feats*2, base_feats,   2, 2)
        self.dec1 = self._block(base_feats*2, base_feats)

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

class BCBM3DDataset(Dataset):
    def __init__(self, preproc_dir, metadata_csv, target_z):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"]=="+").astype(np.float32)
        self.cases = df[["nnUNet_ID","HER2"]].to_records(index=False)
        self.preproc = preproc_dir
        self.target_z = target_z

    def __len__(self): 
        return len(self.cases)

    def __getitem__(self,i):
        case_id, her2 = self.cases[i]
        img_p = os.path.join(self.preproc,"imagesTr", f"{case_id}_0000.nii.gz")
        seg_p = os.path.join(self.preproc,"labelsTr", f"{case_id}.nii.gz")

        img = nib.load(img_p).get_fdata().astype(np.float32)
        seg = nib.load(seg_p).get_fdata().astype(np.float32)

        # ensure (H,W,Z) and then center-crop or pad Z to target_z
        H,W,Z = img.shape
        tz    = self.target_z
        if Z > tz:
            start = (Z - tz)//2
            img = img[:,:,start:start+tz]
            seg = seg[:,:,start:start+tz]
        elif Z < tz:
            pad = tz - Z
            # pad equally front/back
            f = pad//2; b = pad - f
            img = np.pad(img, ((0,0),(0,0),(f,b)), mode="constant")
            seg = np.pad(seg, ((0,0),(0,0),(f,b)), mode="constant")

        # to torch, add channel
        img = torch.from_numpy(img)[None]
        seg = torch.from_numpy(seg)[None]
        return img, seg, torch.tensor(her2)

def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

def remap_braTS_encoder(raw_sd):
    mapped = {}
    for k,v in raw_sd.items():
        if not k.startswith("encoder.stages."): continue
        parts = k.split(".")
        stage = int(parts[2])       # 0..4
        ci    = int(parts[5])       # conv index 0 or 1
        kind  = parts[6]            # 'conv' or 'norm'
        param = parts[7]            # 'weight' or 'bias'
        # map stage->prefix
        if stage < 4:
            prefix = f"enc{stage+1}"
        else:
            prefix = "bot"
        # conv->idx 0, norm->idx1; second conv->3, its norm->4
        idx = ci*3 + (0 if kind=="conv" else 1)
        new_k = f"{prefix}.{idx}.{param}"
        mapped[new_k] = v
    return mapped

def train_epoch(model, loader, opt, device):
    model.train()
    seg_loss = nn.BCEWithLogitsLoss()
    cls_loss = nn.BCEWithLogitsLoss()
    total = 0.0
    for img, seg_gt, her2 in loader:
        img, seg_gt, her2 = img.to(device), seg_gt.to(device), her2.to(device)
        opt.zero_grad()
        seg_logits, cls_logits = model(img)
        loss = seg_loss(seg_logits, seg_gt) + 0.5*cls_loss(cls_logits, her2)
        loss.backward()
        opt.step()
        total += loss.item()
    return total/len(loader)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir",    required=True,
                   help="nnUNet_preprocessed/Dataset002_BCBM")
    p.add_argument("--metadata_csv",   required=True)
    p.add_argument("--pretrained_unet",required=True)
    p.add_argument("--target_z",       type=int, default=176)
    p.add_argument("--base_feats",     type=int, default=32)
    p.add_argument("--freeze",         action="store_true")
    p.add_argument("--epochs",         type=int, default=100)
    p.add_argument("--batch_size",     type=int, default=1)
    p.add_argument("--lr",             type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(in_ch=4, base_feats=args.base_feats).to(device)

    # 1) load raw, 2) remap keys, 3) load into model
    raw = torch.load(args.pretrained_unet, map_location="cpu")
    enc = raw.get("network_weights", raw)  # if wrapped
    mapped = remap_braTS_encoder(enc)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("Loaded encoder ✓  missing:", missing, "unexpected:", unexpected)

    if args.freeze:
        for name,p in model.named_parameters():
            if name.startswith("enc") or name.startswith("bot"):
                p.requires_grad = False

    ds = BCBM3DDataset(args.preproc_dir, args.metadata_csv, args.target_z)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate_fn)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr)

    for epoch in range(1, args.epochs+1):
        try:
            avg = train_epoch(model, dl, opt, device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(" OOM on epoch",epoch,"— try reducing batch_size/base_feats/target_z")
                torch.cuda.empty_cache()
                break
            else:
                raise
        print(f"Epoch {epoch:03d}  Loss {avg:.4f}")

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Saved → her2_bcbm_unet_transfer.pth")

if __name__=="__main__":
    main()
