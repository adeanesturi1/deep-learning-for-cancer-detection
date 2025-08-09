#!/usr/bin/env python
# scripts/trainher2unettransfer.py (2D version)

import argparse, os, torch
import torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, nibabel as nib
from torch.utils.data import Dataset, DataLoader

class UNet2D(nn.Module):
    def __init__(self, in_ch, base_feats=16):  # reduced default for 2D
        super().__init__()
        def block(ic, oc):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            )
        self.enc1 = block(in_ch,      base_feats)
        self.enc2 = block(base_feats, base_feats*2)
        self.enc3 = block(base_feats*2, base_feats*4)
        self.enc4 = block(base_feats*4, base_feats*8)
        self.bot  = block(base_feats*8, base_feats*16)

        self.up4, self.dec4 = nn.ConvTranspose2d(base_feats*16, base_feats*8, 2, 2), block(base_feats*16, base_feats*8)
        self.up3, self.dec3 = nn.ConvTranspose2d(base_feats*8,  base_feats*4, 2, 2), block(base_feats*8,  base_feats*4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base_feats*4,  base_feats*2, 2, 2), block(base_feats*4,  base_feats*2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base_feats*2,  base_feats,   2, 2), block(base_feats*2,  base_feats)

        self.seg_head   = nn.Conv2d(base_feats, 1, 1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_feats*16, 1)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool2d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool2d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool2d(e3, 2))
        b  = self.bot(nn.functional.max_pool2d(e4, 2))

        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))

        return self.seg_head(d1), self.classifier(b).squeeze(1)

class BCBM2DDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, meta_csv):
        df = pd.read_csv(meta_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"] == "+").astype(np.float32)
        self.cases = df[["nnUNet_ID","HER2"]].to_records(False)
        self.raw   = raw_dir
        self.gt    = gt_dir

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, her2 = self.cases[i]
        img_p = os.path.join(self.raw, "imagesTr", f"{cid}_0000.nii.gz")
        seg_p = os.path.join(self.gt,  f"{cid}.nii.gz")

        vol = nib.load(img_p).get_fdata().astype(np.float32)  # (H,W,Z)
        seg = nib.load(seg_p).get_fdata().astype(np.float32)

        # extract central axial slice
        z_mid = vol.shape[2] // 2
        img   = vol[:, :, z_mid]
        seg   = seg[:, :, z_mid]

        # to torch (C,H,W)
        img = torch.from_numpy(img)[None]
        seg = torch.from_numpy(seg)[None]
        return img, seg, torch.tensor(her2)


def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)


def train_epoch(model, loader, opt, device):
    model.train()
    seg_loss = nn.BCEWithLogitsLoss()
    cls_loss = nn.BCEWithLogitsLoss()
    total = 0.0
    for img, seg_gt, her2 in loader:
        img, seg_gt, her2 = img.to(device), seg_gt.to(device), her2.to(device)
        opt.zero_grad()
        seg_out, cls_out = model(img)
        loss = seg_loss(seg_out, seg_gt) + 0.5 * cls_loss(cls_out, her2)
        loss.backward()
        opt.step()
        total += loss.item()
    return total / len(loader)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",      required=True)
    p.add_argument("--gt_dir",       required=True)
    p.add_argument("--metadata_csv", required=True)
    p.add_argument("--base_feats",   type=int, default=16)
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet2D(1, args.base_feats).to(device)

    ds = BCBM2DDataset(args.raw_dir, args.gt_dir, args.metadata_csv)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate_fn)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    for e in range(1, args.epochs+1):
        avg = train_epoch(model, dl, opt, device)
        print(f"Epoch {e:03d}  Loss {avg:.4f}")
        if device.type == "cuda": torch.cuda.empty_cache()

    torch.save(model.state_dict(), "her2_bcbm_unet2d.pth")
    print("✓ Done. Saved → her2_bcbm_unet2d.pth")
