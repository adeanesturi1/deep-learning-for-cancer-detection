#!/usr/bin/env python
import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=32):
        super().__init__()
        # encoder
        self.enc1 = self._block(in_ch, base_feats)
        self.enc2 = self._block(base_feats, base_feats*2)
        self.enc3 = self._block(base_feats*2, base_feats*4)
        self.enc4 = self._block(base_feats*4, base_feats*8)
        # bottleneck
        self.bot  = self._block(base_feats*8, base_feats*16)
        # decoder
        self.up4  = nn.ConvTranspose3d(base_feats*16, base_feats*8, 2, 2)
        self.dec4 = self._block(base_feats*16, base_feats*8)
        self.up3  = nn.ConvTranspose3d(base_feats*8, base_feats*4, 2, 2)
        self.dec3 = self._block(base_feats*8, base_feats*4)
        self.up2  = nn.ConvTranspose3d(base_feats*4, base_feats*2, 2, 2)
        self.dec2 = self._block(base_feats*4, base_feats*2)
        self.up1  = nn.ConvTranspose3d(base_feats*2, base_feats,   2, 2)
        self.dec1 = self._block(base_feats*2, base_feats)
        # segmentation head
        self.seg_head   = nn.Conv3d(base_feats, 1, 1)
        # classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_feats*16, 1),
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
        e2 = self.enc2(nn.functional.max_pool3d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool3d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool3d(e3, 2))
        b  = self.bot(nn.functional.max_pool3d(e4, 2))

        # decode
        d4 = self.dec4(torch.cat([self.up4(b),  e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        seg = self.seg_head(d1)

        cls = self.classifier(b).squeeze(1)
        return seg, cls

class BCBM3DDataset(Dataset):
    def __init__(self, preproc_dir, metadata_csv, target_z):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+", "-"])].copy()
        df["HER2"] = (df["HER2_Status"] == "+").astype(np.float32)
        self.cases = df[["nnUNet_ID", "HER2"]].to_records(index=False)
        self.img_dir = os.path.join(preproc_dir, "imagesTr")
        self.seg_dir = os.path.join(preproc_dir, "gt_segmentations")
        self.Z = target_z

    def __len__(self):
        return len(self.cases)

    def _pad_or_crop(self, vol):
        # vol: numpy array shape (H,W,Z0)
        z0 = vol.shape[2]
        if z0 == self.Z:
            return vol
        if z0 < self.Z:
            pad = (0, 0, 0, 0, 0, self.Z - z0)
            return np.pad(vol, pad, mode="constant", constant_values=0)
        # else crop centrally
        start = (z0 - self.Z)//2
        return vol[:, :, start:start+self.Z]

    def __getitem__(self, i):
        case_id, her2 = self.cases[i]
        # load image volume
        img_path = os.path.join(self.img_dir, f"{case_id}_0000.nii.gz")
        img = nib.load(img_path).get_fdata().astype(np.float32)
        img = self._pad_or_crop(img)
        img = torch.from_numpy(img)[None]  # (1,H,W,Z)

        # load gt segmentation
        seg_path = os.path.join(self.seg_dir, f"{case_id}.nii.gz")
        seg = nib.load(seg_path).get_fdata().astype(np.float32)
        seg = self._pad_or_crop(seg)
        seg = torch.from_numpy(seg)[None]

        return img, seg, torch.tensor(her2, dtype=torch.float32)

def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

def train_epoch(model, loader, opt, device):
    model.train()
    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for img, seg_gt, her2 in loader:
        img, seg_gt, her2 = img.to(device), seg_gt.to(device), her2.to(device)
        opt.zero_grad()
        seg_logits, cls_logits = model(img)
        loss = seg_loss_fn(seg_logits, seg_gt) + 0.5 * cls_loss_fn(cls_logits, her2)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir",    required=True,
                   help="nnUNet_preprocessed/Dataset002_BCBM")
    p.add_argument("--metadata_csv",   required=True,
                   help="nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv")
    p.add_argument("--pretrained_unet",required=True,
                   help="path/to/braTS_encoder.pth")
    p.add_argument("--target_z",       type=int, default=176,
                   help="depth to pad/crop all volumes to")
    p.add_argument("--freeze",         action="store_true",
                   help="freeze encoder & bottleneck")
    p.add_argument("--epochs",         type=int, default=100)
    p.add_argument("--batch_size",     type=int, default=2)
    p.add_argument("--lr",             type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3D(in_ch=1).to(device)

    # load only encoder & bottleneck weights
    enc_sd = torch.load(args.pretrained_unet, map_location="cpu")
    # BraTS extractor saved keys starting with "encoder." and "bot."
    subdict = {k.replace("encoder.", "enc").replace("bot.", "bot."): v
               for k, v in enc_sd.items() if k.startswith("encoder.") or k.startswith("bot.")}
    missing, unexpected = model.load_state_dict(subdict, strict=False)
    print("Loaded encoder ✓  missing:", missing, "unexpected:", unexpected)

    if args.freeze:
        for name, p_ in model.named_parameters():
            if name.startswith("enc") or name.startswith("bot"):
                p_.requires_grad = False

    ds = BCBM3DDataset(args.preproc_dir, args.metadata_csv, args.target_z)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate_fn)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dl, optimizer, device)
        print(f"Epoch {epoch:03d}  Loss {avg_loss:.4f}")

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Saved → her2_bcbm_unet_transfer.pth")

if __name__ == "__main__":
    main()
