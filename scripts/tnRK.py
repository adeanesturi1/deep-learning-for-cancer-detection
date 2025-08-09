#!/usr/bin/env python
# scripts/train_her2_unet_transfer.py

import argparse
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib

# -----------------------------------------------------------------------------
# 3D U-Net + classification head
# -----------------------------------------------------------------------------
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
        self.seg_head = nn.Conv3d(base_feats, 1, 1)
        # classification head off the bottleneck
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
        e2 = self.enc2(nn.functional.max_pool3d(e1, 2))
        e3 = self.enc3(nn.functional.max_pool3d(e2, 2))
        e4 = self.enc4(nn.functional.max_pool3d(e3, 2))
        b  = self.bot(nn.functional.max_pool3d(e4, 2))

        # segmentation path
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg = self.seg_head(d1)

        # classification path
        cls = self.classifier(b)
        return seg, cls.squeeze(1)


# -----------------------------------------------------------------------------
# Dataset for BCBM (preprocessed nnU-Net layout)
# -----------------------------------------------------------------------------
class BCBM3DDataset(Dataset):
    def __init__(self, data_dir, metadata_csv):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+", "-"])]
        df["HER2"] = (df["HER2_Status"] == "+").astype(float)
        self.cases = df[["nnUNet_ID", "HER2"]].to_records(index=False)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        case_id, her2 = self.cases[i]
        # nnU-Net preprocessed expects imagesTr/{case_id}_0000.nii.gz
        img_path = os.path.join(self.data_dir, "imagesTr", f"{case_id}_0000.nii.gz")
        seg_path = os.path.join(self.data_dir, "labelsTr", f"{case_id}.nii.gz")
        img = nib.load(img_path).get_fdata().astype("float32")
        seg = nib.load(seg_path).get_fdata().astype("float32")
        # channel dim
        img = torch.from_numpy(img)[None]
        seg = torch.from_numpy(seg)[None]
        return img, seg, torch.tensor(her2, dtype=torch.float32)


def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)


# -----------------------------------------------------------------------------
# training / epoch
# -----------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for img, seg_gt, her2 in loader:
        img, seg_gt, her2 = img.to(device), seg_gt.to(device), her2.to(device)
        optimizer.zero_grad()
        seg_logits, cls_logits = model(img)
        l1 = seg_loss_fn(seg_logits, seg_gt)
        l2 = cls_loss_fn(cls_logits, her2)
        loss = l1 + 0.5 * l2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a simple 3D U-Net (seg + HER2-status)"
    )
    parser.add_argument("--data_dir",        required=True,
                        help="nnUNet_preprocessed/Dataset002_BCBM")
    parser.add_argument("--metadata_csv",    required=True,
                        help="BCBM metadata CSV with nnUNet_ID + HER2_Status")
    parser.add_argument("--pretrained_unet", required=True,
                        help="path to braTS_encoder.pth")
    parser.add_argument("--freeze",          action="store_true",
                        help="freeze encoder weights")
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--batch_size",      type=int, default=2)
    parser.add_argument("--lr",              type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(in_ch=4).to(device)

    # load only encoder weights
    enc_sd = torch.load(args.pretrained_unet, map_location="cpu")
    missing, unexpected = model.load_state_dict(enc_sd, strict=False)
    print("Loaded encoder ✓  missing keys:", missing, " unexpected:", unexpected)

    if args.freeze:
        for name, p in model.named_parameters():
            if name.startswith("enc") or name.startswith("bot"):
                p.requires_grad = False

    ds = BCBM3DDataset(args.data_dir, args.metadata_csv)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=collate_fn)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)

    for epoch in range(1, args.epochs+1):
        avg_loss = train_epoch(model, dl, optimizer, device)
        print(f"[Epoch {epoch:03d}]  avg loss: {avg_loss:.4f}")

    out_path = "her2_bcbm_unet_transfer.pth"
    torch.save(model.state_dict(), out_path)
    print("✓ Done. Saved model to", out_path)


if __name__ == "__main__":
    main()
