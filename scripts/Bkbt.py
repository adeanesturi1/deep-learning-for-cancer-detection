#!/usr/bin/env python
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=32):
        super().__init__()
        # encoder
        self.enc1 = self._block(in_ch,       base_feats)
        self.enc2 = self._block(base_feats,  base_feats*2)
        self.enc3 = self._block(base_feats*2,base_feats*4)
        self.enc4 = self._block(base_feats*4,base_feats*8)
        # bottleneck
        self.bot  = self._block(base_feats*8,base_feats*16)
        # decoder
        self.up4  = nn.ConvTranspose3d(base_feats*16, base_feats*8, 2, 2)
        self.dec4 = self._block(base_feats*16, base_feats*8)
        self.up3  = nn.ConvTranspose3d(base_feats*8,  base_feats*4, 2, 2)
        self.dec3 = self._block(base_feats*8,  base_feats*4)
        self.up2  = nn.ConvTranspose3d(base_feats*4,  base_feats*2, 2, 2)
        self.dec2 = self._block(base_feats*4,  base_feats*2)
        self.up1  = nn.ConvTranspose3d(base_feats*2,  base_feats,   2, 2)
        self.dec1 = self._block(base_feats*2,  base_feats)
        # seg head
        self.seg_head = nn.Conv3d(base_feats, 1, 1)
        # cls head
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

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        seg = self.seg_head(d1)
        cls = self.classifier(b).squeeze(1)
        return seg, cls


class BCBM3DDataset(Dataset):
    def __init__(self, preproc_dir, metadata_csv):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+", "-"])]
        df["HER2"] = (df["HER2_Status"] == "+").astype(float)
        self.entries = df[["nnUNet_ID", "HER2"]].to_records(index=False)
        self.p = preproc_dir

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        case_id, her2 = self.entries[i]
        img_np = np.load(os.path.join(self.p, "imagesTr", f"{case_id}.npz"))["data"]
        seg_np = np.load(os.path.join(self.p, "labelsTr", f"{case_id}.npz"))["seg"]
        # data === (C, X, Y, Z), seg === (1, X, Y, Z)
        img = torch.from_numpy(img_np).float()
        seg = torch.from_numpy(seg_np).float()
        return img, seg, torch.tensor(her2, dtype=torch.float32)


def train_epoch(model, loader, optimizer, device):
    model.train()
    seg_loss_fn = nn.BCEWithLogitsLoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()
    total = 0.0
    for img, seg_gt, her2 in loader:
        img, seg_gt, her2 = img.to(device), seg_gt.to(device), her2.to(device)
        optimizer.zero_grad()
        seg_logits, cls_logits = model(img)
        l1 = seg_loss_fn(seg_logits, seg_gt)
        l2 = cls_loss_fn(cls_logits, her2)
        loss = l1 + 0.5 * l2
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def map_braTS_encoder(pretrained_dict, model):
    mapped = {}
    for k, v in pretrained_dict.items():
        # only look at encoder.stages.<stage> blocks
        if not k.startswith("encoder.stages"):
            continue
        parts = k.split(".")
        # parts = ["encoder","stages","<stage>","<block>","convs","<conv_idx>","<type>","...","weight|bias"]
        stage = int(parts[2])
        conv_idx = int(parts[5])
        layer_type = parts[6]  # "conv" or "norm"
        wn = parts[-1]         # "weight" or "bias"

        # pick which prefix in our UNet3D
        if   stage == 0: prefix = "enc1"
        elif stage == 1: prefix = "enc2"
        elif stage == 2: prefix = "enc3"
        elif stage == 3: prefix = "enc4"
        elif stage == 4: prefix = "bot"
        else: continue

        # in our block, conv0→layer 0, norm0→layer 1; conv1→layer 3, norm1→layer 4
        if layer_type == "conv":
            layer_idx = 0 if conv_idx == 0 else 3
        else:  # "norm"
            layer_idx = 1 if conv_idx == 0 else 4

        new_key = f"{prefix}.{layer_idx}.{wn}"
        mapped[new_key] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("✓ Loaded encoder; missing keys:", missing)
    print("✓ unexpected keys from pretrained:", unexpected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preproc_dir",    required=True,
                   help="nnUNet_preprocessed/Dataset002_BCBM")
    p.add_argument("--metadata_csv",   required=True)
    p.add_argument("--pretrained_unet",required=True,
                   help="braTS_encoder.pth")
    p.add_argument("--freeze",         action="store_true")
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--batch_size",type=int,  default=2)
    p.add_argument("--lr",       type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if your BCBM preprocessed has only 1 channel, change in_ch=1
    model = UNet3D(in_ch=1).to(device)

    # --- load & map BraTS encoder ---
    raw = torch.load(args.pretrained_unet, map_location="cpu")
    # if it was saved as a bare state_dict:
    sd = raw if not ("network_weights" in raw) else raw["network_weights"]
    map_braTS_encoder(sd, model)

    if args.freeze:
        for n, p in model.named_parameters():
            if n.startswith("enc") or n.startswith("bot"):
                p.requires_grad = False

    ds = BCBM3DDataset(args.preproc_dir, args.metadata_csv)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, dl, optimizer, device)
        print(f"Epoch {ep:03d}  ⇢  loss {loss:.4f}")

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Model saved → her2_bcbm_unet_transfer.pth")


if __name__ == "__main__":
    main()
