#!/usr/bin/env python
# scripts/trainher2_unet2d_transfer_tweaked.py

import argparse
import os
import random

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------
#  2D U-Net + classifier
# ---------------------------------------------------------------------
class UNet2D(nn.Module):
    def __init__(self, in_ch, base_feats=16, dropout=0.0):
        super().__init__()
        def block(ic, oc):
            layers = [
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            ]
            layers += [
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

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
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_feats*16, 1)
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

# ---------------------------------------------------------------------
#  Dataset: central axial slice + augment + normalize
# ---------------------------------------------------------------------
class BCBM2DDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, meta_csv, augment=False):
        df = pd.read_csv(meta_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"] == "+").astype(np.float32)
        self.cases = df[["nnUNet_ID","HER2"]].to_records(index=False)
        self.raw   = raw_dir
        self.gt    = gt_dir
        self.augment = augment

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        cid, her2 = self.cases[i]
        img_p = os.path.join(self.raw, "imagesTr", f"{cid}_0000.nii.gz")
        seg_p = os.path.join(self.gt,  f"{cid}.nii.gz")

        vol = nib.load(img_p).get_fdata().astype(np.float32)
        seg = nib.load(seg_p).get_fdata().astype(np.float32)

        z_mid = vol.shape[2] // 2
        img2d = vol[:, :, z_mid]
        seg2d = seg[:, :, z_mid]

        # intensity normalization
        img2d = (img2d - img2d.mean()) / (img2d.std() + 1e-6)

        # to torch
        img_t = torch.from_numpy(img2d)[None]
        seg_t = torch.from_numpy(seg2d)[None]
        cls_t = torch.tensor(her2, dtype=torch.float32)

        # augment flip
        if self.augment and random.random() < 0.5:
            img_t = torch.flip(img_t, dims=[2])
            seg_t = torch.flip(seg_t, dims=[2])

        return img_t, seg_t, cls_t

# ---------------------------------------------------------------------
#  Collate
# ---------------------------------------------------------------------
def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

# ---------------------------------------------------------------------
#  Loss: Dice + BCE
# ---------------------------------------------------------------------
def dice_loss(logits, targets, eps=1e-6):
    preds = torch.sigmoid(logits)
    inter = (preds * targets).sum(dim=[2,3])
    union = preds.sum(dim=[2,3]) + targets.sum(dim=[2,3])
    return 1 - ((2*inter + eps)/(union + eps)).mean()

# ---------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",      required=True)
    parser.add_argument("--gt_dir",       required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--base_feats",   type=int, default=16)
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--patience",     type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout",      type=float, default=0.0)
    parser.add_argument("--augment",      action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet2D(1, args.base_feats, dropout=args.dropout).to(device)

    # data
    ds = BCBM2DDataset(args.raw_dir, args.gt_dir, args.metadata_csv, augment=args.augment)
    idx = np.arange(len(ds))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_ds = Subset(ds, tr_idx)
    val_ds   = Subset(ds, val_idx)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, collate_fn=collate_fn)

    # optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler: OneCycleLR over total steps
    total_steps = args.epochs * len(train_dl)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps,
                           pct_start=0.1, div_factor=10, final_div_factor=1e4)

    best_dice = 0.0
    no_improve = 0
    history = {'train_loss':[], 'val_soft_dice':[], 'val_auc':[]}

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss=0.0
        for img, seg, her2 in train_dl:
            img, seg, her2 = img.to(device), seg.to(device), her2.to(device)
            optimizer.zero_grad()
            s_out, c_out = model(img)
            seg_loss = dice_loss(s_out, seg) + nn.BCEWithLogitsLoss()(s_out, seg)
            cls_loss = nn.BCEWithLogitsLoss()(c_out, her2)
            loss = seg_loss + 0.5 * cls_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)
        history['train_loss'].append(train_loss)

        # validation
        model.eval()
        dice_sum=0.0; probs=[]; labels=[]
        with torch.no_grad():
            for img, seg, her2 in val_dl:
                img, seg = img.to(device), seg.to(device)
                s_out, c_out = model(img)
                p = torch.sigmoid(s_out)
                # accumulate soft dice
                batch_dice = ((2*(p*seg).sum([2,3]) + 1e-6) /
                              (p.sum([2,3]) + seg.sum([2,3]) + 1e-6)).mean().item()
                dice_sum += batch_dice * img.size(0)
                probs.extend(torch.sigmoid(c_out).cpu().tolist())
                labels.extend(her2.tolist())
        val_soft_dice = dice_sum / len(val_ds)
        val_auc       = roc_auc_score(labels, probs)
        history['val_soft_dice'].append(val_soft_dice)
        history['val_auc'].append(val_auc)

        print(f"Epoch {epoch:03d} TrainLoss {train_loss:.4f}" +
              f" ValSoftDice {val_soft_dice:.4f} ValAUC {val_auc:.4f}", flush=True)

        # checkpoint & early stop on soft dice
        if val_soft_dice > best_dice:
            best_dice = val_soft_dice; no_improve = 0
            torch.save(model.state_dict(), "her2_bcbm_unet2d_transfer_best.pth")
            print("→ New best checkpoint saved (soft dice).", flush=True)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"No improvement in Soft Dice for {no_improve} epochs. Early stopping.", flush=True)
                break
        torch.cuda.empty_cache()

    # plot and save progress
    plt.figure()
    epochs = np.arange(1, len(history['train_loss'])+1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_soft_dice'], label='Val Soft Dice')
    plt.plot(epochs, history['val_auc'],       label='Val AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('training_progress.png')
    print("✓ Training complete. Best Val Soft Dice =", best_dice)
    print("Progress plot saved → training_progress.png")
