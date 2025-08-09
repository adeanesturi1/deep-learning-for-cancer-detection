#!/usr/bin/env python
# scripts/trainher2unettransfer.py

import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

try:
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode
except ImportError:
    TF = None
    InterpolationMode = None

class UNet2D(nn.Module):
    def __init__(self, in_ch, base_feats=16, dropout=0.0):
        super().__init__()
        def block(ic, oc):
            layers = [
                nn.Conv2d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)

        self.enc1 = block(in_ch, base_feats)
        self.enc2 = block(base_feats, base_feats*2)
        self.enc3 = block(base_feats*2, base_feats*4)
        self.enc4 = block(base_feats*4, base_feats*8)
        self.bot  = block(base_feats*8, base_feats*16)

        self.up4, self.dec4 = nn.ConvTranspose2d(base_feats*16, base_feats*8, 2, 2), block(base_feats*16, base_feats*8)
        self.up3, self.dec3 = nn.ConvTranspose2d(base_feats*8, base_feats*4, 2, 2), block(base_feats*8, base_feats*4)
        self.up2, self.dec2 = nn.ConvTranspose2d(base_feats*4, base_feats*2, 2, 2), block(base_feats*4, base_feats*2)
        self.up1, self.dec1 = nn.ConvTranspose2d(base_feats*2, base_feats,   2, 2), block(base_feats*2, base_feats)

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

class BCBM2DDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, meta_csv, augment=False):
        df = pd.read_csv(meta_csv)
        df = df[df["HER2_Status"].isin(["+", "-"])]
        df["HER2"] = (df["HER2_Status"] == "+").astype(np.float32)
        self.ids, self.labels = zip(*df[["nnUNet_ID", "HER2"]].to_records(index=False))
        self.raw = raw_dir
        self.gt = gt_dir
        self.augment = augment

        self.img_slices = []
        self.seg_slices = []
        for cid in self.ids:
            img_p = os.path.join(self.raw, "imagesTr", f"{cid}_0000.nii.gz")
            seg_p = os.path.join(self.gt, f"{cid}.nii.gz")
            vol = nib.load(img_p).get_fdata().astype(np.float32)
            seg = nib.load(seg_p).get_fdata().astype(np.float32)

            tumor_z = np.where(seg.sum(axis=(0, 1)) > 0)[0]
            if len(tumor_z) == 0:
                z = vol.shape[2] // 2
            else:
                z = int(np.median(tumor_z))

            self.img_slices.append(vol[:, :, z])
            self.seg_slices.append(seg[:, :, z])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img2d = self.img_slices[idx]
        seg2d = self.seg_slices[idx]
        label = self.labels[idx]

        img2d = (img2d - img2d.mean()) / (img2d.std() + 1e-6)
        img_t = torch.from_numpy(img2d)[None]
        seg_t = torch.from_numpy(seg2d)[None]
        cls_t = torch.tensor(label, dtype=torch.float32)

        if self.augment and TF:
            angle = random.uniform(-15, 15)
            img_t = TF.rotate(img_t, angle, fill=0)
            seg_t = TF.rotate(seg_t, angle, fill=0)
            scale = random.uniform(0.9, 1.1)
            h, w = img_t.shape[-2:]
            nh, nw = int(h*scale), int(w*scale)
            img_t = TF.resize(img_t, (nh, nw))
            seg_t = TF.resize(seg_t, (nh, nw), interpolation=InterpolationMode.NEAREST)
            img_t = TF.center_crop(img_t, (h, w))
            seg_t = TF.center_crop(seg_t, (h, w))
            img_t = TF.adjust_gamma(img_t, random.uniform(0.75, 1.25))
        elif self.augment:
            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[-1])
                seg_t = torch.flip(seg_t, dims=[-1])

        return img_t, seg_t, cls_t

def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

def dice_loss(logits, targets, eps=1e-6):
    preds = torch.sigmoid(logits) > 0.5
    inter = (preds * targets).sum(dim=[2,3])
    union = preds.sum(dim=[2,3]) + targets.sum(dim=[2,3])
    return 1 - ((2*inter + eps)/(union + eps)).mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_workers", type=int, default=0)
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--base_feats", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--freeze_epochs", type=int, default=5)
    args = parser.parse_args()

    logdir = os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(logdir)

    print(f">>> RUNNING: {__file__}", flush=True)
    print(">>> STARTING TRAINING SCRIPT <<<", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2D(1, args.base_feats, dropout=args.dropout).to(device)

    for name, param in model.named_parameters():
        if any(name.startswith(pref) for pref in ["enc1", "enc2", "enc3", "enc4", "bot"]):
            param.requires_grad = False

    ds = BCBM2DDataset(args.raw_dir, args.gt_dir, args.metadata_csv, augment=args.augment)
    tr_idx, val_idx = train_test_split(np.arange(len(ds)), test_size=0.2, random_state=42)
    train_dl = DataLoader(Subset(ds, tr_idx), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=args.val_workers, pin_memory=True, collate_fn=collate_fn)

    print(">>> Dataloader created. Sampling 1 batch now...", flush=True)
    _ = next(iter(train_dl))
    print(">>> Batch sample successful.", flush=True)

    scaler = torch.cuda.amp.GradScaler()
    dec_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(dec_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, div_factor=10, final_div_factor=1e4)

    best_auc, no_improve = 0.0, 0
    unfroze = False

    for epoch in range(1, args.epochs+1):
        print(f"\n>>> Starting Epoch {epoch:03d}")

        if not unfroze and epoch == args.freeze_epochs + 1:
            print(f"→ Unfreezing encoder at epoch {epoch}")
            enc, dec = [], []
            for name, param in model.named_parameters():
                if any(name.startswith(pref) for pref in ["enc1", "enc2", "enc3", "enc4", "bot"]):
                    param.requires_grad = True
                    enc.append(param)
                else:
                    dec.append(param)
            optimizer = optim.AdamW([
                {'params': dec, 'lr': args.lr},
                {'params': enc, 'lr': args.lr * 0.1}
            ], weight_decay=args.weight_decay)
            rem_steps = (args.epochs - epoch + 1) * len(train_dl)
            scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=rem_steps, pct_start=0.1, div_factor=10, final_div_factor=1e4)
            unfroze = True

        model.train()
        train_loss = 0.0
        for imgs, segs, labs in train_dl:
            imgs, segs, labs = imgs.to(device), segs.to(device), labs.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                s_out, c_out = model(imgs)
                seg_loss = dice_loss(s_out, segs) + nn.BCEWithLogitsLoss()(s_out, segs)
                cls_loss = nn.BCEWithLogitsLoss()(c_out, labs)
                loss = 2.0 * seg_loss + cls_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        dice_sum, probs, labels = 0.0, [], []
        with torch.no_grad():
            for imgs, segs, labs in val_dl:
                imgs, segs = imgs.to(device), segs.to(device)
                s_out, c_out = model(imgs)
                p_seg = torch.sigmoid(s_out) > 0.5
                batch_dice = ((2*(p_seg*segs).sum([2,3]) + 1e-6) / ((p_seg + segs).sum([2,3]) + 1e-6)).mean().item()
                dice_sum += batch_dice * imgs.size(0)
                p_clf = torch.sigmoid(c_out)
                probs.extend(p_clf.cpu().tolist())
                labels.extend(labs.tolist())
        val_dice = dice_sum / len(val_dl.dataset)
        val_auc = roc_auc_score(labels, probs)

        print(f"Epoch {epoch:03d} TrainLoss {train_loss:.4f} ValDice {val_dice:.4f} ValAUC {val_auc:.4f}")
        print(f"[E{epoch:03d}] Best AUC so far: {best_auc:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('AUC/val', val_auc, epoch)

        if epoch in {1, args.epochs}:
            im_np = imgs[0, 0].cpu().numpy()
            gt_np = segs[0, 0].cpu().numpy()
            pm_np = (torch.sigmoid(s_out[0, 0]) > 0.5).float().cpu().numpy()
            overlay = np.stack([im_np, pm_np, gt_np], axis=0)
            writer.add_image('Overlay_RedPred_BlueGT', overlay, epoch)

        if val_auc > best_auc + 1e-4:
            best_auc, no_improve = val_auc, 0
            torch.save(model.state_dict(), 'best.pth')
            writer.add_text('Model', f'New best at {epoch}', epoch)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print("Early stopping.")
                break

        torch.cuda.empty_cache()

    writer.close()
