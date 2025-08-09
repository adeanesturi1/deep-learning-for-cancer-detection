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

# --------------------------------------------------
# If you still hit OOM: lower --batch_size, --base_feats or --target_z
# --------------------------------------------------

class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=32):
        super().__init__()
        self.enc1 = self._block(in_ch, base_feats)
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
    def __init__(self, raw_dir, gt_dir, metadata_csv, target_z):
        df = pd.read_csv(metadata_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"]=="+").astype(np.float32)
        self.cases = df[["nnUNet_ID","HER2"]].to_records(index=False)
        self.raw_dir = raw_dir
        self.gt_dir  = gt_dir
        self.Z       = target_z

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        case_id, her2 = self.cases[i]

        # raw  images: case_id_0000.nii.gz  (first modality)
        img_path = os.path.join(self.raw_dir, "imagesTr", f"{case_id}_0000.nii.gz")
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Missing image: {img_path}")

        # GT segs: either gt_dir/gt_segmentations/<case_id>.nii.gz
        # or gt_dir/labelsTr/<case_id>.nii.gz
        seg_path1 = os.path.join(self.gt_dir, "gt_segmentations", f"{case_id}.nii.gz")
        seg_path2 = os.path.join(self.gt_dir, "labelsTr",         f"{case_id}.nii.gz")
        if   os.path.isfile(seg_path1): seg_path = seg_path1
        elif os.path.isfile(seg_path2): seg_path = seg_path2
        else: raise FileNotFoundError(f"Missing seg: {seg_path1} or {seg_path2}")

        # load & cast
        img = nib.load(img_path).get_fdata().astype(np.float32)  # (H,W,Z)
        seg = nib.load(seg_path).get_fdata().astype(np.float32)

        # center‐crop or pad in Z
        H,W,Z0 = img.shape; Zt = self.Z
        if Z0 > Zt:
            s = (Z0 - Zt)//2
            img = img[:,:,s:s+Zt]
            seg = seg[:,:,s:s+Zt]
        elif Z0 < Zt:
            p = Zt - Z0; f=p//2; b=p-f
            img = np.pad(img, ((0,0),(0,0),(f,b)), mode="constant")
            seg = np.pad(seg, ((0,0),(0,0),(f,b)), mode="constant")

        # to torch
        img_t = torch.from_numpy(img)[None]  # 1×H×W×Z
        seg_t = torch.from_numpy(seg)[None]
        return img_t, seg_t, torch.tensor(her2)

def collate_fn(batch):
    imgs, segs, cls = zip(*batch)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

def remap_braTS_encoder(raw_sd):
    mapped = {}
    for k,v in raw_sd.items():
        if not k.startswith("encoder.stages."): continue
        parts = k.split(".")
        stage = int(parts[2])   # 0..4
        conv_i= int(parts[5])   # 0 or 1
        kind  = parts[6]        # 'conv' or 'norm'
        ptype = parts[7]        # 'weight' or 'bias'

        prefix = f"enc{stage+1}" if stage<4 else "bot"
        idx    = conv_i*3 + (0 if kind=="conv" else 1)
        new_k  = f"{prefix}.{idx}.{ptype}"
        mapped[new_k] = v
    return mapped

def train_epoch(model, loader, opt, device):
    model.train()
    seg_loss = nn.BCEWithLogitsLoss()
    cls_loss = nn.BCEWithLogitsLoss()
    total = 0.
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
    p.add_argument("--raw_dir",       required=True,
                   help="nnUNet_raw/Dataset002_BCBM")
    p.add_argument("--gt_dir",        required=True,
                   help="nnUNet_preprocessed/... (gt_segmentations or labelsTr)")
    p.add_argument("--metadata_csv",  required=True)
    p.add_argument("--pretrained_unet", required=True)
    p.add_argument("--target_z",      type=int, default=176)
    p.add_argument("--base_feats",    type=int, default=32)
    p.add_argument("--freeze",        action="store_true")
    p.add_argument("--epochs",        type=int, default=100)
    p.add_argument("--batch_size",    type=int, default=1)
    p.add_argument("--lr",            type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(in_ch=1, base_feats=args.base_feats).to(device)

    # load + remap BraTS encoder
    raw_ckpt = torch.load(args.pretrained_unet, map_location="cpu")
    enc_raw  = raw_ckpt.get("network_weights", raw_ckpt)
    mapped   = remap_braTS_encoder(enc_raw)
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("Loaded encoder ✓  missing:", missing, "unexpected:", unexpected)

    if args.freeze:
        for name,p in model.named_parameters():
            if name.startswith("enc") or name.startswith("bot"):
                p.requires_grad=False

    ds = BCBM3DDataset(args.raw_dir, args.gt_dir, args.metadata_csv, args.target_z)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate_fn)
    opt= optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)

    for e in range(1, args.epochs+1):
        try:
            avg = train_epoch(model, dl, opt, device)
        except RuntimeError as ex:
            if "out of memory" in str(ex).lower():
                print("OOM at epoch", e, "— reduce batch_size/base_feats/target_z")
                torch.cuda.empty_cache()
                break
            else:
                raise
        print(f"Epoch {e:03d} Loss {avg:.4f}")

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Saved → her2_bcbm_unet_transfer.pth")

if __name__=="__main__":
    main()
