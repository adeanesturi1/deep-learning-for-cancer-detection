#!/usr/bin/env python
# scripts/trainher2unettransfer.py

import argparse, os, torch
import torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, nibabel as nib
from torch.utils.data import Dataset, DataLoader

class UNet3D(nn.Module):
    def __init__(self, in_ch, base_feats=16):  # reduced default base_feats to 16
        super().__init__()
        def block(ic,oc):
            return nn.Sequential(
                nn.Conv3d(ic, oc, 3, padding=1, bias=False),
                nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
                nn.Conv3d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm3d(oc), nn.ReLU(inplace=True),
            )
        self.enc1 = block(in_ch,      base_feats)
        self.enc2 = block(base_feats, base_feats*2)
        self.enc3 = block(base_feats*2, base_feats*4)
        self.enc4 = block(base_feats*4, base_feats*8)
        self.bot  = block(base_feats*8, base_feats*16)
        self.up4, self.dec4 = nn.ConvTranspose3d(base_feats*16, base_feats*8,2,2), block(base_feats*16, base_feats*8)
        self.up3, self.dec3 = nn.ConvTranspose3d(base_feats*8,  base_feats*4,2,2), block(base_feats*8,  base_feats*4)
        self.up2, self.dec2 = nn.ConvTranspose3d(base_feats*4,  base_feats*2,2,2), block(base_feats*4,  base_feats*2)
        self.up1, self.dec1 = nn.ConvTranspose3d(base_feats*2,  base_feats,  2,2), block(base_feats*2,  base_feats)
        self.seg_head      = nn.Conv3d(base_feats,1,1)
        self.classifier    = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(base_feats*16,1)
        )

    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(nn.functional.max_pool3d(e1,2))
        e3 = self.enc3(nn.functional.max_pool3d(e2,2))
        e4 = self.enc4(nn.functional.max_pool3d(e3,2))
        b  = self.bot(nn.functional.max_pool3d(e4,2))
        d4 = self.dec4(torch.cat([self.up4(b),  e4],1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3],1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],1))
        return self.seg_head(d1), self.classifier(b).squeeze(1)

class BCBM3DDataset(Dataset):
    def __init__(self, raw_dir, gt_dir, meta_csv, target_z):
        df = pd.read_csv(meta_csv)
        df = df[df["HER2_Status"].isin(["+","-"])]
        df["HER2"] = (df["HER2_Status"]=="+").astype(np.float32)
        self.cases = df[["nnUNet_ID","HER2"]].to_records(False)
        self.raw = raw_dir; self.gt = gt_dir; self.tz = target_z

    def __len__(self): return len(self.cases)

    def __getitem__(self,i):
        cid, her2 = self.cases[i]
        img_p = os.path.join(self.raw,"imagesTr",f"{cid}_0000.nii.gz")
        img   = nib.load(img_p).get_fdata().astype(np.float32)
        seg_p = os.path.join(self.gt, f"{cid}.nii.gz")
        seg   = nib.load(seg_p).get_fdata().astype(np.float32)

        # center-crop / pad Z
        H,W,Z = img.shape; tz = self.tz
        if Z>tz:
            s = (Z-tz)//2
            img = img[:,:,s:s+tz]; seg = seg[:,:,s:s+tz]
        elif Z<tz:
            pad = tz-Z; f = pad//2; b = pad-f
            img = np.pad(img,((0,0),(0,0),(f,b)),"constant")
            seg = np.pad(seg,((0,0),(0,0),(f,b)),"constant")

        return torch.from_numpy(img)[None], torch.from_numpy(seg)[None], torch.tensor(her2)

def collate_fn(b): 
    imgs, segs, cls = zip(*b)
    return torch.stack(imgs), torch.stack(segs), torch.stack(cls)

def remap_first_conv(w, in_ch):
    m = w.mean(dim=1, keepdim=True)
    return m if in_ch==1 else m.repeat(1,in_ch,1,1,1)

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir",        required=True)
    p.add_argument("--gt_dir",         required=True)
    p.add_argument("--metadata_csv",   required=True)
    p.add_argument("--pretrained_unet",required=True)
    p.add_argument("--target_z",       type=int, default=176)
    p.add_argument("--base_feats",     type=int, default=16)  # reduced default
    p.add_argument("--freeze",         action="store_true")
    p.add_argument("--epochs",         type=int, default=100)
    p.add_argument("--batch_size",     type=int, default=1)
    p.add_argument("--lr",             type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D(1, args.base_feats).to(device)

    # load BraTS encoder
    ckpt = torch.load(args.pretrained_unet, map_location="cpu")
    sd   = ckpt.get("network_weights", ckpt)
    # remap first conv
    if "encoder.stages.0.0.convs.0.conv.weight" in sd:
        w4 = sd.pop("encoder.stages.0.0.convs.0.conv.weight")
        model.state_dict()["enc1.0.weight"].copy_(remap_first_conv(w4,1))
    # copy others
    for k,v in sd.items():
        if not k.startswith("encoder.stages."): continue
        parts = k.split('.')
        st,ci,kind,param = int(parts[2]), int(parts[5]), parts[6], parts[7]
        if kind not in ("conv","norm") or param not in ("weight","bias"): continue
        prefix = "bot" if st==4 else f"enc{st+1}"
        idx    = ci*3 + (0 if kind=="conv" else 1)
        model.state_dict()[f"{prefix}.{idx}.{param}"].copy_(v)
    print("✓ Loaded BraTS encoder into UNet3D")

    if args.freeze:
        for n,p in model.named_parameters():
            if n.startswith("enc") or n.startswith("bot"): p.requires_grad=False

    ds = BCBM3DDataset(args.raw_dir, args.gt_dir, args.metadata_csv, args.target_z)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, collate_fn=collate_fn)
    opt = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr)

    for e in range(1,args.epochs+1):
        model.train(); tot=0.0
        try:
            for img,seg,her2 in dl:
                img,seg,her2 = img.to(device),seg.to(device),her2.to(device)
                opt.zero_grad()
                s_out,c_out = model(img)
                loss = nn.BCEWithLogitsLoss()(s_out,seg) + 0.5*nn.BCEWithLogitsLoss()(c_out,her2)
                loss.backward(); opt.step(); tot+=loss.item()
        except RuntimeError as oom:
            if "out of memory" in str(oom):
                print(f"OOM at epoch {e} – aborting. try smaller base_feats/target_z.")
                break
            else: raise
        print(f"Epoch {e:03d}  Loss {tot/len(dl):.4f}")
        if device.type=="cuda": torch.cuda.empty_cache()

    torch.save(model.state_dict(), "her2_bcbm_unet_transfer.pth")
    print("✓ Done. Saved → her2_bcbm_unet_transfer.pth")
