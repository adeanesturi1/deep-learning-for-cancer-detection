#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------
# 1) ARGS & CONFIG
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--mid_slice_dir",
    default="/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/mid_slices/",
    help="folder with BCBM_xxxx_T1.png")
parser.add_argument("--metadata_csv",
    default="/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv",
    help="csv with nnUNet_ID,HER2_Status")
parser.add_argument("--pretrained_unet",
    required=True,
    help=".pth of your BraTS nnU-Net 2D encoder")
parser.add_argument("--output_dir",
    default="/sharedscratch/an252/cancerdetectiondataset/her2_unet_transfer",
    help="where to save model + logs")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs",     type=int, default=20)
parser.add_argument("--lr",         type=float, default=1e-4)
parser.add_argument("--freeze",     action="store_true",
    help="freeze U-Net encoder weights")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
log_csv = os.path.join(args.output_dir, "training_log.csv")

# ----------------------------
# 2) DATASET
# ----------------------------
class HER2Dataset(Dataset):
    def __init__(self, paths, labels, tfm=None):
        self.paths, self.labels, self.tfm = paths, labels, tfm

    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.tfm: img = self.tfm(img)
        return img, torch.tensor(self.labels[i], dtype=torch.float32)

# load & filter metadata
df = pd.read_csv(args.metadata_csv)
df = df[df["HER2_Status"].isin(["+", "-"])].copy()
df["label"] = df["HER2_Status"].map({"+":1, "-":0})

# build image list
imgs, labs = [], []
for _,r in df.iterrows():
    p = os.path.join(args.mid_slice_dir, f"{r.nnUNet_ID}_T1.png")
    if os.path.isfile(p):
        imgs.append(p); labs.append(r.label)

train_i, val_i, train_l, val_l = train_test_split(
    imgs, labs, test_size=0.2, random_state=42, stratify=labs
)

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
train_ds = HER2Dataset(train_i, train_l, tfm)
val_ds   = HER2Dataset(val_i,   val_l,   tfm)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ----------------------------
# 3) MODEL DEFINITION
# ----------------------------
def double_conv(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )

class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, filters=[64,128,256,512]):
        super().__init__()
        self.conv1 = double_conv(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = double_conv(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = double_conv(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = double_conv(filters[2], filters[3])
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        return x  # bottleneck feature map

class HER2Classifier(nn.Module):
    def __init__(self, encoder:UNetEncoder, freeze:bool=True):
        super().__init__()
        self.encoder = encoder
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(512, 1)
    def forward(self, x):
        feats = self.encoder(x)            # [B,512,H/8,W/8]
        v     = self.pool(feats).view(x.size(0), -1)
        return self.fc(v)

# instantiate
enc = UNetEncoder(in_channels=3, filters=[64,128,256,512])
model = HER2Classifier(enc, freeze=args.freeze)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ----------------------------
# 4) LOAD PRETRAINED ENCODER
# ----------------------------
pt = torch.load(args.pretrained_unet, map_location="cpu")
# assume `pt` is a dict of nnUNet state_dict with keys like 'network.encoder.stages...'
# you may need to adapt the key filtering to your saved format:
enc_sd = { k.replace("encoder.", ""):v
           for k,v in pt.items()
           if k.startswith("encoder.") }
enc.load_state_dict(enc_sd, strict=False)
print("Loaded pretrained encoder weights.")

# ----------------------------
# 5) TRAIN / EVAL
# ----------------------------
criterion = nn.BCEWithLogitsLoss()
opt_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(opt_params, lr=args.lr)

best_val_acc = 0.0
logs = []

for epoch in range(1, args.epochs+1):
    # train
    model.train()
    running_loss, corr, tot = 0,0,0
    for x,y in tqdm(train_loader, desc=f"Train E{epoch}"):
        x,y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*x.size(0)
        preds = (torch.sigmoid(logits)>0.5).float()
        corr += (preds==y).sum().item(); tot += x.size(0)

    train_loss = running_loss/tot
    train_acc  = corr/tot

    # val
    model.eval()
    corr, tot = 0,0
    with torch.no_grad():
        for x,y in tqdm(val_loader, desc=f" Val E{epoch}"):
            x,y = x.to(device), y.to(device).unsqueeze(1)
            preds = (torch.sigmoid(model(x))>0.5).float()
            corr += (preds==y).sum().item()
            tot += x.size(0)
    val_acc = corr/tot

    # save best
    if val_acc>best_val_acc:
        best_val_acc=val_acc
        torch.save(model.state_dict(), os.path.join(args.output_dir,"best_transfer.pth"))

    logs.append({
        "epoch":epoch,
        "train_loss":train_loss,
        "train_acc":train_acc,
        "val_acc":val_acc,
        "freeze":args.freeze
    })
    print(f"E{epoch}: loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

# dump CSV
pd.DataFrame(logs).to_csv(log_csv, index=False)
print("Logs written to", log_csv)
