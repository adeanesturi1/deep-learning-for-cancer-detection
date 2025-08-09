# finetune_bcbm.py
import os
import json
import importlib
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

DATA_DIR = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM"
GT_DIR = os.path.join(DATA_DIR, "gt_segmentations")
IMG_DIR = os.path.join(DATA_DIR, "imagesTr")  # or  folder with 4D NIfTI images
PLANS_PATH = os.path.join(DATA_DIR, "nnUNetPlans.json")
PRETRAINED_PATH = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset001_BraTS/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_3/checkpoint_final.pth"
CONFIG = "3d_fullres"
SAVE_PATH = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_results/Dataset002_BCBM/finetuned_nnunet_bcbm.pth"
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BCBMDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.label_paths = sorted([os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith(".nii.gz")])
        self.image_paths = [os.path.join(image_dir, os.path.basename(p)) for p in self.label_paths]

    def __len__(self):
        return len(self.label_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx]).get_fdata().astype(np.float32)
        label = nib.load(self.label_paths[idx]).get_fdata().astype(np.int64)

        image = torch.tensor(image).unsqueeze(0)  # [C, D, H, W]
        label = torch.tensor(label)

        return image, label

with open(PLANS_PATH, 'r') as f:
    plans = json.load(f)

arch_info = plans["configurations"][CONFIG]["architecture"]
arch_kwargs = arch_info["arch_kwargs"]

for key in arch_info.get("_kw_requires_import", []):
    if isinstance(arch_kwargs[key], str):
        module_path, class_name = arch_kwargs[key].rsplit(".", 1)
        mod = importlib.import_module(module_path)
        arch_kwargs[key] = getattr(mod, class_name)

module_path, class_name = arch_info["network_class_name"].rsplit(".", 1)
ModelModule = importlib.import_module(module_path)
ModelClass = getattr(ModelModule, class_name)

model = ModelClass(
    input_channels=1,
    num_classes=1,
    **arch_kwargs
)
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE)['network_state_dict'])
model.to(DEVICE)

# Freeze encoder blocks 0–2
for name, param in model.named_parameters():
    if any(f"encoder.stages.{i}" in name for i in [0, 1, 2]):
        param.requires_grad = False

# Training setup
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()
dataloader = DataLoader(BCBMDataset(IMG_DIR, GT_DIR), batch_size=BATCH_SIZE, shuffle=True)

model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        logits = model(images)
        loss = loss_fn(logits, masks.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), SAVE_PATH)
print(f" Model saved to: {SAVE_PATH}")
