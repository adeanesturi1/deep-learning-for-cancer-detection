import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

mid_slice_dir = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_preprocessed/Dataset002_BCBM/mid_slices/"
metadata_csv = "/sharedscratch/an252/cancerdetectiondataset/nnUNet_raw/Dataset002_BCBM/bcbm_metadata.csv"
output_base = "/sharedscratch/an252/cancerdetectiondataset"
os.makedirs(output_base, exist_ok=True)
df = pd.read_csv(metadata_csv)
df = df[df["HER2_Status"].isin(["+", "-"])].copy()
df["label"] = df["HER2_Status"].map({"+": 1, "-": 0})

image_paths = []
labels = []

for _, row in df.iterrows():
    case_id = row["nnUNet_ID"]
    img_path = os.path.join(mid_slice_dir, f"{case_id}.png")
    if os.path.isfile(img_path):
        image_paths.append(img_path)
        labels.append(row["label"])

train_imgs, val_imgs, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

class HER2Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

train_ds = HER2Dataset(train_imgs, train_labels, transform=transform)
val_ds = HER2Dataset(val_imgs, val_labels, transform=transform)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_acc = 0

for epoch in range(10):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        x, y = x.to(device), y.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = torch.sigmoid(logits) > 0.5
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            logits = model(x)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == y).sum().item()
            total += y.size(0)
    val_acc = correct / total
    print(f"Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(output_base, "best_model.pth"))
        print("Saved best model!")

print("Done.")
