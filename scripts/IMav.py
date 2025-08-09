import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torchvision import datasets, transforms, models # type: ignore
from torch.utils.data import DataLoader # type: ignore
import os # type: ignore

class SimCLRTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def contrastive_loss(z1, z2, temperature=0.5):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(representations, representations.T)
    labels = torch.cat([torch.arange(z1.size(0))] * 2).to(z1.device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    logits = similarity / temperature
    loss = nn.functional.cross_entropy(logits, labels.argmax(dim=1))
    return loss

# main
def main():
    data_path = "/sharedscratch/an252/cancerdetectiondataset/simclr_input/"
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.ImageFolder(root=data_path, transform=SimCLRTransform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Identity()  # remove final classifier
    model = model.to(device)

    projection_head = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).to(device)

    optimizer = optim.Adam(list(model.parameters()) + list(projection_head.parameters()), lr=1e-3)

    for epoch in range(20):
        total_loss = 0
        model.train()
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = projection_head(model(x1))
            z2 = projection_head(model(x2))
            loss = contrastive_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

if __name__ == "__main__":
    main()
