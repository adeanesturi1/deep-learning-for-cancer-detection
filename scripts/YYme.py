import torch
import torch.nn as nn
from torchvision import models

class SimCLRNet(nn.Module):
    def __init__(self):
        super(SimCLRNet, self).__init__()
        self.encoder = models.resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        return x

output_path = "/sharedscratch/an252/cancerdetectiondataset/simclr_input/simclr_model.onnx"
dummy_input = torch.randn(1, 3, 224, 224)
model = SimCLRNet()
model.eval()
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["embedding"],
    dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
    opset_version=11
)

print(f"ONNX model saved to {output_path}")
