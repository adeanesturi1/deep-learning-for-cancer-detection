#!/usr/bin/env python
import torch, argparse
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from trainher2unettransfer import UNet2D

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', nargs='?', default='her2_bcbm_unet2d_transfer_best.pth')
parser.add_argument('--onnx_out',    nargs='?', default='her2_bcbm_unet2d_transfer_best.onnx')
parser.add_argument('--input_size',  nargs=2,  type=int, default=[240,240])
parser.add_argument('--base_feats',  type=int, default=16)
parser.add_argument('--dropout',     type=float, default=0.0)
args = parser.parse_args()

# load model
device = torch.device('cpu')
model  = UNet2D(1, args.base_feats, dropout=args.dropout).to(device)
sd     = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(sd)
model.eval()

# dummy input matching your slice dims
H, W = args.input_size
dummy = torch.randn(1, 1, H, W, device=device)

# export
torch.onnx.export(
    model, dummy, args.onnx_out,
    input_names=['input'],
    output_names=['segmentation','classification'],
    dynamic_axes={
        'input':        {0:'batch', 2:'H',   3:'W'},
        'segmentation': {0:'batch', 2:'H',   3:'W'},
        'classification': {0:'batch'}
    },
    opset_version=11
)
print(f"ONNX model saved to {args.onnx_out}")
