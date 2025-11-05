# Pretrained Model Weights

## Available Models


### BACKBONE Models

| Model | Priority | Size | Description |
|-------|----------|------|-------------|
| **resnet50_imagenet** | 1 | 98 MB | ResNet-50 pretrained on ImageNet |
| **resnet101_imagenet** | 2 | 171 MB | ResNet-101 pretrained on ImageNet |
| **vit_base_patch16_224** | 2 | 330 MB | ViT-Base/16 pretrained on ImageNet-21K |

### HRNET Models

| Model | Priority | Size | Description |
|-------|----------|------|-------------|
| **hrnet_w48_coco** | 2 | 250 MB | HRNet-W48 pretrained on COCO (384x288) |

### OPENPOSE Models

| Model | Priority | Size | Description |
|-------|----------|------|-------------|
| **openpose_coco** | 3 | 200 MB | OpenPose model trained on COCO |

### VITPOSE Models

| Model | Priority | Size | Description |
|-------|----------|------|-------------|
| **vitpose_small_coco** | 1 | 90 MB | ViTPose-Small pretrained on COCO |

## Download Instructions

```powershell
# Download all Priority 1 weights (essentials)
python scripts/download_pretrained_weights.py --priority 1

# Download only backbones
python scripts/download_pretrained_weights.py --types backbone

# Download everything
python scripts/download_pretrained_weights.py
```

## Model Directory Structure

```
data/pretrained/
├── vitpose/
│   └── vitpose_small_coco.pth
├── hrnet/
│   └── hrnet_w48_384x288.pth
├── backbone/
│   ├── resnet50_imagenet.pth
│   ├── resnet101_imagenet.pth
│   └── vit_base_patch16_224.npz
└── openpose/
    └── openpose_coco.caffemodel
```

## Usage in Code

```python
import torch

# Load backbone for fine-tuning
backbone = torch.load('data/pretrained/backbone/resnet50_imagenet.pth')

# Load pretrained pose model
model = torch.load('data/pretrained/vitpose/vitpose_small_coco.pth')
```
