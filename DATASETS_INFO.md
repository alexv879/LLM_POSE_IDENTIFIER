# Available Datasets for Download

## Dataset Overview


### Priority 1: Essential

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| **coco_annotations** | annotations | 241 MB | COCO 2017 Annotations (Train/Val keypoints, captions, instances) |
| **coco_sample_images** | test_data | 1 GB | COCO Val 2017 (can extract just a few for testing) |
| **coco_train_2017** | images | 19 GB | COCO 2017 Training Images (118K images) |
| **coco_val_2017** | images | 1 GB | COCO 2017 Validation Images (5K images) |
| **vitpose_base_coco** | weights | 360 MB | ViTPose-Base pretrained on COCO (publicly available) |

### Priority 2: Important

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| **coco_unlabeled_2017** | images | 19 GB | COCO 2017 Unlabeled Images (123K images for SSL) |
| **hrnet_w32_coco** | weights | 120 MB | HRNet-W32 pretrained on COCO (256x192) |
| **vit_base_patch16** | weights | 330 MB | ViT-Base/16 pretrained on ImageNet-21K |
| **vitpose_large_coco** | weights | 1.1 GB | ViTPose-Large pretrained on COCO |

### Priority 3: Optional

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| **coco_test_2017** | images | 6 GB | COCO 2017 Test Images (41K images) |
| **mpii_images** | images | 12 GB | MPII Human Pose Dataset (25K images) |
| **vitpose_huge_coco** | weights | 2.3 GB | ViTPose-Huge pretrained on COCO |

## Download Instructions

### Quick Start

```powershell
# Download ONLY essentials (Priority 1: ~20 GB)
python scripts/download_datasets.py --priority 1 --types annotations weights

# Download annotations and weights (no large images)
python scripts/download_datasets.py --types annotations weights

# Download everything (Priority 1-3: ~60+ GB)
python scripts/download_datasets.py
```

### Options

- `--priority N`: Only download priority ≤ N
- `--types TYPE1 TYPE2`: Only download specific types (images, annotations, weights)
- `--no-extract`: Download without extracting archives
- `--output DIR`: Change output directory (default: data)

## Manual Download Links

### coco_train_2017
- **Description**: COCO 2017 Training Images (118K images)
- **Size**: 19 GB
- **Type**: images
- **URL**: http://images.cocodataset.org/zips/train2017.zip

### coco_val_2017
- **Description**: COCO 2017 Validation Images (5K images)
- **Size**: 1 GB
- **Type**: images
- **URL**: http://images.cocodataset.org/zips/val2017.zip

### coco_annotations
- **Description**: COCO 2017 Annotations (Train/Val keypoints, captions, instances)
- **Size**: 241 MB
- **Type**: annotations
- **URL**: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

### vitpose_base_coco
- **Description**: ViTPose-Base pretrained on COCO (publicly available)
- **Size**: 360 MB
- **Type**: weights
- **URL**: https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-b-multi-coco.pth

### coco_sample_images
- **Description**: COCO Val 2017 (can extract just a few for testing)
- **Size**: 1 GB
- **Type**: test_data
- **URL**: http://images.cocodataset.org/zips/val2017.zip
- **Note**: For testing, extract only a few images

### coco_unlabeled_2017
- **Description**: COCO 2017 Unlabeled Images (123K images for SSL)
- **Size**: 19 GB
- **Type**: images
- **URL**: http://images.cocodataset.org/zips/unlabeled2017.zip

### vitpose_large_coco
- **Description**: ViTPose-Large pretrained on COCO
- **Size**: 1.1 GB
- **Type**: weights
- **URL**: https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-l-multi-coco.pth

### hrnet_w32_coco
- **Description**: HRNet-W32 pretrained on COCO (256x192)
- **Size**: 120 MB
- **Type**: weights
- **URL**: https://drive.google.com/uc?export=download&id=1zYC7go9EV0XaSlSBjMaiyJ4j5Q4UOOr8

### vit_base_patch16
- **Description**: ViT-Base/16 pretrained on ImageNet-21K
- **Size**: 330 MB
- **Type**: weights
- **URL**: https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth

### coco_test_2017
- **Description**: COCO 2017 Test Images (41K images)
- **Size**: 6 GB
- **Type**: images
- **URL**: http://images.cocodataset.org/zips/test2017.zip

### vitpose_huge_coco
- **Description**: ViTPose-Huge pretrained on COCO
- **Size**: 2.3 GB
- **Type**: weights
- **URL**: https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-h-multi-coco.pth

### mpii_images
- **Description**: MPII Human Pose Dataset (25K images)
- **Size**: 12 GB
- **Type**: images
- **URL**: https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz
- **Note**: Requires registration at MPI website

## Dataset Structure

```
data/
├── downloads/                 (Downloaded archives)
│   ├── train2017.zip
│   ├── val2017.zip
│   └── annotations_trainval2017.zip
│
├── coco/                      (COCO dataset)
│   ├── images/
│   │   ├── train2017/        (118K images)
│   │   ├── val2017/          (5K images)
│   │   └── unlabeled2017/    (123K images)
│   └── annotations/
│       ├── person_keypoints_train2017.json
│       └── person_keypoints_val2017.json
│
└── pretrained/                (Pretrained weights)
    └── sapiens/
        └── sapiens_2b_pytorch_model.bin
```
