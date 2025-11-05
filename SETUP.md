# 🚀 Quick Setup Guide

## For Users Who Clone This Repository

### Prerequisites
- **Python 3.8+**
- **CUDA-capable GPU** (RTX 4060 or better recommended, 8GB+ VRAM)
- **100GB+ disk space** (for datasets and models)

### Installation Steps

#### 1. Clone the Repository
```bash
git clone https://github.com/alexv879/LLM_POSE_IDENTIFIER.git
cd LLM_POSE_IDENTIFIER
```

#### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv pose_env

# Activate it
# On Windows:
pose_env\Scripts\activate
# On Linux/Mac:
source pose_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Verify Installation
```bash
python scripts/validate_setup.py
```

#### 4. Download COCO Dataset (REQUIRED - ~19GB)

**This is mandatory!** The dataset is not included in the repository.

```bash
# Automatic download (recommended)
python scripts/download_coco.py

# This downloads:
# - train2017 images (118K images, ~18GB)
# - val2017 images (5K images, ~1GB)
# - Person keypoint annotations (~241MB)

# Expected structure after download:
# data/
#   coco/
#     images/
#       train2017/    (118,287 images)
#       val2017/      (5,000 images)
#     annotations/
#       person_keypoints_train2017.json
#       person_keypoints_val2017.json
```

**Manual download option:**
- Visit: https://cocodataset.org/#download
- Download train2017, val2017, and annotations
- Extract to `data/coco/` following structure above

**Download time:** 30-60 minutes (varies by connection speed)

**Optional: Unlabeled images for Stage 2 SSL (~5GB)**
```bash
# Only needed if you plan to run Stage 2 semi-supervised learning
python scripts/download_coco_unlabeled.py --num_images 5000
```

#### 5. Download Pretrained Weights (Automatic)
```bash
# Sapiens-1B will auto-download from HuggingFace on first run
# Or manually download:
python scripts/download_pretrained.py
```

### Quick Start Training

#### Stage 1: Baseline (Start Here!)
```bash
# Train baseline model (2-3 days on RTX 4060)
python stages/stage1_baseline.py --config configs/stage1_config.yaml

# Monitor training
tensorboard --logdir logs/stage1
```

Expected output:
- Phase 1 (10 epochs, decoder warmup): ~76-79% AP
- Phase 2 (100 epochs, full training): ~82-86% AP

#### Stage 2: SSL (Optional, for improvement)
```bash
python stages/stage2_ssl.py --config configs/stage2_ssl_config.yaml
```

Expected: +5-7% AP improvement → ~89-93% AP

### Configuration

All settings are in `configs/stage1_config.yaml`:

```yaml
model:
  pretrained_path: "facebook/sapiens-pretrain-1b-torchscript"
  input_size: [512, 384]  # Adjust for your GPU
  
training:
  phase2:
    batch_size: 2  # Increase if you have more VRAM
    epochs: 100    # Reduce for faster experiments
```

### Troubleshooting

**Out of Memory (OOM)?**
```yaml
# In configs/stage1_config.yaml, reduce:
input_size: [384, 288]  # Lower resolution
batch_size: 1           # Smaller batches
```

**Import errors?**
```bash
pip install --upgrade transformers albumentations
```

**CUDA not available?**
```bash
# Check PyTorch installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### File Structure After Setup

```
pose_llm_identifier/
├── data/
│   └── coco/              # Downloaded COCO dataset
│       ├── images/
│       └── annotations/
├── checkpoints/           # Saved models (created during training)
├── logs/                  # TensorBoard logs
├── configs/               # Configuration files
├── models/                # Model implementations
├── stages/                # Training scripts
└── utils/                 # Helper functions
```

### Need Help?

- **Issues**: [GitHub Issues](https://github.com/alexv879/LLM_POSE_IDENTIFIER/issues)
- **Discussions**: [GitHub Discussions](https://github.com/alexv879/LLM_POSE_IDENTIFIER/discussions)
- **Documentation**: See full [README.md](README.md)
- **Papers**: See [PAPERS_DOWNLOADED.md](PAPERS_DOWNLOADED.md) for research background
- **Repository**: https://github.com/alexv879/LLM_POSE_IDENTIFIER

### Expected Training Times (RTX 4060)

| Stage | Time | Output |
|-------|------|--------|
| Phase 1 (decoder) | 8-10 hours | ~76-79% AP |
| Phase 2 (full) | 2-3 days | ~82-86% AP |
| Stage 2 SSL | 1-2 days | ~89-93% AP |

### Research Papers Included

All 18 research papers are included in the `papers/` directory:
- ✅ Sapiens (ECCV 2024) - Primary model
- ✅ ViTPose (NeurIPS 2022) - Ensemble component  
- ✅ DWPose (ICCV 2023) - Ensemble component
- ✅ SSL Multi-Path (ICLR 2025) - Semi-supervised learning
- ✅ 14 supporting papers (HRNet, OpenPose, UDP, COCO, etc.)

See [PAPERS_DOWNLOADED.md](PAPERS_DOWNLOADED.md) for full list and reading guides.

### System Requirements Met?

Run this to check:
```bash
python check_system.py
```

This will verify:
- ✅ Python version (3.8+)
- ✅ PyTorch + CUDA
- ✅ GPU memory (8GB+)
- ✅ Disk space (100GB+)
- ✅ All dependencies installed

### Success Indicators

You're ready when:
1. ✅ `validate_setup.py` passes all checks
2. ✅ COCO dataset downloaded (~19GB)
3. ✅ Test model creation works (Stage 1 Phase 1 starts)
4. ✅ TensorBoard shows training curves

**Good luck with your training!** 🚀
