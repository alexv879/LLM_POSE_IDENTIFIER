# Installation Instructions

## Quick Setup

Install all required dependencies with a single command:

```powershell
pip install torch torchvision numpy opencv-python pillow tqdm pyyaml pycocotools timm einops
```

## Verify Installation

Run the validation script to check everything is working:

```powershell
python scripts/validate_setup.py
```

## What's Installed

✅ **Downloaded** (No images yet):
- COCO 2017 Annotations (Train/Val) - 241 MB ✓
- ResNet-50 ImageNet Weights - 98 MB ✓
- ViTPose-Small COCO Weights - 0.3 MB ✓

⏳ **Not Downloaded** (Large files - download when needed):
- COCO 2017 Training Images - 19 GB
- COCO 2017 Validation Images - 1 GB
- Additional pretrained models

## Download Options

### For Testing (Small downloads):
```powershell
# Already done! You have:
# - Annotations (241 MB) ✓
# - Pretrained weights (~100 MB) ✓
```

### For Training (Large downloads):
```powershell
# Download COCO training images (19 GB)
python scripts/download_datasets.py --types images --priority 1

# Or download everything
python scripts/download_datasets.py
```

## Dependencies Status

| Package | Status | Purpose |
|---------|--------|---------|
| torch | ✓ Installed | Deep learning framework |
| torchvision | ✗ Missing | Image transforms and models |
| numpy | ✓ Installed | Numerical computing |
| opencv-python | ✗ Missing | Image processing |
| pillow | ✓ Installed | Image I/O |
| tqdm | ✓ Installed | Progress bars |
| pyyaml | ✓ Installed | Config files |
| pycocotools | ✗ Missing | COCO dataset API |
| timm | Optional | Pretrained vision models |
| einops | Optional | Tensor operations |

## Quick Test Without Full Dataset

Once dependencies are installed, you can test the models even without images:

```python
import torch
from src.stage1_baseline_model import SimplePoseNet

# Create model
model = SimplePoseNet(num_keypoints=17, backbone='resnet50')

# Test with dummy input
dummy_input = torch.randn(1, 3, 256, 192)
output = model(dummy_input)

print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
print("✓ Model works!")
```

## Next Steps After Installing Dependencies

1. **Install missing packages**:
   ```powershell
   pip install torchvision opencv-python pycocotools
   ```

2. **Verify setup**:
   ```powershell
   python scripts/validate_setup.py
   ```

3. **Test model loading**:
   ```powershell
   python scripts/test_model_loading.py
   ```

4. **Download sample images** (when ready):
   ```powershell
   # Download just val images for testing (1 GB)
   python scripts/download_datasets.py --types images --priority 1
   ```

5. **Run inference on a test image**:
   ```powershell
   python scripts/test_inference.py --image path/to/image.jpg
   ```

## Disk Space Requirements

| Component | Size | Status |
|-----------|------|--------|
| Annotations | 241 MB | ✓ Downloaded |
| Pretrained Weights | ~100 MB | ✓ Downloaded |
| **Total Downloaded** | **~350 MB** | **✓ Ready** |
| | | |
| Training Images | 19 GB | ⏳ Not downloaded |
| Validation Images | 1 GB | ⏳ Not downloaded |
| Test Images | 6 GB | ⏳ Not downloaded |
| Unlabeled Images | 19 GB | ⏳ Not downloaded |
| **Total Available** | **~45 GB** | **Optional** |

## Troubleshooting

### Import Errors
If you see "No module named 'stageX_...'", make sure you're running scripts from the project root:
```powershell
cd "d:\Research Paper Pose LLM Identifier\pose_llm_identifier"
python scripts/validate_setup.py
```

### PyTorch Version Issues
The pretrained weights show warnings because they're in older PyTorch format. This is normal and won't affect functionality. If needed:
```powershell
# Load with weights_only=False in code
torch.load(path, map_location='cpu', weights_only=False)
```

### COCO API on Windows
If pycocotools fails to install:
```powershell
# Install Visual C++ Build Tools first, then:
pip install pycocotools-windows
```

Or use pre-built wheels:
```powershell
pip install pycocotools --no-build-isolation
```
