"""
Download Pretrained Model Weights
Downloads publicly available pretrained weights for pose estimation models
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Pretrained weights with working URLs
PRETRAINED_WEIGHTS = {
    # ViTPose Models (from OneDrive/GitHub mirrors)
    "vitpose_small_coco": {
        "url": "https://1drv.ms/u/s!AimBgYV7JjTlgccFuHevO5Q_vV0Jgg?e=U1lnLd",
        "filename": "vitpose_small_coco.pth",
        "size": "90 MB",
        "description": "ViTPose-Small pretrained on COCO",
        "priority": 1,
        "source": "onedrive",
        "model_type": "vitpose"
    },
    
    # HRNet Models (from official repo)
    "hrnet_w48_coco": {
        "url": "https://drive.google.com/uc?export=download&id=1UoJhTtjHNJxELOcYm9wJLXFvUPvPRhkg",
        "filename": "hrnet_w48_384x288.pth",
        "size": "250 MB",
        "description": "HRNet-W48 pretrained on COCO (384x288)",
        "priority": 2,
        "source": "gdrive",
        "model_type": "hrnet"
    },
    
    # OpenPose Model (COCO format)
    "openpose_coco": {
        "url": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
        "filename": "openpose_coco.caffemodel",
        "size": "200 MB",
        "description": "OpenPose model trained on COCO",
        "priority": 3,
        "source": "direct",
        "model_type": "openpose"
    },
    
    # ImageNet Pretrained Backbones
    "resnet50_imagenet": {
        "url": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
        "filename": "resnet50_imagenet.pth",
        "size": "98 MB",
        "description": "ResNet-50 pretrained on ImageNet",
        "priority": 1,
        "source": "pytorch",
        "model_type": "backbone"
    },
    
    "resnet101_imagenet": {
        "url": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
        "filename": "resnet101_imagenet.pth",
        "size": "171 MB",
        "description": "ResNet-101 pretrained on ImageNet",
        "priority": 2,
        "source": "pytorch",
        "model_type": "backbone"
    },
    
    # ViT Backbones (from timm/PyTorch)
    "vit_base_patch16_224": {
        "url": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
        "filename": "vit_base_patch16_224.npz",
        "size": "330 MB",
        "description": "ViT-Base/16 pretrained on ImageNet-21K",
        "priority": 2,
        "source": "google",
        "model_type": "backbone"
    },
}


def download_file(url, filepath, description, chunk_size=8192):
    """Download file with progress bar"""
    try:
        logger.info(f"📥 Downloading: {description}")
        logger.info(f"   URL: {url}")
        logger.info(f"   Saving to: {filepath}")
        
        # Check if already exists
        if filepath.exists():
            logger.info(f"✓ File already exists: {filepath.name}")
            return True
        
        # Create directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✓ Downloaded: {filepath.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def download_weights(output_dir="data/pretrained", priority_filter=None, model_types=None):
    """Download pretrained weights"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("PRETRAINED WEIGHTS DOWNLOADER")
    logger.info("="*60)
    logger.info(f"Output: {output_path.absolute()}")
    logger.info(f"Total weights: {len(PRETRAINED_WEIGHTS)}")
    
    # Filter
    weights_to_download = {}
    for weight_id, info in PRETRAINED_WEIGHTS.items():
        if priority_filter and info['priority'] > priority_filter:
            continue
        if model_types and info['model_type'] not in model_types:
            continue
        weights_to_download[weight_id] = info
    
    if priority_filter:
        logger.info(f"Priority filter: ≤{priority_filter}")
    if model_types:
        logger.info(f"Type filter: {', '.join(model_types)}")
    
    logger.info(f"Weights to download: {len(weights_to_download)}")
    logger.info("="*60 + "\n")
    
    # Download
    success_count = 0
    failed = []
    
    sorted_weights = sorted(weights_to_download.items(), key=lambda x: x[1]['priority'])
    
    for i, (weight_id, info) in enumerate(sorted_weights, 1):
        logger.info(f"\n[{i}/{len(sorted_weights)}] {weight_id}")
        logger.info(f"Priority: {info['priority']} | Type: {info['model_type']}")
        logger.info(f"Size: {info['size']}")
        
        # Determine output path based on model type
        model_dir = output_path / info['model_type']
        filepath = model_dir / info['filename']
        
        success = download_file(info['url'], filepath, info['description'])
        
        if success:
            success_count += 1
        else:
            failed.append((weight_id, info))
        
        logger.info("")
    
    # Summary
    logger.info("="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"✓ Downloaded: {success_count}/{len(sorted_weights)}")
    logger.info(f"❌ Failed: {len(failed)}")
    
    if failed:
        logger.info("\nFailed downloads (may need manual download):")
        for weight_id, info in failed:
            logger.info(f"  - {weight_id}")
            logger.info(f"    URL: {info['url']}")
    
    logger.info(f"\nWeights saved to: {output_path.absolute()}")
    logger.info("="*60 + "\n")
    
    return success_count, failed


def create_weights_info():
    """Create info file about available weights"""
    
    with open("PRETRAINED_WEIGHTS.md", 'w', encoding='utf-8') as f:
        f.write("# Pretrained Model Weights\n\n")
        f.write("## Available Models\n\n")
        
        # Group by model type
        by_type = {}
        for weight_id, info in PRETRAINED_WEIGHTS.items():
            model_type = info['model_type']
            if model_type not in by_type:
                by_type[model_type] = []
            by_type[model_type].append((weight_id, info))
        
        for model_type in sorted(by_type.keys()):
            f.write(f"\n### {model_type.upper()} Models\n\n")
            f.write("| Model | Priority | Size | Description |\n")
            f.write("|-------|----------|------|-------------|\n")
            
            for weight_id, info in sorted(by_type[model_type], key=lambda x: x[1]['priority']):
                f.write(f"| **{weight_id}** | {info['priority']} | {info['size']} | {info['description']} |\n")
        
        f.write("\n## Download Instructions\n\n")
        f.write("```powershell\n")
        f.write("# Download all Priority 1 weights (essentials)\n")
        f.write("python scripts/download_pretrained_weights.py --priority 1\n\n")
        f.write("# Download only backbones\n")
        f.write("python scripts/download_pretrained_weights.py --types backbone\n\n")
        f.write("# Download everything\n")
        f.write("python scripts/download_pretrained_weights.py\n")
        f.write("```\n\n")
        
        f.write("## Model Directory Structure\n\n")
        f.write("```\n")
        f.write("data/pretrained/\n")
        f.write("├── vitpose/\n")
        f.write("│   └── vitpose_small_coco.pth\n")
        f.write("├── hrnet/\n")
        f.write("│   └── hrnet_w48_384x288.pth\n")
        f.write("├── backbone/\n")
        f.write("│   ├── resnet50_imagenet.pth\n")
        f.write("│   ├── resnet101_imagenet.pth\n")
        f.write("│   └── vit_base_patch16_224.npz\n")
        f.write("└── openpose/\n")
        f.write("    └── openpose_coco.caffemodel\n")
        f.write("```\n\n")
        
        f.write("## Usage in Code\n\n")
        f.write("```python\n")
        f.write("import torch\n\n")
        f.write("# Load backbone for fine-tuning\n")
        f.write("backbone = torch.load('data/pretrained/backbone/resnet50_imagenet.pth')\n\n")
        f.write("# Load pretrained pose model\n")
        f.write("model = torch.load('data/pretrained/vitpose/vitpose_small_coco.pth')\n")
        f.write("```\n")
    
    logger.info("✓ Created: PRETRAINED_WEIGHTS.md")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pretrained weights")
    parser.add_argument('--output', '-o', default='data/pretrained',
                       help='Output directory')
    parser.add_argument('--priority', '-p', type=int, default=None,
                       help='Only download priority ≤ N')
    parser.add_argument('--types', '-t', nargs='+',
                       choices=['vitpose', 'hrnet', 'openpose', 'backbone'],
                       help='Only download specific model types')
    parser.add_argument('--list', action='store_true',
                       help='List available weights')
    parser.add_argument('--info', action='store_true',
                       help='Create info file only')
    
    args = parser.parse_args()
    
    if args.list:
        logger.info("\n📋 AVAILABLE WEIGHTS:\n")
        for weight_id, info in sorted(PRETRAINED_WEIGHTS.items(), key=lambda x: x[1]['priority']):
            logger.info(f"  {weight_id}")
            logger.info(f"    Priority: {info['priority']}")
            logger.info(f"    Type: {info['model_type']}")
            logger.info(f"    Size: {info['size']}")
            logger.info(f"    Description: {info['description']}")
            logger.info("")
        return
    
    if args.info:
        create_weights_info()
        return
    
    # Create info file
    create_weights_info()
    
    # Download
    logger.info("\n💡 TIP: Priority 1 weights are essential for testing")
    logger.info("💡 TIP: Backbones are needed for training from scratch\n")
    
    success_count, failed = download_weights(
        output_dir=args.output,
        priority_filter=args.priority,
        model_types=args.types
    )
    
    if success_count > 0:
        logger.info("\n📚 NEXT STEPS:")
        logger.info("1. Verify downloaded weights")
        logger.info("2. Run: python scripts/test_model_loading.py")
        logger.info("3. Update config with model paths")


if __name__ == "__main__":
    main()
