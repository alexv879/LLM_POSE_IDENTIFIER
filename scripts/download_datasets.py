"""
Download All Referenced Datasets
Automatically downloads datasets mentioned in the research documents
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import logging
import hashlib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# All datasets with their download URLs
DATASETS = {
    # COCO Dataset (Primary)
    "coco_train_2017": {
        "url": "http://images.cocodataset.org/zips/train2017.zip",
        "filename": "train2017.zip",
        "size": "19 GB",
        "description": "COCO 2017 Training Images (118K images)",
        "extract_to": "coco/images",
        "priority": 1,
        "type": "images"
    },
    "coco_val_2017": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "filename": "val2017.zip",
        "size": "1 GB",
        "description": "COCO 2017 Validation Images (5K images)",
        "extract_to": "coco/images",
        "priority": 1,
        "type": "images"
    },
    "coco_test_2017": {
        "url": "http://images.cocodataset.org/zips/test2017.zip",
        "filename": "test2017.zip",
        "size": "6 GB",
        "description": "COCO 2017 Test Images (41K images)",
        "extract_to": "coco/images",
        "priority": 3,
        "type": "images"
    },
    "coco_annotations": {
        "url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "filename": "annotations_trainval2017.zip",
        "size": "241 MB",
        "description": "COCO 2017 Annotations (Train/Val keypoints, captions, instances)",
        "extract_to": "coco",
        "priority": 1,
        "type": "annotations"
    },
    
    # COCO Unlabeled (for SSL)
    "coco_unlabeled_2017": {
        "url": "http://images.cocodataset.org/zips/unlabeled2017.zip",
        "filename": "unlabeled2017.zip",
        "size": "19 GB",
        "description": "COCO 2017 Unlabeled Images (123K images for SSL)",
        "extract_to": "coco/images",
        "priority": 2,
        "type": "images"
    },
    
    # Pretrained Model Weights (ViTPose - publicly available)
    "vitpose_base_coco": {
        "url": "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-b-multi-coco.pth",
        "filename": "vitpose_base_coco.pth",
        "size": "360 MB",
        "description": "ViTPose-Base pretrained on COCO (publicly available)",
        "extract_to": "pretrained/vitpose",
        "priority": 1,
        "type": "weights"
    },
    "vitpose_large_coco": {
        "url": "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-l-multi-coco.pth",
        "filename": "vitpose_large_coco.pth",
        "size": "1.1 GB",
        "description": "ViTPose-Large pretrained on COCO",
        "extract_to": "pretrained/vitpose",
        "priority": 2,
        "type": "weights"
    },
    "vitpose_huge_coco": {
        "url": "https://github.com/ViTAE-Transformer/ViTPose/releases/download/v0.1/vitpose-h-multi-coco.pth",
        "filename": "vitpose_huge_coco.pth",
        "size": "2.3 GB",
        "description": "ViTPose-Huge pretrained on COCO",
        "extract_to": "pretrained/vitpose",
        "priority": 3,
        "type": "weights"
    },
    
    # HRNet Pretrained Weights
    "hrnet_w32_coco": {
        "url": "https://drive.google.com/uc?export=download&id=1zYC7go9EV0XaSlSBjMaiyJ4j5Q4UOOr8",
        "filename": "hrnet_w32_256x192.pth",
        "size": "120 MB",
        "description": "HRNet-W32 pretrained on COCO (256x192)",
        "extract_to": "pretrained/hrnet",
        "priority": 2,
        "type": "weights"
    },
    
    # ImageNet Pretrained Backbones (for fine-tuning)
    "vit_base_patch16": {
        "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth",
        "filename": "vit_base_patch16_224.pth",
        "size": "330 MB",
        "description": "ViT-Base/16 pretrained on ImageNet-21K",
        "extract_to": "pretrained/vit",
        "priority": 2,
        "type": "weights"
    },
    
    # Sample Test Images (for testing without full dataset)
    "coco_sample_images": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "filename": "val2017_sample.zip",
        "size": "1 GB",
        "description": "COCO Val 2017 (can extract just a few for testing)",
        "extract_to": "coco/images",
        "priority": 1,
        "type": "test_data",
        "note": "For testing, extract only a few images"
    },
    
    # MPII Dataset (Optional)
    "mpii_images": {
        "url": "https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz",
        "filename": "mpii_human_pose_v1.tar.gz",
        "size": "12 GB",
        "description": "MPII Human Pose Dataset (25K images)",
        "extract_to": "mpii",
        "priority": 3,
        "type": "images",
        "note": "Requires registration at MPI website"
    },
}


def download_file(url, filepath, description):
    """
    Download a file with progress bar
    
    Args:
        url: Download URL
        filepath: Local file path to save
        description: Description for progress bar
    """
    try:
        logger.info(f"📥 Downloading: {description}")
        logger.info(f"   URL: {url}")
        logger.info(f"   Saving to: {filepath}")
        
        # Check if file already exists
        if filepath.exists():
            logger.info(f"✓ File already exists, skipping download")
            return True
        
        # Create parent directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, 
                     desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        logger.info(f"✓ Successfully downloaded: {filepath.name}")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to download: {e}")
        if filepath.exists():
            filepath.unlink()
        return False
    
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if filepath.exists():
            filepath.unlink()
        return False


def extract_archive(archive_path, extract_to):
    """
    Extract zip or tar.gz archive
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
    """
    try:
        logger.info(f"📦 Extracting: {archive_path.name}")
        logger.info(f"   Destination: {extract_to}")
        
        extract_path = Path(extract_to)
        extract_path.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                # Get total size for progress bar
                total_size = sum(info.file_size for info in zip_ref.filelist)
                
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc="Extracting") as pbar:
                    for member in zip_ref.filelist:
                        zip_ref.extract(member, extract_path)
                        pbar.update(member.file_size)
        
        elif archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_path)
        
        else:
            logger.warning(f"⚠️  Unknown archive format: {archive_path.suffix}")
            return False
        
        logger.info(f"✓ Successfully extracted: {archive_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to extract {archive_path.name}: {e}")
        return False


def download_dataset(dataset_id, dataset_info, output_dir, extract=True):
    """
    Download and optionally extract a dataset
    
    Args:
        dataset_id: Unique identifier
        dataset_info: Dict with url, filename, description, etc.
        output_dir: Base output directory
        extract: Whether to extract after download
    """
    url = dataset_info['url']
    filename = dataset_info['filename']
    description = dataset_info['description']
    extract_to = dataset_info.get('extract_to', '')
    
    # Determine file paths
    download_path = Path(output_dir) / "downloads" / filename
    
    # Download
    success = download_file(url, download_path, description)
    
    if not success:
        return False
    
    # Extract if requested and file is archive
    if extract and filename.endswith(('.zip', '.tar.gz')):
        extract_path = Path(output_dir) / extract_to
        success = extract_archive(download_path, extract_path)
        
        if success:
            logger.info(f"💾 Archive kept at: {download_path}")
            logger.info(f"📁 Extracted to: {extract_path}")
    
    return success


def download_datasets(output_dir="data", priority_filter=None, 
                     types_filter=None, extract=True):
    """
    Download datasets filtered by priority and type
    
    Args:
        output_dir: Base directory for downloads
        priority_filter: Only download priority <= N (1=essential)
        types_filter: List of types to download ['images', 'annotations', 'weights']
        extract: Whether to extract archives
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("DATASET DOWNLOADER")
    logger.info("="*60)
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Total datasets: {len(DATASETS)}")
    
    # Filter datasets
    datasets_to_download = {}
    
    for dataset_id, info in DATASETS.items():
        # Priority filter
        if priority_filter and info['priority'] > priority_filter:
            continue
        
        # Type filter
        if types_filter and info['type'] not in types_filter:
            continue
        
        datasets_to_download[dataset_id] = info
    
    if priority_filter:
        logger.info(f"Priority filter: ≤{priority_filter}")
    if types_filter:
        logger.info(f"Type filter: {', '.join(types_filter)}")
    
    logger.info(f"Datasets to download: {len(datasets_to_download)}")
    logger.info("="*60 + "\n")
    
    # Sort by priority
    sorted_datasets = sorted(datasets_to_download.items(), 
                           key=lambda x: x[1]['priority'])
    
    # Calculate total size
    total_size_gb = sum(
        float(info['size'].split()[0]) 
        for _, info in sorted_datasets 
        if info['size'].split()[0].replace('.', '').isdigit()
    )
    
    logger.info(f"📊 Estimated total download: ~{total_size_gb:.1f} GB")
    logger.info(f"⏱️  Estimated time: ~{int(total_size_gb * 5)} minutes (at 10 MB/s)")
    logger.info("")
    
    # Confirm large downloads
    if total_size_gb > 5:
        logger.warning(f"⚠️  Large download size: {total_size_gb:.1f} GB")
        logger.info("💡 TIP: Use --priority 1 to download only essentials first")
        logger.info("💡 TIP: Use --types annotations weights to skip large images")
        logger.info("")
    
    # Download datasets
    success_count = 0
    failed_datasets = []
    
    for i, (dataset_id, info) in enumerate(sorted_datasets, 1):
        logger.info(f"\n[{i}/{len(sorted_datasets)}] Processing: {dataset_id}")
        logger.info(f"Priority: {info['priority']}")
        logger.info(f"Type: {info['type']}")
        logger.info(f"Size: {info['size']}")
        
        if 'note' in info:
            logger.warning(f"⚠️  NOTE: {info['note']}")
        
        success = download_dataset(dataset_id, info, output_path, extract=extract)
        
        if success:
            success_count += 1
        else:
            failed_datasets.append((dataset_id, info))
        
        logger.info("")
    
    # Summary
    logger.info("="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"✓ Successfully downloaded: {success_count}/{len(sorted_datasets)}")
    logger.info(f"❌ Failed: {len(failed_datasets)}")
    
    if failed_datasets:
        logger.info("\nFailed downloads:")
        for dataset_id, info in failed_datasets:
            logger.info(f"  - {dataset_id}: {info['url']}")
    
    logger.info(f"\nData saved to: {output_path.absolute()}")
    logger.info("="*60 + "\n")
    
    return success_count, failed_datasets


def create_dataset_info(output_file="DATASETS_INFO.md"):
    """Create a markdown file with dataset information"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Available Datasets for Download\n\n")
        f.write("## Dataset Overview\n\n")
        
        # Group by priority
        for priority in [1, 2, 3]:
            priority_name = {1: "Essential", 2: "Important", 3: "Optional"}[priority]
            f.write(f"\n### Priority {priority}: {priority_name}\n\n")
            
            priority_datasets = {k: v for k, v in DATASETS.items() 
                               if v['priority'] == priority}
            
            if not priority_datasets:
                continue
            
            f.write("| Dataset | Type | Size | Description |\n")
            f.write("|---------|------|------|-------------|\n")
            
            for dataset_id, info in sorted(priority_datasets.items()):
                f.write(f"| **{dataset_id}** | {info['type']} | {info['size']} | {info['description']} |\n")
        
        f.write("\n## Download Instructions\n\n")
        f.write("### Quick Start\n\n")
        f.write("```powershell\n")
        f.write("# Download ONLY essentials (Priority 1: ~20 GB)\n")
        f.write("python scripts/download_datasets.py --priority 1 --types annotations weights\n\n")
        f.write("# Download annotations and weights (no large images)\n")
        f.write("python scripts/download_datasets.py --types annotations weights\n\n")
        f.write("# Download everything (Priority 1-3: ~60+ GB)\n")
        f.write("python scripts/download_datasets.py\n")
        f.write("```\n\n")
        
        f.write("### Options\n\n")
        f.write("- `--priority N`: Only download priority ≤ N\n")
        f.write("- `--types TYPE1 TYPE2`: Only download specific types (images, annotations, weights)\n")
        f.write("- `--no-extract`: Download without extracting archives\n")
        f.write("- `--output DIR`: Change output directory (default: data)\n\n")
        
        f.write("## Manual Download Links\n\n")
        
        for dataset_id, info in sorted(DATASETS.items(), key=lambda x: x[1]['priority']):
            f.write(f"### {dataset_id}\n")
            f.write(f"- **Description**: {info['description']}\n")
            f.write(f"- **Size**: {info['size']}\n")
            f.write(f"- **Type**: {info['type']}\n")
            f.write(f"- **URL**: {info['url']}\n")
            if 'note' in info:
                f.write(f"- **Note**: {info['note']}\n")
            f.write("\n")
        
        f.write("## Dataset Structure\n\n")
        f.write("```\n")
        f.write("data/\n")
        f.write("├── downloads/                 (Downloaded archives)\n")
        f.write("│   ├── train2017.zip\n")
        f.write("│   ├── val2017.zip\n")
        f.write("│   └── annotations_trainval2017.zip\n")
        f.write("│\n")
        f.write("├── coco/                      (COCO dataset)\n")
        f.write("│   ├── images/\n")
        f.write("│   │   ├── train2017/        (118K images)\n")
        f.write("│   │   ├── val2017/          (5K images)\n")
        f.write("│   │   └── unlabeled2017/    (123K images)\n")
        f.write("│   └── annotations/\n")
        f.write("│       ├── person_keypoints_train2017.json\n")
        f.write("│       └── person_keypoints_val2017.json\n")
        f.write("│\n")
        f.write("└── pretrained/                (Pretrained weights)\n")
        f.write("    └── sapiens/\n")
        f.write("        └── sapiens_2b_pytorch_model.bin\n")
        f.write("```\n")
    
    logger.info(f"✓ Dataset info created: {output_file}")


def main():
    """Main function with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download research datasets")
    parser.add_argument('--output', '-o', default='data', 
                       help='Output directory (default: data)')
    parser.add_argument('--priority', '-p', type=int, default=None,
                       help='Only download priority ≤ N (1=essential, 2=important, 3=optional)')
    parser.add_argument('--types', '-t', nargs='+', 
                       choices=['images', 'annotations', 'weights'],
                       help='Only download specific types')
    parser.add_argument('--no-extract', action='store_true',
                       help='Download without extracting archives')
    parser.add_argument('--info', action='store_true',
                       help='Create dataset info file only (no download)')
    parser.add_argument('--list', action='store_true',
                       help='List all available datasets')
    
    args = parser.parse_args()
    
    # List datasets
    if args.list:
        logger.info("\n📋 AVAILABLE DATASETS:\n")
        for dataset_id, info in sorted(DATASETS.items(), key=lambda x: x[1]['priority']):
            logger.info(f"  {dataset_id}")
            logger.info(f"    Priority: {info['priority']}")
            logger.info(f"    Type: {info['type']}")
            logger.info(f"    Size: {info['size']}")
            logger.info(f"    Description: {info['description']}")
            if 'note' in info:
                logger.info(f"    Note: {info['note']}")
            logger.info("")
        return
    
    # Create info file
    if args.info:
        create_dataset_info()
        return
    
    # Create info file anyway
    create_dataset_info()
    
    # Download datasets
    extract = not args.no_extract
    
    logger.info("\n⚠️  IMPORTANT NOTES:")
    logger.info("1. COCO images are very large (20+ GB each)")
    logger.info("2. Sapiens weights require HuggingFace token")
    logger.info("3. MPII requires registration at MPI website")
    logger.info("4. Ensure you have enough disk space!\n")
    
    if args.priority == 1 and not args.types:
        logger.info("💡 RECOMMENDATION: For Priority 1, use --types annotations weights")
        logger.info("   This downloads only annotations & weights (~8 GB)")
        logger.info("   Skip large image datasets until needed\n")
    
    success_count, failed_datasets = download_datasets(
        output_dir=args.output,
        priority_filter=args.priority,
        types_filter=args.types,
        extract=extract
    )
    
    if success_count > 0:
        logger.info("\n📚 WHAT'S NEXT:")
        logger.info("1. Check data/ directory for downloaded files")
        logger.info("2. Verify extracted files are correct")
        logger.info("3. Update config files with data paths")
        logger.info("4. Run: python scripts/validate_annotations.py")


if __name__ == "__main__":
    main()
