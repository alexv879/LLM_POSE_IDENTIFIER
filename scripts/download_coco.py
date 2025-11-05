"""
Download COCO 2017 dataset for pose estimation training
Downloads train2017, val2017 images and annotations
"""

import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# COCO 2017 URLs
COCO_URLS = {
    'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
    'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
    'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
}

def download_file(url, destination, desc=None):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc or url) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return destination

def extract_zip(zip_path, extract_to):
    """Extract zip file with progress bar"""
    print(f"Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {extract_to}")

def main():
    # Base directory
    base_dir = Path(__file__).parent / 'data' / 'coco'
    images_dir = base_dir / 'images'
    annotations_dir = base_dir / 'annotations'
    downloads_dir = Path(__file__).parent / 'data' / 'downloads'
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COCO 2017 Dataset Download")
    print("=" * 60)
    print()
    
    # Check what's already downloaded
    train_exists = (images_dir / 'train2017').exists()
    val_exists = (images_dir / 'val2017').exists()
    annot_exists = (annotations_dir / 'person_keypoints_train2017.json').exists()
    
    print("Current status:")
    print(f"  Train images: {'✓' if train_exists else '✗'}")
    print(f"  Val images: {'✓' if val_exists else '✗'}")
    print(f"  Annotations: {'✓' if annot_exists else '✗'}")
    print()
    
    if train_exists and val_exists and annot_exists:
        print("✅ All COCO data already downloaded!")
        return
    
    print("⚠️  WARNING: This will download ~19GB of data")
    print("Sizes:")
    print("  - Train images: ~18GB")
    print("  - Val images: ~1GB")
    print("  - Annotations: ~252MB")
    print()
    
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    print()
    
    # Download and extract train images
    if not train_exists:
        print("Downloading train2017 images (~18GB)...")
        train_zip = downloads_dir / 'train2017.zip'
        download_file(COCO_URLS['train_images'], train_zip, 'Train images')
        extract_zip(train_zip, images_dir)
        train_zip.unlink()  # Delete zip after extraction
        print()
    else:
        print("✓ Train images already exist")
        print()
    
    # Download and extract val images
    if not val_exists:
        print("Downloading val2017 images (~1GB)...")
        val_zip = downloads_dir / 'val2017.zip'
        download_file(COCO_URLS['val_images'], val_zip, 'Val images')
        extract_zip(val_zip, images_dir)
        val_zip.unlink()  # Delete zip after extraction
        print()
    else:
        print("✓ Val images already exist")
        print()
    
    # Download and extract annotations
    if not annot_exists:
        print("Downloading annotations (~252MB)...")
        annot_zip = downloads_dir / 'annotations_trainval2017.zip'
        download_file(COCO_URLS['annotations'], annot_zip, 'Annotations')
        extract_zip(annot_zip, base_dir)
        annot_zip.unlink()  # Delete zip after extraction
        print()
    else:
        print("✓ Annotations already exist")
        print()
    
    print("=" * 60)
    print("✅ COCO 2017 dataset download complete!")
    print("=" * 60)
    print()
    print("Dataset location:")
    print(f"  Images: {images_dir}")
    print(f"  Annotations: {annotations_dir}")
    print()
    
    # Verify
    train_count = len(list((images_dir / 'train2017').glob('*.jpg'))) if (images_dir / 'train2017').exists() else 0
    val_count = len(list((images_dir / 'val2017').glob('*.jpg'))) if (images_dir / 'val2017').exists() else 0
    
    print("Dataset statistics:")
    print(f"  Train images: {train_count}")
    print(f"  Val images: {val_count}")
    print()

if __name__ == "__main__":
    main()
