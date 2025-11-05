"""
Download Sapiens pretrained weights from HuggingFace
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download

def main():
    print("=" * 60)
    print("Sapiens Pretrained Weights Download")
    print("=" * 60)
    print()
    
    # Model configuration
    model_id = "facebook/sapiens-pretrain-1b-torchscript"
    local_dir = Path(__file__).parent.parent / "data" / "pretrained" / "sapiens_1b"
    
    print(f"Model: {model_id}")
    print(f"Destination: {local_dir}")
    print()
    
    # Create directory
    local_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    model_files = list(local_dir.glob("*.pt")) + list(local_dir.glob("*.pth"))
    if model_files:
        print(f"✓ Pretrained weights already exist ({len(model_files)} files)")
        print()
        for f in model_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print()
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Using existing weights.")
            return
    
    print("Downloading pretrained weights...")
    print("This may take several minutes (~2-3GB)...")
    print()
    
    try:
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )
        
        print()
        print("=" * 60)
        print("✅ Download complete!")
        print("=" * 60)
        print()
        print(f"Weights saved to: {local_dir}")
        print()
        
        # List downloaded files
        downloaded_files = list(local_dir.glob("*"))
        print(f"Downloaded {len(downloaded_files)} files:")
        for f in sorted(downloaded_files):
            if f.is_file():
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  - {f.name} ({size_mb:.1f} MB)")
        print()
        
    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Download failed!")
        print("=" * 60)
        print()
        print(f"Error: {e}")
        print()
        print("Troubleshooting:")
        print("1. Check your internet connection")
        print("2. Verify HuggingFace Hub is accessible")
        print("3. Try manual download from:")
        print(f"   https://huggingface.co/{model_id}")
        print()

if __name__ == "__main__":
    main()
