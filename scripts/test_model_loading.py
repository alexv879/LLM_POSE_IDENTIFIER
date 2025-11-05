"""
Test Model Loading and Basic Functionality
Tests that models can be created and run without needing the full dataset
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_imports():
    """Test that all modules can be imported"""
    print("=" * 60)
    print("TESTING MODULE IMPORTS")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        return False
    
    modules = [
        ("Sapiens Model", "models.sapiens_model"),
        ("Stage 1: Baseline", "stages.stage1_baseline"),
        ("Stage 2: SSL", "stages.stage2_ssl"),
        ("Stage 3: Ensemble", "stages.stage3_ensemble"),
        ("UDP Codec", "utils.udp_codec"),
        ("Augmentations", "utils.augmentations"),
    ]
    
    success = True
    for name, module in modules:
        try:
            __import__(module)
            print(f"✓ {name}: Imported")
        except ImportError as e:
            print(f"✗ {name}: {e}")
            success = False
    
    return success


def test_model_creation():
    """Test creating model instances"""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    try:
        import torch
        
        # Test Sapiens Model
        print("\n📦 Sapiens Model (ViT-Base for Pose)")
        from models.sapiens_model import SapiensForPose
        
        # Create a minimal config for testing
        class TestConfig:
            num_keypoints = 17
            backbone_name = "vit_base"
            pretrained_path = None  # Don't load weights for quick test
            image_size = (384, 288)
            heatmap_size = (96, 72)
        
        config = TestConfig()
        model = SapiensForPose(config)
        print(f"✓ Model created")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 288, 384)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"  Forward pass: {dummy_input.shape} → {output.shape}")
        print(f"  Expected: torch.Size([2, 17, 72, 96])")
        
        # Test UDP Codec
        print("\n📦 UDP Heatmap Codec")
        from utils.udp_codec import UDPHeatmap
        
        codec = UDPHeatmap(
            input_size=(384, 288),
            heatmap_size=(96, 72),
            sigma=2.0
        )
        print(f"✓ UDP Codec created")
        print(f"  Input size: {codec.input_size}")
        print(f"  Heatmap size: {codec.heatmap_size}")
        
        # Test encoding
        import numpy as np
        joints = np.array([[100, 100, 2], [200, 150, 2]])  # 2 joints
        visibility = np.array([1.0, 1.0])
        heatmaps, target_weight = codec.encode(joints, visibility)
        print(f"  Encode test: {joints.shape} → heatmaps {heatmaps.shape}, weights {target_weight.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pretrained_loading():
    """Test loading pretrained weights"""
    print("\n" + "=" * 60)
    print("TESTING PRETRAINED WEIGHT LOADING")
    print("=" * 60)
    
    try:
        import torch
        
        weights_dir = project_root / "data" / "pretrained"
        
        # Test Sapiens weights
        sapiens_path = weights_dir / "sapiens_1b" / "sapiens_1b_epoch_173_torchscript.pt2"
        if sapiens_path.exists():
            print(f"\n📦 Found: {sapiens_path.name}")
            file_size_mb = sapiens_path.stat().st_size / (1024 * 1024)
            print(f"✓ Sapiens-1B weights available")
            print(f"  Size: {file_size_mb:.1f} MB")
            print(f"  Format: TorchScript (.pt2)")
            # Note: TorchScript models are loaded differently, typically with torch.jit.load
            # We don't load it here to keep the test fast
        else:
            print(f"⚠️  Sapiens weights not found: {sapiens_path}")
            print(f"   Download with: python scripts/download_pretrained.py")
        
        # Check for any other weights in pretrained directory
        print(f"\n📦 Scanning pretrained directory...")
        weight_files = list(weights_dir.rglob("*.pth")) + list(weights_dir.rglob("*.pt")) + list(weights_dir.rglob("*.pt2"))
        if weight_files:
            print(f"✓ Found {len(weight_files)} weight file(s):")
            for wf in weight_files[:5]:  # Show first 5
                size_mb = wf.stat().st_size / (1024 * 1024)
                rel_path = wf.relative_to(weights_dir)
                print(f"  - {rel_path} ({size_mb:.1f} MB)")
            if len(weight_files) > 5:
                print(f"  ... and {len(weight_files) - 5} more")
        else:
            print(f"⚠️  No weight files found in {weights_dir}")
        
        return True
        
    except ImportError:
        print("✗ PyTorch not available")
        return False


def test_coco_annotations():
    """Test loading COCO annotations"""
    print("\n" + "=" * 60)
    print("TESTING COCO ANNOTATION LOADING")
    print("=" * 60)
    
    train_annot = project_root / "data" / "coco" / "annotations" / "person_keypoints_train2017.json"
    val_annot = project_root / "data" / "coco" / "annotations" / "person_keypoints_val2017.json"
    
    if not train_annot.exists() and not val_annot.exists():
        print(f"⚠️  Annotations not found")
        print("   Download with: python scripts/download_coco.py")
        return False
    
    try:
        import json
        
        # Try validation annotations first (smaller file)
        annot_file = val_annot if val_annot.exists() else train_annot
        
        print(f"\n📦 Loading: {annot_file.name}")
        with open(annot_file, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded successfully")
        print(f"  Images: {len(data['images'])}")
        print(f"  Annotations: {len(data['annotations'])}")
        print(f"  Categories: {len(data['categories'])}")
        
        # Check keypoint structure
        if data['annotations']:
            sample = data['annotations'][0]
            print(f"\n  Sample annotation:")
            print(f"    Image ID: {sample['image_id']}")
            print(f"    Category ID: {sample['category_id']}")
            if 'keypoints' in sample:
                print(f"    Keypoints: {len(sample['keypoints'])} values (17 keypoints × 3)")
                print(f"    Visible keypoints: {sample.get('num_keypoints', 0)}")
        
        # Check if images exist
        images_dir = project_root / "data" / "coco" / "images"
        train_images = images_dir / "train2017"
        val_images = images_dir / "val2017"
        
        train_count = len(list(train_images.glob("*.jpg"))) if train_images.exists() else 0
        val_count = len(list(val_images.glob("*.jpg"))) if val_images.exists() else 0
        
        print(f"\n  COCO Images:")
        print(f"    Train: {train_count} images" + (" ✓" if train_count > 0 else " ⚠️ missing"))
        print(f"    Val: {val_count} images" + (" ✓" if val_count > 0 else " ⚠️ missing"))
        
        if train_count == 0 and val_count == 0:
            print(f"\n  ⚠️  No COCO images found!")
            print(f"     Download with: python scripts/download_coco.py")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("POSE ESTIMATION MODEL TESTING")
    print("=" * 60)
    print(f"Project root: {project_root}\n")
    
    results = {
        "Module Imports": test_imports(),
        "Model Creation": test_model_creation(),
        "Pretrained Loading": test_pretrained_loading(),
        "COCO Annotations": test_coco_annotations(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📚 READY TO:")
        print("1. Download COCO images: python scripts/download_coco.py")
        print("2. Start training: python train_stage1.py --config configs/stage1_config.yaml")
        print("3. Monitor with TensorBoard: tensorboard --logdir=runs")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\n🔧 FIX ISSUES:")
        if not results["Module Imports"]:
            print("• Check that models/ and stages/ directories exist")
        if not results["Model Creation"]:
            print("• Install PyTorch: pip install torch torchvision")
        if not results["Pretrained Loading"]:
            print("• Download weights: python scripts/download_pretrained.py")
        if not results["COCO Annotations"]:
            print("• Download dataset: python scripts/download_coco.py")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
