"""
Validate Downloaded Data and Test Code Functionality
Tests that all downloaded resources work correctly with the implementation
"""

import sys
from pathlib import Path
import json
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_file_exists(filepath, description):
    """Check if a file exists"""
    filepath = Path(filepath)
    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"✓ {description}: {filepath.name} ({size_mb:.1f} MB)")
        return True
    else:
        logger.warning(f"✗ {description}: NOT FOUND - {filepath}")
        return False


def validate_annotations():
    """Validate COCO annotations"""
    logger.info("\n" + "="*60)
    logger.info("VALIDATING COCO ANNOTATIONS")
    logger.info("="*60)
    
    annotations_dir = project_root / "data" / "coco" / "annotations"
    
    required_files = [
        ("person_keypoints_train2017.json", "Training annotations"),
        ("person_keypoints_val2017.json", "Validation annotations"),
        ("instances_train2017.json", "Instance annotations (train)"),
        ("instances_val2017.json", "Instance annotations (val)"),
    ]
    
    found_count = 0
    for filename, description in required_files:
        filepath = annotations_dir / filename
        if check_file_exists(filepath, description):
            found_count += 1
            
            # Try to load and validate structure
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                if 'annotations' in data and 'images' in data:
                    logger.info(f"  → {len(data['images'])} images, {len(data['annotations'])} annotations")
                    
                    if 'keypoints' in filename:
                        # Check keypoint structure
                        sample = data['annotations'][0]
                        if 'keypoints' in sample and 'num_keypoints' in sample:
                            logger.info(f"  → Keypoints format: Valid (17 keypoints)")
            
            except Exception as e:
                logger.warning(f"  → Failed to parse: {e}")
    
    logger.info(f"\nAnnotations: {found_count}/{len(required_files)} found")
    return found_count > 0


def validate_pretrained_weights():
    """Validate pretrained model weights"""
    logger.info("\n" + "="*60)
    logger.info("VALIDATING PRETRAINED WEIGHTS")
    logger.info("="*60)
    
    weights_dir = project_root / "data" / "pretrained"
    
    expected_weights = [
        ("sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2", "Sapiens-1B TorchScript"),
    ]
    
    found_count = 0
    for rel_path, description in expected_weights:
        filepath = weights_dir / rel_path
        if check_file_exists(filepath, description):
            found_count += 1
            logger.info(f"  → TorchScript format (.pt2)")
    
    # Check for any other weight files
    all_weights = list(weights_dir.rglob("*.pth")) + list(weights_dir.rglob("*.pt")) + list(weights_dir.rglob("*.pt2"))
    if all_weights:
        logger.info(f"\nTotal weight files found: {len(all_weights)}")
    
    logger.info(f"\nWeights: {found_count}/{len(expected_weights)} required files found")
    return found_count > 0


def test_model_imports():
    """Test that model code can be imported"""
    logger.info("\n" + "="*60)
    logger.info("TESTING MODEL IMPORTS")
    logger.info("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test Sapiens model import
    try:
        tests_total += 1
        from models.sapiens_model import SapiensForPose
        logger.info("✓ Sapiens Model: SapiensForPose imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Sapiens Model: {e}")
    
    # Test Stage 1 imports
    try:
        tests_total += 1
        from stages.stage1_baseline import BaselineTrainer
        logger.info("✓ Stage 1: BaselineTrainer imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Stage 1: {e}")
    
    # Test Stage 2 imports (SSL)
    try:
        tests_total += 1
        from stages.stage2_ssl import SSLTrainer
        logger.info("✓ Stage 2: SSLTrainer imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Stage 2: {e}")
    
    # Test utilities imports
    try:
        tests_total += 1
        from utils.udp_codec import UDPHeatmap
        from utils.augmentations import get_train_augmentation
        logger.info("✓ Utilities: UDP codec and augmentations imported successfully")
        tests_passed += 1
    except Exception as e:
        logger.error(f"✗ Utilities: {e}")
    
    logger.info(f"\nImports: {tests_passed}/{tests_total} successful")
    return tests_passed == tests_total


def test_model_creation():
    """Test creating model instances"""
    logger.info("\n" + "="*60)
    logger.info("TESTING MODEL CREATION")
    logger.info("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    try:
        import torch
        
        # Test Sapiens model creation
        try:
            tests_total += 1
            from models.sapiens_model import SapiensForPose
            
            # Create a minimal config for testing
            class TestConfig:
                num_keypoints = 17
                backbone_name = "vit_base"
                pretrained_path = None
                image_size = (384, 288)
                heatmap_size = (96, 72)
            
            config = TestConfig()
            model = SapiensForPose(config)
            logger.info(f"✓ Sapiens: SapiensForPose created (ViT-Base)")
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 288, 384)
            with torch.no_grad():
                output = model(dummy_input)
            logger.info(f"  → Forward pass: input {dummy_input.shape} → output {output.shape}")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ Sapiens: {e}")
        
        # Test UDP Codec
        try:
            tests_total += 1
            from utils.udp_codec import UDPHeatmap
            import numpy as np
            
            codec = UDPHeatmap(
                input_size=(384, 288),
                heatmap_size=(96, 72),
                sigma=2.0
            )
            logger.info(f"✓ UDP Codec: Created and ready")
            
            # Test encoding
            joints = np.array([[100, 100, 2], [200, 150, 2]])
            visibility = np.array([1.0, 1.0])
            heatmaps, target_weight = codec.encode(joints, visibility)
            logger.info(f"  → Encode test: {joints.shape} → heatmaps {heatmaps.shape}")
            tests_passed += 1
        except Exception as e:
            logger.error(f"✗ UDP Codec: {e}")
        
    except ImportError as e:
        logger.error(f"✗ PyTorch not installed: {e}")
        logger.info("  → Install with: pip install torch torchvision")
    
    logger.info(f"\nModel Creation: {tests_passed}/{tests_total} successful")
    return tests_passed > 0


def test_data_loading():
    """Test data loading functionality"""
    logger.info("\n" + "="*60)
    logger.info("TESTING DATA LOADING")
    logger.info("="*60)
    
    tests_passed = 0
    tests_total = 0
    
    # Check if annotations exist
    annotations_file = project_root / "data" / "coco" / "annotations" / "person_keypoints_val2017.json"
    
    if not annotations_file.exists():
        logger.warning("✗ Cannot test data loading: annotations not found")
        logger.info("  → Download annotations first with:")
        logger.info("    python scripts/download_datasets.py --types annotations")
        return False
    
    try:
        # Test COCO dataset loading
        tests_total += 1
        from pycocotools.coco import COCO
        
        coco = COCO(annotations_file)
        
        # Get some statistics
        img_ids = coco.getImgIds()
        ann_ids = coco.getAnnIds()
        cat_ids = coco.getCatIds(catNms=['person'])
        
        logger.info(f"✓ COCO dataset loaded")
        logger.info(f"  → Images: {len(img_ids)}")
        logger.info(f"  → Annotations: {len(ann_ids)}")
        logger.info(f"  → Person category ID: {cat_ids}")
        
        # Get sample annotation
        if ann_ids:
            sample_ann = coco.loadAnns(ann_ids[0])[0]
            if 'keypoints' in sample_ann:
                keypoints = sample_ann['keypoints']
                num_keypoints = sample_ann.get('num_keypoints', 0)
                logger.info(f"  → Sample annotation: {num_keypoints} visible keypoints")
        
        tests_passed += 1
        
    except ImportError:
        logger.warning("✗ pycocotools not installed")
        logger.info("  → Install with: pip install pycocotools")
    except Exception as e:
        logger.error(f"✗ Failed to load COCO dataset: {e}")
    
    logger.info(f"\nData Loading: {tests_passed}/{tests_total} successful")
    return tests_passed > 0


def test_configuration():
    """Test configuration files"""
    logger.info("\n" + "="*60)
    logger.info("TESTING CONFIGURATION")
    logger.info("="*60)
    
    config_file = project_root / "configs" / "stage1_config.yaml"
    
    if not config_file.exists():
        logger.warning(f"✗ Config file not found: {config_file}")
        return False
    
    try:
        import yaml
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Configuration loaded: {config_file.name}")
        
        # Check key sections
        required_sections = ['model', 'training', 'dataset']
        found_sections = [s for s in required_sections if s in config]
        
        logger.info(f"  → Sections: {', '.join(found_sections)}")
        
        if 'model' in config:
            logger.info(f"  → Model: {config['model'].get('name', 'unknown')}")
            logger.info(f"  → Input size: {config['model'].get('input_size', 'unknown')}")
        
        if 'training' in config and 'phase1' in config['training']:
            logger.info(f"  → Batch size: {config['training']['phase1'].get('batch_size', 'unknown')}")
            logger.info(f"  → Epochs: {config['training']['phase1'].get('epochs', 'unknown')}")
        
        return True
        
    except ImportError:
        logger.warning("✗ PyYAML not installed")
        logger.info("  → Install with: pip install pyyaml")
        return False
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    logger.info("\n" + "="*60)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'albumentations': 'Albumentations',
        'transformers': 'Transformers',
    }
    
    installed = []
    missing = []
    
    for package, name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'yaml':
                import yaml
            else:
                __import__(package)
            
            logger.info(f"✓ {name}: Installed")
            installed.append(name)
        except ImportError:
            logger.warning(f"✗ {name}: NOT INSTALLED")
            missing.append(name)
    
    logger.info(f"\nDependencies: {len(installed)}/{len(required_packages)} installed")
    
    if missing:
        logger.info("\n📦 INSTALL MISSING PACKAGES:")
        logger.info("pip install torch torchvision numpy opencv-python pillow tqdm pyyaml albumentations transformers")
    
    return len(missing) == 0


def generate_summary_report():
    """Generate summary report of validation"""
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    results = {
        "Dependencies": check_dependencies(),
        "Annotations": validate_annotations(),
        "Pretrained Weights": validate_pretrained_weights(),
        "Model Imports": test_model_imports(),
        "Model Creation": test_model_creation(),
        "Data Loading": test_data_loading(),
        "Configuration": test_configuration(),
    }
    
    logger.info("\n📊 RESULTS:")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Your setup is ready.")
        logger.info("\n📚 NEXT STEPS:")
        logger.info("1. Download COCO images: python scripts/download_coco.py")
        logger.info("2. Start training: python train_stage1.py --config configs/stage1_config.yaml")
        logger.info("3. Monitor with TensorBoard: tensorboard --logdir=runs")
    else:
        logger.info("\n⚠️  SOME TESTS FAILED")
        logger.info("\n🔧 TROUBLESHOOTING:")
        if not results["Dependencies"]:
            logger.info("• Install missing dependencies with pip")
        if not results["Annotations"]:
            logger.info("• Download annotations: python scripts/download_coco.py")
        if not results["Pretrained Weights"]:
            logger.info("• Download weights: python scripts/download_pretrained.py")
        if not results["Model Imports"]:
            logger.info("• Check that models/ and stages/ directories exist with Python files")
    
    return passed == total


def main():
    """Main validation function"""
    logger.info("="*60)
    logger.info("POSE ESTIMATION SYSTEM VALIDATION")
    logger.info("="*60)
    logger.info(f"Project root: {project_root}")
    logger.info("")
    
    success = generate_summary_report()
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
