"""
System Check: Verify Installation and Setup
Run this script to verify all dependencies and configurations are correct
"""

import sys
from pathlib import Path
import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    logger.info("✓ Python version OK")
    return True


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'transformers': 'HuggingFace Transformers',
        'albumentations': 'Albumentations',
        'opencv-cv2': 'OpenCV (cv2)',
        'PIL': 'Pillow (PIL)',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'pyyaml': 'PyYAML',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    
    for package, name in required_packages.items():
        try:
            if package == 'opencv-cv2':
                importlib.import_module('cv2')
            elif package == 'pyyaml':
                importlib.import_module('yaml')
            else:
                importlib.import_module(package)
            logger.info(f"✓ {name} installed")
        except ImportError:
            logger.error(f"❌ {name} NOT installed (package: {package})")
            all_ok = False
    
    return all_ok


def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA available")
            logger.info(f"  - CUDA version: {torch.version.cuda}")
            logger.info(f"  - Device count: {torch.cuda.device_count()}")
            logger.info(f"  - Device name: {torch.cuda.get_device_name(0)}")
            
            # Check VRAM
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"  - Total VRAM: {total_memory:.1f} GB")
            
            if total_memory < 8:
                logger.warning("⚠️  Low VRAM (<8GB). Consider reducing batch size.")
            
            return True
        else:
            logger.warning("⚠️  CUDA not available. Training will be slow on CPU.")
            return False
    
    except Exception as e:
        logger.error(f"❌ Error checking CUDA: {e}")
        return False


def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = [
        'configs',
        'stages',
        'models',
        'utils',
        'scripts'
    ]
    
    required_files = [
        'run_pipeline.py',
        'README.md',
        'requirements.txt',
        'configs/stage1_config.yaml',
        'configs/stage2_config.yaml',
        'configs/stage3_config.yaml',
        'configs/stage4_config.yaml',
        'configs/stage5_config.yaml',
        'stages/stage1_baseline.py',
        'stages/stage2_ssl.py',
        'stages/stage3_ensemble.py',
        'stages/stage4_vae.py',
        'stages/stage5_postprocess.py',
        'models/sapiens_model.py',
        'utils/coco_dataset.py',
        'utils/metrics.py',
        'utils/visualization.py',
        'scripts/validate_annotations.py'
    ]
    
    all_ok = True
    
    # Check directories
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            logger.info(f"✓ Directory '{dir_name}/' exists")
        else:
            logger.error(f"❌ Directory '{dir_name}/' NOT found")
            all_ok = False
    
    # Check files
    for file_name in required_files:
        file_path = Path(file_name)
        if file_path.exists() and file_path.is_file():
            logger.info(f"✓ File '{file_name}' exists")
        else:
            logger.error(f"❌ File '{file_name}' NOT found")
            all_ok = False
    
    return all_ok


def check_data_setup():
    """Check if data directories are set up"""
    data_dirs = ['data', 'data/raw', 'data/annotations']
    
    all_exist = True
    for dir_name in data_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"✓ Data directory '{dir_name}/' exists")
        else:
            logger.warning(f"⚠️  Data directory '{dir_name}/' NOT found (will be needed for training)")
            all_exist = False
    
    if not all_exist:
        logger.info("\n📝 To set up data directories:")
        logger.info("   mkdir data\\raw data\\annotations")
        logger.info("   # Then place your images and annotations there")
    
    return True  # Don't fail on missing data dirs


def check_configs():
    """Check if config files are valid YAML"""
    import yaml
    
    config_files = [
        'configs/stage1_config.yaml',
        'configs/stage2_config.yaml',
        'configs/stage3_config.yaml',
        'configs/stage4_config.yaml',
        'configs/stage5_config.yaml'
    ]
    
    all_ok = True
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                yaml.safe_load(f)
            logger.info(f"✓ Config '{config_file}' is valid YAML")
        except FileNotFoundError:
            logger.error(f"❌ Config '{config_file}' NOT found")
            all_ok = False
        except yaml.YAMLError as e:
            logger.error(f"❌ Config '{config_file}' has YAML syntax error: {e}")
            all_ok = False
    
    return all_ok


def check_imports():
    """Check if project modules can be imported"""
    modules_to_check = [
        ('models.sapiens_model', 'SapiensForPose'),
        ('utils.coco_dataset', 'COCOPoseDataset'),
        ('utils.metrics', 'COCOEvaluator'),
        ('utils.visualization', 'visualize_pose'),
    ]
    
    all_ok = True
    
    for module_name, class_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
            logger.info(f"✓ Can import {class_name} from {module_name}")
        except Exception as e:
            logger.error(f"❌ Cannot import {class_name} from {module_name}: {e}")
            all_ok = False
    
    return all_ok


def print_summary(checks):
    """Print summary of all checks"""
    logger.info("\n" + "="*60)
    logger.info("SYSTEM CHECK SUMMARY")
    logger.info("="*60)
    
    total = len(checks)
    passed = sum(checks.values())
    
    for check_name, passed_check in checks.items():
        status = "✓ PASS" if passed_check else "❌ FAIL"
        logger.info(f"{status} - {check_name}")
    
    logger.info("="*60)
    logger.info(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("\n🎉 ALL CHECKS PASSED! System ready to use.")
        logger.info("\n📚 Quick Start:")
        logger.info("   1. Prepare your data (see QUICKSTART.md)")
        logger.info("   2. Update config files with data paths")
        logger.info("   3. Run: python run_pipeline.py --all")
    else:
        logger.warning("\n⚠️  Some checks failed. Please fix the issues above.")
        logger.info("\n💡 Installation help:")
        logger.info("   pip install -r requirements.txt")
        logger.info("   # Or for CUDA support:")
        logger.info("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    logger.info("="*60 + "\n")


def main():
    """Run all checks"""
    logger.info("="*60)
    logger.info("POSE LLM IDENTIFIER - SYSTEM CHECK")
    logger.info("="*60 + "\n")
    
    checks = {
        "Python Version": check_python_version(),
        "Dependencies": check_dependencies(),
        "CUDA Support": check_cuda(),
        "Project Structure": check_project_structure(),
        "Config Files": check_configs(),
        "Module Imports": check_imports(),
        "Data Setup": check_data_setup()
    }
    
    print_summary(checks)


if __name__ == "__main__":
    main()
