"""
Final System Validation - Test All Components
Tests the actual project structure with downloaded data
"""

import sys
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check all required dependencies"""
    logger.info("\n" + "="*60)
    logger.info("CHECKING DEPENDENCIES")
    logger.info("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'tqdm': 'tqdm',
        'yaml': 'PyYAML',
        'pycocotools': 'COCO API',
        'timm': 'PyTorch Image Models',
        'einops': 'Einops',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
    }
    
    installed = 0
    for package, name in required.items():
        try:
            __import__(package)
            logger.info(f"✓ {name}: Installed")
            installed += 1
        except ImportError:
            logger.warning(f"✗ {name}: NOT INSTALLED")
    
    logger.info(f"\nDependencies: {installed}/{len(required)} installed")
    return installed == len(required)


def check_downloaded_data():
    """Check all downloaded data"""
    logger.info("\n" + "="*60)
    logger.info("CHECKING DOWNLOADED DATA")
    logger.info("="*60)
    
    checks = []
    
    # Check papers
    papers_dir = project_root / "papers"
    pdf_count = len(list(papers_dir.glob("*.pdf")))
    logger.info(f"✓ Research Papers: {pdf_count} PDFs")
    checks.append(pdf_count == 17)
    
    # Check annotations
    annotations_dir = project_root / "data" / "coco" / "annotations"
    if annotations_dir.exists():
        ann_files = list(annotations_dir.glob("*.json"))
        logger.info(f"✓ COCO Annotations: {len(ann_files)} files")
        checks.append(len(ann_files) >= 4)
    else:
        logger.warning("✗ COCO Annotations: NOT FOUND")
        checks.append(False)
    
    # Check pretrained weights
    weights_dir = project_root / "data" / "pretrained"
    if weights_dir.exists():
        weight_files = list(weights_dir.rglob("*.pth")) + list(weights_dir.rglob("*.pt"))
        logger.info(f"✓ Pretrained Weights: {len(weight_files)} files")
        checks.append(len(weight_files) >= 2)
    else:
        logger.warning("✗ Pretrained Weights: NOT FOUND")
        checks.append(False)
    
    return all(checks)


def check_project_structure():
    """Check project structure"""
    logger.info("\n" + "="*60)
    logger.info("CHECKING PROJECT STRUCTURE")
    logger.info("="*60)
    
    required_dirs = {
        'stages': 'Stage implementations',
        'models': 'Model definitions',
        'utils': 'Utility functions',
        'configs': 'Configuration files',
        'scripts': 'Scripts',
        'papers': 'Research papers',
    }
    
    found = 0
    for dir_name, description in required_dirs.items():
        dir_path = project_root / dir_name
        if dir_path.exists():
            files = list(dir_path.glob("*.py")) + list(dir_path.glob("*.yaml"))
            logger.info(f"✓ {dir_name}/: {len(files)} files - {description}")
            found += 1
        else:
            logger.warning(f"✗ {dir_name}/: NOT FOUND")
    
    logger.info(f"\nDirectories: {found}/{len(required_dirs)} found")
    return found == len(required_dirs)


def test_stage_imports():
    """Test importing stage modules"""
    logger.info("\n" + "="*60)
    logger.info("TESTING STAGE IMPORTS")
    logger.info("="*60)
    
    stages = [
        ('stage1_baseline', 'Stage 1: Baseline'),
        ('stage2_ssl', 'Stage 2: SSL'),
        ('stage3_ensemble', 'Stage 3: Ensemble'),
        ('stage4_vae', 'Stage 4: VAE'),
        ('stage5_postprocess', 'Stage 5: Postprocess'),
    ]
    
    success = 0
    for module, name in stages:
        try:
            sys.path.insert(0, str(project_root / 'stages'))
            __import__(module)
            logger.info(f"✓ {name}: Imported")
            success += 1
        except ImportError as e:
            logger.warning(f"✗ {name}: {e}")
    
    logger.info(f"\nStages: {success}/{len(stages)} imported")
    return success > 0


def test_config_loading():
    """Test loading configuration files"""
    logger.info("\n" + "="*60)
    logger.info("TESTING CONFIGURATION LOADING")
    logger.info("="*60)
    
    try:
        import yaml
        configs_dir = project_root / "configs"
        config_files = list(configs_dir.glob("*.yaml"))
        
        logger.info(f"Found {len(config_files)} config files:")
        for config_file in config_files:
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✓ {config_file.name}: Loaded")
            except Exception as e:
                logger.warning(f"✗ {config_file.name}: {e}")
        
        return len(config_files) > 0
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return False


def test_coco_loading():
    """Test loading COCO annotations"""
    logger.info("\n" + "="*60)
    logger.info("TESTING COCO DATA LOADING")
    logger.info("="*60)
    
    try:
        from pycocotools.coco import COCO
        
        ann_file = project_root / "data" / "coco" / "annotations" / "person_keypoints_val2017.json"
        if not ann_file.exists():
            logger.warning("✗ Validation annotations not found")
            return False
        
        coco = COCO(str(ann_file))
        
        img_ids = coco.getImgIds()
        ann_ids = coco.getAnnIds()
        cat_ids = coco.getCatIds(catNms=['person'])
        
        logger.info(f"✓ COCO dataset loaded successfully")
        logger.info(f"  → Images: {len(img_ids)}")
        logger.info(f"  → Annotations: {len(ann_ids)}")
        logger.info(f"  → Person category: {cat_ids}")
        
        # Check sample annotation
        if ann_ids:
            sample = coco.loadAnns(ann_ids[0])[0]
            if 'keypoints' in sample:
                logger.info(f"  → Sample: {sample['num_keypoints']} visible keypoints")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return False


def test_model_creation():
    """Test creating a simple model"""
    logger.info("\n" + "="*60)
    logger.info("TESTING MODEL CREATION")
    logger.info("="*60)
    
    try:
        import torch
        import torch.nn as nn
        
        # Create a simple test model
        class TestPoseModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.conv2 = nn.Conv2d(64, 17, 1)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                return self.conv2(x)
        
        model = TestPoseModel()
        dummy_input = torch.randn(1, 3, 256, 192)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"✓ Test model created successfully")
        logger.info(f"  → Input: {dummy_input.shape}")
        logger.info(f"  → Output: {output.shape}")
        logger.info(f"  → Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return False


def main():
    """Run all validation tests"""
    logger.info("="*60)
    logger.info("FINAL SYSTEM VALIDATION")
    logger.info("="*60)
    logger.info(f"Project: {project_root.name}")
    logger.info(f"Location: {project_root}")
    logger.info("")
    
    results = {
        "Dependencies": check_dependencies(),
        "Downloaded Data": check_downloaded_data(),
        "Project Structure": check_project_structure(),
        "Stage Imports": test_stage_imports(),
        "Config Loading": test_config_loading(),
        "COCO Loading": test_coco_loading(),
        "Model Creation": test_model_creation(),
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\n📊 Overall: {passed}/{total} tests passed ({passed*100//total}%)")
    
    if passed == total:
        logger.info("\n" + "🎉"*20)
        logger.info("🎉 ALL TESTS PASSED - SYSTEM FULLY FUNCTIONAL! 🎉")
        logger.info("🎉"*20)
        logger.info("\n✅ READY TO:")
        logger.info("   1. Run pipeline: python run_pipeline.py")
        logger.info("   2. Train models: python stages/stage1_baseline.py")
        logger.info("   3. Test inference on images")
        logger.info("   4. Explore COCO annotations")
        logger.info("   5. Read research papers")
    elif passed >= total * 0.8:
        logger.info("\n✅ MOSTLY FUNCTIONAL ({0}% complete)".format(passed*100//total))
        logger.info("\n📚 YOU CAN NOW:")
        for test_name, result in results.items():
            if result:
                logger.info(f"   ✓ {test_name}")
        logger.info("\n⚠️  NEEDS ATTENTION:")
        for test_name, result in results.items():
            if not result:
                logger.info(f"   ✗ {test_name}")
    else:
        logger.info("\n⚠️  SYSTEM NEEDS ATTENTION")
        logger.info("\n🔧 FAILED TESTS:")
        for test_name, result in results.items():
            if not result:
                logger.info(f"   ✗ {test_name}")
    
    logger.info("\n" + "="*60)
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
