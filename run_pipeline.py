"""
Master Pipeline: Run All 5 Stages Sequentially
Complete pose estimation pipeline from baseline to LLM integration
"""

import argparse
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stages.stage1_baseline import Stage1Trainer
from stages.stage2_ssl import Stage2SSLTrainer
from stages.stage3_ensemble import Stage3EnsembleTrainer
from stages.stage4_vae import Stage4VAETrainer
from stages.stage5_postprocess import Stage5Pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseLLMPipeline:
    """Complete 5-stage pose estimation pipeline"""
    
    def __init__(self, configs_dir: str = "configs"):
        self.configs_dir = Path(configs_dir)
        
        # Config paths
        self.stage_configs = {
            1: self.configs_dir / "stage1_config.yaml",
            2: self.configs_dir / "stage2_config.yaml",
            3: self.configs_dir / "stage3_config.yaml",
            4: self.configs_dir / "stage4_config.yaml",
            5: self.configs_dir / "stage5_config.yaml"
        }
        
        # Verify configs exist
        for stage, config_path in self.stage_configs.items():
            if not config_path.exists():
                raise FileNotFoundError(f"Config not found: {config_path}")
        
        logger.info("Master pipeline initialized")
        logger.info(f"Configs directory: {self.configs_dir}")
    
    def run_stage1(self) -> bool:
        """
        Run Stage 1: Baseline Fine-Tuning with Sapiens-2B
        Expected: 82-85% AP
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 1: BASELINE FINE-TUNING (Sapiens-2B)")
        logger.info("="*70)
        logger.info("Goal: Establish strong baseline with foundation model")
        logger.info("Expected performance: 82-85% AP")
        logger.info("="*70 + "\n")
        
        try:
            trainer = Stage1Trainer(str(self.stage_configs[1]))
            trainer.train()
            logger.info("✓ Stage 1 completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Stage 1 failed: {e}")
            return False
    
    def run_stage2(self) -> bool:
        """
        Run Stage 2: Semi-Supervised Learning with Multi-Path Augmentation
        Expected: 89-93% AP (+6-8% over Stage 1)
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 2: SEMI-SUPERVISED LEARNING")
        logger.info("="*70)
        logger.info("Goal: Leverage unlabeled data with multi-path consistency")
        logger.info("Expected performance: 89-93% AP (+6-8% improvement)")
        logger.info("="*70 + "\n")
        
        try:
            trainer = Stage2SSLTrainer(str(self.stage_configs[2]))
            trainer.train()
            logger.info("✓ Stage 2 completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Stage 2 failed: {e}")
            return False
    
    def run_stage3(self) -> bool:
        """
        Run Stage 3: Ensemble Fusion
        Expected: 92-95% AP (+3-2% over Stage 2)
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 3: ENSEMBLE FUSION")
        logger.info("="*70)
        logger.info("Goal: Combine Sapiens-2B, DWPose, ViTPose predictions")
        logger.info("Expected performance: 92-95% AP (+3-2% improvement)")
        logger.info("="*70 + "\n")
        
        try:
            trainer = Stage3EnsembleTrainer(str(self.stage_configs[3]))
            trainer.train()
            logger.info("✓ Stage 3 completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Stage 3 failed: {e}")
            return False
    
    def run_stage4(self) -> bool:
        """
        Run Stage 4: VAE-Based Anatomical Plausibility
        Expected: 94-97% AP (+2% over Stage 3)
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 4: VAE ANATOMICAL REFINEMENT")
        logger.info("="*70)
        logger.info("Goal: Ensure anatomically plausible poses")
        logger.info("Expected performance: 94-97% AP (+2% improvement)")
        logger.info("="*70 + "\n")
        
        try:
            trainer = Stage4VAETrainer(str(self.stage_configs[4]))
            trainer.train()
            logger.info("✓ Stage 4 completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Stage 4 failed: {e}")
            return False
    
    def run_stage5(self) -> bool:
        """
        Run Stage 5: Post-Processing and LLM Integration
        Expected: 95-98% AP (+1% over Stage 4)
        """
        logger.info("\n" + "="*70)
        logger.info("STAGE 5: POST-PROCESSING & LLM INTEGRATION")
        logger.info("="*70)
        logger.info("Goal: Final refinement and interpretability")
        logger.info("Expected performance: 95-98% AP (+1% improvement)")
        logger.info("="*70 + "\n")
        
        try:
            pipeline = Stage5Pipeline(str(self.stage_configs[5]))
            pipeline.run()
            logger.info("✓ Stage 5 completed successfully")
            return True
        except Exception as e:
            logger.error(f"✗ Stage 5 failed: {e}")
            return False
    
    def run_all_stages(self, start_stage: int = 1, end_stage: int = 5):
        """
        Run all stages sequentially.
        
        Args:
            start_stage: Stage to start from (1-5)
            end_stage: Stage to end at (1-5)
        """
        logger.info("\n" + "🚀"*35)
        logger.info("POSE LLM IDENTIFIER - MASTER PIPELINE")
        logger.info("🚀"*35)
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Running stages {start_stage} to {end_stage}")
        logger.info("🚀"*35 + "\n")
        
        # Map stages to functions
        stage_functions = {
            1: self.run_stage1,
            2: self.run_stage2,
            3: self.run_stage3,
            4: self.run_stage4,
            5: self.run_stage5
        }
        
        # Run requested stages
        results = {}
        for stage in range(start_stage, end_stage + 1):
            success = stage_functions[stage]()
            results[stage] = success
            
            if not success:
                logger.error(f"Pipeline halted at Stage {stage} due to failure")
                break
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*70)
        
        for stage, success in results.items():
            status = "✓ SUCCESS" if success else "✗ FAILED"
            logger.info(f"Stage {stage}: {status}")
        
        logger.info("="*70)
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if all(results.values()):
            logger.info("\n🎉 ALL STAGES COMPLETED SUCCESSFULLY! 🎉")
            logger.info("Final system ready for deployment.")
            logger.info("Expected performance: 95-98% AP on COCO test set")
        else:
            logger.warning("\n⚠️  Some stages failed. Please check logs above.")
        
        logger.info("\n" + "="*70 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Master pipeline for Pose LLM Identifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python run_pipeline.py --all
  
  # Run specific stage
  python run_pipeline.py --stage 1
  
  # Run stages 2-4
  python run_pipeline.py --start 2 --end 4
  
  # Use custom config directory
  python run_pipeline.py --all --configs custom_configs/
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all 5 stages sequentially'
    )
    
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run a specific stage (1-5)'
    )
    
    parser.add_argument(
        '--start',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help='Starting stage (default: 1)'
    )
    
    parser.add_argument(
        '--end',
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help='Ending stage (default: 5)'
    )
    
    parser.add_argument(
        '--configs',
        type=str,
        default='configs',
        help='Path to configs directory (default: configs)'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PoseLLMPipeline(configs_dir=args.configs)
    
    # Run requested stages
    if args.all:
        pipeline.run_all_stages(start_stage=1, end_stage=5)
    elif args.stage:
        pipeline.run_all_stages(start_stage=args.stage, end_stage=args.stage)
    else:
        pipeline.run_all_stages(start_stage=args.start, end_stage=args.end)


if __name__ == "__main__":
    main()
