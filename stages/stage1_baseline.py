"""
Stage 1: Baseline Fine-Tuning with Sapiens-2B
Implementation based on Meta Sapiens paper (ECCV 2024)

Two-phase training:
1. Phase 1: Train decoder only (2-3 epochs, lr=1e-3)
2. Phase 2: Fine-tune full model (15-20 epochs, lr=2e-4, cosine annealing)

Expected Performance: 82-85% AP on COCO format validation set
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.coco_dataset import COCOPoseDataset, get_train_transforms, get_val_transforms, collate_fn
from models.sapiens_model import SapiensForPose
from utils.metrics import compute_metrics, COCOEvaluator
from utils.visualization import visualize_predictions

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stage1Trainer:
    """
    Stage 1 Trainer: Baseline Sapiens-2B Fine-tuning
    
    Implements two-phase training protocol from Sapiens paper.
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("="*60)
        logger.info("Stage 1: Baseline Fine-Tuning")
        logger.info("="*60)
        
        # Set device and seed
        self.device = torch.device(
            self.config['hardware']['device'] 
            if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
        
        # Set random seed for reproducibility
        self._set_seed(self.config['hardware']['seed'])
        
        # Create output directories
        self.checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = self._build_model()
        
        # Initialize datasets and dataloaders
        logger.info("Loading datasets...")
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Initialize evaluator
        self.evaluator = COCOEvaluator(
            oks_sigmas=self.config['evaluation']['oks_sigmas']
        )
        
        # Training state
        self.global_step = 0
        self.best_val_ap = 0.0
        
        logger.info("Initialization complete!")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
        if self.config['hardware']['deterministic']:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def _build_model(self) -> nn.Module:
        """Build and initialize Sapiens-2B model."""
        model_config = self.config['model']
        
        model = SapiensForPose(
            backbone_name=model_config['backbone'],
            num_keypoints=model_config['num_keypoints'],
            pretrained=model_config['pretrained'],
            pretrained_path=model_config.get('pretrained_path'),
            input_size=tuple(model_config['input_size']),
            heatmap_size=tuple(model_config['heatmap_size'])
        )
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {model_config['name']}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _build_dataloaders(self) -> tuple:
        """Build training and validation dataloaders."""
        dataset_config = self.config['dataset']
        
        # Training dataset
        train_dataset = COCOPoseDataset(
            image_dir=dataset_config['image_dir'],
            annotations_file=dataset_config['train_annotations'],
            split='train',
            train_ratio=dataset_config['train_ratio'],
            transform=get_train_transforms(
                tuple(self.config['model']['input_size'])
            ),
            image_size=tuple(self.config['model']['input_size']),
            heatmap_size=tuple(self.config['model']['heatmap_size']),
            sigma=self.config['model']['sigma']
        )
        
        # Validation dataset
        val_dataset = COCOPoseDataset(
            image_dir=dataset_config['image_dir'],
            annotations_file=dataset_config['train_annotations'],
            split='val',
            train_ratio=dataset_config['train_ratio'],
            transform=get_val_transforms(
                tuple(self.config['model']['input_size'])
            ),
            image_size=tuple(self.config['model']['input_size']),
            heatmap_size=tuple(self.config['model']['heatmap_size']),
            sigma=self.config['model']['sigma']
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['phase2']['batch_size'],
            shuffle=True,
            num_workers=dataset_config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=dataset_config.get('prefetch_factor', 2)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['phase2']['batch_size'],
            shuffle=False,
            num_workers=dataset_config['num_workers'],
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        return train_loader, val_loader
    
    def _build_optimizer_and_scheduler(self, phase_config: dict):
        """Build optimizer and learning rate scheduler."""
        # Optimizer
        if phase_config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=phase_config['learning_rate'],
                weight_decay=phase_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {phase_config['optimizer']}")
        
        # Scheduler
        scheduler_config = self.config['training']['scheduler']
        total_epochs = phase_config['epochs']
        
        if scheduler_config['type'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs - scheduler_config['warmup_epochs'],
                eta_min=scheduler_config['min_lr']
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_config['type']}")
        
        return optimizer, scheduler
    
    def _freeze_backbone(self):
        """Freeze backbone parameters (for Phase 1)."""
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False
        
        logger.info("Backbone frozen - only training decoder")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters (for Phase 2)."""
        for param in self.model.parameters():
            param.requires_grad = True
        
        logger.info("Backbone unfrozen - training full model")
    
    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        scaler: Optional[GradScaler],
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch['image'].to(self.device)
            target_heatmaps = batch['heatmaps'].to(self.device)
            
            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if scaler is not None:  # Mixed precision training
                with autocast():
                    pred_heatmaps = self.model(images)
                    loss = nn.functional.mse_loss(pred_heatmaps, target_heatmaps)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Gradient clipping
                if 'gradient_clip' in self.config['training']['phase2']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['phase2']['gradient_clip']
                    )
                
                scaler.step(optimizer)
                scaler.update()
            else:  # Full precision training
                pred_heatmaps = self.model(images)
                loss = nn.functional.mse_loss(pred_heatmaps, target_heatmaps)
                
                loss.backward()
                
                if 'gradient_clip' in self.config['training']['phase2']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['phase2']['gradient_clip']
                    )
                
                optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            target_heatmaps = batch['heatmaps'].to(self.device)
            target_keypoints = batch['keypoints']
            
            # Forward pass
            pred_heatmaps = self.model(images)
            loss = nn.functional.mse_loss(pred_heatmaps, target_heatmaps)
            
            total_loss += loss.item()
            
            # Convert heatmaps to keypoints
            pred_keypoints = self._heatmaps_to_keypoints(pred_heatmaps)
            
            all_predictions.append(pred_keypoints.cpu())
            all_targets.append(target_keypoints)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.evaluator.compute_metrics(all_predictions, all_targets)
        metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def _heatmaps_to_keypoints(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Convert heatmaps to keypoint coordinates using soft-argmax.
        
        Args:
            heatmaps: (B, 17, H, W) tensor
        
        Returns:
            keypoints: (B, 17, 3) tensor with [x, y, confidence]
        """
        B, num_joints, H, W = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.view(B, num_joints, -1)
        
        # Softmax to get probability distribution
        probs = torch.softmax(heatmaps_flat, dim=-1)
        
        # Get coordinates
        coords = probs.view(B, num_joints, H, W)
        
        # Compute expected coordinates (soft-argmax)
        x_coords = torch.arange(W, device=heatmaps.device).float()
        y_coords = torch.arange(H, device=heatmaps.device).float()
        
        x = (coords.sum(dim=2) * x_coords.view(1, 1, -1)).sum(dim=-1)
        y = (coords.sum(dim=3) * y_coords.view(1, 1, -1)).sum(dim=-1)
        
        # Get confidence (max value in heatmap)
        confidence = heatmaps.view(B, num_joints, -1).max(dim=-1)[0]
        
        # Scale to image coordinates
        image_w, image_h = self.config['model']['input_size']
        x = x * (image_w / W)
        y = y * (image_h / H)
        
        # Stack into (B, 17, 3)
        keypoints = torch.stack([x, y, confidence], dim=-1)
        
        return keypoints
    
    def train_phase(self, phase_name: str, phase_config: dict):
        """Train a single phase."""
        logger.info("\n" + "="*60)
        logger.info(f"Starting {phase_name}")
        logger.info("="*60)
        
        # Freeze/unfreeze backbone
        if phase_config['freeze_backbone']:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()
        
        # Build optimizer and scheduler
        optimizer, scheduler = self._build_optimizer_and_scheduler(phase_config)
        
        # Mixed precision scaler
        scaler = GradScaler() if self.config['training']['mixed_precision'] else None
        
        # Training loop
        for epoch in range(1, phase_config['epochs'] + 1):
            logger.info(f"\nEpoch {epoch}/{phase_config['epochs']}")
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Train
            train_metrics = self.train_epoch(optimizer, scaler, epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"Val AP: {val_metrics['AP']:.2f}%")
            logger.info(f"Val AP50: {val_metrics['AP50']:.2f}%")
            logger.info(f"Val AP75: {val_metrics['AP75']:.2f}%")
            
            # Save checkpoint
            if val_metrics['AP'] > self.best_val_ap:
                self.best_val_ap = val_metrics['AP']
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"✓ New best model saved! AP: {self.best_val_ap:.2f}%")
            
            # Step scheduler
            scheduler.step()
        
        logger.info(f"\n{phase_name} complete!")
        logger.info(f"Best validation AP: {self.best_val_ap:.2f}%")
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'best_val_ap': self.best_val_ap,
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            save_path = self.checkpoint_dir / 'best_model.pth'
        else:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved: {save_path}")
    
    def run(self):
        """Run complete Stage 1 training."""
        try:
            # Phase 1: Decoder warmup
            phase1_config = self.config['training']['phase1']
            self.train_phase("Phase 1: Decoder Warmup", phase1_config)
            
            # Phase 2: Full fine-tuning
            phase2_config = self.config['training']['phase2']
            self.train_phase("Phase 2: Full Fine-tuning", phase2_config)
            
            logger.info("\n" + "="*60)
            logger.info("STAGE 1 TRAINING COMPLETE!")
            logger.info("="*60)
            logger.info(f"Best validation AP: {self.best_val_ap:.2f}%")
            logger.info(f"Expected range: 82-85% AP")
            logger.info(f"Best model saved at: {self.checkpoint_dir / 'best_model.pth'}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 1: Baseline Fine-Tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_config.yaml",
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Create and run trainer
    trainer = Stage1Trainer(args.config)
    trainer.run()
