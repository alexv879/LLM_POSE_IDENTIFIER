"""
Stage 3: Ensemble Fusion with Multiple Models
Combines Sapiens-2B, DWPose, and ViTPose predictions with confidence-weighted fusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm
import json

from models.sapiens_model import SapiensForPose
from utils.coco_dataset import COCOPoseDataset
from utils.metrics import COCOEvaluator
from utils.visualization import visualize_predictions, create_comparison_grid, plot_training_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SEAttentionModule(nn.Module):
    """Squeeze-and-Excitation attention for refinement"""
    
    def __init__(self, num_keypoints: int, reduction: int = 4):
        super().__init__()
        self.num_keypoints = num_keypoints
        
        self.fc1 = nn.Linear(num_keypoints * 3, num_keypoints * 3 // reduction)
        self.fc2 = nn.Linear(num_keypoints * 3 // reduction, num_keypoints * 3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, 3) keypoint tensor
        Returns:
            Attention-weighted keypoints (B, K, 3)
        """
        B, K, _ = x.shape
        
        # Flatten
        x_flat = x.reshape(B, -1)  # (B, K*3)
        
        # Squeeze-and-Excitation
        se = self.fc1(x_flat)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        
        # Apply attention
        x_weighted = x_flat * se
        
        return x_weighted.reshape(B, K, 3)


class IterativeRefinementModule(nn.Module):
    """Iterative refinement with SE attention"""
    
    def __init__(self, num_keypoints: int, hidden_dim: int = 256, num_iterations: int = 3):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Refinement network
        self.refine_net = nn.Sequential(
            nn.Linear(num_keypoints * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_keypoints * 3)
        )
        
        # SE attention
        self.se_attention = SEAttentionModule(num_keypoints)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, K, 3) initial keypoints
        Returns:
            Refined keypoints (B, K, 3)
        """
        B, K, _ = x.shape
        
        for i in range(self.num_iterations):
            # Flatten
            x_flat = x.reshape(B, -1)
            
            # Refinement
            residual = self.refine_net(x_flat)
            x_flat = x_flat + residual
            
            # Reshape
            x = x_flat.reshape(B, K, 3)
            
            # Apply attention
            x = self.se_attention(x)
        
        return x


class EnsembleModel(nn.Module):
    """Ensemble of multiple pose estimation models"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Load models (placeholders for DWPose and ViTPose)
        self.models = nn.ModuleDict()
        self.model_weights = {}
        
        # Sapiens model
        sapiens_config = config['models']['sapiens']
        self.models['sapiens'] = SapiensForPose(
            num_keypoints=sapiens_config['num_keypoints']
        )
        self._load_checkpoint(self.models['sapiens'], sapiens_config['checkpoint'])
        self.model_weights['sapiens'] = sapiens_config['weight']
        
        logger.info(f"Loaded Sapiens model with weight {sapiens_config['weight']}")
        
        # DWPose (placeholder - would need actual DWPose implementation)
        # For now, we'll use a copy of Sapiens as a stand-in
        dwpose_config = config['models']['dwpose']
        self.models['dwpose'] = SapiensForPose(
            num_keypoints=dwpose_config['num_keypoints']
        )
        self.model_weights['dwpose'] = dwpose_config['weight']
        logger.info(f"DWPose placeholder initialized with weight {dwpose_config['weight']}")
        
        # ViTPose (placeholder - would need actual ViTPose implementation)
        vitpose_config = config['models']['vitpose']
        self.models['vitpose'] = SapiensForPose(
            num_keypoints=vitpose_config['num_keypoints']
        )
        self.model_weights['vitpose'] = vitpose_config['weight']
        logger.info(f"ViTPose placeholder initialized with weight {vitpose_config['weight']}")
        
        # Refinement module
        if config['ensemble']['refinement']['enabled']:
            self.refinement = IterativeRefinementModule(
                num_keypoints=sapiens_config['num_keypoints'],
                hidden_dim=config['ensemble']['refinement']['hidden_dim'],
                num_iterations=config['ensemble']['refinement']['num_iterations']
            )
        else:
            self.refinement = None
    
    def _load_checkpoint(self, model: nn.Module, checkpoint_path: str):
        """Load model checkpoint"""
        ckpt_path = Path(checkpoint_path)
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    def _heatmaps_to_keypoints(
        self,
        heatmaps: torch.Tensor,
        input_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Convert heatmaps to keypoint coordinates"""
        B, K, H, W = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.reshape(B, K, -1)
        
        # Softmax to get probability distribution
        probs = torch.softmax(heatmaps_flat, dim=-1)
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=heatmaps.device).float()
        x_coords = torch.arange(W, device=heatmaps.device).float()
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
        
        # Weighted average
        keypoints = torch.matmul(probs, coords)
        
        # Get confidence
        confidence = heatmaps_flat.max(dim=-1)[0].unsqueeze(-1)
        
        # Concatenate
        keypoints = torch.cat([keypoints, confidence], dim=-1)
        
        # Scale to original image size
        scale = torch.tensor(
            [input_size[1] / W, input_size[0] / H, 1.0],
            device=heatmaps.device
        )
        keypoints = keypoints * scale
        
        return keypoints
    
    def confidence_weighted_fusion(
        self,
        predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse predictions using confidence-weighted averaging.
        
        Final = Σ(weight_i * pred_i * conf_i) / Σ(weight_i * conf_i)
        
        Args:
            predictions: Dict of {model_name: (B, K, 3) keypoints}
        Returns:
            Fused keypoints (B, K, 3)
        """
        # Extract coordinates and confidences
        weighted_sum = None
        weight_sum = None
        
        for model_name, pred in predictions.items():
            coords = pred[:, :, :2]  # (B, K, 2)
            conf = pred[:, :, 2:3]  # (B, K, 1)
            
            model_weight = self.model_weights[model_name]
            
            # Weighted contribution
            weighted_coords = coords * conf * model_weight
            weighted_conf = conf * model_weight
            
            if weighted_sum is None:
                weighted_sum = weighted_coords
                weight_sum = weighted_conf
            else:
                weighted_sum += weighted_coords
                weight_sum += weighted_conf
        
        # Normalize
        fused_coords = weighted_sum / (weight_sum + 1e-8)
        fused_conf = weight_sum / sum(self.model_weights.values())
        
        # Concatenate
        fused_keypoints = torch.cat([fused_coords, fused_conf], dim=-1)
        
        return fused_keypoints
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            images: (B, 3, H, W) input images
        Returns:
            Fused keypoints (B, K, 3)
        """
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                heatmaps = model(images)
                keypoints = self._heatmaps_to_keypoints(
                    heatmaps,
                    input_size=(images.shape[2], images.shape[3])
                )
                predictions[model_name] = keypoints
        
        # Fuse predictions
        fused = self.confidence_weighted_fusion(predictions)
        
        # Apply refinement if enabled
        if self.refinement is not None:
            fused = self.refinement(fused)
        
        return fused


class Stage3EnsembleTrainer:
    """Stage 3: Ensemble training and evaluation"""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create ensemble model
        self.model = EnsembleModel(self.config).to(self.device)
        
        # Create datasets
        self._create_datasets()
        
        # Create optimizer (only for refinement module if enabled)
        if self.config['ensemble']['refinement']['enabled']:
            self.optimizer = optim.AdamW(
                self.model.refinement.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
            
            self.scaler = GradScaler()
            self.criterion = nn.MSELoss()
        else:
            self.optimizer = None
        
        # Evaluator
        self.evaluator = COCOEvaluator()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_ap': []
        }
        
        logger.info("Stage 3 Ensemble Trainer initialized")
    
    def _create_datasets(self):
        """Create validation and test datasets"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        val_transforms = A.Compose([
            A.Resize(
                self.config['models']['sapiens']['input_size'][0],
                self.config['models']['sapiens']['input_size'][1]
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        
        # Validation dataset
        val_dataset = COCOPoseDataset(
            image_dir=self.config['dataset']['image_dir'],
            annotation_file=self.config['dataset']['val_annotations'],
            split='val',
            transforms=val_transforms
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        # Test dataset
        test_dataset = COCOPoseDataset(
            image_dir=self.config['dataset']['image_dir'],
            annotation_file=self.config['dataset']['test_annotations'],
            split='test',
            transforms=val_transforms
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Validation samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train refinement module for one epoch"""
        if self.optimizer is None:
            logger.info("No trainable parameters, skipping training")
            return {'train_loss': 0.0}
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(
            self.val_loader,  # Using val as train for refinement
            desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}"
        )
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            target_keypoints = batch['keypoints'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                pred_keypoints = self.model(images)
                loss = self.criterion(pred_keypoints, target_keypoints)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.refinement.parameters(),
                max_norm=self.config['training']['gradient_clip']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {'train_loss': total_loss / num_batches}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate ensemble model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                images = batch['image'].to(self.device)
                target_keypoints = batch['keypoints']
                
                with autocast():
                    pred_keypoints = self.model(images)
                
                all_predictions.append(pred_keypoints.cpu().numpy())
                all_targets.append(target_keypoints.numpy())
        
        # Compute metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.evaluator.compute_metrics(predictions, targets)
        
        return metrics
    
    def run(self):
        """Full training loop (if refinement enabled)"""
        logger.info("Starting Stage 3 Ensemble training...")
        
        if self.optimizer is None:
            logger.info("No trainable parameters. Evaluating ensemble only.")
            metrics = self.evaluate(self.val_loader)
            logger.info(
                f"Validation - AP: {metrics['AP']:.2f}% | "
                f"AP50: {metrics['AP50']:.2f}% | AP75: {metrics['AP75']:.2f}%"
            )
            
            test_metrics = self.evaluate(self.test_loader)
            logger.info(
                f"Test - AP: {test_metrics['AP']:.2f}% | "
                f"AP50: {test_metrics['AP50']:.2f}% | AP75: {test_metrics['AP75']:.2f}%"
            )
            return
        
        best_ap = 0.0
        output_dir = Path(self.config['checkpoint']['save_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val AP: {val_metrics['AP']:.2f}% | "
                f"Val AP50: {val_metrics['AP50']:.2f}% | "
                f"Val AP75: {val_metrics['AP75']:.2f}%"
            )
            
            # Save history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['val_ap'].append(val_metrics['AP'])
            
            # Save checkpoint
            is_best = val_metrics['AP'] > best_ap
            if is_best:
                best_ap = val_metrics['AP']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_ap': best_ap,
                'val_metrics': val_metrics,
                'config': self.config
            }
            
            torch.save(checkpoint, output_dir / 'stage3_latest.pth')
            
            if is_best:
                torch.save(checkpoint, output_dir / 'stage3_best.pth')
                logger.info(f"✓ New best model! AP: {best_ap:.2f}%")
        
        # Final test evaluation
        test_metrics = self.evaluate(self.test_loader)
        logger.info(
            f"Final Test - AP: {test_metrics['AP']:.2f}% | "
            f"AP50: {test_metrics['AP50']:.2f}% | AP75: {test_metrics['AP75']:.2f}%"
        )
        
        logger.info(f"Training complete! Best AP: {best_ap:.2f}%")
        logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 3 Ensemble Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage3_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Train
    trainer = Stage3EnsembleTrainer(args.config)
    trainer.run()
