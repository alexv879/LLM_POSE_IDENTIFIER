"""
Stage 2: Semi-Supervised Learning with Multi-Path Augmentation
Implements ICLR 2025 multi-path consistency approach with 3 synergistic augmentation variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.cuda.amp import autocast, GradScaler
import yaml
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm

from models.sapiens_model import SapiensForPose
from utils.coco_dataset import COCOPoseDataset
from utils.metrics import COCOEvaluator
from utils.visualization import visualize_predictions, plot_training_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiPathAugmentation:
    """Multi-path hard augmentation variants for SSL"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def get_weak_augmentation(self):
        """Weak augmentation: minimal transforms"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(self.config['input_size'][0], self.config['input_size'][1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_geometry_augmentation(self):
        """Hard augmentation path 1: Geometric transformations"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(self.config['input_size'][0], self.config['input_size'][1]),
            A.Rotate(limit=60, p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.4,
                rotate_limit=60,
                p=0.8
            ),
            A.Perspective(scale=(0.05, 0.15), p=0.6),
            A.ElasticTransform(alpha=50, sigma=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_appearance_augmentation(self):
        """Hard augmentation path 2: Appearance transformations"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(self.config['input_size'][0], self.config['input_size'][1]),
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=0.8),
            A.GaussianBlur(blur_limit=(7, 15), p=0.5),
            A.GaussNoise(var_limit=(20, 80), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.ToGray(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    def get_occlusion_augmentation(self):
        """Hard augmentation path 3: Occlusion and cutout"""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        return A.Compose([
            A.Resize(self.config['input_size'][0], self.config['input_size'][1]),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=3,
                fill_value=0,
                p=0.7
            ),
            A.GridDropout(ratio=0.3, p=0.5),
            A.RandomCrop(
                height=int(self.config['input_size'][0] * 0.85),
                width=int(self.config['input_size'][1] * 0.85),
                p=0.6
            ),
            A.Resize(self.config['input_size'][0], self.config['input_size'][1]),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


class UnlabeledDataset(COCOPoseDataset):
    """Dataset for unlabeled images"""
    
    def __init__(self, image_dir: str, transforms=None):
        # Call parent init without annotations
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        
        # Load all images
        self.image_paths = list(self.image_dir.glob("*.jpg")) + \
                          list(self.image_dir.glob("*.png"))
        
        logger.info(f"Loaded {len(self.image_paths)} unlabeled images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'image_id': img_path.stem
        }


class Stage2SSLTrainer:
    """Stage 2: SSL training with multi-path augmentation"""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create model
        self.model = SapiensForPose(
            num_keypoints=self.config['model']['num_keypoints'],
            pretrained_model_name=self.config['model'].get('pretrained_model_name')
        ).to(self.device)
        
        # Load Stage 1 checkpoint
        self._load_stage1_checkpoint()
        
        # Create augmentation paths
        self.aug_handler = MultiPathAugmentation(self.config['data'])
        
        # Create datasets
        self._create_datasets()
        
        # Create optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Create scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler()
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Evaluator
        self.evaluator = COCOEvaluator()
        
        # Training history
        self.history = {
            'train_loss': [],
            'supervised_loss': [],
            'ssl_loss': [],
            'val_loss': [],
            'val_ap': []
        }
        
        logger.info("Stage 2 SSL Trainer initialized")
    
    def _load_stage1_checkpoint(self):
        """Load Stage 1 checkpoint as initialization"""
        ckpt_path = Path(self.config['training']['stage1_checkpoint'])
        
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded Stage 1 checkpoint from {ckpt_path}")
            logger.info(f"Stage 1 Best AP: {checkpoint.get('best_ap', 'N/A'):.2f}%")
        else:
            logger.warning(f"Stage 1 checkpoint not found at {ckpt_path}")
            logger.warning("Starting from scratch!")
    
    def _create_datasets(self):
        """Create labeled and unlabeled datasets"""
        # Labeled dataset
        labeled_dataset = COCOPoseDataset(
            image_dir=self.config['data']['image_dir'],
            annotation_file=self.config['data']['annotation_file'],
            split='train',
            transforms=self.aug_handler.get_weak_augmentation()
        )
        
        # Unlabeled dataset with 3 augmentation variants
        unlabeled_dir = self.config['data']['unlabeled_image_dir']
        
        unlabeled_geo = UnlabeledDataset(
            unlabeled_dir,
            transforms=self.aug_handler.get_geometry_augmentation()
        )
        
        unlabeled_app = UnlabeledDataset(
            unlabeled_dir,
            transforms=self.aug_handler.get_appearance_augmentation()
        )
        
        unlabeled_occ = UnlabeledDataset(
            unlabeled_dir,
            transforms=self.aug_handler.get_occlusion_augmentation()
        )
        
        # Create dataloaders
        self.labeled_loader = DataLoader(
            labeled_dataset,
            batch_size=self.config['training']['batch_size'] // 2,
            shuffle=True,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        self.unlabeled_loaders = [
            DataLoader(ds, batch_size=self.config['training']['batch_size'] // 6,
                      shuffle=True, num_workers=2, pin_memory=True)
            for ds in [unlabeled_geo, unlabeled_app, unlabeled_occ]
        ]
        
        # Validation dataset
        val_dataset = COCOPoseDataset(
            image_dir=self.config['data']['image_dir'],
            annotation_file=self.config['data']['annotation_file'],
            split='val',
            transforms=self.aug_handler.get_weak_augmentation()
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Labeled samples: {len(labeled_dataset)}")
        logger.info(f"Unlabeled samples: {len(unlabeled_geo)} x 3 paths")
        logger.info(f"Validation samples: {len(val_dataset)}")
    
    def _compute_consistency_loss(
        self,
        predictions: List[torch.Tensor],
        temperature: float = 0.5,
        confidence_threshold: float = 0.7
    ) -> torch.Tensor:
        """
        Compute consistency loss between multiple augmentation paths.
        Uses confidence thresholding and sharpening for better pseudo-labels.
        
        Implementation based on SSL Multi-Path paper (ICLR 2025):
        1. Ensemble prediction as pseudo-label
        2. Sharpen with temperature
        3. Confidence-based weighting
        4. KL divergence loss
        
        Args:
            predictions: List of [B, K, H, W] heatmap predictions from different aug paths
            temperature: Temperature for sharpening (lower = sharper)
            confidence_threshold: Minimum confidence to use for loss
        
        Returns:
            Weighted consistency loss scalar
        """
        # Stack predictions: (num_paths, B, K, H, W)
        pred_stack = torch.stack(predictions)
        B, K, H, W = predictions[0].shape
        
        # Compute ensemble (pseudo-label) with sharpening
        with torch.no_grad():
            # Average across augmentation paths
            ensemble_pred = pred_stack.mean(dim=0)  # (B, K, H, W)
            
            # Sharpen with temperature (flatten spatial dims for softmax)
            ensemble_flat = ensemble_pred.view(B, K, H * W)
            ensemble_sharp = torch.softmax(ensemble_flat / temperature, dim=-1)
            ensemble_sharp = ensemble_sharp.view(B, K, H, W)
            
            # Get confidence (max value per keypoint heatmap)
            confidence = ensemble_sharp.view(B, K, -1).max(dim=-1)[0]  # (B, K)
            
            # Confidence mask for high-confidence predictions only
            conf_mask = (confidence > confidence_threshold).float()  # (B, K)
            
            # Detach pseudo-labels
            ensemble_sharp = ensemble_sharp.detach()
        
        # Compute consistency loss with confidence weighting
        consistency_loss = 0.0
        
        for pred in predictions:
            # Apply softmax to prediction
            pred_flat = pred.view(B, K, H * W)
            pred_softmax = torch.softmax(pred_flat, dim=-1)
            pred_softmax = pred_softmax.view(B, K, H, W)
            
            # Flatten for KL computation
            pred_log_softmax = torch.log_softmax(pred.view(B, K, -1), dim=-1)
            ensemble_softmax_flat = ensemble_sharp.view(B, K, -1)
            
            # KL divergence loss (per keypoint)
            # KL(P||Q) = sum(P * log(P/Q))
            kl_loss = F.kl_div(
                pred_log_softmax,
                ensemble_softmax_flat,
                reduction='none',
                log_target=False
            ).sum(dim=-1)  # (B, K)
            
            # Weight by confidence
            weighted_loss = (kl_loss * conf_mask).sum() / (conf_mask.sum() + 1e-8)
            consistency_loss += weighted_loss
        
        # Average across augmentation paths
        consistency_loss = consistency_loss / len(predictions)
        
        return consistency_loss
    
    def _get_ssl_weight(self, epoch: int) -> float:
        """
        Ramp-up schedule for SSL weight.
        
        λ(t) = t / rampup_epochs for first rampup_epochs
        λ(t) = 1.0 thereafter
        """
        rampup = self.config['ssl']['rampup_epochs']
        max_weight = self.config['ssl']['ssl_weight']
        
        if epoch < rampup:
            return max_weight * (epoch / rampup)
        else:
            return max_weight
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with SSL"""
        self.model.train()
        
        total_loss = 0.0
        supervised_loss_sum = 0.0
        ssl_loss_sum = 0.0
        num_batches = 0
        
        # Get SSL weight for this epoch
        ssl_weight = self._get_ssl_weight(epoch)
        
        # Create iterators
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iters = [iter(loader) for loader in self.unlabeled_loaders]
        
        # Progress bar
        pbar = tqdm(
            range(len(self.labeled_loader)),
            desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} (λ={ssl_weight:.3f})"
        )
        
        for batch_idx in pbar:
            # Get labeled batch
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                labeled_batch = next(labeled_iter)
            
            # Get unlabeled batches (one from each augmentation path)
            unlabeled_batches = []
            for i, unlabeled_iter in enumerate(unlabeled_iters):
                try:
                    batch = next(unlabeled_iter)
                    unlabeled_batches.append(batch)
                except StopIteration:
                    unlabeled_iters[i] = iter(self.unlabeled_loaders[i])
                    batch = next(unlabeled_iters[i])
                    unlabeled_batches.append(batch)
            
            # Move labeled data to device
            images = labeled_batch['image'].to(self.device)
            target_heatmaps = labeled_batch['heatmap'].to(self.device)
            
            # Forward pass on labeled data
            self.optimizer.zero_grad()
            
            with autocast():
                # Supervised loss
                pred_heatmaps = self.model(images)
                supervised_loss = self.criterion(pred_heatmaps, target_heatmaps)
                
                # SSL loss on unlabeled data
                if ssl_weight > 0:
                    unlabeled_predictions = []
                    
                    for unlabeled_batch in unlabeled_batches:
                        unlabeled_images = unlabeled_batch['image'].to(self.device)
                        unlabeled_pred = self.model(unlabeled_images)
                        unlabeled_predictions.append(unlabeled_pred)
                    
                    ssl_loss = self._compute_consistency_loss(unlabeled_predictions)
                else:
                    ssl_loss = torch.tensor(0.0).to(self.device)
                
                # Total loss
                loss = supervised_loss + ssl_weight * ssl_loss
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            supervised_loss_sum += supervised_loss.item()
            ssl_loss_sum += ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else 0.0
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sup': f'{supervised_loss.item():.4f}',
                'ssl': f'{ssl_loss.item() if isinstance(ssl_loss, torch.Tensor) else 0.0:.4f}'
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_sup_loss = supervised_loss_sum / num_batches
        avg_ssl_loss = ssl_loss_sum / num_batches
        
        return {
            'train_loss': avg_loss,
            'supervised_loss': avg_sup_loss,
            'ssl_loss': avg_ssl_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                target_heatmaps = batch['heatmap'].to(self.device)
                target_keypoints = batch['keypoints']
                
                # Forward pass
                with autocast():
                    pred_heatmaps = self.model(images)
                    loss = self.criterion(pred_heatmaps, target_heatmaps)
                
                total_loss += loss.item()
                
                # Convert heatmaps to keypoints
                pred_keypoints = self._heatmaps_to_keypoints(pred_heatmaps)
                
                all_predictions.append(pred_keypoints.cpu().numpy())
                all_targets.append(target_keypoints.numpy())
        
        # Compute metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        metrics = self.evaluator.compute_metrics(predictions, targets)
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_ap': metrics['AP'],
            'val_ap50': metrics['AP50'],
            'val_ap75': metrics['AP75'],
            'val_ar': metrics['AR']
        }
    
    def _heatmaps_to_keypoints(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Convert heatmaps to keypoint coordinates using soft-argmax"""
        B, K, H, W = heatmaps.shape
        
        # Flatten spatial dimensions
        heatmaps_flat = heatmaps.reshape(B, K, -1)
        
        # Softmax to get probability distribution
        probs = torch.softmax(heatmaps_flat, dim=-1)
        
        # Create coordinate grids
        y_coords = torch.arange(H, device=heatmaps.device).float()
        x_coords = torch.arange(W, device=heatmaps.device).float()
        
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (H*W, 2)
        
        # Weighted average
        keypoints = torch.matmul(probs, coords)  # (B, K, 2)
        
        # Get confidence from max heatmap value
        confidence = heatmaps_flat.max(dim=-1)[0].unsqueeze(-1)  # (B, K, 1)
        
        # Concatenate [x, y, conf]
        keypoints = torch.cat([keypoints, confidence], dim=-1)  # (B, K, 3)
        
        # Scale to original image size
        scale = torch.tensor(
            [self.config['data']['input_size'][1] / W,
             self.config['data']['input_size'][0] / H,
             1.0],
            device=heatmaps.device
        )
        keypoints = keypoints * scale
        
        return keypoints
    
    def run(self):
        """Full training loop"""
        logger.info("Starting Stage 2 SSL training...")
        
        best_ap = 0.0
        output_dir = Path(self.config['training']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_metrics['train_loss']:.4f} "
                f"(Sup: {train_metrics['supervised_loss']:.4f}, "
                f"SSL: {train_metrics['ssl_loss']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val AP: {val_metrics['val_ap']:.2f}% | "
                f"Val AP50: {val_metrics['val_ap50']:.2f}% | "
                f"Val AP75: {val_metrics['val_ap75']:.2f}%"
            )
            
            # Save history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['supervised_loss'].append(train_metrics['supervised_loss'])
            self.history['ssl_loss'].append(train_metrics['ssl_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['val_ap'].append(val_metrics['val_ap'])
            
            # Save checkpoint
            is_best = val_metrics['val_ap'] > best_ap
            if is_best:
                best_ap = val_metrics['val_ap']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_ap': best_ap,
                'val_metrics': val_metrics,
                'config': self.config
            }
            
            # Save latest
            torch.save(checkpoint, output_dir / 'stage2_latest.pth')
            
            # Save best
            if is_best:
                torch.save(checkpoint, output_dir / 'stage2_best.pth')
                logger.info(f"✓ New best model! AP: {best_ap:.2f}%")
        
        # Plot training curves
        plot_training_curves(
            self.history['train_loss'],
            self.history['val_loss'],
            self.history['val_ap'],
            save_path=output_dir / 'stage2_training_curves.png'
        )
        
        logger.info(f"Training complete! Best AP: {best_ap:.2f}%")
        logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 2 SSL Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage2_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Train
    trainer = Stage2SSLTrainer(args.config)
    trainer.run()
