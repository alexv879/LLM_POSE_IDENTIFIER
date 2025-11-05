"""
Stage 4: VAE-Based Anatomical Plausibility Refinement
Uses denoising variational autoencoder to ensure anatomically plausible poses
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import yaml
from pathlib import Path
import logging
from typing import Dict, Tuple
import numpy as np
from tqdm import tqdm
import json

from utils.coco_dataset import COCOPoseDataset
from utils.metrics import COCOEvaluator
from utils.visualization import visualize_predictions, plot_training_curves

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PoseVAE(nn.Module):
    """Variational Autoencoder for pose keypoints"""
    
    def __init__(
        self,
        input_dim: int = 51,  # 17 keypoints × 3
        latent_dim: int = 32,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batch_norm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent space.
        
        Args:
            x: (B, 51) flattened keypoints
        Returns:
            mu, logvar: (B, latent_dim) each
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to keypoints.
        
        Args:
            z: (B, latent_dim) latent vectors
        Returns:
            Reconstructed keypoints (B, 51)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: (B, 51) flattened keypoints
        Returns:
            reconstruction, mu, logvar
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar


class AnatomicalConstraints:
    """Check anatomical plausibility of poses"""
    
    # COCO bone connections
    BONES = [
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
    ]
    
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(
        self,
        max_bone_length: float = 500.0,
        min_bone_length: float = 20.0,
        max_bone_length_ratio: float = 2.5,
        symmetry_tolerance: float = 0.3
    ):
        self.max_bone_length = max_bone_length
        self.min_bone_length = min_bone_length
        self.max_bone_length_ratio = max_bone_length_ratio
        self.symmetry_tolerance = symmetry_tolerance
        
        # Build keypoint index mapping
        self.kpt_to_idx = {name: i for i, name in enumerate(self.KEYPOINT_NAMES)}
    
    def compute_bone_length(
        self,
        keypoints: np.ndarray,
        kpt1_name: str,
        kpt2_name: str
    ) -> float:
        """Compute length of bone between two keypoints"""
        idx1 = self.kpt_to_idx[kpt1_name]
        idx2 = self.kpt_to_idx[kpt2_name]
        
        pt1 = keypoints[idx1, :2]
        pt2 = keypoints[idx2, :2]
        
        return np.linalg.norm(pt1 - pt2)
    
    def check_bone_lengths(self, keypoints: np.ndarray) -> bool:
        """Check if bone lengths are within reasonable range"""
        for kpt1, kpt2 in self.BONES:
            length = self.compute_bone_length(keypoints, kpt1, kpt2)
            
            if length < self.min_bone_length or length > self.max_bone_length:
                return False
        
        return True
    
    def check_symmetry(self, keypoints: np.ndarray) -> bool:
        """Check if left/right limbs have similar lengths"""
        symmetric_bones = [
            (('left_shoulder', 'left_elbow'), ('right_shoulder', 'right_elbow')),
            (('left_elbow', 'left_wrist'), ('right_elbow', 'right_wrist')),
            (('left_hip', 'left_knee'), ('right_hip', 'right_knee')),
            (('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')),
        ]
        
        for left_bone, right_bone in symmetric_bones:
            left_length = self.compute_bone_length(keypoints, *left_bone)
            right_length = self.compute_bone_length(keypoints, *right_bone)
            
            ratio = max(left_length, right_length) / (min(left_length, right_length) + 1e-8)
            
            if ratio > (1 + self.symmetry_tolerance):
                return False
        
        return True
    
    def is_plausible(self, keypoints: np.ndarray) -> bool:
        """Check overall plausibility"""
        return self.check_bone_lengths(keypoints) and self.check_symmetry(keypoints)


class KeypointDataset(Dataset):
    """Dataset of keypoints for VAE training"""
    
    def __init__(self, keypoints: np.ndarray):
        """
        Args:
            keypoints: (N, 17, 3) array of keypoints
        """
        self.keypoints = keypoints
    
    def __len__(self):
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        kpts = self.keypoints[idx]  # (17, 3)
        kpts_flat = kpts.reshape(-1)  # (51,)
        return torch.from_numpy(kpts_flat).float()


class Stage4VAETrainer:
    """Stage 4: VAE refinement trainer"""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create VAE model
        vae_config = self.config['model']['vae']
        self.vae = PoseVAE(
            input_dim=vae_config['input_dim'],
            latent_dim=vae_config['latent_dim'],
            hidden_dims=vae_config['hidden_dims'],
            dropout=vae_config['dropout'],
            use_batch_norm=vae_config['use_batch_norm']
        ).to(self.device)
        
        # Anatomical constraints checker
        plausibility_config = self.config['model']['plausibility']['anatomical_constraints']
        self.anatomical_checker = AnatomicalConstraints(
            max_bone_length_ratio=plausibility_config['max_bone_length_ratio']
        )
        
        # Load Stage 3 predictions for validation
        self._load_stage3_predictions()
        
        # Create datasets
        self._create_datasets()
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.vae.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs']
        )
        
        # Scaler for mixed precision
        self.scaler = GradScaler()
        
        # Beta schedule for KL annealing
        self.beta_schedule = self._create_beta_schedule()
        
        # Evaluator
        self.evaluator = COCOEvaluator()
        
        # Training history
        self.history = {
            'train_loss': [],
            'recon_loss': [],
            'kl_loss': [],
            'val_loss': [],
            'val_ap': [],
            'plausibility_rate': []
        }
        
        logger.info("Stage 4 VAE Trainer initialized")
    
    def _load_stage3_predictions(self):
        """Load Stage 3 checkpoint"""
        ckpt_path = Path(self.config['model']['input_checkpoint'])
        
        if ckpt_path.exists():
            self.stage3_checkpoint = torch.load(ckpt_path, map_location=self.device)
            logger.info(f"Loaded Stage 3 checkpoint from {ckpt_path}")
        else:
            logger.warning(f"Stage 3 checkpoint not found: {ckpt_path}")
            self.stage3_checkpoint = None
    
    def _create_beta_schedule(self) -> np.ndarray:
        """Create beta annealing schedule"""
        if not self.config['training']['vae_training']['beta_annealing']:
            return np.ones(self.config['training']['epochs']) * \
                   self.config['training']['vae_training']['beta']
        
        schedule_config = self.config['training']['vae_training']['beta_schedule']
        anneal_epochs = schedule_config['anneal_epochs']
        start = schedule_config['start']
        end = schedule_config['end']
        total_epochs = self.config['training']['epochs']
        
        # Linear annealing
        schedule = np.ones(total_epochs) * end
        schedule[:anneal_epochs] = np.linspace(start, end, anneal_epochs)
        
        return schedule
    
    def _create_datasets(self):
        """Create datasets from pseudo-labeled COCO data"""
        # For demonstration, create synthetic keypoint data
        # In practice, load from Stage 3 predictions on COCO
        logger.info("Creating synthetic pseudo-labeled dataset...")
        
        # Generate synthetic keypoints
        num_samples = self.config['dataset']['train']['num_samples']
        keypoints = np.random.rand(num_samples, 17, 3)
        keypoints[:, :, :2] *= [192, 256]  # Scale to image size
        keypoints[:, :, 2] = np.random.rand(num_samples, 17) * 0.5 + 0.5  # Confidence
        
        train_dataset = KeypointDataset(keypoints)
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        # Validation dataset (smaller)
        val_keypoints = np.random.rand(1000, 17, 3)
        val_keypoints[:, :, :2] *= [192, 256]
        val_keypoints[:, :, 2] = np.random.rand(1000, 17) * 0.5 + 0.5
        
        val_dataset = KeypointDataset(val_keypoints)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
    
    def vae_loss(
        self,
        reconstruction: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute VAE loss.
        
        Loss = Reconstruction_Loss + β * KL_Divergence
        """
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(reconstruction, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = kl_loss.mean()
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train VAE for one epoch"""
        self.vae.train()
        
        total_loss = 0.0
        recon_loss_sum = 0.0
        kl_loss_sum = 0.0
        num_batches = 0
        
        # Get beta for this epoch
        beta = self.beta_schedule[epoch]
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} (β={beta:.3f})"
        )
        
        for batch in pbar:
            x = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast():
                reconstruction, mu, logvar = self.vae(x)
                loss, recon_loss, kl_loss = self.vae_loss(
                    reconstruction, x, mu, logvar, beta
                )
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.vae.parameters(),
                max_norm=self.config['training']['gradient_clip']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'recon_loss': recon_loss_sum / num_batches,
            'kl_loss': kl_loss_sum / num_batches
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate VAE"""
        self.vae.eval()
        
        total_loss = 0.0
        recon_errors = []
        plausible_count = 0
        total_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                x = batch.to(self.device)
                
                with autocast():
                    reconstruction, mu, logvar = self.vae(x)
                    loss, recon_loss, kl_loss = self.vae_loss(
                        reconstruction, x, mu, logvar,
                        beta=self.config['training']['vae_training']['beta']
                    )
                
                total_loss += loss.item()
                
                # Check plausibility
                x_np = x.cpu().numpy().reshape(-1, 17, 3)
                recon_np = reconstruction.cpu().numpy().reshape(-1, 17, 3)
                
                for i in range(len(x_np)):
                    recon_error = np.mean((x_np[i] - recon_np[i]) ** 2)
                    recon_errors.append(recon_error)
                    
                    if self.anatomical_checker.is_plausible(recon_np[i]):
                        plausible_count += 1
                    total_count += 1
        
        plausibility_rate = plausible_count / total_count * 100
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'mean_recon_error': np.mean(recon_errors),
            'plausibility_rate': plausibility_rate
        }
    
    def train(self):
        """Full training loop"""
        logger.info("Starting Stage 4 VAE training...")
        
        best_loss = float('inf')
        output_dir = Path(self.config['checkpoint']['save_dir'])
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
                f"(Recon: {train_metrics['recon_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Plausibility: {val_metrics['plausibility_rate']:.1f}%"
            )
            
            # Save history
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['recon_loss'].append(train_metrics['recon_loss'])
            self.history['kl_loss'].append(train_metrics['kl_loss'])
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['plausibility_rate'].append(val_metrics['plausibility_rate'])
            
            # Save checkpoint
            is_best = val_metrics['val_loss'] < best_loss
            if is_best:
                best_loss = val_metrics['val_loss']
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.vae.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_loss': best_loss,
                'val_metrics': val_metrics,
                'config': self.config
            }
            
            torch.save(checkpoint, output_dir / 'stage4_latest.pth')
            
            if is_best:
                torch.save(checkpoint, output_dir / 'stage4_best.pth')
                logger.info(f"✓ New best model! Loss: {best_loss:.4f}")
        
        logger.info(f"Training complete! Best loss: {best_loss:.4f}")
        logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 4 VAE Training")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage4_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Train
    trainer = Stage4VAETrainer(args.config)
    trainer.train()
