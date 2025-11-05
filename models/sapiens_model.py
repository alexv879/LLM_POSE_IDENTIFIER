"""
Sapiens-2B Model Implementation for Pose Estimation
Based on Meta Sapiens paper (ECCV 2024)

Uses Vision Transformer (ViT) backbone with MAE pretraining
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SapiensForPose(nn.Module):
    """
    Sapiens-2B adapted for 2D human pose estimation.
    
    Architecture:
    - Backbone: Vision Transformer (ViT-2B) pretrained with MAE
    - Decoder: Convolutional upsampling layers
    - Output: Heatmaps for 17 COCO keypoints
    
    Based on: facebook/sapiens_2b (HuggingFace model)
    """
    
    def __init__(
        self,
        backbone_name: str = "vit_2b",
        num_keypoints: int = 17,
        pretrained: bool = True,
        pretrained_path: Optional[str] = None,
        input_size: Tuple[int, int] = (256, 192),
        heatmap_size: Tuple[int, int] = (64, 48),
        freeze_backbone: bool = False
    ):
        """
        Args:
            backbone_name: Name of backbone architecture
            num_keypoints: Number of keypoints to predict (17 for COCO)
            pretrained: Whether to load pretrained weights
            pretrained_path: Path to pretrained model (HuggingFace or local)
            input_size: Input image size (width, height)
            heatmap_size: Output heatmap size (width, height)
            freeze_backbone: Whether to freeze backbone initially
        """
        super().__init__()
        
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        
        # Build backbone
        logger.info(f"Building backbone: {backbone_name}")
        self.backbone = self._build_backbone(
            backbone_name, pretrained, pretrained_path
        )
        
        # Get backbone output dimensions
        self.backbone_output_dim = self.backbone.config.hidden_size
        self.num_patches = (input_size[1] // 16) * (input_size[0] // 16)  # 16x16 patches
        
        # Build decoder
        logger.info("Building decoder...")
        self.decoder = self._build_decoder()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        logger.info(f"Model initialized:")
        logger.info(f"  Input size: {input_size}")
        logger.info(f"  Heatmap size: {heatmap_size}")
        logger.info(f"  Num keypoints: {num_keypoints}")
        logger.info(f"  Backbone output dim: {self.backbone_output_dim}")
    
    def _build_backbone(
        self,
        backbone_name: str,
        pretrained: bool,
        pretrained_path: Optional[str]
    ) -> nn.Module:
        """Build Vision Transformer backbone."""
        if pretrained and pretrained_path:
            try:
                # Try loading from HuggingFace
                logger.info(f"Loading pretrained model from: {pretrained_path}")
                backbone = ViTModel.from_pretrained(pretrained_path)
            except Exception as e:
                logger.warning(f"Failed to load from HuggingFace: {e}")
                logger.info("Initializing from scratch...")
                config = ViTConfig(
                    hidden_size=1024,  # ViT-Base
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    intermediate_size=4096,
                    image_size=self.input_size[1],  # height
                    patch_size=16
                )
                backbone = ViTModel(config)
        else:
            # Initialize from scratch
            logger.info("Initializing ViT from scratch...")
            config = ViTConfig(
                hidden_size=1024,  # ViT-Base (use 2048 for ViT-2B if available)
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
                image_size=self.input_size[1],
                patch_size=16
            )
            backbone = ViTModel(config)
        
        return backbone
    
    def _build_decoder(self) -> nn.Module:
        """
        Build decoder for converting ViT features to heatmaps.
        Matches official Sapiens HeatmapHead architecture from facebook/sapiens.
        
        Architecture (Official):
        - Reshape patch embeddings to spatial grid
        - 2x deconvolution layers (768 channels each, 4x4 kernel, stride 2)
        - 2x 1x1 convolution layers (768 channels) for feature refinement
        - Final 1x1 conv to num_keypoints channels
        
        Total upsampling: 4x (2x from each deconv)
        """
        # Calculate spatial dimensions after ViT
        patch_h = self.input_size[1] // 16  # 192 // 16 = 12
        patch_w = self.input_size[0] // 16  # 256 // 16 = 16
        
        decoder_layers = []
        
        # Deconv Layer 1: upsample by 2x (12x16 -> 24x32)
        decoder_layers.extend([
            nn.ConvTranspose2d(
                self.backbone_output_dim, 768,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        ])
        
        # Conv Layer 1: feature refinement at 24x32
        decoder_layers.extend([
            nn.Conv2d(768, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        ])
        
        # Deconv Layer 2: upsample by 2x (24x32 -> 48x64)
        decoder_layers.extend([
            nn.ConvTranspose2d(
                768, 768,
                kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        ])
        
        # Conv Layer 2: feature refinement at 48x64
        decoder_layers.extend([
            nn.Conv2d(768, 768, kernel_size=1, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
        ])
        
        # Final layer: output heatmaps for num_keypoints
        decoder_layers.append(
            nn.Conv2d(768, self.num_keypoints, kernel_size=1, stride=1, padding=0)
        )
        
        decoder = nn.Sequential(*decoder_layers)
        
        logger.info(f"Decoder built with 768-channel HeatmapHead architecture (official Sapiens)")
        
        return decoder
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            heatmaps: (B, num_keypoints, heatmap_h, heatmap_w)
        """
        B = x.shape[0]
        
        # Pass through ViT backbone
        # Output: (B, num_patches + 1, hidden_dim) where +1 is CLS token
        outputs = self.backbone(pixel_values=x)
        features = outputs.last_hidden_state
        
        # Remove CLS token and reshape to spatial grid
        patch_embeddings = features[:, 1:, :]  # (B, num_patches, hidden_dim)
        
        # Reshape to spatial: (B, hidden_dim, patch_h, patch_w)
        patch_h = self.input_size[1] // 16
        patch_w = self.input_size[0] // 16
        
        spatial_features = patch_embeddings.permute(0, 2, 1).reshape(
            B, self.backbone_output_dim, patch_h, patch_w
        )
        
        # Pass through decoder
        heatmaps = self.decoder(spatial_features)
        
        # Ensure output matches target heatmap size
        if heatmaps.shape[-2:] != self.heatmap_size[::-1]:  # (H, W)
            heatmaps = nn.functional.interpolate(
                heatmaps,
                size=self.heatmap_size[::-1],  # (height, width)
                mode='bilinear',
                align_corners=False
            )
        
        return heatmaps


class PoseDecoder(nn.Module):
    """
    Alternative lightweight decoder for pose estimation.
    Can be swapped with the decoder in SapiensForPose.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_keypoints: int,
        input_spatial_size: Tuple[int, int],
        output_spatial_size: Tuple[int, int]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_keypoints = num_keypoints
        self.input_spatial_size = input_spatial_size
        self.output_spatial_size = output_spatial_size
        
        # Simple decoder with transposed convolutions
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, num_keypoints, 1, 1, 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


if __name__ == "__main__":
    # Test model
    logging.basicConfig(level=logging.INFO)
    
    model = SapiensForPose(
        backbone_name="vit_base",
        num_keypoints=17,
        pretrained=False,
        input_size=(256, 192),
        heatmap_size=(64, 48)
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 192, 256)  # (B, C, H, W)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: (2, 17, 48, 64)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
