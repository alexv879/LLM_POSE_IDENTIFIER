"""
Test-Time Augmentation (TTA) for Pose Estimation
Implements horizontal flip test as per official Sapiens/ViTPose
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


# COCO keypoint pairs (left-right symmetric joints)
COCO_FLIP_PAIRS = [
    (1, 2),    # left_eye <-> right_eye
    (3, 4),    # left_ear <-> right_ear
    (5, 6),    # left_shoulder <-> right_shoulder
    (7, 8),    # left_elbow <-> right_elbow
    (9, 10),   # left_wrist <-> right_wrist
    (11, 12),  # left_hip <-> right_hip
    (13, 14),  # left_knee <-> right_knee
    (15, 16),  # left_ankle <-> right_ankle
]

# Keypoint 0 (nose) is center, no flip needed


class FlipTest:
    """
    Horizontal flip test-time augmentation for pose estimation.
    
    Based on official Sapiens/ViTPose implementation:
    - Flip image horizontally
    - Get prediction from flipped image
    - Flip heatmaps back
    - Swap left-right keypoints
    - Average with original prediction
    
    Args:
        flip_pairs: List of (left_idx, right_idx) tuples for keypoint swapping
        mode: 'heatmap' (flip heatmaps) or 'coord' (flip coordinates)
    """
    
    def __init__(
        self,
        flip_pairs: list = None,
        mode: str = 'heatmap'
    ):
        self.flip_pairs = flip_pairs if flip_pairs is not None else COCO_FLIP_PAIRS
        self.mode = mode
        
    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor,
        return_individual: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Apply flip test-time augmentation.
        
        Args:
            model: PyTorch model
            images: (B, 3, H, W) input images
            return_individual: If True, return both original and flipped predictions
        
        Returns:
            averaged_pred: (B, K, H, W) averaged heatmaps
            (optional) (original_pred, flipped_pred): Individual predictions
        """
        # Original prediction
        with torch.no_grad():
            pred_orig = model(images)
        
        # Flip images horizontally
        images_flip = torch.flip(images, dims=[-1])  # Flip along width dimension
        
        # Flipped prediction
        with torch.no_grad():
            pred_flip = model(images_flip)
        
        # Flip heatmaps back
        pred_flip = torch.flip(pred_flip, dims=[-1])
        
        # Swap left-right keypoints in flipped prediction
        pred_flip = self._swap_left_right_keypoints(pred_flip)
        
        # Average predictions
        pred_avg = (pred_orig + pred_flip) / 2.0
        
        if return_individual:
            return pred_avg, (pred_orig, pred_flip)
        else:
            return pred_avg
    
    def _swap_left_right_keypoints(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Swap left-right symmetric keypoints in heatmaps.
        
        Args:
            heatmaps: (B, K, H, W) keypoint heatmaps
        
        Returns:
            heatmaps_swapped: (B, K, H, W) with left-right keypoints swapped
        """
        heatmaps_swapped = heatmaps.clone()
        
        # Swap each pair
        for left_idx, right_idx in self.flip_pairs:
            # Swap channels
            heatmaps_swapped[:, left_idx] = heatmaps[:, right_idx]
            heatmaps_swapped[:, right_idx] = heatmaps[:, left_idx]
        
        return heatmaps_swapped
    
    def swap_keypoint_coordinates(
        self,
        keypoints: np.ndarray,
        image_width: int
    ) -> np.ndarray:
        """
        Swap left-right keypoints for coordinate predictions.
        
        Args:
            keypoints: (N, K, 2) or (K, 2) keypoint coordinates
            image_width: Width of image for flipping x-coordinates
        
        Returns:
            keypoints_swapped: Same shape as input with swapped keypoints
        """
        keypoints_swapped = keypoints.copy()
        
        # Flip x-coordinates
        keypoints_swapped[..., 0] = image_width - keypoints_swapped[..., 0]
        
        # Swap left-right keypoint pairs
        if keypoints.ndim == 3:
            # Batch mode (N, K, 2)
            for left_idx, right_idx in self.flip_pairs:
                # Swap keypoints
                temp = keypoints_swapped[:, left_idx].copy()
                keypoints_swapped[:, left_idx] = keypoints_swapped[:, right_idx]
                keypoints_swapped[:, right_idx] = temp
        else:
            # Single sample (K, 2)
            for left_idx, right_idx in self.flip_pairs:
                # Swap keypoints
                temp = keypoints_swapped[left_idx].copy()
                keypoints_swapped[left_idx] = keypoints_swapped[right_idx]
                keypoints_swapped[right_idx] = temp
        
        return keypoints_swapped


class MultiScaleTest:
    """
    Multi-scale test-time augmentation.
    
    Tests model at multiple scales and averages predictions.
    Common scales: [0.5, 1.0, 1.5, 2.0]
    
    Args:
        scales: List of scale factors to test
        flip_test: Whether to also apply flip test at each scale
    """
    
    def __init__(
        self,
        scales: list = None,
        flip_test: bool = True
    ):
        self.scales = scales if scales is not None else [0.5, 1.0, 1.5, 2.0]
        self.flip_test_module = FlipTest() if flip_test else None
    
    def __call__(
        self,
        model: nn.Module,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply multi-scale test-time augmentation.
        
        Args:
            model: PyTorch model
            images: (B, 3, H, W) input images
        
        Returns:
            pred_avg: (B, K, H, W) averaged heatmaps across scales
        """
        B, C, H, W = images.shape
        predictions = []
        
        for scale in self.scales:
            # Resize images
            if scale != 1.0:
                new_h = int(H * scale)
                new_w = int(W * scale)
                images_scaled = nn.functional.interpolate(
                    images,
                    size=(new_h, new_w),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                images_scaled = images
            
            # Get prediction at this scale
            if self.flip_test_module is not None:
                pred = self.flip_test_module(model, images_scaled)
            else:
                with torch.no_grad():
                    pred = model(images_scaled)
            
            # Resize prediction back to original heatmap size
            if scale != 1.0:
                pred = nn.functional.interpolate(
                    pred,
                    size=(H, W),  # Back to original heatmap size
                    mode='bilinear',
                    align_corners=False
                )
            
            predictions.append(pred)
        
        # Average across scales
        pred_avg = torch.stack(predictions).mean(dim=0)
        
        return pred_avg


def apply_flip_test(
    model: nn.Module,
    images: torch.Tensor,
    flip_pairs: list = None
) -> torch.Tensor:
    """
    Convenience function for flip test.
    
    Args:
        model: PyTorch model
        images: (B, 3, H, W) input images
        flip_pairs: List of (left_idx, right_idx) for keypoint swapping
    
    Returns:
        pred_avg: (B, K, H, W) averaged predictions
    """
    flip_test = FlipTest(flip_pairs=flip_pairs)
    return flip_test(model, images)


def apply_multi_scale_test(
    model: nn.Module,
    images: torch.Tensor,
    scales: list = None,
    flip_test: bool = True
) -> torch.Tensor:
    """
    Convenience function for multi-scale test.
    
    Args:
        model: PyTorch model
        images: (B, 3, H, W) input images
        scales: List of scale factors
        flip_test: Whether to apply flip test at each scale
    
    Returns:
        pred_avg: (B, K, H, W) averaged predictions
    """
    ms_test = MultiScaleTest(scales=scales, flip_test=flip_test)
    return ms_test(model, images)


def test_flip_test():
    """Test flip test functionality."""
    print("Testing Flip Test TTA...")
    
    # Create dummy model and data
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 17, 1)
        
        def forward(self, x):
            return self.conv(x)
    
    model = DummyModel()
    model.eval()
    
    # Create test input
    images = torch.randn(2, 3, 192, 256)
    
    # Apply flip test
    flip_test = FlipTest()
    pred_avg, (pred_orig, pred_flip) = flip_test(
        model, images, return_individual=True
    )
    
    print(f"Input shape: {images.shape}")
    print(f"Original prediction shape: {pred_orig.shape}")
    print(f"Flipped prediction shape: {pred_flip.shape}")
    print(f"Averaged prediction shape: {pred_avg.shape}")
    
    # Test coordinate swapping
    keypoints = np.array([
        [100, 50],   # 0: nose
        [90, 40],    # 1: left_eye
        [110, 40],   # 2: right_eye
        [80, 45],    # 3: left_ear
        [120, 45],   # 4: right_ear
        [85, 80],    # 5: left_shoulder
        [115, 80],   # 6: right_shoulder
        # ... etc
    ] + [[0, 0]] * 10)  # Fill rest with zeros
    
    keypoints_flipped = flip_test.swap_keypoint_coordinates(
        keypoints, image_width=256
    )
    
    print(f"\nOriginal keypoints (first 7):")
    print(keypoints[:7])
    print(f"\nFlipped keypoints (first 7):")
    print(keypoints_flipped[:7])
    
    # Verify left-right swap
    assert np.allclose(keypoints[1], [90, 40]) and np.allclose(keypoints_flipped[2], [256-90, 40])
    assert np.allclose(keypoints[2], [110, 40]) and np.allclose(keypoints_flipped[1], [256-110, 40])
    
    print("\n✅ Flip test TTA working correctly!")
    
    # Test multi-scale
    print("\nTesting Multi-Scale TTA...")
    ms_test = MultiScaleTest(scales=[0.5, 1.0, 2.0], flip_test=False)
    pred_ms = ms_test(model, images)
    
    print(f"Multi-scale prediction shape: {pred_ms.shape}")
    print("✅ Multi-scale TTA working correctly!")


if __name__ == "__main__":
    test_flip_test()
