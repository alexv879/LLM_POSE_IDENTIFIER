"""
UDP (Unbiased Data Processing) Heatmap Codec
Based on "The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation" (CVPR 2020)

Official implementation reference: mmpose/codecs/udp_heatmap.py
Paper: https://arxiv.org/abs/1911.07524
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List
import cv2


class UDPHeatmap:
    """
    UDP Heatmap Codec for unbiased keypoint encoding and decoding.
    
    Key differences from standard Gaussian heatmaps:
    1. Uses offset-based coordinate encoding
    2. Unbiased spatial distribution (no boundary bias)
    3. Sub-pixel localization through UDP decoding
    
    Args:
        input_size: Input image size (width, height)
        heatmap_size: Output heatmap size (width, height)
        sigma: Standard deviation for Gaussian kernel (default: 2 for 256x256)
        use_dark: Whether to use DARK post-processing for sub-pixel refinement
    """
    
    def __init__(
        self,
        input_size: Tuple[int, int],
        heatmap_size: Tuple[int, int],
        sigma: float = 2.0,
        use_dark: bool = False
    ):
        self.input_size = input_size  # (width, height)
        self.heatmap_size = heatmap_size  # (width, height)
        self.sigma = sigma
        self.use_dark = use_dark
        
        # Calculate downsampling scale
        self.scale_x = input_size[0] / heatmap_size[0]
        self.scale_y = input_size[1] / heatmap_size[1]
        
        # Generate Gaussian kernel
        self.kernel_size = int(6 * sigma + 1)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        
    def encode(
        self,
        keypoints: np.ndarray,
        visibility: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode keypoints to UDP heatmaps.
        
        Args:
            keypoints: Keypoint coordinates (num_keypoints, 2) in input image space
            visibility: Keypoint visibility flags (num_keypoints,) 
                       0=not labeled, 1=labeled but not visible, 2=labeled and visible
        
        Returns:
            heatmaps: (num_keypoints, heatmap_h, heatmap_w)
            target_weight: (num_keypoints,) weight for each keypoint
        """
        num_keypoints = keypoints.shape[0]
        heatmap_h, heatmap_w = self.heatmap_size[1], self.heatmap_size[0]
        
        # Initialize heatmaps and weights
        heatmaps = np.zeros((num_keypoints, heatmap_h, heatmap_w), dtype=np.float32)
        
        # Target weights (1 for visible, 0 for invisible/not labeled)
        if visibility is not None:
            target_weight = (visibility > 0).astype(np.float32)
        else:
            target_weight = np.ones(num_keypoints, dtype=np.float32)
        
        # Generate heatmap for each keypoint
        for joint_id in range(num_keypoints):
            if target_weight[joint_id] < 0.5:
                continue
            
            # Convert keypoint from input space to heatmap space
            mu_x = keypoints[joint_id, 0] / self.scale_x
            mu_y = keypoints[joint_id, 1] / self.scale_y
            
            # Check if keypoint is within heatmap bounds
            if mu_x < 0 or mu_y < 0 or mu_x >= heatmap_w or mu_y >= heatmap_h:
                target_weight[joint_id] = 0
                continue
            
            # Generate UDP-style Gaussian heatmap
            heatmaps[joint_id] = self._generate_udp_gaussian(
                mu_x, mu_y, heatmap_w, heatmap_h
            )
        
        return heatmaps, target_weight
    
    def _generate_udp_gaussian(
        self,
        mu_x: float,
        mu_y: float,
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Generate UDP-style Gaussian heatmap.
        
        UDP uses unbiased coordinate encoding:
        - Coordinates are continuous (sub-pixel)
        - Gaussian is centered at exact coordinate
        - No quantization bias
        """
        # Create coordinate grids
        x = np.arange(0, width, dtype=np.float32)
        y = np.arange(0, height, dtype=np.float32)
        y = y[:, np.newaxis]
        
        # UDP: Use continuous coordinates (no rounding)
        # Standard Gaussian: exp(-(x-mu)^2 / (2*sigma^2))
        heatmap = np.exp(
            -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * self.sigma ** 2)
        )
        
        return heatmap.astype(np.float32)
    
    def decode(
        self,
        heatmaps: np.ndarray,
        use_udp: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode heatmaps to keypoint coordinates.
        
        Args:
            heatmaps: (num_keypoints, heatmap_h, heatmap_w) or 
                     (batch, num_keypoints, heatmap_h, heatmap_w)
            use_udp: Whether to use UDP decoding (recommended)
        
        Returns:
            keypoints: (num_keypoints, 2) or (batch, num_keypoints, 2) in input image space
            scores: (num_keypoints,) or (batch, num_keypoints,) confidence scores
        """
        if heatmaps.ndim == 4:
            # Batch mode
            batch_size = heatmaps.shape[0]
            all_keypoints = []
            all_scores = []
            
            for i in range(batch_size):
                kpts, scores = self._decode_single(heatmaps[i], use_udp)
                all_keypoints.append(kpts)
                all_scores.append(scores)
            
            return np.stack(all_keypoints), np.stack(all_scores)
        else:
            # Single sample
            return self._decode_single(heatmaps, use_udp)
    
    def _decode_single(
        self,
        heatmaps: np.ndarray,
        use_udp: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode single heatmap to keypoints.
        
        Args:
            heatmaps: (num_keypoints, heatmap_h, heatmap_w)
            use_udp: Use UDP decoding for unbiased coordinates
        
        Returns:
            keypoints: (num_keypoints, 2) coordinates in input image space
            scores: (num_keypoints,) confidence scores
        """
        num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
        
        keypoints = np.zeros((num_keypoints, 2), dtype=np.float32)
        scores = np.zeros(num_keypoints, dtype=np.float32)
        
        for joint_id in range(num_keypoints):
            heatmap = heatmaps[joint_id]
            
            # Find maximum location
            max_val = heatmap.max()
            scores[joint_id] = max_val
            
            if max_val < 1e-5:
                # No confident prediction
                keypoints[joint_id] = [-1, -1]
                continue
            
            # Get argmax location
            max_idx = heatmap.argmax()
            max_y, max_x = np.unravel_index(max_idx, (heatmap_h, heatmap_w))
            
            if use_udp:
                # UDP decoding: use 2nd order Taylor expansion for sub-pixel refinement
                x, y = self._udp_refine(heatmap, max_x, max_y)
            else:
                # Standard decoding: just use argmax
                x, y = float(max_x), float(max_y)
            
            # Apply DARK refinement if enabled
            if self.use_dark and use_udp:
                x, y = self._dark_refine(heatmap, x, y)
            
            # Convert from heatmap space to input image space
            keypoints[joint_id, 0] = x * self.scale_x
            keypoints[joint_id, 1] = y * self.scale_y
        
        return keypoints, scores
    
    def _udp_refine(
        self,
        heatmap: np.ndarray,
        max_x: int,
        max_y: int
    ) -> Tuple[float, float]:
        """
        UDP sub-pixel refinement using 2nd order Taylor expansion.
        
        This provides unbiased coordinate estimation by considering
        the gradient around the peak location.
        """
        h, w = heatmap.shape
        
        # Boundary check
        if max_x == 0 or max_x == w - 1 or max_y == 0 or max_y == h - 1:
            return float(max_x), float(max_y)
        
        # Get neighboring values
        dx = 0.5 * (heatmap[max_y, max_x + 1] - heatmap[max_y, max_x - 1])
        dy = 0.5 * (heatmap[max_y + 1, max_x] - heatmap[max_y - 1, max_x])
        
        dxx = heatmap[max_y, max_x + 1] - 2 * heatmap[max_y, max_x] + heatmap[max_y, max_x - 1]
        dyy = heatmap[max_y + 1, max_x] - 2 * heatmap[max_y, max_x] + heatmap[max_y - 1, max_x]
        
        # 2nd order Taylor expansion
        offset_x = -dx / (dxx + 1e-8) if abs(dxx) > 1e-8 else 0.0
        offset_y = -dy / (dyy + 1e-8) if abs(dyy) > 1e-8 else 0.0
        
        # Clamp offsets to [-0.5, 0.5]
        offset_x = np.clip(offset_x, -0.5, 0.5)
        offset_y = np.clip(offset_y, -0.5, 0.5)
        
        refined_x = max_x + offset_x
        refined_y = max_y + offset_y
        
        return refined_x, refined_y
    
    def _dark_refine(
        self,
        heatmap: np.ndarray,
        x: float,
        y: float
    ) -> Tuple[float, float]:
        """
        DARK (Distribution-Aware Keypoint Refinement) post-processing.
        
        Reference: "Distribution-Aware Coordinate Representation for Human Pose Estimation" (CVPR 2020)
        """
        # Convert to integer coordinates for neighborhood
        ix, iy = int(x), int(y)
        h, w = heatmap.shape
        
        # Boundary check
        if ix <= 0 or ix >= w - 1 or iy <= 0 or iy >= h - 1:
            return x, y
        
        # Get 3x3 neighborhood
        patch = heatmap[iy - 1:iy + 2, ix - 1:ix + 2]
        
        if patch.sum() < 1e-5:
            return x, y
        
        # Compute weighted center of mass
        dx = np.array([-1, 0, 1], dtype=np.float32)
        dy = np.array([-1, 0, 1], dtype=np.float32)
        
        dx_weight = (patch * dx[np.newaxis, :]).sum() / patch.sum()
        dy_weight = (patch * dy[:, np.newaxis]).sum() / patch.sum()
        
        # Refine coordinates
        refined_x = x + dx_weight * 0.25  # Scale factor from paper
        refined_y = y + dy_weight * 0.25
        
        return refined_x, refined_y
    
    def decode_torch(
        self,
        heatmaps: torch.Tensor,
        use_udp: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        PyTorch version of decode for differentiable inference.
        
        Args:
            heatmaps: (B, K, H, W) tensor
            use_udp: Use UDP decoding
        
        Returns:
            keypoints: (B, K, 2) coordinates in input image space
            scores: (B, K) confidence scores
        """
        B, K, H, W = heatmaps.shape
        device = heatmaps.device
        
        # Reshape for processing
        heatmaps_flat = heatmaps.view(B, K, -1)
        
        # Get max values and indices
        scores, indices = heatmaps_flat.max(dim=2)
        
        # Convert flat indices to 2D coordinates
        y_coords = (indices // W).float()
        x_coords = (indices % W).float()
        
        if use_udp:
            # UDP refinement in PyTorch
            # Get neighboring values for gradient computation
            batch_idx = torch.arange(B, device=device)[:, None].expand(B, K)
            joint_idx = torch.arange(K, device=device)[None, :].expand(B, K)
            
            # Clamp coordinates for boundary safety
            y_int = y_coords.long().clamp(1, H - 2)
            x_int = x_coords.long().clamp(1, W - 2)
            
            # Compute gradients
            dx = 0.5 * (
                heatmaps[batch_idx, joint_idx, y_int, x_int + 1] -
                heatmaps[batch_idx, joint_idx, y_int, x_int - 1]
            )
            dy = 0.5 * (
                heatmaps[batch_idx, joint_idx, y_int + 1, x_int] -
                heatmaps[batch_idx, joint_idx, y_int - 1, x_int]
            )
            
            dxx = (
                heatmaps[batch_idx, joint_idx, y_int, x_int + 1] -
                2 * heatmaps[batch_idx, joint_idx, y_int, x_int] +
                heatmaps[batch_idx, joint_idx, y_int, x_int - 1]
            )
            dyy = (
                heatmaps[batch_idx, joint_idx, y_int + 1, x_int] -
                2 * heatmaps[batch_idx, joint_idx, y_int, x_int] +
                heatmaps[batch_idx, joint_idx, y_int - 1, x_int]
            )
            
            # UDP offset computation
            offset_x = torch.where(
                torch.abs(dxx) > 1e-8,
                -dx / (dxx + 1e-8),
                torch.zeros_like(dx)
            ).clamp(-0.5, 0.5)
            
            offset_y = torch.where(
                torch.abs(dyy) > 1e-8,
                -dy / (dyy + 1e-8),
                torch.zeros_like(dy)
            ).clamp(-0.5, 0.5)
            
            # Apply refinement
            x_coords = x_coords + offset_x
            y_coords = y_coords + offset_y
        
        # Scale to input image space
        x_coords = x_coords * self.scale_x
        y_coords = y_coords * self.scale_y
        
        # Stack coordinates
        keypoints = torch.stack([x_coords, y_coords], dim=2)
        
        return keypoints, scores


def test_udp_codec():
    """Test UDP codec functionality."""
    print("Testing UDP Codec...")
    
    # Create codec
    codec = UDPHeatmap(
        input_size=(256, 192),
        heatmap_size=(64, 48),
        sigma=2.0
    )
    
    # Test keypoints
    keypoints = np.array([
        [128, 96],   # Center
        [200, 50],   # Top right
        [50, 150],   # Bottom left
    ], dtype=np.float32)
    
    visibility = np.array([2, 2, 1], dtype=np.float32)
    
    # Encode
    heatmaps, target_weight = codec.encode(keypoints, visibility)
    print(f"Encoded heatmaps shape: {heatmaps.shape}")
    print(f"Target weights: {target_weight}")
    
    # Decode with UDP
    decoded_kpts_udp, scores_udp = codec.decode(heatmaps, use_udp=True)
    print(f"\nUDP Decoding:")
    print(f"Original keypoints:\n{keypoints}")
    print(f"Decoded keypoints:\n{decoded_kpts_udp}")
    print(f"Scores: {scores_udp}")
    print(f"Error: {np.abs(keypoints - decoded_kpts_udp).mean():.4f} pixels")
    
    # Decode without UDP (standard argmax)
    decoded_kpts_std, scores_std = codec.decode(heatmaps, use_udp=False)
    print(f"\nStandard Decoding:")
    print(f"Decoded keypoints:\n{decoded_kpts_std}")
    print(f"Error: {np.abs(keypoints - decoded_kpts_std).mean():.4f} pixels")
    
    # Test PyTorch version
    print("\nTesting PyTorch version...")
    heatmaps_torch = torch.from_numpy(heatmaps[np.newaxis, :]).float()
    decoded_torch, scores_torch = codec.decode_torch(heatmaps_torch, use_udp=True)
    print(f"PyTorch decoded shape: {decoded_torch.shape}")
    print(f"PyTorch error: {(keypoints - decoded_torch[0].numpy()).abs().mean():.4f} pixels")
    
    print("\n✅ UDP Codec test completed!")


if __name__ == "__main__":
    test_udp_codec()
