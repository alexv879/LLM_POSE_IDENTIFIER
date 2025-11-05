"""
COCO Pose Dataset Loader
Implements complete COCO format dataset loading with support for
17 keypoint annotations and proper train/val splitting.

Based on research from:
- COCO Keypoint Format (Lin et al., 2014)
- Best practices from Sapiens-2B, ViTPose, HRNet papers
- UDP (Unbiased Data Processing) from CVPR 2020
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import defaultdict
from PIL import Image
import logging
from typing import Dict, List, Optional, Tuple, Any
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import UDP codec
from utils.udp_codec import UDPHeatmap

logger = logging.getLogger(__name__)


class COCOPoseDataset(Dataset):
    """
    Load images and keypoints from COCO format JSON annotation.
    
    COCO Format Structure:
      - images: List of image info (id, file_name, height, width)
      - annotations: List of annotations (image_id, keypoints [51 values], bbox)
      - categories: Category info (keypoint names, skeleton connections)
    
    Keypoints format: [x1,y1,v1, x2,y2,v2, ..., x17,y17,v17]
      where v (visibility) = 0 (not labeled), 1 (labeled but hidden), 2 (visible)
    
    17 COCO Keypoints:
      0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
      5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
      9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
      13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        split: str = 'train',
        train_ratio: float = 0.8,
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (256, 192),  # (width, height)
        heatmap_size: Tuple[int, int] = (64, 48),  # (width, height)
        sigma: float = 2.0,  # Gaussian sigma for heatmap generation
        use_udp: bool = True,  # Use UDP encoding (recommended)
    ):
        """
        Args:
            image_dir: Path to images folder
            annotations_file: Path to COCO JSON annotations
            split: 'train' or 'val'
            train_ratio: Train/val split ratio (default 0.8)
            transform: Optional albumentations augmentations
            image_size: Target image size (width, height)
            heatmap_size: Target heatmap size (width, height)
            sigma: Gaussian sigma for heatmap generation
            use_udp: Whether to use UDP (Unbiased Data Processing) encoding
        """
        self.image_dir = Path(image_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.use_udp = use_udp
        
        # Initialize UDP codec if enabled
        if self.use_udp:
            self.udp_codec = UDPHeatmap(
                input_size=image_size,
                heatmap_size=heatmap_size,
                sigma=sigma
            )
            logger.info("Using UDP heatmap encoding for unbiased data processing")
        else:
            self.udp_codec = None
            logger.info("Using standard Gaussian heatmap encoding")
        self.sigma = sigma
        
        # Load COCO annotations
        logger.info(f"Loading annotations from {annotations_file}")
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data.get('images', [])
        self.annotations = self.coco_data.get('annotations', [])
        
        # Get keypoint names and skeleton
        self.categories = self.coco_data.get('categories', [{}])[0]
        self.keypoint_names = self.categories.get('keypoints', [])
        self.skeleton = self.categories.get('skeleton', [])
        
        # Map image_id → annotations (can have multiple people per image)
        self.image_annotations = defaultdict(list)
        for ann in self.annotations:
            self.image_annotations[ann['image_id']].append(ann)
        
        # Train/val split (fixed seed for reproducibility)
        np.random.seed(42)
        indices = np.arange(len(self.images))
        np.random.shuffle(indices)
        
        split_point = int(len(indices) * train_ratio)
        
        if split == 'train':
            self.indices = indices[:split_point]
        elif split == 'val':
            self.indices = indices[split_point:]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'val'")
        
        logger.info(f"Dataset '{split}': {len(self.indices)} images loaded")
        logger.info(f"Total annotations: {len(self.annotations)}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def _generate_heatmap(
        self,
        keypoints: np.ndarray,
        heatmap_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate heatmaps from keypoint coordinates.
        Uses UDP encoding if enabled, otherwise standard Gaussian.
        
        Args:
            keypoints: (17, 3) array with [x, y, visibility]
            heatmap_size: (width, height) of output heatmap
        
        Returns:
            heatmaps: (17, height, width) array with heatmaps
            target_weight: (17,) array with per-keypoint weights for loss
        """
        num_joints = keypoints.shape[0]
        
        if self.use_udp and self.udp_codec is not None:
            # Use UDP encoding
            # Extract coordinates and visibility
            coords = keypoints[:, :2]  # (17, 2)
            visibility = keypoints[:, 2]  # (17,)
            
            # Encode with UDP
            heatmaps, target_weight = self.udp_codec.encode(
                coords, visibility
            )
            
        else:
            # Use standard Gaussian encoding
            heatmaps = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
            target_weight = np.ones(num_joints, dtype=np.float32)
            
            # Create meshgrid for Gaussian computation
            x = np.arange(0, heatmap_size[0], 1, dtype=np.float32)
            y = np.arange(0, heatmap_size[1], 1, dtype=np.float32)
            yy, xx = np.meshgrid(y, x, indexing='ij')
            
            for idx in range(num_joints):
                x_coord, y_coord, visibility = keypoints[idx]
                
                # Skip if keypoint is not visible or not labeled
                if visibility == 0:
                    target_weight[idx] = 0.0
                    continue
                
                # Scale coordinates to heatmap size
                x_hm = x_coord * heatmap_size[0] / self.image_size[0]
                y_hm = y_coord * heatmap_size[1] / self.image_size[1]
                
                # Check if coordinate is within heatmap bounds
                if x_hm < 0 or x_hm >= heatmap_size[0] or y_hm < 0 or y_hm >= heatmap_size[1]:
                    target_weight[idx] = 0.0
                    continue
                
                # Generate 2D Gaussian: exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))
                gaussian = np.exp(-((xx - x_hm) ** 2 + (yy - y_hm) ** 2) / (2 * self.sigma ** 2))
                
                # Apply visibility weighting
                if visibility == 1:  # Labeled but not visible (occluded)
                    target_weight[idx] = 0.5  # Reduce weight for occluded keypoints
                
                heatmaps[idx] = gaussian
        
        return heatmaps, target_weight
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
              - 'image': (3, H, W) torch tensor, normalized
              - 'keypoints': (17, 3) torch tensor with [x, y, visibility]
              - 'heatmaps': (17, H_hm, W_hm) torch tensor with Gaussian heatmaps
              - 'image_id': int (for tracking)
              - 'image_name': str (filename)
              - 'bbox': (4,) torch tensor with [x, y, w, h]
        """
        img_idx = self.indices[idx]
        img_info = self.images[img_idx]
        
        # Load image
        img_path = self.image_dir / img_info['file_name']
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            logger.error(f"Failed to load image {img_path}: {e}")
            raise
        
        # Get annotations (use person with largest bounding box)
        annotations = self.image_annotations[img_info['id']]
        
        if annotations:
            # Select largest person (by bbox area)
            best_ann = max(annotations, key=lambda x: x['bbox'][2] * x['bbox'][3])
        else:
            # No annotations for this image - create dummy
            best_ann = {
                'keypoints': [0] * 51,
                'bbox': [0, 0, img_info['width'], img_info['height']]
            }
        
        # Extract 17 COCO keypoints: [x1,y1,v1, x2,y2,v2, ..., x17,y17,v17]
        keypoints_raw = np.array(
            best_ann.get('keypoints', [0] * 51),
            dtype=np.float32
        ).reshape(17, 3)
        
        bbox = np.array(best_ann.get('bbox', [0, 0, image.shape[1], image.shape[0]]),
                       dtype=np.float32)
        
        # Apply augmentations if provided
        if self.transform:
            # Convert keypoints to format expected by albumentations
            kpts_xy = keypoints_raw[:, :2].tolist()
            
            try:
                aug_result = self.transform(image=image, keypoints=kpts_xy)
                image = aug_result['image']
                kpts_xy_aug = np.array(aug_result['keypoints'], dtype=np.float32)
                
                # Reconstruct with visibility information
                keypoints = np.concatenate([kpts_xy_aug, keypoints_raw[:, 2:3]], axis=1)
            except Exception as e:
                logger.warning(f"Augmentation failed for image {img_info['file_name']}: {e}")
                # Fall back to simple resize
                image = np.array(Image.fromarray(image).resize(self.image_size))
                keypoints = keypoints_raw
        else:
            # Simple resize without augmentation
            image = np.array(Image.fromarray(image).resize(self.image_size))
            keypoints = keypoints_raw
        
        # Normalize image to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        # Generate heatmaps (returns heatmaps and target_weight)
        heatmaps, target_weight = self._generate_heatmap(keypoints, self.heatmap_size)
        
        # Convert to torch tensors
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # CHW format
        
        keypoints = torch.from_numpy(keypoints).float()
        heatmaps = torch.from_numpy(heatmaps).float()
        target_weight = torch.from_numpy(target_weight).float()
        bbox = torch.from_numpy(bbox).float()
        
        return {
            'image': image,
            'keypoints': keypoints,
            'heatmaps': heatmaps,
            'target_weight': target_weight,  # ADDED: For weighted loss
            'image_id': img_info['id'],
            'image_name': img_info['file_name'],
            'bbox': bbox
        }


def get_train_transforms(image_size: Tuple[int, int] = (256, 192)) -> A.Compose:
    """
    Training augmentations following best practices from ViTPose and Sapiens.
    
    Args:
        image_size: Target image size (width, height)
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(height=image_size[1], width=image_size[0]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=45,
            p=0.7,
            border_mode=0
        ),
        A.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet statistics
            std=[0.229, 0.224, 0.225]
        ),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def get_val_transforms(image_size: Tuple[int, int] = (256, 192)) -> A.Compose:
    """
    Validation transforms (no augmentation, only resize and normalize).
    
    Args:
        image_size: Target image size (width, height)
    
    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(height=image_size[1], width=image_size[0]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batching of variable-sized data.
    
    Args:
        batch: List of dataset items
    
    Returns:
        Batched dictionary
    """
    images = torch.stack([item['image'] for item in batch])
    keypoints = torch.stack([item['keypoints'] for item in batch])
    heatmaps = torch.stack([item['heatmaps'] for item in batch])
    bboxes = torch.stack([item['bbox'] for item in batch])
    image_ids = [item['image_id'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    return {
        'image': images,
        'keypoints': keypoints,
        'heatmaps': heatmaps,
        'bbox': bboxes,
        'image_id': image_ids,
        'image_name': image_names
    }


if __name__ == "__main__":
    # Test dataset loader
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    dataset = COCOPoseDataset(
        image_dir="data/raw",
        annotations_file="data/annotations/train_keypoints.json",
        split='train',
        transform=get_train_transforms()
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading one sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Keypoints shape: {sample['keypoints'].shape}")
    print(f"Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"Image ID: {sample['image_id']}")
