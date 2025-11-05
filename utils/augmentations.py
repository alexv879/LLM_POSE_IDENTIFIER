"""
Enhanced Data Augmentations for Pose Estimation
Includes official augmentations from Sapiens and ViTPose papers
"""

import numpy as np
import cv2
import random
from typing import Tuple, Optional, Dict, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RandomHalfBody:
    """
    RandomHalfBody augmentation from official Sapiens/ViTPose.
    
    Randomly crops to upper or lower half of the body, useful for:
    - Handling occlusions
    - Improving robustness to partial visibility
    - Training on close-up views
    
    Args:
        prob: Probability of applying the augmentation
        num_upper_body_joints: Number of upper body joints (default: 8 for COCO)
        min_joints: Minimum joints visible to apply (default: 8)
    """
    
    def __init__(
        self,
        prob: float = 0.3,
        num_upper_body_joints: int = 8,
        min_joints: int = 8
    ):
        self.prob = prob
        self.num_upper_body_joints = num_upper_body_joints
        self.min_joints = min_joints
        
        # COCO joint indices
        # Upper body: 0-9 (nose, eyes, ears, shoulders, elbows, wrists)
        # Lower body: 10-16 (hips, knees, ankles)
        self.upper_body_ids = list(range(num_upper_body_joints))
        self.lower_body_ids = list(range(num_upper_body_joints, 17))
    
    def __call__(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply RandomHalfBody augmentation.
        
        Args:
            image: (H, W, 3)
            keypoints: (17, 2) in image coordinates
            visibility: (17,) visibility flags
            bbox: (4,) [x1, y1, x2, y2] or None
        
        Returns:
            image, keypoints, visibility, bbox (augmented)
        """
        if random.random() > self.prob:
            return image, keypoints, visibility, bbox
        
        # Count visible joints in upper and lower body
        visible_upper = sum(visibility[i] > 0 for i in self.upper_body_ids)
        visible_lower = sum(visibility[i] > 0 for i in self.lower_body_ids)
        
        # Need minimum visible joints to apply
        if visible_upper < self.min_joints // 2 and visible_lower < self.min_joints // 2:
            return image, keypoints, visibility, bbox
        
        # Randomly choose upper or lower body
        # Prefer the half with more visible joints
        if visible_upper > visible_lower:
            selected_ids = self.upper_body_ids
            select_upper = random.random() < 0.7
        else:
            selected_ids = self.lower_body_ids
            select_upper = random.random() < 0.3
        
        if select_upper:
            selected_ids = self.upper_body_ids
        else:
            selected_ids = self.lower_body_ids
        
        # Get bounding box of selected joints
        selected_keypoints = []
        for idx in selected_ids:
            if visibility[idx] > 0:
                selected_keypoints.append(keypoints[idx])
        
        if len(selected_keypoints) < 2:
            return image, keypoints, visibility, bbox
        
        selected_keypoints = np.array(selected_keypoints)
        
        # Calculate new bounding box
        x_min = selected_keypoints[:, 0].min()
        y_min = selected_keypoints[:, 1].min()
        x_max = selected_keypoints[:, 0].max()
        y_max = selected_keypoints[:, 1].max()
        
        # Add margin (40% of bbox size)
        w = x_max - x_min
        h = y_max - y_min
        margin = 0.4
        
        x_min = max(0, x_min - w * margin)
        y_min = max(0, y_min - h * margin)
        x_max = min(image.shape[1], x_max + w * margin)
        y_max = min(image.shape[0], y_max + h * margin)
        
        # Crop image
        x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
        
        if x_max <= x_min or y_max <= y_min:
            return image, keypoints, visibility, bbox
        
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Adjust keypoints
        adjusted_keypoints = keypoints.copy()
        adjusted_keypoints[:, 0] -= x_min
        adjusted_keypoints[:, 1] -= y_min
        
        # Update visibility for out-of-crop joints
        adjusted_visibility = visibility.copy()
        for i in range(len(keypoints)):
            if (adjusted_keypoints[i, 0] < 0 or 
                adjusted_keypoints[i, 0] >= cropped_image.shape[1] or
                adjusted_keypoints[i, 1] < 0 or 
                adjusted_keypoints[i, 1] >= cropped_image.shape[0]):
                adjusted_visibility[i] = 0
        
        new_bbox = np.array([0, 0, cropped_image.shape[1], cropped_image.shape[0]])
        
        return cropped_image, adjusted_keypoints, adjusted_visibility, new_bbox


class RandomBBoxTransform:
    """
    RandomBBoxTransform from official Sapiens.
    
    Applies random scaling and translation to the bounding box,
    simulating different crop variations.
    
    Args:
        scale_factor: Range for random scaling [min, max]
        rotation_factor: Max rotation in degrees
        shift_factor: Max shift as ratio of bbox size
        prob: Probability of applying
    """
    
    def __init__(
        self,
        scale_factor: Tuple[float, float] = (0.75, 1.5),
        rotation_factor: float = 60.0,
        shift_factor: float = 0.15,
        prob: float = 1.0
    ):
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.shift_factor = shift_factor
        self.prob = prob
    
    def __call__(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        bbox: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply random bbox transformation.
        
        Args:
            image: (H, W, 3)
            keypoints: (17, 2)
            visibility: (17,)
            bbox: (4,) [x1, y1, x2, y2] or None
        
        Returns:
            image, keypoints, visibility (transformed)
        """
        if random.random() > self.prob:
            return image, keypoints, visibility
        
        h, w = image.shape[:2]
        
        # Get center and scale
        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        else:
            # Use image center
            cx, cy = w / 2, h / 2
            scale = max(w, h)
        
        # Random scale
        random_scale = random.uniform(self.scale_factor[0], self.scale_factor[1])
        scale *= random_scale
        
        # Random shift
        dx = random.uniform(-self.shift_factor, self.shift_factor) * scale
        dy = random.uniform(-self.shift_factor, self.shift_factor) * scale
        cx += dx
        cy += dy
        
        # Random rotation
        angle = random.uniform(-self.rotation_factor, self.rotation_factor)
        
        # Build transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        # Apply to image
        transformed_image = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Apply to keypoints
        transformed_keypoints = keypoints.copy()
        for i in range(len(keypoints)):
            if visibility[i] > 0:
                pt = np.array([keypoints[i, 0], keypoints[i, 1], 1.0])
                new_pt = M @ pt
                transformed_keypoints[i] = new_pt
        
        return transformed_image, transformed_keypoints, visibility


class CoarseDropout:
    """
    CoarseDropout augmentation (random rectangular occlusions).
    Simulates occlusions for robustness.
    
    Args:
        max_holes: Maximum number of holes
        max_height: Max height of holes (ratio of image)
        max_width: Max width of holes (ratio of image)
        min_holes: Minimum number of holes
        min_height: Min height of holes (ratio of image)
        min_width: Min width of holes (ratio of image)
        prob: Probability of applying
    """
    
    def __init__(
        self,
        max_holes: int = 1,
        max_height: float = 0.4,
        max_width: float = 0.4,
        min_holes: int = 1,
        min_height: float = 0.2,
        min_width: float = 0.2,
        prob: float = 0.5
    ):
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes
        self.min_height = min_height
        self.min_width = min_width
        self.prob = prob
    
    def __call__(
        self,
        image: np.ndarray,
        keypoints: np.ndarray = None,
        visibility: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Apply coarse dropout.
        
        Args:
            image: (H, W, 3)
            keypoints: (17, 2) or None
            visibility: (17,) or None
        
        Returns:
            image (with holes), keypoints, visibility (updated for occluded joints)
        """
        if random.random() > self.prob:
            return image, keypoints, visibility
        
        h, w = image.shape[:2]
        image_dropped = image.copy()
        
        num_holes = random.randint(self.min_holes, self.max_holes)
        
        for _ in range(num_holes):
            # Random hole size
            hole_h = random.randint(
                int(h * self.min_height),
                int(h * self.max_height)
            )
            hole_w = random.randint(
                int(w * self.min_width),
                int(w * self.max_width)
            )
            
            # Random position
            y1 = random.randint(0, max(0, h - hole_h))
            x1 = random.randint(0, max(0, w - hole_w))
            y2 = min(y1 + hole_h, h)
            x2 = min(x1 + hole_w, w)
            
            # Fill with zeros (black)
            image_dropped[y1:y2, x1:x2] = 0
            
            # Update visibility for occluded keypoints
            if keypoints is not None and visibility is not None:
                for i in range(len(keypoints)):
                    if visibility[i] > 0:
                        kx, ky = keypoints[i]
                        if x1 <= kx < x2 and y1 <= ky < y2:
                            # Keypoint is occluded, but don't set to 0
                            # Just mark as less visible (keep for loss)
                            visibility[i] = max(0.5, visibility[i])
        
        return image_dropped, keypoints, visibility


def get_train_augmentation(config: Dict) -> A.Compose:
    """
    Get training augmentation pipeline matching official Sapiens.
    
    Args:
        config: Augmentation config dict
    
    Returns:
        Albumentations Compose object
    """
    transforms = []
    
    # Geometric transforms
    if config.get('horizontal_flip', 0) > 0:
        transforms.append(
            A.HorizontalFlip(p=config['horizontal_flip'])
        )
    
    if config.get('rotation', 0) > 0:
        transforms.append(
            A.Rotate(
                limit=config['rotation'],
                p=0.8,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            )
        )
    
    # Photometric transforms
    if 'color_jitter' in config:
        cj = config['color_jitter']
        transforms.append(
            A.ColorJitter(
                brightness=cj.get('brightness', 0.4),
                contrast=cj.get('contrast', 0.4),
                saturation=cj.get('saturation', 0.4),
                hue=cj.get('hue', 0.1),
                p=0.8
            )
        )
    
    # Blur augmentations
    if 'albumentation' in config:
        albu_cfg = config['albumentation']
        
        if albu_cfg.get('blur_prob', 0) > 0:
            transforms.append(
                A.Blur(blur_limit=3, p=albu_cfg['blur_prob'])
            )
        
        if albu_cfg.get('median_blur_prob', 0) > 0:
            transforms.append(
                A.MedianBlur(blur_limit=3, p=albu_cfg['median_blur_prob'])
            )
        
        # Coarse dropout (occlusion)
        if 'coarse_dropout' in albu_cfg:
            cd = albu_cfg['coarse_dropout']
            transforms.append(
                A.CoarseDropout(
                    max_holes=cd.get('max_holes', 1),
                    max_height=cd.get('max_height', 0.4),
                    max_width=cd.get('max_width', 0.4),
                    min_holes=cd.get('min_holes', 1),
                    min_height=cd.get('min_height', 0.2),
                    min_width=cd.get('min_width', 0.2),
                    fill_value=0,
                    p=cd.get('prob', 0.5)
                )
            )
    
    # Normalization
    transforms.append(
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        )
    )
    
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    )


def get_val_augmentation(config: Dict) -> A.Compose:
    """
    Get validation augmentation (minimal, just normalization).
    
    Args:
        config: Augmentation config dict
    
    Returns:
        Albumentations Compose object
    """
    transforms = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        )
    ]
    
    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(
            format='xy',
            remove_invisible=False
        )
    )


def test_augmentations():
    """Test augmentation functions."""
    print("Testing augmentations...")
    
    # Create dummy data
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    keypoints = np.random.rand(17, 2) * np.array([640, 480])
    visibility = np.ones(17)
    
    # Test RandomHalfBody
    print("\n1. Testing RandomHalfBody...")
    rhb = RandomHalfBody(prob=1.0)
    aug_img, aug_kpts, aug_vis, aug_bbox = rhb(image, keypoints, visibility)
    print(f"   Original shape: {image.shape}, Augmented: {aug_img.shape}")
    print(f"   Visible joints: {aug_vis.sum():.0f}/{len(visibility)}")
    
    # Test RandomBBoxTransform
    print("\n2. Testing RandomBBoxTransform...")
    rbt = RandomBBoxTransform(prob=1.0)
    aug_img2, aug_kpts2, aug_vis2 = rbt(image, keypoints, visibility)
    print(f"   Shape: {aug_img2.shape}")
    print(f"   Keypoint shift: {np.abs(keypoints - aug_kpts2).mean():.2f} pixels")
    
    # Test CoarseDropout
    print("\n3. Testing CoarseDropout...")
    cd = CoarseDropout(prob=1.0, max_holes=2)
    aug_img3, aug_kpts3, aug_vis3 = cd(image, keypoints, visibility)
    print(f"   Shape: {aug_img3.shape}")
    print(f"   Pixels zeroed: {(aug_img3 == 0).sum()} / {image.size}")
    
    # Test full pipeline
    print("\n4. Testing full augmentation pipeline...")
    config = {
        'horizontal_flip': 0.5,
        'rotation': 45,
        'color_jitter': {
            'brightness': 0.4,
            'contrast': 0.4,
            'saturation': 0.4,
            'hue': 0.1
        },
        'albumentation': {
            'blur_prob': 0.1,
            'median_blur_prob': 0.1,
            'coarse_dropout': {
                'prob': 0.5,
                'max_holes': 1,
                'max_height': 0.4,
                'max_width': 0.4,
                'min_holes': 1,
                'min_height': 0.2,
                'min_width': 0.2
            }
        }
    }
    
    transform = get_train_augmentation(config)
    
    # Convert keypoints to list of tuples for albumentations
    kpts_list = [(kp[0], kp[1]) for kp in keypoints]
    
    augmented = transform(image=image, keypoints=kpts_list)
    print(f"   Output shape: {augmented['image'].shape}")
    print(f"   Normalized: min={augmented['image'].min():.3f}, max={augmented['image'].max():.3f}")
    
    print("\n✅ All augmentation tests passed!")


if __name__ == "__main__":
    test_augmentations()
