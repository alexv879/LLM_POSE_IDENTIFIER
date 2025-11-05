"""
Visualization Utilities for Pose Estimation
Generate annotated images with keypoints, skeletons, and confidence scores
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# COCO Skeleton connections (0-indexed)
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6]
]

# COCO Keypoint names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Color palette for keypoints
KEYPOINT_COLORS = [
    [255, 0, 0],     # nose - red
    [255, 85, 0],    # left_eye - orange
    [255, 170, 0],   # right_eye - yellow-orange
    [255, 255, 0],   # left_ear - yellow
    [170, 255, 0],   # right_ear - yellow-green
    [85, 255, 0],    # left_shoulder - green
    [0, 255, 0],     # right_shoulder - bright green
    [0, 255, 85],    # left_elbow - cyan-green
    [0, 255, 170],   # right_elbow - cyan
    [0, 255, 255],   # left_wrist - cyan
    [0, 170, 255],   # right_wrist - light blue
    [0, 85, 255],    # left_hip - blue
    [0, 0, 255],     # right_hip - dark blue
    [85, 0, 255],    # left_knee - purple-blue
    [170, 0, 255],   # right_knee - purple
    [255, 0, 255],   # left_ankle - magenta
    [255, 0, 170],   # right_ankle - pink
]


def visualize_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    confidence_threshold: float = 0.3,
    show_labels: bool = True,
    show_skeleton: bool = True,
    title: Optional[str] = None
) -> np.ndarray:
    """
    Visualize pose on image with keypoints and skeleton.
    
    Args:
        image: RGB image (H, W, 3)
        keypoints: (17, 3) array with [x, y, confidence]
        confidence_threshold: Minimum confidence to show keypoint
        show_labels: Whether to show keypoint labels
        show_skeleton: Whether to draw skeleton connections
        title: Optional title for the image
    
    Returns:
        Annotated image (H, W, 3)
    """
    # Create copy of image
    vis_img = image.copy()
    
    # Ensure image is uint8
    if vis_img.dtype != np.uint8:
        vis_img = (vis_img * 255).astype(np.uint8)
    
    # Draw skeleton first (so keypoints appear on top)
    if show_skeleton:
        for connection in COCO_SKELETON:
            pt1_idx, pt2_idx = connection
            
            # Check if both keypoints are confident
            if (keypoints[pt1_idx, 2] > confidence_threshold and 
                keypoints[pt2_idx, 2] > confidence_threshold):
                
                pt1 = tuple(keypoints[pt1_idx, :2].astype(int))
                pt2 = tuple(keypoints[pt2_idx, :2].astype(int))
                
                # Draw line
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            # Get color
            color = KEYPOINT_COLORS[i]
            
            # Draw circle
            center = (int(x), int(y))
            cv2.circle(vis_img, center, 5, color, -1)
            cv2.circle(vis_img, center, 6, (255, 255, 255), 1)
            
            # Draw label if requested
            if show_labels:
                label = f"{COCO_KEYPOINTS[i]}: {conf:.2f}"
                cv2.putText(
                    vis_img,
                    label,
                    (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
    
    # Add title if provided
    if title:
        cv2.putText(
            vis_img,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    
    return vis_img


def visualize_predictions(
    images: np.ndarray,
    predictions: np.ndarray,
    targets: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
    num_samples: int = 10
) -> None:
    """
    Visualize batch of predictions.
    
    Args:
        images: (B, 3, H, W) or (B, H, W, 3) tensor/array
        predictions: (B, 17, 3) keypoint predictions
        targets: Optional (B, 17, 3) ground truth keypoints
        save_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Convert to numpy if tensor
    if hasattr(images, 'cpu'):
        images = images.cpu().numpy()
    if hasattr(predictions, 'cpu'):
        predictions = predictions.cpu().numpy()
    if targets is not None and hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    
    # Handle channel ordering
    if images.shape[1] == 3:  # (B, C, H, W)
        images = images.transpose(0, 2, 3, 1)  # (B, H, W, C)
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
    
    # Visualize samples
    num_samples = min(num_samples, images.shape[0])
    
    for i in range(num_samples):
        img = images[i]
        pred = predictions[i]
        
        # Denormalize image if needed
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        
        # Create figure
        if targets is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Prediction
            vis_pred = visualize_pose(
                img.copy(),
                pred,
                show_labels=False,
                title=f"Prediction {i+1}"
            )
            axes[0].imshow(vis_pred)
            axes[0].set_title("Prediction")
            axes[0].axis('off')
            
            # Ground truth
            gt = targets[i]
            vis_gt = visualize_pose(
                img.copy(),
                gt,
                show_labels=False,
                title=f"Ground Truth {i+1}"
            )
            axes[1].imshow(vis_gt)
            axes[1].set_title("Ground Truth")
            axes[1].axis('off')
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            vis_pred = visualize_pose(
                img.copy(),
                pred,
                show_labels=True,
                title=f"Prediction {i+1}"
            )
            ax.imshow(vis_pred)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save or show
        if save_dir:
            save_file = save_path / f"prediction_{i+1:03d}.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization: {save_file}")
        else:
            plt.show()
        
        plt.close()


def create_comparison_grid(
    images: np.ndarray,
    predictions_list: List[np.ndarray],
    labels: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison grid of multiple prediction methods.
    
    Args:
        images: (B, H, W, 3) images
        predictions_list: List of (B, 17, 3) predictions from different methods
        labels: Labels for each method
        save_path: Path to save comparison image
    """
    num_methods = len(predictions_list)
    num_samples = min(5, images.shape[0])
    
    fig, axes = plt.subplots(
        num_samples, num_methods + 1,
        figsize=(4 * (num_methods + 1), 4 * num_samples)
    )
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row in range(num_samples):
        # Original image
        axes[row, 0].imshow(images[row])
        if row == 0:
            axes[row, 0].set_title("Original")
        axes[row, 0].axis('off')
        
        # Each method's prediction
        for col, (preds, label) in enumerate(zip(predictions_list, labels), 1):
            vis_img = visualize_pose(
                images[row].copy(),
                preds[row],
                show_labels=False,
                show_skeleton=True
            )
            axes[row, col].imshow(vis_img)
            if row == 0:
                axes[row, col].set_title(label)
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison grid: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_aps: List[float],
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_aps: List of validation APs per epoch
        save_path: Path to save plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # AP curve
    ax2.plot(epochs, val_aps, 'g-', linewidth=2, marker='o')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('AP (%)', fontsize=12)
    ax2.set_title('Validation AP', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Highlight best AP
    best_ap = max(val_aps)
    best_epoch = val_aps.index(best_ap) + 1
    ax2.axhline(y=best_ap, color='r', linestyle='--', alpha=0.5, label=f'Best: {best_ap:.2f}%')
    ax2.scatter([best_epoch], [best_ap], color='r', s=100, zorder=5)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves: {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    # Test visualization
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy data
    img = np.random.randint(0, 255, (256, 192, 3), dtype=np.uint8)
    keypoints = np.random.rand(17, 3)
    keypoints[:, :2] *= [192, 256]  # Scale to image size
    keypoints[:, 2] = np.random.rand(17) * 0.5 + 0.5  # Confidence [0.5, 1.0]
    
    # Visualize
    vis_img = visualize_pose(img, keypoints, show_labels=True)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_img)
    plt.title("Test Visualization")
    plt.axis('off')
    plt.show()
    
    print("Visualization test complete!")
