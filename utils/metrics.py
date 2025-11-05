"""
Evaluation Metrics for Pose Estimation
Implements COCO evaluation metrics including OKS and AP
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class COCOEvaluator:
    """
    COCO-style evaluation for pose estimation.
    
    Implements:
    - OKS (Object Keypoint Similarity)
    - AP (Average Precision) at various IoU thresholds
    - AR (Average Recall)
    """
    
    def __init__(self, oks_sigmas: List[float]):
        """
        Args:
            oks_sigmas: Per-joint standard deviations for OKS computation
                       (17 values for COCO keypoints)
        """
        self.oks_sigmas = np.array(oks_sigmas)
        self.oks_thresholds = np.linspace(0.5, 0.95, 10)  # 10 thresholds
        
        assert len(self.oks_sigmas) == 17, "Must provide 17 sigma values for COCO"
    
    def compute_oks(
        self,
        pred_keypoints: np.ndarray,
        gt_keypoints: np.ndarray,
        bbox_area: float = None
    ) -> float:
        """
        Compute Object Keypoint Similarity between prediction and ground truth.
        
        Formula:
        OKS = Σ_i [exp(-d_i^2 / (2*s^2*σ_i^2)) * v_i] / Σ_i v_i
        
        Args:
            pred_keypoints: (17, 3) array with [x, y, confidence]
            gt_keypoints: (17, 3) array with [x, y, visibility]
            bbox_area: Area of person's bounding box (for scale)
        
        Returns:
            oks: Object Keypoint Similarity score (0-1)
        """
        # Extract coordinates and visibility
        pred_xy = pred_keypoints[:, :2]
        gt_xy = gt_keypoints[:, :2]
        gt_vis = gt_keypoints[:, 2]
        
        # Compute distances
        distances = np.linalg.norm(pred_xy - gt_xy, axis=1)
        
        # Scale factor (use bbox area if provided, else use 1.0)
        if bbox_area is not None and bbox_area > 0:
            scale = np.sqrt(bbox_area)
        else:
            scale = 1.0
        
        # Compute OKS for each keypoint
        # exp(-d_i^2 / (2 * s^2 * σ_i^2))
        oks_per_joint = np.exp(
            -distances ** 2 / (2 * scale ** 2 * self.oks_sigmas ** 2)
        )
        
        # Weight by visibility (only consider labeled keypoints)
        visible_mask = gt_vis > 0
        
        if visible_mask.sum() == 0:
            return 0.0
        
        # OKS = weighted average over visible keypoints
        oks = (oks_per_joint * visible_mask).sum() / visible_mask.sum()
        
        return float(oks)
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        bbox_areas: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: (N, 17, 3) tensor with [x, y, confidence]
            targets: (N, 17, 3) tensor with [x, y, visibility]
            bbox_areas: (N,) tensor with bounding box areas
        
        Returns:
            Dictionary with metrics:
            - AP: Average Precision @ IoU=0.50:0.95
            - AP50: AP @ IoU=0.50
            - AP75: AP @ IoU=0.75
            - AR: Average Recall
        """
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        if bbox_areas is not None:
            bbox_areas = bbox_areas.cpu().numpy()
        else:
            bbox_areas = np.ones(predictions.shape[0])
        
        N = predictions.shape[0]
        
        # Compute OKS for each sample
        oks_scores = []
        for i in range(N):
            oks = self.compute_oks(
                predictions[i],
                targets[i],
                bbox_areas[i]
            )
            oks_scores.append(oks)
        
        oks_scores = np.array(oks_scores)
        
        # Compute AP at different thresholds
        ap_scores = []
        for threshold in self.oks_thresholds:
            # Count samples with OKS >= threshold
            correct = (oks_scores >= threshold).sum()
            ap = 100.0 * correct / N
            ap_scores.append(ap)
        
        # Average over all thresholds
        ap = np.mean(ap_scores)
        
        # AP at specific thresholds
        ap50 = ap_scores[0]  # Threshold 0.5
        ap75 = ap_scores[5]  # Threshold 0.75
        
        # Average Recall (average OKS score)
        ar = 100.0 * np.mean(oks_scores)
        
        return {
            'AP': ap,
            'AP50': ap50,
            'AP75': ap75,
            'AR': ar,
            'mean_OKS': np.mean(oks_scores)
        }
    
    def compute_per_joint_accuracy(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 2.0
    ) -> Dict[str, float]:
        """
        Compute per-joint accuracy (PCK - Percentage of Correct Keypoints).
        
        Args:
            predictions: (N, 17, 3) tensor
            targets: (N, 17, 3) tensor
            threshold: Distance threshold in pixels
        
        Returns:
            Dictionary with per-joint accuracies
        """
        predictions = predictions.cpu().numpy()
        targets = targets.cpu().numpy()
        
        pred_xy = predictions[:, :, :2]
        gt_xy = targets[:, :, :2]
        gt_vis = targets[:, :, 2]
        
        # Compute distances
        distances = np.linalg.norm(pred_xy - gt_xy, axis=2)
        
        # Count correct predictions (distance < threshold and visible)
        correct = (distances < threshold) & (gt_vis > 0)
        total = (gt_vis > 0).sum(axis=0)
        
        # Avoid division by zero
        total = np.maximum(total, 1)
        
        # Compute accuracy per joint
        joint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        per_joint_acc = {}
        for i, name in enumerate(joint_names):
            acc = 100.0 * correct[:, i].sum() / total[i]
            per_joint_acc[name] = acc
        
        # Overall accuracy
        per_joint_acc['overall'] = 100.0 * correct.sum() / total.sum()
        
        return per_joint_acc


def compute_pck(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    normalize_by: str = 'bbox'
) -> float:
    """
    Compute PCK (Percentage of Correct Keypoints).
    
    Args:
        predictions: (N, 17, 2) or (N, 17, 3) keypoints
        targets: (N, 17, 2) or (N, 17, 3) keypoints
        threshold: Threshold as fraction of normalization factor
        normalize_by: 'bbox', 'torso', or 'headsize'
    
    Returns:
        PCK score (0-100)
    """
    pred_xy = predictions[:, :, :2].cpu().numpy()
    gt_xy = targets[:, :, :2].cpu().numpy()
    gt_vis = targets[:, :, 2].cpu().numpy() if targets.shape[-1] == 3 else np.ones_like(pred_xy[:, :, 0])
    
    # Compute distances
    distances = np.linalg.norm(pred_xy - gt_xy, axis=2)
    
    # Compute normalization factor
    if normalize_by == 'bbox':
        # Use max dimension of bbox
        mins = gt_xy.min(axis=1)
        maxs = gt_xy.max(axis=1)
        norm = np.maximum(maxs[:, 0] - mins[:, 0], maxs[:, 1] - mins[:, 1])
    elif normalize_by == 'torso':
        # Use torso diagonal (shoulders to hips)
        # Shoulders: indices 5, 6; Hips: indices 11, 12
        left_shoulder = gt_xy[:, 5]
        right_hip = gt_xy[:, 12]
        norm = np.linalg.norm(left_shoulder - right_hip, axis=1)
    elif normalize_by == 'headsize':
        # Use head diagonal (eyes to ears)
        left_eye = gt_xy[:, 1]
        right_ear = gt_xy[:, 4]
        norm = np.linalg.norm(left_eye - right_ear, axis=1)
    else:
        norm = np.ones(pred_xy.shape[0])
    
    # Normalize distances
    norm = np.maximum(norm, 1e-6)  # Avoid division by zero
    normalized_distances = distances / norm[:, np.newaxis]
    
    # Count correct (distance < threshold and visible)
    correct = (normalized_distances < threshold) & (gt_vis > 0)
    total = (gt_vis > 0).sum()
    
    if total == 0:
        return 0.0
    
    pck = 100.0 * correct.sum() / total
    
    return float(pck)


if __name__ == "__main__":
    # Test evaluator
    logging.basicConfig(level=logging.INFO)
    
    # COCO sigmas
    oks_sigmas = [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079,
        0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087,
        0.087, 0.089, 0.089
    ]
    
    evaluator = COCOEvaluator(oks_sigmas)
    
    # Generate dummy data
    N = 100
    predictions = torch.randn(N, 17, 3)
    predictions[:, :, :2] = torch.abs(predictions[:, :, :2]) * 100  # x, y in [0, 100]
    predictions[:, :, 2] = torch.sigmoid(predictions[:, :, 2])  # confidence in [0, 1]
    
    targets = predictions.clone()
    targets[:, :, :2] += torch.randn(N, 17, 2) * 5  # Add noise
    targets[:, :, 2] = torch.randint(0, 3, (N, 17)).float()  # visibility
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, targets)
    
    print("Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}")
    
    # Compute per-joint accuracy
    per_joint = evaluator.compute_per_joint_accuracy(predictions, targets)
    print("\nPer-Joint Accuracy:")
    for joint, acc in per_joint.items():
        print(f"  {joint}: {acc:.2f}%")
