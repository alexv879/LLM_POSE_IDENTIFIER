"""
COCO Keypoint Validation Script
Comprehensive validation of COCO format annotations with visualization.

Based on:
- COCO Keypoint Format Specification (Lin et al., 2014)
- Best practices from MMPose evaluation toolkit
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# COCO Skeleton connections (1-indexed as in COCO format)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

# COCO Keypoint names
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Per-joint sigma values for OKS computation (COCO standard)
COCO_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
    1.07, 1.07, .87, .87, .89, .89
]) / 10.0


def validate_coco_keypoints(
    img_dir: str,
    ann_file: str,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Comprehensive COCO keypoint validation.
    
    Args:
        img_dir: Path to images directory
        ann_file: Path to COCO JSON annotations
        verbose: Whether to print detailed error messages
    
    Returns:
        Dictionary with validation results and statistics
    """
    logger.info("="*60)
    logger.info("COCO Keypoint Validation")
    logger.info("="*60)
    
    # Load annotations
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    logger.info(f"Images: {len(images)}")
    logger.info(f"Annotations: {len(annotations)}")
    logger.info(f"Categories: {len(categories)}")
    
    errors = []
    warnings = []
    stats = defaultdict(int)
    
    # Check 1: Image/annotation ID uniqueness
    img_ids = [i['id'] for i in images]
    if len(img_ids) != len(set(img_ids)):
        errors.append("ERROR: Duplicate image IDs found")
    
    ann_ids = [a['id'] for a in annotations]
    if len(ann_ids) != len(set(ann_ids)):
        errors.append("ERROR: Duplicate annotation IDs found")
    
    # Check 2: Image existence
    img_dir_path = Path(img_dir)
    for img in images:
        img_path = img_dir_path / img['file_name']
        if not img_path.exists():
            errors.append(f"ERROR: Image not found: {img['file_name']}")
            stats['missing_images'] += 1
    
    # Check 3: Validate each annotation
    for ann_idx, ann in enumerate(annotations):
        ann_id = ann.get('id', ann_idx)
        kpts = ann.get('keypoints', [])
        
        # Check keypoint count (must be 51: 17 joints × 3 values)
        if len(kpts) != 51:
            errors.append(
                f"ERROR: Annotation {ann_id} has {len(kpts)} keypoint values "
                f"(expected 51 = 17 joints × 3)"
            )
            continue
        
        # Reshape keypoints
        kpts_array = np.array(kpts).reshape(17, 3)
        
        # Check visibility values
        valid_visibilities = {0, 1, 2}
        for j in range(17):
            v = kpts_array[j, 2]
            if v not in valid_visibilities:
                errors.append(
                    f"ERROR: Annotation {ann_id}, joint {j} has invalid "
                    f"visibility {v} (expected 0, 1, or 2)"
                )
        
        # Count visible/labeled keypoints
        visible_count = np.sum(kpts_array[:, 2] == 2)
        labeled_count = np.sum(kpts_array[:, 2] > 0)
        stats['total_visible_keypoints'] += visible_count
        stats['total_labeled_keypoints'] += labeled_count
        
        # Check coordinates within image bounds
        img_id = ann.get('image_id')
        img_info = next((i for i in images if i['id'] == img_id), None)
        
        if img_info:
            h, w = img_info['height'], img_info['width']
            
            for j in range(17):
                x, y, v = kpts_array[j]
                
                if v > 0:  # Only check visible/labeled keypoints
                    if not (0 <= x <= w and 0 <= y <= h):
                        warnings.append(
                            f"WARNING: Annotation {ann_id}, joint {j} "
                            f"({COCO_KEYPOINTS[j]}) out of bounds: "
                            f"({x:.1f}, {y:.1f}) vs image size ({w}, {h})"
                        )
                        stats['out_of_bounds_keypoints'] += 1
        
        # Check bbox validity
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, w_box, h_box = bbox
            if w_box <= 0 or h_box <= 0:
                warnings.append(
                    f"WARNING: Annotation {ann_id} has invalid bbox: "
                    f"width={w_box}, height={h_box}"
                )
                stats['invalid_bboxes'] += 1
        
        # Update stats
        if visible_count == 0:
            stats['annotations_no_visible_keypoints'] += 1
        elif visible_count < 5:
            stats['annotations_few_visible_keypoints'] += 1
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("VALIDATION RESULTS")
    logger.info("="*60)
    
    if errors:
        logger.error(f"\n❌ Found {len(errors)} ERRORS:")
        if verbose:
            for error in errors[:20]:  # Print first 20 errors
                logger.error(f"  {error}")
            if len(errors) > 20:
                logger.error(f"  ... and {len(errors) - 20} more errors")
        else:
            logger.error(f"  Run with verbose=True to see details")
    else:
        logger.info("\n✅ No critical errors found!")
    
    if warnings:
        logger.warning(f"\n⚠️  Found {len(warnings)} WARNINGS:")
        if verbose:
            for warning in warnings[:20]:
                logger.warning(f"  {warning}")
            if len(warnings) > 20:
                logger.warning(f"  ... and {len(warnings) - 20} more warnings")
    
    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("STATISTICS")
    logger.info("="*60)
    logger.info(f"Total annotations: {len(annotations)}")
    logger.info(f"Average visible keypoints per annotation: "
                f"{stats['total_visible_keypoints'] / max(len(annotations), 1):.2f}")
    logger.info(f"Average labeled keypoints per annotation: "
                f"{stats['total_labeled_keypoints'] / max(len(annotations), 1):.2f}")
    logger.info(f"Annotations with no visible keypoints: "
                f"{stats['annotations_no_visible_keypoints']}")
    logger.info(f"Annotations with <5 visible keypoints: "
                f"{stats['annotations_few_visible_keypoints']}")
    logger.info(f"Out of bounds keypoints: {stats['out_of_bounds_keypoints']}")
    logger.info(f"Invalid bboxes: {stats['invalid_bboxes']}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'statistics': dict(stats)
    }


def visualize_sample_annotations(
    img_dir: str,
    ann_file: str,
    output_dir: str = "validation_outputs",
    num_samples: int = 10,
    random_seed: int = 42
) -> None:
    """
    Visualize random annotation samples with keypoints and skeleton.
    
    Args:
        img_dir: Path to images directory
        ann_file: Path to COCO JSON annotations
        output_dir: Directory to save visualization outputs
        num_samples: Number of samples to visualize
        random_seed: Random seed for reproducibility
    """
    logger.info("\n" + "="*60)
    logger.info("GENERATING VISUALIZATION SAMPLES")
    logger.info("="*60)
    
    # Load annotations
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [{}])[0]
    
    kpt_names = categories.get('keypoints', COCO_KEYPOINTS)
    skeleton = categories.get('skeleton', COCO_SKELETON)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample random annotations
    np.random.seed(random_seed)
    sample_indices = np.random.choice(
        len(annotations),
        min(num_samples, len(annotations)),
        replace=False
    )
    
    colors = plt.cm.rainbow(np.linspace(0, 1, 17))
    
    for idx in sample_indices:
        ann = annotations[idx]
        img_info = next(i for i in images if i['id'] == ann['image_id'])
        
        # Load image
        img_path = Path(img_dir) / img_info['file_name']
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        
        img = np.array(Image.open(img_path).convert('RGB'))
        kpts = np.array(ann['keypoints']).reshape(17, 3)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(img)
        
        # Draw skeleton connections
        for connection in skeleton:
            p1_idx, p2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-indexed
            kp1, kp2 = kpts[p1_idx], kpts[p2_idx]
            
            if kp1[2] > 0 and kp2[2] > 0:  # Both keypoints visible/labeled
                ax.plot(
                    [kp1[0], kp2[0]],
                    [kp1[1], kp2[1]],
                    'b-',
                    linewidth=2,
                    alpha=0.6
                )
        
        # Draw keypoints
        for i, (x, y, v) in enumerate(kpts):
            if v > 0:
                # Color based on visibility
                color = 'lime' if v == 2 else 'yellow'
                marker = 'o' if v == 2 else 'x'
                
                ax.scatter(x, y, c=color, s=100, marker=marker, edgecolors='black', linewidths=2)
                ax.text(
                    x + 5, y + 5,
                    kpt_names[i],
                    fontsize=8,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6)
                )
        
        # Add title with statistics
        visible_count = np.sum(kpts[:, 2] == 2)
        labeled_count = np.sum(kpts[:, 2] > 0)
        ax.set_title(
            f"Image ID: {ann['image_id']} | Annotation ID: {ann['id']}\n"
            f"Visible: {visible_count}/17 | Labeled: {labeled_count}/17",
            fontsize=12,
            fontweight='bold'
        )
        ax.axis('off')
        
        # Save figure
        output_file = output_path / f"annotation_sample_{ann['id']}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Saved visualization: {output_file}")
    
    logger.info(f"\n✓ All visualizations saved to: {output_path}")


def compute_dataset_statistics(ann_file: str) -> Dict:
    """
    Compute detailed statistics about the dataset.
    
    Args:
        ann_file: Path to COCO JSON annotations
    
    Returns:
        Dictionary with comprehensive statistics
    """
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    
    stats = {
        'total_images': len(images),
        'total_annotations': len(annotations),
        'annotations_per_image': len(annotations) / max(len(images), 1),
    }
    
    # Per-joint visibility statistics
    joint_visibility = np.zeros((17, 3))  # (17 joints, 3 visibility states)
    
    for ann in annotations:
        kpts = np.array(ann['keypoints']).reshape(17, 3)
        for j in range(17):
            v = int(kpts[j, 2])
            if 0 <= v <= 2:
                joint_visibility[j, v] += 1
    
    stats['per_joint_visibility'] = {
        COCO_KEYPOINTS[j]: {
            'not_labeled': int(joint_visibility[j, 0]),
            'labeled_hidden': int(joint_visibility[j, 1]),
            'visible': int(joint_visibility[j, 2]),
            'total': int(joint_visibility[j].sum())
        }
        for j in range(17)
    }
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate COCO keypoint annotations")
    parser.add_argument("--img_dir", type=str, default="data/raw",
                       help="Path to images directory")
    parser.add_argument("--ann_file", type=str, 
                       default="data/annotations/train_keypoints.json",
                       help="Path to COCO JSON annotations")
    parser.add_argument("--output_dir", type=str, default="validation_outputs",
                       help="Directory to save visualization outputs")
    parser.add_argument("--num_samples", type=int, default=10,
                       help="Number of samples to visualize")
    parser.add_argument("--no_viz", action="store_true",
                       help="Skip visualization generation")
    
    args = parser.parse_args()
    
    # Run validation
    results = validate_coco_keypoints(args.img_dir, args.ann_file, verbose=True)
    
    if results['valid']:
        logger.info("\n" + "="*60)
        logger.info("✅ VALIDATION PASSED - Dataset is ready to use!")
        logger.info("="*60)
        
        # Generate visualizations
        if not args.no_viz:
            visualize_sample_annotations(
                args.img_dir,
                args.ann_file,
                args.output_dir,
                args.num_samples
            )
        
        # Compute and display statistics
        stats = compute_dataset_statistics(args.ann_file)
        logger.info("\nDataset Statistics:")
        logger.info(f"  Total images: {stats['total_images']}")
        logger.info(f"  Total annotations: {stats['total_annotations']}")
        logger.info(f"  Annotations per image: {stats['annotations_per_image']:.2f}")
    else:
        logger.error("\n" + "="*60)
        logger.error("❌ VALIDATION FAILED - Please fix errors before proceeding!")
        logger.error("="*60)
        exit(1)
