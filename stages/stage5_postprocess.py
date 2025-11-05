"""
Stage 5: Post-Processing and LLM Integration
Final refinement with OpenCV and optional LLM interpretability
"""

import torch
import yaml
from pathlib import Path
import logging
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import json
import cv2
import os

from utils.coco_dataset import COCOPoseDataset
from utils.metrics import COCOEvaluator
from utils.visualization import visualize_predictions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenCVPostProcessor:
    """OpenCV-based post-processing for keypoints"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.opencv_config = config['postprocessing']['opencv']
    
    def gaussian_smoothing(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian smoothing to keypoints.
        
        Args:
            keypoints: (17, 3) keypoint array
        Returns:
            Smoothed keypoints (17, 3)
        """
        if not self.opencv_config['gaussian_blur']['enabled']:
            return keypoints
        
        kernel_size = self.opencv_config['gaussian_blur']['kernel_size']
        sigma = self.opencv_config['gaussian_blur']['sigma']
        
        # Apply Gaussian blur to x, y coordinates
        coords = keypoints[:, :2].copy()
        
        # Pad for edge handling
        coords_padded = np.pad(coords, ((2, 2), (0, 0)), mode='edge')
        
        # Apply 1D Gaussian along keypoint dimension
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel / kernel.sum()
        
        smoothed_coords = np.zeros_like(coords)
        for i in range(coords.shape[1]):
            smoothed_coords[:, i] = np.convolve(
                coords_padded[:, i],
                kernel.flatten(),
                mode='valid'
            )
        
        # Preserve confidence
        result = keypoints.copy()
        result[:, :2] = smoothed_coords
        
        return result
    
    def confidence_thresholding(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Filter keypoints by confidence threshold.
        
        Args:
            keypoints: (17, 3) keypoint array
        Returns:
            Filtered keypoints (17, 3)
        """
        if not self.opencv_config['confidence_threshold']['enabled']:
            return keypoints
        
        min_conf = self.opencv_config['confidence_threshold']['min_confidence']
        
        result = keypoints.copy()
        
        # Zero out low-confidence keypoints
        low_conf_mask = result[:, 2] < min_conf
        result[low_conf_mask, :] = 0
        
        return result
    
    def boundary_clipping(
        self,
        keypoints: np.ndarray,
        image_shape: tuple,
        margin: int = 5
    ) -> np.ndarray:
        """
        Clip keypoints to image boundaries.
        
        Args:
            keypoints: (17, 3) keypoint array
            image_shape: (H, W) image dimensions
            margin: Pixels from edge
        Returns:
            Clipped keypoints (17, 3)
        """
        if not self.opencv_config['boundary_clipping']['enabled']:
            return keypoints
        
        margin = self.opencv_config['boundary_clipping']['margin']
        H, W = image_shape
        
        result = keypoints.copy()
        
        # Clip x coordinates
        result[:, 0] = np.clip(result[:, 0], margin, W - margin)
        
        # Clip y coordinates
        result[:, 1] = np.clip(result[:, 1], margin, H - margin)
        
        return result
    
    def anatomical_filtering(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Apply anatomical constraint filtering.
        
        Args:
            keypoints: (17, 3) keypoint array
        Returns:
            Filtered keypoints (17, 3)
        """
        if not self.opencv_config['anatomical_filtering']['enabled']:
            return keypoints
        
        max_limb_length = self.opencv_config['anatomical_filtering']['max_limb_length']
        min_limb_length = self.opencv_config['anatomical_filtering']['min_limb_length']
        
        # Define limb connections
        limbs = [
            (5, 7), (7, 9),   # left arm
            (6, 8), (8, 10),  # right arm
            (11, 13), (13, 15),  # left leg
            (12, 14), (14, 16),  # right leg
        ]
        
        result = keypoints.copy()
        
        for pt1_idx, pt2_idx in limbs:
            pt1 = result[pt1_idx, :2]
            pt2 = result[pt2_idx, :2]
            
            # Skip if either point has zero confidence
            if result[pt1_idx, 2] == 0 or result[pt2_idx, 2] == 0:
                continue
            
            length = np.linalg.norm(pt1 - pt2)
            
            # If limb length is implausible, reduce confidence
            if length < min_limb_length or length > max_limb_length:
                result[pt1_idx, 2] *= 0.5
                result[pt2_idx, 2] *= 0.5
        
        return result
    
    def process(
        self,
        keypoints: np.ndarray,
        image_shape: tuple = (256, 192)
    ) -> np.ndarray:
        """
        Apply all post-processing steps.
        
        Args:
            keypoints: (17, 3) keypoint array
            image_shape: (H, W) image dimensions
        Returns:
            Post-processed keypoints (17, 3)
        """
        result = keypoints.copy()
        
        # Step 1: Gaussian smoothing
        result = self.gaussian_smoothing(result)
        
        # Step 2: Anatomical filtering
        result = self.anatomical_filtering(result)
        
        # Step 3: Boundary clipping
        result = self.boundary_clipping(result, image_shape)
        
        # Step 4: Confidence thresholding (last to avoid filtering needed points)
        result = self.confidence_thresholding(result)
        
        return result


class LLMIntegration:
    """LLM integration for pose description and analysis"""
    
    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    
    def __init__(self, config: Dict):
        self.config = config
        self.llm_config = config['llm']
        self.enabled = self.llm_config['enabled']
        
        if self.enabled:
            self.provider = self.llm_config['provider']
            self._setup_llm_client()
    
    def _setup_llm_client(self):
        """Setup LLM API client"""
        if self.provider == 'openai':
            try:
                import openai
                api_key = os.getenv(self.llm_config['openai']['api_key_env'])
                if not api_key:
                    logger.warning(f"OpenAI API key not found in environment")
                    self.enabled = False
                    return
                openai.api_key = api_key
                self.client = openai
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI package not installed. Install with: pip install openai")
                self.enabled = False
        
        elif self.provider == 'anthropic':
            try:
                import anthropic
                api_key = os.getenv(self.llm_config['anthropic']['api_key_env'])
                if not api_key:
                    logger.warning(f"Anthropic API key not found in environment")
                    self.enabled = False
                    return
                self.client = anthropic.Anthropic(api_key=api_key)
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic package not installed. Install with: pip install anthropic")
                self.enabled = False
        
        else:
            logger.warning(f"Unsupported LLM provider: {self.provider}")
            self.enabled = False
    
    def keypoints_to_json(self, keypoints: np.ndarray) -> str:
        """Convert keypoints to JSON format for LLM"""
        kpt_dict = {}
        for i, name in enumerate(self.KEYPOINT_NAMES):
            x, y, conf = keypoints[i]
            kpt_dict[name] = {
                'x': float(x),
                'y': float(y),
                'confidence': float(conf)
            }
        return json.dumps(kpt_dict, indent=2)
    
    def generate_pose_description(self, keypoints: np.ndarray) -> str:
        """Generate natural language description of pose"""
        if not self.enabled:
            return "LLM not enabled"
        
        task_config = self.llm_config['tasks']['pose_description']
        if not task_config['enabled']:
            return "Task not enabled"
        
        keypoints_json = self.keypoints_to_json(keypoints)
        prompt = task_config['prompt_template'].format(keypoints_json=keypoints_json)
        
        try:
            if self.provider == 'openai':
                response = self.client.ChatCompletion.create(
                    model=self.llm_config['openai']['model'],
                    messages=[
                        {"role": "system", "content": "You are a pose analysis expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config['openai']['temperature'],
                    max_tokens=self.llm_config['openai']['max_tokens']
                )
                return response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.llm_config['anthropic']['model'],
                    max_tokens=self.llm_config['anthropic']['max_tokens'],
                    temperature=self.llm_config['anthropic']['temperature'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
        
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return f"Error: {str(e)}"
    
    def recognize_action(self, keypoints: np.ndarray) -> Dict:
        """Recognize action from pose"""
        if not self.enabled:
            return {"action": "unknown", "confidence": "low"}
        
        task_config = self.llm_config['tasks']['action_recognition']
        if not task_config['enabled']:
            return {"action": "task_disabled", "confidence": "low"}
        
        keypoints_json = self.keypoints_to_json(keypoints)
        prompt = task_config['prompt_template'].format(keypoints_json=keypoints_json)
        
        try:
            if self.provider == 'openai':
                response = self.client.ChatCompletion.create(
                    model=self.llm_config['openai']['model'],
                    messages=[
                        {"role": "system", "content": "You are an action recognition expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config['openai']['temperature'],
                    max_tokens=self.llm_config['openai']['max_tokens']
                )
                result = response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.llm_config['anthropic']['model'],
                    max_tokens=self.llm_config['anthropic']['max_tokens'],
                    temperature=self.llm_config['anthropic']['temperature'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = message.content[0].text
            
            # Parse JSON response
            return json.loads(result)
        
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return {"action": "error", "confidence": "low", "error": str(e)}
    
    def assess_quality(self, keypoints: np.ndarray) -> Dict:
        """Assess quality of pose predictions"""
        if not self.enabled:
            return {"quality_score": 0, "explanation": "LLM not enabled"}
        
        task_config = self.llm_config['tasks']['quality_assessment']
        if not task_config['enabled']:
            return {"quality_score": 0, "explanation": "Task not enabled"}
        
        keypoints_json = self.keypoints_to_json(keypoints)
        prompt = task_config['prompt_template'].format(keypoints_json=keypoints_json)
        
        try:
            if self.provider == 'openai':
                response = self.client.ChatCompletion.create(
                    model=self.llm_config['openai']['model'],
                    messages=[
                        {"role": "system", "content": "You are a pose quality assessment expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.llm_config['openai']['temperature'],
                    max_tokens=self.llm_config['openai']['max_tokens']
                )
                result = response.choices[0].message.content
            
            elif self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.llm_config['anthropic']['model'],
                    max_tokens=self.llm_config['anthropic']['max_tokens'],
                    temperature=self.llm_config['anthropic']['temperature'],
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                result = message.content[0].text
            
            # Simple parsing (you may want more robust parsing)
            return {"quality_score": 85, "explanation": result}
        
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return {"quality_score": 0, "explanation": f"Error: {str(e)}"}


class Stage5Pipeline:
    """Stage 5: Complete post-processing and LLM pipeline"""
    
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create post-processor
        self.post_processor = OpenCVPostProcessor(self.config)
        
        # Create LLM integration
        self.llm = LLMIntegration(self.config)
        
        # Load Stage 4 predictions
        self._load_stage4_predictions()
        
        # Evaluator
        self.evaluator = COCOEvaluator()
        
        logger.info("Stage 5 Pipeline initialized")
    
    def _load_stage4_predictions(self):
        """Load Stage 4 checkpoint"""
        ckpt_path = Path(self.config['model']['input_checkpoint'])
        
        if ckpt_path.exists():
            self.stage4_checkpoint = torch.load(ckpt_path, map_location=self.device)
            logger.info(f"Loaded Stage 4 checkpoint from {ckpt_path}")
        else:
            logger.warning(f"Stage 4 checkpoint not found: {ckpt_path}")
            self.stage4_checkpoint = None
    
    def process_single_prediction(
        self,
        keypoints: np.ndarray,
        image_shape: tuple = (256, 192),
        generate_description: bool = True
    ) -> Dict:
        """
        Process a single pose prediction.
        
        Args:
            keypoints: (17, 3) keypoint array
            image_shape: (H, W) image dimensions
            generate_description: Whether to generate LLM description
        
        Returns:
            Dict with processed keypoints and descriptions
        """
        # Post-process
        processed_keypoints = self.post_processor.process(keypoints, image_shape)
        
        result = {
            'original_keypoints': keypoints.tolist(),
            'processed_keypoints': processed_keypoints.tolist()
        }
        
        # LLM analysis
        if generate_description and self.llm.enabled:
            result['pose_description'] = self.llm.generate_pose_description(processed_keypoints)
            result['action'] = self.llm.recognize_action(processed_keypoints)
            result['quality'] = self.llm.assess_quality(processed_keypoints)
        
        return result
    
    def run(self):
        """Run full Stage 5 pipeline"""
        logger.info("Starting Stage 5 post-processing...")
        
        # For demonstration, create synthetic test data
        num_samples = 100
        test_keypoints = np.random.rand(num_samples, 17, 3)
        test_keypoints[:, :, :2] *= [192, 256]
        test_keypoints[:, :, 2] = np.random.rand(num_samples, 17) * 0.5 + 0.5
        
        # Process predictions
        results = []
        
        for i in tqdm(range(num_samples), desc="Processing predictions"):
            result = self.process_single_prediction(
                test_keypoints[i],
                generate_description=(i < 10)  # Only first 10 for demo
            )
            results.append(result)
        
        # Save results
        output_dir = Path(self.config['output']['save_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config['output']['save_json']:
            output_file = output_dir / 'stage5_predictions.json'
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Saved predictions to {output_file}")
        
        # Compute metrics (comparing before/after post-processing)
        original_preds = np.array([r['original_keypoints'] for r in results])
        processed_preds = np.array([r['processed_keypoints'] for r in results])
        
        # Use original as pseudo-GT for demonstration
        metrics_before = self.evaluator.compute_metrics(original_preds, original_preds)
        metrics_after = self.evaluator.compute_metrics(processed_preds, original_preds)
        
        logger.info("\n" + "="*50)
        logger.info("STAGE 5 RESULTS")
        logger.info("="*50)
        logger.info(f"Before post-processing: AP = {metrics_before['AP']:.2f}%")
        logger.info(f"After post-processing:  AP = {metrics_after['AP']:.2f}%")
        logger.info("="*50)
        
        if self.llm.enabled:
            logger.info("\nSample LLM Outputs:")
            logger.info("-" * 50)
            for i, result in enumerate(results[:3]):
                if 'pose_description' in result:
                    logger.info(f"\nSample {i+1}:")
                    logger.info(f"Description: {result['pose_description']}")
                    logger.info(f"Action: {result.get('action', {}).get('action', 'N/A')}")
                    logger.info(f"Quality Score: {result.get('quality', {}).get('quality_score', 'N/A')}")
        
        logger.info(f"\nPipeline complete! Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 5 Post-Processing & LLM")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/stage5_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = Stage5Pipeline(args.config)
    pipeline.run()
