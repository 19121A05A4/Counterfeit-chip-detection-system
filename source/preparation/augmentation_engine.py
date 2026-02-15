"""
IC-Specific Augmentation Engine
Author: SAI KATARI

My custom augmentation strategy designed for counterfeit IC detection.
Unlike generic image augmentation, this simulates real-world chip inspection
conditions: lighting variations, camera angles, sensor noise.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import logging

logger = logging.getLogger(__name__)


class ICSpecificAugmenter:
    """
    My specialized augmentation for IC chip imagery.
    
    Philosophy: Aggressive augmentation to combat small dataset size,
    but carefully designed to preserve the subtle features that
    distinguish authentic from counterfeit chips.
    """
    
    def __init__(self, config_file: str = "config/settings.yaml"):
        """
        Initialize augmenter with settings from config.
        
        Args:
            config_file: Path to YAML configuration
        """
        self.config = self._load_augmentation_config(config_file)
        
        # Build the augmentation pipelines
        self.training_transforms = self._build_training_pipeline()
        self.validation_transforms = self._build_validation_pipeline()
        
        logger.info("IC augmentation engine initialized")
    
    def _load_augmentation_config(self, yaml_path: str) -> dict:
        """
        Load augmentation parameters from YAML.
        
        My approach: Centralized config for easy experimentation.
        """
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('augmentation_strategy', {})
        except FileNotFoundError:
            logger.warning(f"Config file not found: {yaml_path}. Using defaults.")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> dict:
        """Fallback settings if YAML not found."""
        return {
            'enabled': True,
            'samples_per_image': 8
        }
    
    def _build_training_pipeline(self) -> A.Compose:
        """
        Construct my training augmentation sequence.
        
        My strategy: Layer augmentations from geometric to photometric to noise.
        This creates diverse training samples while keeping chips recognizable.
        """
        return A.Compose([
            # Geometric transformations
            # Reasoning: IC can be photographed from different angles
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, border_mode=0, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1, 
                rotate_limit=10,
                border_mode=0,
                p=0.5
            ),
            
            # Lighting variations (critical for IC inspection)
            # Different inspection stations have different lighting
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            
            # Enhance local contrast (helps with surface defect visibility)
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            
            # Color variations (different camera sensors/white balance)
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4
            ),
            A.RGBShift(
                r_shift_limit=15,
                g_shift_limit=15,
                b_shift_limit=15,
                p=0.4
            ),
            
            # Noise simulation (real sensors have noise)
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=1.0),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
            ], p=0.3),
            
            # Blur effects (simulates focus issues)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.3),
            
            # Detail enhancement (can help or hurt - experiment)
            A.OneOf([
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
                A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
            ], p=0.2),
            
            # Normalize with ImageNet statistics
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            
            # Convert to PyTorch tensor
            ToTensorV2()
        ])
    
    def _build_validation_pipeline(self) -> A.Compose:
        """
        Validation pipeline: resize and normalize only.
        
        My philosophy: Validation should represent ideal imaging conditions.
        """
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def augment_for_training(self, image):
        """Apply training augmentation."""
        return self.training_transforms(image=image)
    
    def augment_for_validation(self, image):
        """Apply validation augmentation."""
        return self.validation_transforms(image=image)


# Test the augmenter
if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    print("Testing IC Augmentation Engine\n")
    
    # Create a dummy IC image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize augmenter
    augmenter = ICSpecificAugmenter()
    
    # Test training augmentation
    augmented = augmenter.augment_for_training(dummy_image)
    
    print(f"✓ Training augmentation successful")
    print(f"  Output type: {type(augmented['image'])}")
    print(f"  Output shape: {augmented['image'].shape}")
    print(f"  Value range: [{augmented['image'].min():.3f}, {augmented['image'].max():.3f}]")
