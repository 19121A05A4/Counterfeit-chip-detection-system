"""
PyTorch Data Pipeline for IC Authentication
Author: [Your Name]

My custom PyTorch Dataset and DataLoader setup specifically
optimized for small IC authentication datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
import logging

# Import my custom modules
import sys
sys.path.append('.')
from source.preparation.dataset_manager import KaggleICDatasetManager
from source.preparation.augmentation_engine import ICSpecificAugmenter

logger = logging.getLogger(__name__)


class HardwareAuthenticationDataset(Dataset):
    """
    My PyTorch Dataset for IC chip authentication.
    
    Design philosophy: Keep it simple but robust.
    Each __getitem__ call returns a properly augmented image tensor.
    """
    
    def __init__(self, 
                 image_file_paths: np.ndarray,
                 ground_truth_labels: np.ndarray,
                 augmentation_pipeline: Optional[object] = None,
                 is_training_mode: bool = True):
        """
        Initialize the dataset.
        
        Args:
            image_file_paths: Array of image file paths
            ground_truth_labels: Array of binary labels (0 or 1)
            augmentation_pipeline: Augmenter object (my custom implementation)
            is_training_mode: Whether to apply training augmentations
        """
        self.file_paths = image_file_paths
        self.labels = ground_truth_labels
        self.augmenter = augmentation_pipeline
        self.training_mode = is_training_mode
        
        logger.info(f"Dataset initialized: {len(self.file_paths)} samples, "
                   f"Training mode: {is_training_mode}")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.file_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process a single IC image.
        
        My pipeline:
        1. Load image from disk
        2. Convert to RGB (handle grayscale/RGBA)
        3. Apply augmentation if available
        4. Return as tensor
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (image_tensor, label_tensor)
        """
        # Load the image
        image_path = self.file_paths[index]
        label = self.labels[index]
        
        try:
            # Open with PIL
            ic_chip_image = Image.open(image_path)
            
            # Force RGB conversion
            if ic_chip_image.mode != 'RGB':
                ic_chip_image = ic_chip_image.convert('RGB')
            
            # Convert to numpy for augmentation
            image_array = np.array(ic_chip_image)
            
            # Apply augmentation
            if self.augmenter is not None:
                if self.training_mode:
                    augmented = self.augmenter.augment_for_training(image_array)
                else:
                    augmented = self.augmenter.augment_for_validation(image_array)
                
                image_tensor = augmented['image']
            else:
                # No augmentation - just convert to tensor
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
            
            # Convert label to tensor
            label_tensor = torch.tensor(label, dtype=torch.float32)
            
            return image_tensor, label_tensor
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a zero tensor on error to prevent crashes
            return torch.zeros(3, 224, 224), torch.tensor(0.0)


class ICDataLoaderFactory:
    """
    My factory class for creating train/val/test data loaders.
    
    Why a factory? Encapsulates all the splitting logic and configuration
    in one place, making experiments easier to reproduce.
    """
    
    def __init__(self, 
                 dataset_directory: str,
                 batch_size: int = 16,
                 validation_fraction: float = 0.15,
                 test_fraction: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize the factory.
        
        Args:
            dataset_directory: Path to Kaggle dataset
            batch_size: Samples per batch
            validation_fraction: Fraction for validation set
            test_fraction: Fraction for test set
            random_seed: For reproducible splits
        """
        self.dataset_dir = dataset_directory
        self.batch_size = batch_size
        self.val_fraction = validation_fraction
        self.test_fraction = test_fraction
        self.seed = random_seed
        
        # Initialize my dataset manager
        self.manager = KaggleICDatasetManager(dataset_directory)
        
        # Initialize my augmenter
        self.augmentation_engine = ICSpecificAugmenter()
        
        logger.info("DataLoader factory initialized")
    
    def create_stratified_splits(self) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Create train/val/test splits with stratification.
        
        My approach: Two-stage splitting to ensure balanced classes.
        Critical for small datasets where class imbalance matters.
        
        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        # Load complete dataset
        all_paths, all_labels = self.manager.load_complete_dataset()
        
        # First split: Separate out test set
        paths_temp, paths_test, labels_temp, labels_test = train_test_split(
            all_paths,
            all_labels,
            test_size=self.test_fraction,
            stratify=all_labels,
            random_state=self.seed
        )
        
        # Second split: Separate train and validation
        # Adjust validation fraction relative to remaining data
        adjusted_val_fraction = self.val_fraction / (1 - self.test_fraction)
        
        paths_train, paths_val, labels_train, labels_val = train_test_split(
            paths_temp,
            labels_temp,
            test_size=adjusted_val_fraction,
            stratify=labels_temp,
            random_state=self.seed
        )
        
        # Log split statistics
        logger.info(f"\n{'='*60}")
        logger.info("Dataset Split Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Training:   {len(paths_train)} samples "
                   f"({np.sum(labels_train==1)} normal, {np.sum(labels_train==0)} defective)")
        logger.info(f"Validation: {len(paths_val)} samples "
                   f"({np.sum(labels_val==1)} normal, {np.sum(labels_val==0)} defective)")
        logger.info(f"Test:       {len(paths_test)} samples "
                   f"({np.sum(labels_test==1)} normal, {np.sum(labels_test==0)} defective)")
        logger.info(f"{'='*60}\n")
        
        return ((paths_train, labels_train),
                (paths_val, labels_val),
                (paths_test, labels_test))
    
    def build_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Build all three data loaders.
        
        My configuration:
        - Training: Shuffle ON, augmentation ON, drop_last ON
        - Validation: Shuffle OFF, augmentation minimal, drop_last OFF
        - Test: Shuffle OFF, augmentation minimal, drop_last OFF
        
        Returns:
            (train_loader, val_loader, test_loader)
        """
        # Get splits
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.create_stratified_splits()
        
        # Create PyTorch datasets
        train_dataset = HardwareAuthenticationDataset(
            X_train, y_train,
            augmentation_pipeline=self.augmentation_engine,
            is_training_mode=True
        )
        
        val_dataset = HardwareAuthenticationDataset(
            X_val, y_val,
            augmentation_pipeline=self.augmentation_engine,
            is_training_mode=False
        )
        
        test_dataset = HardwareAuthenticationDataset(
            X_test, y_test,
            augmentation_pipeline=self.augmentation_engine,
            is_training_mode=False
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Important for training
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Avoid small last batch
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Deterministic validation
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )
        
        logger.info("✓ All DataLoaders created successfully")
        
        return train_loader, val_loader, test_loader


# Test the data pipeline
if __name__ == "__main__":
    print("Testing IC Authentication Data Pipeline\n")
    
    # Create factory
    factory = ICDataLoaderFactory(
        dataset_directory="dataset/raw",
        batch_size=8
    )
    
    try:
        # Build loaders
        train_loader, val_loader, test_loader = factory.build_data_loaders()
        
        # Test loading a batch
        images, labels = next(iter(train_loader))
        
        print(f"✓ Successfully loaded a batch")
        print(f"  Image batch shape: {images.shape}")
        print(f"  Label batch shape: {labels.shape}")
        print(f"  Image value range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Labels in batch: {labels.tolist()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")