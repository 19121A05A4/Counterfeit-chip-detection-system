"""
Dataset Manager for Kaggle IC Datasets
Author: SAI KATARI
Date: February 2026

My custom approach to loading and organizing IC authentication datasets.
Designed specifically for Kaggle PCB/IC defect datasets with automatic
structure detection.
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
import logging

# My logging configuration for tracking data pipeline
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KaggleICDatasetManager:
    """
    My custom dataset manager for Kaggle IC/PCB datasets.
    
    Philosophy: Handle multiple folder structures automatically,
    making the system flexible for different Kaggle dataset formats.
    
    Supported structures:
    1. Binary folders: defective/ and normal/
    2. Binary folders: counterfeit/ and authentic/
    3. Flat structure with labeled filenames (A-*.png, C-*.png)
    """
    
    def __init__(self, dataset_root: str):
        """
        Initialize the dataset manager.
        
        Args:
            dataset_root: Path to the root folder containing IC images
        """
        self.dataset_path = Path(dataset_root)
        self.image_locations = []
        self.classification_labels = []
        
        # My supported image formats
        self.valid_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        logger.info(f"Initializing dataset manager for: {dataset_root}")
    
    def _identify_dataset_structure(self) -> str:
        """
        Automatically detect how the dataset is organized.
        
        My approach: Check for subdirectories first, then check filename patterns.
        This makes the system work with various Kaggle datasets.
        
        Returns:
            'folder_based' or 'filename_based'
        """
        subdirectories = [item for item in self.dataset_path.iterdir() 
                         if item.is_dir()]
        
        if len(subdirectories) >= 2:
            logger.info(f"Detected folder-based structure with {len(subdirectories)} categories")
            return 'folder_based'
        else:
            logger.info("Detected filename-based labeling structure")
            return 'filename_based'
    
    def _extract_label_from_foldername(self, folder_name: str) -> int:
        """
        Map folder names to binary labels.
        
        My strategy: Use keyword matching to automatically identify
        defective vs normal categories without manual configuration.
        
        Args:
            folder_name: Name of the folder
            
        Returns:
            0 for defective/counterfeit, 1 for normal/authentic
        """
        folder_lower = folder_name.lower()
        
        # Keywords indicating defective/counterfeit class
        defect_keywords = ['defect', 'counterfeit', 'fake', 'bad', 
                          'anomaly', 'faulty', 'ng', 'reject']
        
        # Keywords indicating normal/authentic class
        normal_keywords = ['normal', 'authentic', 'good', 'ok', 
                          'genuine', 'real', 'pass', 'accept']
        
        for keyword in defect_keywords:
            if keyword in folder_lower:
                return 0  # Defective class
        
        for keyword in normal_keywords:
            if keyword in folder_lower:
                return 1  # Normal class
        
        # If no match, log warning and return based on folder index
        logger.warning(f"Could not auto-detect label for folder: {folder_name}")
        return None
    
    def _load_from_folder_structure(self) -> Tuple[List[str], List[int]]:
        """
        Load images organized in separate folders.
        
        My implementation: Scan each subdirectory, extract labels,
        and build complete image path list.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        subdirectories = [item for item in self.dataset_path.iterdir() 
                         if item.is_dir()]
        
        # Create folder to label mapping
        folder_label_map = {}
        
        for subdir in subdirectories:
            predicted_label = self._extract_label_from_foldername(subdir.name)
            
            if predicted_label is not None:
                folder_label_map[subdir.name] = predicted_label
                logger.info(f"  '{subdir.name}' → Label {predicted_label}")
        
        # If auto-detection failed for all folders, use simple mapping
        if not folder_label_map:
            logger.warning("Auto-detection failed. Using first folder as 0, second as 1")
            folder_label_map[subdirectories[0].name] = 0
            folder_label_map[subdirectories[1].name] = 1
        
        # Load images from each labeled folder
        for folder_name, label in folder_label_map.items():
            folder_path = self.dataset_path / folder_name
            
            for image_file in folder_path.iterdir():
                if image_file.suffix.lower() in self.valid_formats:
                    image_paths.append(str(image_file))
                    labels.append(label)
        
        return image_paths, labels
    
    def _load_from_filename_pattern(self) -> Tuple[List[str], List[int]]:
        """
        Load images where labels are encoded in filenames.
        
        My pattern recognition:
        - Files starting with 'A' or containing 'authentic' → Label 1
        - Files starting with 'C' or containing 'counterfeit' → Label 0
        
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        for image_file in self.dataset_path.iterdir():
            if image_file.suffix.lower() not in self.valid_formats:
                continue
            
            filename_lower = image_file.name.lower()
            
            # My label extraction logic
            if filename_lower.startswith('a') or 'authentic' in filename_lower or 'normal' in filename_lower:
                label = 1  # Authentic/Normal
            elif filename_lower.startswith('c') or 'counterfeit' in filename_lower or 'defect' in filename_lower:
                label = 0  # Counterfeit/Defective
            else:
                logger.warning(f"Could not determine label for: {image_file.name}")
                continue
            
            image_paths.append(str(image_file))
            labels.append(label)
        
        return image_paths, labels
    
    def load_complete_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all images and labels from the dataset.
        
        My orchestration: Detect structure, load accordingly, validate.
        
        Returns:
            Tuple of (image_paths as array, labels as array)
        """
        structure_type = self._identify_dataset_structure()
        
        if structure_type == 'folder_based':
            paths, labels = self._load_from_folder_structure()
        else:
            paths, labels = self._load_from_filename_pattern()
        
        # Validate we found images
        if len(paths) == 0:
            raise ValueError(f"No valid images found in {self.dataset_path}")
        
        # Convert to numpy arrays
        paths_array = np.array(paths)
        labels_array = np.array(labels)
        
        # Log dataset statistics
        num_normal = np.sum(labels_array == 1)
        num_defective = np.sum(labels_array == 0)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset Loading Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total images: {len(paths)}")
        logger.info(f"Normal/Authentic (1): {num_normal} ({num_normal/len(paths)*100:.1f}%)")
        logger.info(f"Defective/Counterfeit (0): {num_defective} ({num_defective/len(paths)*100:.1f}%)")
        logger.info(f"{'='*60}\n")
        
        return paths_array, labels_array
    
    def verify_image_integrity(self, sample_size: int = 10) -> Dict[str, int]:
        """
        Check a sample of images for loading errors.
        
        My quality check: Validate images can be opened before training starts.
        
        Args:
            sample_size: Number of random images to test
            
        Returns:
            Dictionary with verification statistics
        """
        if len(self.image_locations) == 0:
            logger.warning("No images loaded yet. Call load_complete_dataset() first.")
            return {}
        
        # Sample random images
        sample_indices = np.random.choice(
            len(self.image_locations), 
            min(sample_size, len(self.image_locations)),
            replace=False
        )
        
        successful = 0
        failed = 0
        
        for idx in sample_indices:
            try:
                img = Image.open(self.image_locations[idx])
                img.verify()  # Check if image is corrupted
                successful += 1
            except Exception as e:
                failed += 1
                logger.error(f"Image verification failed: {self.image_locations[idx]} - {e}")
        
        verification_stats = {
            'tested': len(sample_indices),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(sample_indices)
        }
        
        logger.info(f"Image verification: {successful}/{len(sample_indices)} passed")
        
        return verification_stats


# Quick test of the dataset manager
if __name__ == "__main__":
    # Test with your dataset path
    manager = KaggleICDatasetManager("dataset/raw")
    
    try:
        paths, labels = manager.load_complete_dataset()
        print(f"\n✓ Successfully loaded {len(paths)} images")
        print(f"  Sample paths: {paths[:3]}")
        print(f"  Sample labels: {labels[:3]}")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("  Make sure you've placed your Kaggle dataset in dataset/raw/")
