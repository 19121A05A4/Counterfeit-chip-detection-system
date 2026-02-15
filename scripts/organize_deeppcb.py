"""
Organize DeepPCB Dataset for Binary Classification
Author: SAI KATARI

Handles deeply nested folder structure of DeepPCB dataset.
"""

import os
import shutil
from pathlib import Path

def organize_deeppcb_dataset():
    """
    Organize DeepPCB into binary classification structure.
    
    DeepPCB structure is deeply nested:
    PCBData/group00041/00041/*.jpg
    PCBData/group00041/00041_not/*.jpg
    """
    
    # Paths
    source_dir = Path('temp_deeppcb/PCBData')
    target_base = Path('dataset/raw')
    
    # Create target directories
    normal_dir = target_base / 'normal'
    defective_dir = target_base / 'defective'
    
    normal_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing files
    print("Clearing old dataset...")
    for f in normal_dir.glob('*'):
        if f.is_file():
            f.unlink()
    for f in defective_dir.glob('*'):
        if f.is_file():
            f.unlink()
    
    print("\nOrganizing DeepPCB dataset...")
    print("Recursively scanning ALL folders for images...\n")
    
    template_count = 0
    test_count = 0
    
    # Use rglob to recursively find ALL .jpg files
    all_jpg_files = list(source_dir.rglob('*.jpg'))
    
    print(f"Found {len(all_jpg_files)} total .jpg files")
    print("Processing...\n")
    
    for img_file in all_jpg_files:
        filename = img_file.name
        
        if '_temp.jpg' in filename:
            # Template image (normal)
            # Create unique name using parent folder info
            new_name = f"{img_file.parent.name}_{filename}"
            shutil.copy2(img_file, normal_dir / new_name)
            template_count += 1
            
            if template_count % 200 == 0:
                print(f"  Copied {template_count} normal images...")
                
        elif '_test.jpg' in filename:
            # Test image (defective)
            new_name = f"{img_file.parent.name}_{filename}"
            shutil.copy2(img_file, defective_dir / new_name)
            test_count += 1
            
            if test_count % 200 == 0:
                print(f"  Copied {test_count} defective images...")
    
    print(f"\n{'='*60}")
    print("DeepPCB Organization Complete!")
    print(f"{'='*60}")
    print(f"Normal images (templates): {template_count}")
    print(f"Defective images (tests):  {test_count}")
    print(f"Total images:              {template_count + test_count}")
    print(f"{'='*60}\n")
    
    if template_count > 0 and test_count > 0:
        print(f"✓ Dataset ready at: dataset/raw/")
        print(f"  - dataset/raw/normal/     ({template_count} images)")
        print(f"  - dataset/raw/defective/  ({test_count} images)")
        print("\n✓ You can now run: python scripts/train_model.py")
    else:
        print("⚠ No images found! Check the DeepPCB folder structure.")

if __name__ == "__main__":
    organize_deeppcb_dataset()
