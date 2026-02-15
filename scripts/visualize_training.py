"""
Training Visualization
Author: SAI KATARI

Plot training curves after training completes.
"""

import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_training_history(checkpoint_path: str = 'artifacts/checkpoints/best_model.pth'):
    """Load and visualize training history."""
    
    checkpoint = torch.load(checkpoint_path)
    history = checkpoint['training_history']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history['train_accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('artifacts/visualizations/training_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Training curves saved to artifacts/visualizations/training_curves.png")
    plt.show()


if __name__ == "__main__":
    plot_training_history()
