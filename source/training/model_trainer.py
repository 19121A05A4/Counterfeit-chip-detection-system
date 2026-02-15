"""
Model Training Orchestrator for IC Authentication
Author: [Your Name]
Date: February 2026

My custom training loop designed specifically for small IC datasets.
Implements my personal optimization strategy with early stopping,
learning rate scheduling, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ICAuthenticationTrainer:
    """
    My custom training orchestrator.
    
    Philosophy: Treat training as a carefully controlled experiment.
    Every hyperparameter choice has a reason, every metric is logged.
    """
    
    def __init__(self,
                 neural_network: nn.Module,
                 train_data_loader,
                 validation_data_loader,
                 device: str = 'cuda',
                 learning_rate: float = 0.0001,
                 weight_decay: float = 0.0001):
        """
        Initialize the trainer.
        
        Args:
            neural_network: The model to train
            train_data_loader: Training data
            validation_data_loader: Validation data
            device: 'cuda' or 'cpu'
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
        """
        self.model = neural_network
        self.train_loader = train_data_loader
        self.val_loader = validation_data_loader
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # My loss function choice: BCEWithLogitsLoss
        # Why? Combines sigmoid + BCE in one operation (numerically stable)
        self.loss_function = nn.BCEWithLogitsLoss()
        
        # My optimizer choice: AdamW (Adam with weight decay fix)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # My learning rate scheduler: Cosine Annealing with Warm Restarts
        # Why? Helps escape local minima and explores loss landscape
        self.lr_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs initially
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        
        # Training state tracking
        self.current_epoch = 0
        self.best_validation_loss = float('inf')
        self.best_validation_accuracy = 0.0
        self.epochs_without_improvement = 0
        self.early_stop_patience = 15
        
        # History tracking
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Checkpoint directory
        self.checkpoint_dir = Path('artifacts/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _compute_batch_metrics(self, 
                               predictions: torch.Tensor,
                               ground_truth: torch.Tensor) -> Tuple[float, float]:
        """
        Calculate loss and accuracy for a batch.
        
        My implementation: Clean separation of metric computation.
        
        Args:
            predictions: Raw model outputs (logits)
            ground_truth: True labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        # Compute loss
        batch_loss = self.loss_function(predictions.squeeze(), ground_truth)
        
        # Compute accuracy
        probabilities = torch.sigmoid(predictions.squeeze())
        predicted_classes = (probabilities > 0.5).float()
        correct_predictions = (predicted_classes == ground_truth).sum().item()
        batch_accuracy = correct_predictions / ground_truth.size(0)
        
        return batch_loss.item(), batch_accuracy
    
    def _train_single_epoch(self) -> Tuple[float, float]:
        """
        Execute one complete training epoch.
        
        My approach: Accumulate metrics across all batches,
        then return epoch-level averages.
        
        Returns:
            Tuple of (average_loss, average_accuracy)
        """
        self.model.train()  # Set to training mode
        
        epoch_loss_accumulator = 0.0
        epoch_accuracy_accumulator = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(images)
            
            # Compute loss
            loss = self.loss_function(predictions.squeeze(), labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            batch_loss, batch_acc = self._compute_batch_metrics(predictions, labels)
            epoch_loss_accumulator += batch_loss
            epoch_accuracy_accumulator += batch_acc
        
        # Calculate epoch averages
        avg_loss = epoch_loss_accumulator / num_batches
        avg_accuracy = epoch_accuracy_accumulator / num_batches
        
        return avg_loss, avg_accuracy
    
    def _validate(self) -> Tuple[float, float]:
        """
        Evaluate model on validation set.
        
        My approach: No gradient computation, deterministic evaluation.
        
        Returns:
            Tuple of (validation_loss, validation_accuracy)
        """
        self.model.eval()  # Set to evaluation mode
        
        val_loss_accumulator = 0.0
        val_accuracy_accumulator = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():  # Disable gradient computation
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass only
                predictions = self.model(images)
                
                # Compute metrics
                batch_loss, batch_acc = self._compute_batch_metrics(predictions, labels)
                val_loss_accumulator += batch_loss
                val_accuracy_accumulator += batch_acc
        
        avg_val_loss = val_loss_accumulator / num_batches
        avg_val_accuracy = val_accuracy_accumulator / num_batches
        
        return avg_val_loss, avg_val_accuracy
    
    def _save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        My strategy: Always save latest, also save best separately.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_val_loss': self.best_validation_loss,
            'best_val_accuracy': self.best_validation_accuracy,
            'training_history': self.training_history
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint_data, latest_path)
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint_data, best_path)
            logger.info(f"  ✓ Best model saved (Val Acc: {self.best_validation_accuracy:.4f})")
    
    def _check_early_stopping(self, current_val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        My criteria: Stop if validation loss doesn't improve for N epochs.
        
        Args:
            current_val_loss: Current validation loss
            
        Returns:
            True if training should stop
        """
        if current_val_loss < self.best_validation_loss:
            self.best_validation_loss = current_val_loss
            self.epochs_without_improvement = 0
            return False
        else:
            self.epochs_without_improvement += 1
            
            if self.epochs_without_improvement >= self.early_stop_patience:
                logger.info(f"\nEarly stopping triggered after {self.early_stop_patience} epochs without improvement")
                return True
            
            return False
    
    def train(self, num_epochs: int = 100) -> Dict[str, List[float]]:
        """
        Execute the complete training loop.
        
        My orchestration: Train, validate, track, checkpoint, repeat.
        
        Args:
            num_epochs: Maximum number of epochs
            
        Returns:
            Training history dictionary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting Training: {self.model.model_name}")
        logger.info(f"{'='*70}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"Batch size: {self.train_loader.batch_size}")
        logger.info(f"Max epochs: {num_epochs}")
        logger.info(f"Early stopping patience: {self.early_stop_patience}")
        logger.info(f"{'='*70}\n")
        
        training_start_time = datetime.now()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            # Train one epoch
            train_loss, train_acc = self._train_single_epoch()
            
            # Validate
            val_loss, val_acc = self._validate()
            
            # Update learning rate
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            # Check for best model
            is_best = val_acc > self.best_validation_accuracy
            if is_best:
                self.best_validation_accuracy = val_acc
            
            # Print progress
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            logger.info(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            logger.info(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            logger.info(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            self._save_checkpoint(is_best=is_best)
            
            # Early stopping check
            if self._check_early_stopping(val_loss):
                break
            
            logger.info("")  # Blank line for readability
        
        training_duration = datetime.now() - training_start_time
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Training Complete!")
        logger.info(f"{'='*70}")
        logger.info(f"Total time: {training_duration}")
        logger.info(f"Best validation accuracy: {self.best_validation_accuracy:.4f}")
        logger.info(f"Best validation loss: {self.best_validation_loss:.4f}")
        logger.info(f"Final epoch: {self.current_epoch}")
        logger.info(f"{'='*70}\n")
        
        return self.training_history


# Testing the trainer
if __name__ == "__main__":
    print("Training orchestrator module loaded successfully")
    print("Use this in your main training script")