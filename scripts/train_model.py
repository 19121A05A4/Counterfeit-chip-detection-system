"""
Main Training Script
Author: SAI KATARI

Execute complete training pipeline for IC authentication models.
"""

import sys
sys.path.append('.')

import torch
import yaml
import logging
from pathlib import Path

from source.preparation.torch_data_pipeline import ICDataLoaderFactory
from source.architecture.neural_networks import create_ic_authentication_model
from source.training.model_trainer import ICAuthenticationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('artifacts/logs/training.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_configuration(config_path: str = 'config/settings.yaml') -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Execute the training pipeline."""
    
    # Load configuration
    config = load_configuration()
    
    logger.info("\n" + "="*70)
    logger.info("IC AUTHENTICATION SYSTEM - TRAINING PIPELINE")
    logger.info("="*70 + "\n")
    
    # Create log directory
    Path('artifacts/logs').mkdir(parents=True, exist_ok=True)
    
    # Step 1: Prepare data loaders
    logger.info("Step 1: Preparing data loaders...")
    
    data_factory = ICDataLoaderFactory(
        dataset_directory=config['data_configuration']['raw_images_path'],
        batch_size=config['network_training']['optimization_params']['samples_per_batch'],
        validation_fraction=config['data_configuration']['partitioning']['validation_fraction'],
        test_fraction=config['data_configuration']['partitioning']['test_fraction']
    )
    
    train_loader, val_loader, test_loader = data_factory.build_data_loaders()
    
    logger.info("✓ Data loaders ready\n")
    
    # Step 2: Create model
    logger.info("Step 2: Creating neural network...")
    
    model_architecture = config['network_training']['model_architecture']
    use_pretrained = config['network_training']['use_imagenet_weights']
    
    model = create_ic_authentication_model(
        architecture=model_architecture,
        pretrained=use_pretrained
    )
    
    logger.info("✓ Model created\n")
    
    # Step 3: Initialize trainer
    logger.info("Step 3: Initializing trainer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    trainer = ICAuthenticationTrainer(
        neural_network=model,
        train_data_loader=train_loader,
        validation_data_loader=val_loader,
        device=device,
        learning_rate=config['network_training']['optimization_params']['learning_rate'],
        weight_decay=config['network_training']['optimization_params']['l2_regularization']
    )
    
    logger.info("✓ Trainer ready\n")
    
    # Step 4: Train the model
    logger.info("Step 4: Starting training...\n")
    
    max_epochs = config['network_training']['optimization_params']['maximum_epochs']
    history = trainer.train(num_epochs=max_epochs)
    
    logger.info("✓ Training complete!")
    
    # Step 5: Save final results
    logger.info("\nStep 5: Saving results...")
    
    import json
    results_path = Path('artifacts/training_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'model_architecture': model_architecture,
            'final_train_accuracy': history['train_accuracy'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'best_val_accuracy': trainer.best_validation_accuracy,
            'total_epochs': trainer.current_epoch
        }, f, indent=2)
    
    logger.info(f"✓ Results saved to {results_path}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*70 + "\n")


if __name__ == "__main__":
    main()
