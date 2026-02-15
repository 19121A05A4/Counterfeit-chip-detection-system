"""
Neural Network Architectures for IC Authentication
Author: SAI KATARI

My custom model implementations using transfer learning.
Each architecture is optimized specifically for small IC datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from efficientnet_pytorch import EfficientNet
import logging

logger = logging.getLogger(__name__)


class ICAuthenticationNetworkBase(nn.Module):
    """
    My base class for all IC authentication models.
    
    Why a base class? Ensures consistent interface across architectures
    and makes swapping models easier during experiments.
    """
    
    def __init__(self):
        super().__init__()
        self.model_name = "base"
    
    def get_model_info(self) -> dict:
        """Return model metadata."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }


class VGG16ICClassifier(ICAuthenticationNetworkBase):
    """
    VGG16-based classifier for IC authentication.
    
    My modifications:
    - Custom classifier head with dropout
    - Optional fine-tuning of later conv blocks
    """
    
    def __init__(self, use_pretrained: bool = True, fine_tune_blocks: int = 0):
        """
        Initialize VGG16 model.
        
        Args:
            use_pretrained: Load ImageNet weights
            fine_tune_blocks: Number of conv blocks to unfreeze (0-5)
        """
        super().__init__()
        self.model_name = "VGG16_IC_Classifier"
        
        # Load pre-trained VGG16
        vgg_backbone = models.vgg16(pretrained=use_pretrained)
        
        # Extract feature extractor
        self.feature_layers = vgg_backbone.features
        
        # Freeze feature layers initially
        for param in self.feature_layers.parameters():
            param.requires_grad = False
        
        # Optionally unfreeze later blocks for fine-tuning
        if fine_tune_blocks > 0:
            self._unfreeze_last_blocks(fine_tune_blocks)
        
        # My custom classifier for IC binary classification
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # Binary output (will use BCEWithLogitsLoss)
        )
        
        logger.info(f"VGG16 initialized (fine-tune blocks: {fine_tune_blocks})")
    
    def _unfreeze_last_blocks(self, num_blocks: int):
        """Unfreeze the last N conv blocks."""
        # VGG16 has 5 conv blocks
        blocks = [self.feature_layers[i] for i in range(len(self.feature_layers))]
        
        # Unfreeze from the end
        params_to_unfreeze = blocks[-(num_blocks * 6):]  # Each block ≈ 6 layers
        
        for layer in params_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Unfroze last {num_blocks} conv blocks")
    
    def forward(self, x):
        """Forward pass."""
        features = self.feature_layers(x)
        output = self.classification_head(features)
        return output


class EfficientNetICClassifier(ICAuthenticationNetworkBase):
    """
    EfficientNet-B0 for IC authentication.
    
    My choice: EfficientNet is more parameter-efficient than VGG,
    making it better suited for small datasets.
    """
    
    def __init__(self, use_pretrained: bool = True):
        """
        Initialize EfficientNet-B0.
        
        Args:
            use_pretrained: Load ImageNet weights
        """
        super().__init__()
        self.model_name = "EfficientNetB0_IC_Classifier"
        
        # Load EfficientNet
        if use_pretrained:
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.backbone = EfficientNet.from_name('efficientnet-b0')
        
        # Get number of features from backbone
        num_backbone_features = self.backbone._fc.in_features
        
        # Replace final layer with my custom head
        self.backbone._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_backbone_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        logger.info("EfficientNet-B0 initialized")
    
    def forward(self, x):
        """Forward pass."""
        return self.backbone(x)


class ResNet50ICClassifier(ICAuthenticationNetworkBase):
    """
    ResNet50 for IC authentication.
    
    My implementation: Use ResNet's residual connections
    which help with gradient flow in deeper networks.
    """
    
    def __init__(self, use_pretrained: bool = True):
        """
        Initialize ResNet50.
        
        Args:
            use_pretrained: Load ImageNet weights
        """
        super().__init__()
        self.model_name = "ResNet50_IC_Classifier"
        
        # Load ResNet50
        resnet_backbone = models.resnet50(pretrained=use_pretrained)
        
        # Extract feature extractor (everything except final FC)
        self.feature_extractor = nn.Sequential(*list(resnet_backbone.children())[:-1])
        
        # My custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        logger.info("ResNet50 initialized")
    
    def forward(self, x):
        """Forward pass."""
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output


def create_ic_authentication_model(architecture: str = "efficientnet",
                                   pretrained: bool = True) -> ICAuthenticationNetworkBase:
    """
    Factory function to create models.
    
    My design: Single entry point for model creation makes
    experimentation cleaner.
    
    Args:
        architecture: 'vgg16', 'efficientnet', or 'resnet50'
        pretrained: Use ImageNet weights
        
    Returns:
        Model instance
    """
    architecture = architecture.lower()
    
    if architecture == 'vgg16':
        model = VGG16ICClassifier(use_pretrained=pretrained, fine_tune_blocks=0)
    elif architecture == 'efficientnet':
        model = EfficientNetICClassifier(use_pretrained=pretrained)
    elif architecture == 'resnet50':
        model = ResNet50ICClassifier(use_pretrained=pretrained)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Choose from: vgg16, efficientnet, resnet50")
    
    # Log model info
    info = model.get_model_info()
    logger.info(f"\nModel Created: {info['name']}")
    logger.info(f"  Total parameters: {info['total_parameters']:,}")
    logger.info(f"  Trainable parameters: {info['trainable_parameters']:,}")
    logger.info(f"  Frozen parameters: {info['frozen_parameters']:,}\n")
    
    return model


# Test model creation
if __name__ == "__main__":
    print("Testing Model Architectures\n")
    
    # Test each architecture
    for arch in ['vgg16', 'efficientnet', 'resnet50']:
        print(f"\nTesting {arch.upper()}:")
        
        model = create_ic_authentication_model(arch, pretrained=False)
        
        # Test forward pass with dummy data
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"  ✓ Input shape: {dummy_input.shape}")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Output range: [{output.min():.3f}, {output.max():.3f}]")
