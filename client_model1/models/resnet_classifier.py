"""
ResNet-18 based Cat/Dog Classifier
Uses transfer learning with custom classification head
"""

import torch
import torch.nn as nn
from torchvision import models

class CatDogClassifier(nn.Module):
    """
    ResNet-18 based binary classifier for cats vs dogs
    Includes embedding extraction capability for drift detection
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.5):
        super(CatDogClassifier, self).__init__()
        
        # Load pre-trained ResNet-18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Get the number of features from the last layer
        num_features = self.backbone.fc.in_features
        
        # Replace the final fully connected layer with custom classifier
        self.backbone.fc = nn.Identity()  # Remove original FC
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
        # Store embedding dimension
        self.embedding_dim = num_features
    
    def forward(self, x, return_embedding=False):
        """
        Forward pass
        
        Args:
            x: Input tensor (B, C, H, W)
            return_embedding: If True, return (logits, embeddings)
        
        Returns:
            logits: Class logits (B, num_classes)
            embeddings (optional): Feature embeddings (B, embedding_dim)
        """
        # Extract features from backbone
        embeddings = self.backbone(x)  # (B, 512)
        
        # Classification
        logits = self.classifier(embeddings)  # (B, num_classes)
        
        if return_embedding:
            return logits, embeddings
        else:
            return logits
    
    def get_embeddings(self, x):
        """Extract only embeddings (for drift detection)"""
        with torch.no_grad():
            embeddings = self.backbone(x)
        return embeddings


def create_model(num_classes=2, pretrained=True, dropout=0.5, device='cuda'):
    """
    Create and initialize the model
    
    Args:
        num_classes: Number of output classes (2 for cat/dog)
        pretrained: Use ImageNet pre-trained weights
        dropout: Dropout probability
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Initialized model on specified device
    """
    model = CatDogClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


def freeze_backbone(model, freeze=True):
    """
    Freeze/unfreeze backbone parameters for transfer learning
    
    Args:
        model: CatDogClassifier model
        freeze: If True, freeze backbone; if False, unfreeze
    """
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    print(f"Backbone parameters {status}")


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(device=device)
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test normal forward
    logits = model(dummy_input)
    print(f"\nLogits shape: {logits.shape}")
    
    # Test with embeddings
    logits, embeddings = model(dummy_input, return_embedding=True)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test embedding extraction
    embeddings_only = model.get_embeddings(dummy_input)
    print(f"Embeddings only shape: {embeddings_only.shape}")
    
    print("\n✓ Model test passed!")
