# Cat/Dog Image Classification with Drift Detection

## Project Overview
This project demonstrates model drift in computer vision by training a cat/dog classifier and evaluating its performance degradation under distribution shift.

## Project Structure
```
cat_dog_classifier/
├── data/
│   ├── prepare_dataset.py      # Split dataset into train/val/test
│   └── augmentations.py        # Define clean & drift transforms
├── models/
│   └── resnet_classifier.py    # ResNet-18 model definition
├── train.py                     # Training script
├── evaluate.py                  # Evaluation with drift metrics
├── visualize.py                 # Generate all graphs
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Update the dataset path in `data/prepare_dataset.py`:**
```python
DATASET_PATH = r"C:\Users\SRIVANDHI\CIP\archive\animals"
```

**Run data preparation:**
```bash
cd cat_dog_classifier
python data/prepare_dataset.py
```

This will create:
- `prepared_data/train/` (70% of data)
- `prepared_data/val/` (15% of data)
- `prepared_data/test_clean/` (7.5% of data)
- `prepared_data/test_drift/` (7.5% of data)

### 3. Train the Model

```bash
python train.py
```

Training features:
- Transfer learning from ImageNet-pretrained ResNet-18
- Initial backbone freezing (first 5 epochs)
- Fine-tuning with lower learning rate
- Early stopping with patience=5
- Model checkpointing (saves best model)

Expected training time: 15-30 minutes on GPU

Outputs:
- `outputs/best_model.pth` - Best model checkpoint
- `outputs/last_checkpoint.pth` - Latest checkpoint
- `outputs/history.json` - Training history
- `outputs/config.json` - Training configuration

### 4. Evaluate the Model

```bash
python evaluate.py
```

This script:
1. Loads the trained model
2. Evaluates on clean test set
3. Evaluates on drift test set (with augmentations)
4. Computes drift metrics:
   - Confidence drop
   - Entropy increase
   - Embedding distribution shift (KL divergence)
5. Calculates drift score and reliability score

Outputs:
- `outputs/evaluation_results.json` - Complete evaluation metrics
- `outputs/embeddings.pth` - Saved embeddings for visualization

### 5. Generate Visualizations

```bash
python visualize.py
```

Creates publication-ready graphs:
1. **training_history.png** - Loss and accuracy curves during training
2. **accuracy_comparison.png** - Bar chart showing performance drop
3. **confidence_distribution.png** - Histogram of prediction confidence
4. **entropy_comparison.png** - Uncertainty comparison
5. **confusion_matrices.png** - Side-by-side confusion matrices
6. **embedding_space.png** - t-SNE visualization showing distribution shift
7. **embedding_space_umap.png** - UMAP visualization (if available)
8. **metrics_summary.png** - Comprehensive 4-panel summary

All graphs are saved at 300 DPI for presentation quality.

## Expected Results

### Training Performance
- Training Accuracy: 95-98%
- Validation Accuracy: 93-97%
- Convergence: ~15-25 epochs

### Clean Test Set
- Accuracy: 93-97%
- Mean Confidence: 0.90-0.95
- Mean Entropy: 0.10-0.20

### Drift Test Set
- Accuracy: 70-85% (↓ 8-25%)
- Mean Confidence: 0.65-0.80 (↓ 15-30%)
- Mean Entropy: 0.25-0.40 (↑ 50-100%)

### Drift Detection
- Drift Score: 0.3-0.5 (higher = more drift)
- Reliability Score: 0.5-0.7 (lower = less reliable)

## Understanding the Drift

The model experiences drift due to these applied transformations on the drift test set:

1. **Lighting Shift**: Brightness ±30%, Contrast ±25%
2. **Color Shift**: Hue ±20°, Saturation ±25%
3. **Blur**: Gaussian blur (σ=2)
4. **Rotation**: ±15 degrees
5. **Scaling**: 0.8x to 1.2x
6. **Noise**: Gaussian noise (σ=0.05)

These simulate real-world scenarios:
- Different cameras/sensors
- Varying lighting conditions
- Motion blur or focus issues
- Different viewing angles

## Drift Detection Metrics

### 1. Prediction Confidence
```
Δconfidence = mean(confidence_clean) - mean(confidence_drift)
```
Drop in confidence indicates model uncertainty.

### 2. Output Entropy
```
H(p) = -Σ p_i * log(p_i)
Δentropy = mean(entropy_drift) - mean(entropy_clean)
```
Higher entropy = more uncertain predictions.

### 3. Embedding Distribution Shift
```
KL(P_clean || P_drift) = KL divergence between distributions
```
Measures how much the internal representations have shifted.

### Combined Drift Score
```
drift_score = w1·Δconfidence + w2·Δentropy + w3·KL_divergence
reliability_score = 1 - drift_score
```
Default weights: w1=0.3, w2=0.3, w3=0.4

## Customization

### Modify Drift Intensity
In `data/augmentations.py`, adjust transformation parameters:

```python
# More extreme drift
transforms.ColorJitter(brightness=0.5, contrast=0.4, ...)
transforms.GaussianBlur(kernel_size=7, sigma=(1.0, 3.0))
GaussianNoise(mean=0, std=0.08)

# Milder drift
transforms.ColorJitter(brightness=0.15, contrast=0.15, ...)
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
GaussianNoise(mean=0, std=0.02)
```

### Change Model Architecture
In `models/resnet_classifier.py`, try different backbones:

```python
# Use ResNet-50 instead
self.backbone = models.resnet50(pretrained=pretrained)

# Use EfficientNet
from torchvision.models import efficientnet_b0
self.backbone = efficientnet_b0(pretrained=pretrained)
```

### Adjust Training Hyperparameters
In `train.py`, modify the CONFIG dictionary:

```python
CONFIG = {
    'batch_size': 64,           # Larger batch for faster training
    'learning_rate': 0.0005,    # Lower LR for more stable training
    'num_epochs': 50,           # More epochs
    'freeze_backbone_epochs': 10,  # Freeze longer
}
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `train.py`:
```python
'batch_size': 16,  # or 8
```

### Low Training Accuracy
- Check if dataset loaded correctly
- Increase number of epochs
- Try unfreezing backbone earlier

### Similar Clean and Drift Performance
- Increase drift transformation intensity
- Use `extreme_drift_transform` in `augmentations.py`
- Apply multiple drift types simultaneously

## Next Steps

After completing this baseline:

1. **Implement SDK Wrapper** - Monitor model outputs in real-time
2. **Build Drift Detection Module** - Use the three metrics continuously
3. **Add Local Adaptation** - Fine-tune when drift detected
4. **Create FL Server** - Aggregate updates with reliability weighting
5. **Multi-Client Simulation** - Test with different drift patterns

## Presentation Tips

For your review, focus on:
1. **Accuracy comparison bar chart** - Shows clear degradation
2. **Embedding space t-SNE** - Visual proof of distribution shift
3. **Confidence distribution** - Shows model becoming uncertain
4. **Metrics summary** - Comprehensive overview

Emphasize:
- Real-world relevance (cameras, lighting, sensors)
- Quantifiable degradation (accuracy, confidence, entropy)
- Automated detection (no manual intervention needed)

## Questions?

Common issues and solutions are documented in the code comments.
Check function docstrings for detailed parameter explanations.
