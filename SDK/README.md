# VisionGuard SDK

**Automated Drift Detection, Local Adaptation, and Federated Learning for Computer Vision Models**

## Overview

VisionGuard SDK wraps your existing CV models to provide:
- ✅ **Automatic drift detection** using three metrics (confidence, entropy, embeddings)
- ✅ **Local adaptation** triggered by reliability thresholds
- ✅ **Offline/online mode** with automatic FL server synchronization
- ✅ **Real-time monitoring** dashboard
- ✅ **Privacy-preserving** federated learning

## Project Structure

```
visionguard_sdk/
├── sdk/
│   └── visionguard.py          # Main SDK wrapper
├── ui/
│   └── dashboard.py            # Real-time monitoring dashboard
├── client_demo/
│   └── demo_complete.py        # Complete demonstration
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```python
from sdk.visionguard import VisionGuardSDK
from your_model import YourCVModel

# Load your trained model
model = YourCVModel()
model.load_state_dict(torch.load('model.pth'))

# Wrap with VisionGuard
sdk = VisionGuardSDK(
    model=model,
    device='cuda',
    window_size=500,
    drift_threshold=0.3,
    reliability_threshold=0.7,
    baseline_data=clean_dataloader,  # For establishing baseline
    fl_server_url=None,  # None = offline mode
    auto_adapt=True,
    enable_ui=True
)

# Use normally - drift detection is automatic!
result = sdk.predict(image_tensor)

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Drift Score: {result['drift_score']:.4f}")
print(f"Reliability: {result['reliability_score']:.4f}")
```

### 3. Run Complete Demo

```bash
cd client_demo
python demo_complete.py
```

This will:
1. Load your trained cat/dog classifier
2. Establish baseline from validation data
3. Test on clean data (low drift)
4. Test on drifted data (triggers adaptation)
5. Generate visualizations and reports

## How It Works

### Layer 1: Client CV Model
Your existing model - no changes required!

### Layer 2: SDK Wrapper
Intercepts outputs automatically using PyTorch hooks:
- Predictions
- Confidence scores
- Internal embeddings

### Layer 3: Drift Detection
Three complementary metrics:

**1. Confidence Drop**
```
Δconfidence = baseline_mean - current_mean
```

**2. Entropy Increase**
```
H(p) = -Σ pᵢ log(pᵢ)
Δentropy = current_mean - baseline_mean
```

**3. Embedding Shift (KL Divergence)**
```
KL(P₁||P₂) = log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 0.5
```

**Combined Drift Score:**
```
drift_score = 0.3×Δconf + 0.3×Δent + 0.4×KL
```

### Layer 4: Decision & Reliability
```
reliability_score = 1 - drift_score

IF drift_score > 0.3:
    → Alert drift event
    
IF reliability < 0.7:
    → Trigger local adaptation
    
IF reliability >= 0.7 AND online:
    → Send updates to FL server
ELSE:
    → Store locally until online
```

### Layer 5: FL Server
Reliability-weighted aggregation:
```
Global_ΔW = Σ(reliabilityᵢ × ΔWᵢ) / Σreliabilityᵢ
```

## Features

### Automatic Drift Detection
- Runs continuously during inference
- No manual intervention needed
- Uses sliding window (500-1000 samples)

### Local Adaptation
```python
# Automatically triggered when reliability < threshold
sdk.adapt_locally(
    adaptation_data=recent_dataloader,
    epochs=5,
    lr=0.0001
)
```

### Real-Time Monitoring
```python
from ui.dashboard import MonitoringDashboard

dashboard = MonitoringDashboard(sdk)
dashboard.show()  # Opens live dashboard
```

Displays:
- Confidence over time
- Entropy over time  
- Drift score with threshold zones
- Reliability score with zones
- Drift events and adaptation cycles

### Offline Mode
```python
# Initialize without server
sdk = VisionGuardSDK(model, fl_server_url=None)

# Updates stored locally
# ... inference continues ...

# Later, when online:
sdk.is_online = True
sdk.sync_pending_updates()  # Automatic sync!
```

### State Management
```python
# Save everything
sdk.save_state('visionguard_state.pth')

# Load later
sdk.load_state('visionguard_state.pth')
```

## Generated Outputs

### 1. Real-Time Dashboard
Interactive matplotlib dashboard showing:
- Live metric tracking
- Drift events highlighted
- Adaptation cycles marked
- Status information

### 2. Summary Report
```python
from ui.dashboard import create_summary_report

create_summary_report(sdk, 'drift_report.png')
```

Creates:
- Confidence time series & distribution
- Entropy time series & distribution
- Drift score with threshold
- Summary statistics table

### 3. Logs & Metrics
- `drift_events` - List of all detected drift events
- `adaptation_cycles` - Details of each adaptation
- `monitoring_data` - Complete time series data
- `pending_updates` - Queued FL updates (offline mode)

## Configuration Parameters

### Critical Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `window_size` | 500 | 100-2000 | Samples for drift detection |
| `drift_threshold` | 0.3 | 0.1-0.5 | Trigger for drift alert |
| `reliability_threshold` | 0.7 | 0.5-0.9 | Min for FL participation |
| `auto_adapt` | True | bool | Enable auto-adaptation |

### Drift Metric Weights

| Weight | Default | Description |
|--------|---------|-------------|
| `w_confidence` | 0.3 | Confidence metric weight |
| `w_entropy` | 0.3 | Entropy metric weight |
| `w_embedding` | 0.4 | Embedding metric weight (highest) |

### Adaptation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 5 | Fine-tuning epochs |
| `lr` | 1e-4 | Learning rate |

## Privacy & Security

**Server NEVER receives:**
- ✗ Raw images/videos
- ✗ Embeddings
- ✗ Labels
- ✗ Any PII

**Server ONLY receives:**
- ✓ Weight deltas (ΔW)
- ✓ Reliability scores
- ✓ Drift signatures (metadata)

## Example: Cat/Dog Classifier

See `client_demo/demo_complete.py` for full example:

```python
# 1. Load model
model = create_model(num_classes=2)
model.load_state_dict(torch.load('best_model.pth'))

# 2. Initialize SDK
sdk = VisionGuardSDK(model, baseline_data=val_loader)

# 3. Test on clean data
for image, label in test_clean_loader:
    result = sdk.predict(image.squeeze(0))
    # Low drift expected

# 4. Test on drifted data  
for image, label in test_drift_loader:
    result = sdk.predict(image.squeeze(0))
    # Drift detected! Adaptation triggered automatically
```

## Troubleshooting

### High False Positive Rate
- Increase `drift_threshold` (e.g., 0.4 or 0.5)
- Increase `window_size` for more smoothing

### Missed Drift Events
- Decrease `drift_threshold` (e.g., 0.2)
- Increase `w_embedding` weight
- Ensure baseline is representative

### Slow Performance
- Reduce `window_size`
- Disable UI if not needed: `enable_ui=False`
- Use smaller adaptation epochs

## Technical Documentation

See `VisionGuard_Technical_Documentation.docx` for:
- Detailed algorithm descriptions
- Mathematical formulas
- Parameter tuning guide
- Architecture diagrams
- Implementation examples

## Citation

If you use VisionGuard SDK in your research, please cite:

```
@software{visionguard2025,
  title={VisionGuard SDK: Automated Drift Detection for Computer Vision},
  author={Your Team},
  year={2025}
}
```

## License

MIT License - See LICENSE file

## Support

For issues or questions:
- GitHub Issues: [your-repo]
- Email: support@visionguard.ai
- Documentation: [docs link]
