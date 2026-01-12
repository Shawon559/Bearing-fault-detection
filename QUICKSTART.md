# Quick Start Guide

Get started with the Bearing Fault Detection system in 5 minutes.

## Installation

```bash
git clone https://github.com/Shawon559/Bearing-fault-detection.git
cd Bearing-fault-detection
pip install -r requirements.txt
```

## Using Pre-trained Models

### Load and Test the 3-Class Model (99.26% accuracy)

```python
import torch
from src.models.model_paderborn_3class import MultiScaleAttentionFusion3Class

# Load model
checkpoint = torch.load('checkpoints/model_3class/best_model.pth',
                       map_location='cpu')

# Create model instance (dimensions from checkpoint)
model = MultiScaleAttentionFusion3Class(
    envelope_dim=1024,
    fft_dims=[1024, 4096, 16384],
    stats_dim=8,
    hidden_dim=64,
    fusion_dim=128,
    num_classes=3,
    dropout=0.3
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded! Validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
```

### Run Inference on New Data

```python
import numpy as np
from src.datasets.dataset_paderborn_3class import (
    compute_envelope_spectrum,
    compute_fft_multiscale,
    compute_time_statistics
)

# Your vibration signal (example: 2048 samples at 25.6 kHz)
signal = np.random.randn(2048)  # Replace with actual signal

# Extract features
envelope = compute_envelope_spectrum(signal, sample_rate=25600)
fft_features = compute_fft_multiscale(signal, nfft_scales=[2048, 8192, 32768])
stats = compute_time_statistics(signal)

# Prepare inputs
envelope_tensor = torch.FloatTensor(envelope).unsqueeze(0)
fft1_tensor = torch.FloatTensor(fft_features[0]).unsqueeze(0)
fft2_tensor = torch.FloatTensor(fft_features[1]).unsqueeze(0)
fft3_tensor = torch.FloatTensor(fft_features[2]).unsqueeze(0)
stats_tensor = torch.FloatTensor(stats).unsqueeze(0)

# Predict
with torch.no_grad():
    logits = model(envelope_tensor, fft1_tensor, fft2_tensor, fft3_tensor, stats_tensor)
    probs = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1)

# Interpret results
class_names = ['Healthy', 'Inner Race Fault', 'Outer Race Fault']
confidence = probs[0, prediction[0]].item()

print(f"Prediction: {class_names[prediction[0]]}")
print(f"Confidence: {confidence:.2%}")
print(f"All probabilities: {probs[0].numpy()}")
```

## Training from Scratch

### Prepare Your Dataset

1. Download the Paderborn dataset
2. Extract to `data/` directory
3. Structure should be:
```
data/
├── K001/  (Healthy bearings)
├── K002/
├── KI04/  (Inner race faults)
├── KI14/
├── KA01/  (Outer race faults)
└── ...
```

### Train 3-Class Model

```bash
cd src/training
python train_3class.py
```

This will:
- Load and preprocess the data
- Train for 60 epochs
- Save checkpoints to `checkpoints/model_3class/`
- Save training history to `results/`

### Monitor Training

```python
import json
import matplotlib.pyplot as plt

# Load training history
with open('results/model_3class_results/paderborn_3class_improved_history.json', 'r') as f:
    history = json.load(f)

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(history['train_acc'], label='Train')
plt.plot(history['val_acc'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training Progress')
plt.grid(True)
plt.show()
```

## Validation

### Verify Data Integrity

```bash
cd experiments/validation
python verify_data_leakage.py
```

This confirms no data leakage between train/val/test splits.

### Cross-Validation

```bash
python cross_validate_models.py
```

Tests model with multiple random seeds to verify stability.

## Common Use Cases

### 1. Quick Health Check

```python
def check_bearing_health(signal, model):
    """Quick bearing health assessment."""
    # Extract features
    envelope = compute_envelope_spectrum(signal, sample_rate=25600)
    fft_features = compute_fft_multiscale(signal)
    stats = compute_time_statistics(signal)

    # Predict
    with torch.no_grad():
        # ... prepare tensors ...
        logits = model(envelope, fft1, fft2, fft3, stats)
        prob_healthy = torch.softmax(logits, dim=1)[0, 0].item()

    if prob_healthy > 0.95:
        return "HEALTHY", prob_healthy
    elif prob_healthy > 0.80:
        return "MONITOR", prob_healthy
    else:
        return "FAULT DETECTED", prob_healthy
```

### 2. Batch Processing

```python
from torch.utils.data import DataLoader
from src.datasets.dataset_paderborn_3class import Paderborn3ClassDataset

# Load test dataset
test_dataset = Paderborn3ClassDataset(
    data_root='data/',
    split='test',
    seed=42
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate
model.eval()
correct = 0
total = 0

for batch in test_loader:
    with torch.no_grad():
        outputs = model(
            batch['envelope'],
            batch['fft_scale1'],
            batch['fft_scale2'],
            batch['fft_scale3'],
            batch['stats']
        )
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == batch['label']).sum().item()
        total += batch['label'].size(0)

print(f"Accuracy: {100 * correct / total:.2f}%")
```

### 3. Real-time Monitoring

```python
import time

def monitor_bearing_realtime(signal_source, model, interval=1.0):
    """Monitor bearing in real-time."""
    class_names = ['Healthy', 'Inner Fault', 'Outer Fault']

    while True:
        # Get signal from source (DAQ, file, etc.)
        signal = signal_source.read_window(2048)

        # Extract and predict
        features = extract_features(signal)
        prediction, confidence = predict(model, features)

        # Log/alert
        status = class_names[prediction]
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status} ({confidence:.1%})")

        if status != 'Healthy' and confidence > 0.90:
            print("⚠️  ALERT: Fault detected!")

        time.sleep(interval)
```

## Troubleshooting

### Model not loading
- Ensure checkpoint file exists in `checkpoints/model_3class/`
- Check PyTorch version compatibility

### Dataset errors
- Verify data directory structure
- Check file permissions
- Ensure .mat files are valid

### Low accuracy
- Verify signal preprocessing
- Check sampling rate (should be 25,600 Hz for Paderborn)
- Ensure proper normalization

## Next Steps

1. Review the [API Documentation](docs/API.md) for detailed function references
2. Check [Installation Guide](docs/INSTALLATION.md) for advanced setup
3. Read the main [README.md](README.md) for project overview

## Performance Benchmarks

| Operation | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| Feature Extraction (1 window) | ~5ms | ~2ms |
| Model Inference (batch=32) | ~20ms | ~3ms |
| Full Dataset Evaluation | ~2min | ~15s |

## Support

For issues or questions:
- Check [docs/](docs/) directory
- Open an issue on GitHub
- Review code examples in this guide

Happy fault detecting!
