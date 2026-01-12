# API Documentation

## Models

### MultiScaleAttentionFusion3Class

3-class bearing fault classification model.

```python
from src.models.model_paderborn_3class import MultiScaleAttentionFusion3Class

model = MultiScaleAttentionFusion3Class(
    envelope_dim=1024,
    fft_dims=[1024, 4096, 16384],
    stats_dim=8,
    hidden_dim=64,
    fusion_dim=128,
    num_classes=3,
    dropout=0.3
)
```

**Parameters:**
- `envelope_dim` (int): Dimension of envelope spectrum features
- `fft_dims` (list): Dimensions for each FFT scale [scale1, scale2, scale3]
- `stats_dim` (int): Dimension of statistical features (default: 8)
- `hidden_dim` (int): Hidden layer dimension (default: 64)
- `fusion_dim` (int): Fusion layer dimension (default: 128)
- `num_classes` (int): Number of output classes (default: 3)
- `dropout` (float): Dropout probability (default: 0.3)

**Input:**
- `envelope`: Envelope spectrum features [batch, envelope_dim]
- `fft_scale1`: First scale FFT features [batch, fft_dims[0]]
- `fft_scale2`: Second scale FFT features [batch, fft_dims[1]]
- `fft_scale3`: Third scale FFT features [batch, fft_dims[2]]
- `stats`: Statistical features [batch, stats_dim]

**Output:**
- `logits`: Class logits [batch, num_classes]

**Classes:**
- 0: Healthy bearing
- 1: Inner race fault
- 2: Outer race fault

### MultiScaleAttentionFusion4Class

4-class bearing fault classification model (includes cage faults).

```python
from src.models.model_paderborn_4class import MultiScaleAttentionFusion4Class

model = MultiScaleAttentionFusion4Class(
    envelope_dim=1024,
    fft_dims=[1024, 4096, 16384],
    stats_dim=8,
    hidden_dim=64,
    fusion_dim=128,
    num_classes=4,
    dropout=0.3
)
```

**Classes:**
- 0: Healthy bearing
- 1: Inner race fault
- 2: Outer race fault
- 3: Cage fault

## Datasets

### Paderborn3ClassDataset

Dataset loader for Paderborn bearing data with 3-class classification.

```python
from src.datasets.dataset_paderborn_3class import Paderborn3ClassDataset, create_loader

# Create dataset
dataset = Paderborn3ClassDataset(
    data_root='data/',
    split='train',
    window_size=2048,
    stride=512,
    seed=42
)

# Create dataloader
loader = create_loader(dataset, batch_size=32, shuffle=True)
```

**Parameters:**
- `data_root` (str): Path to Paderborn dataset directory
- `split` (str): Data split - 'train', 'val', or 'test'
- `window_size` (int): Size of sliding window (default: 2048)
- `stride` (int): Stride for sliding window (default: 512)
- `seed` (int): Random seed for reproducibility (default: 42)

**Returns:** Dictionary with keys:
- `envelope`: Envelope spectrum features
- `fft_scale1`: First scale FFT
- `fft_scale2`: Second scale FFT
- `fft_scale3`: Third scale FFT
- `stats`: Time-domain statistics
- `label`: Class label (0, 1, or 2)

### Paderborn4ClassDataset

Dataset loader for Paderborn bearing data with 4-class classification.

```python
from src.datasets.dataset_paderborn_4class import Paderborn4ClassDataset, create_loader

dataset = Paderborn4ClassDataset(
    data_root='data/',
    split='train',
    window_size=2048,
    stride=512,
    seed=42
)
```

**Classes:** Same as 3-class plus cage faults (class 3)

## Feature Extraction

### Envelope Spectrum

```python
from src.datasets.dataset_paderborn_3class import compute_envelope_spectrum

envelope_fft = compute_envelope_spectrum(
    signal,
    sample_rate=25600,
    bandpass_low=1000,
    bandpass_high=5000
)
```

**Parameters:**
- `signal` (np.ndarray): Input vibration signal
- `sample_rate` (int): Sampling rate in Hz
- `bandpass_low` (int): Low cutoff frequency for bandpass filter
- `bandpass_high` (int): High cutoff frequency for bandpass filter

**Returns:** Envelope spectrum FFT magnitude

### Multi-Scale FFT

```python
from src.datasets.dataset_paderborn_3class import compute_fft_multiscale

fft_features = compute_fft_multiscale(
    signal,
    nfft_scales=[2048, 8192, 32768]
)
```

**Parameters:**
- `signal` (np.ndarray): Input vibration signal
- `nfft_scales` (list): FFT sizes for each scale

**Returns:** List of FFT magnitudes for each scale

### Time-Domain Statistics

```python
from src.datasets.dataset_paderborn_3class import compute_time_statistics

stats = compute_time_statistics(signal)
```

**Returns:** Array with 8 statistical features:
- Mean, Standard deviation, RMS, Crest factor
- Skewness, Kurtosis, Maximum, Minimum

## Training

### Training Loop

```python
from src.training.train_3class import train_one_epoch, validate

# Training
train_loss, train_acc = train_one_epoch(
    model, train_loader, optimizer, criterion, device, epoch
)

# Validation
val_loss, val_acc, val_balanced_acc, cm, all_labels = validate(
    model, val_loader, criterion, device
)
```

**Parameters:**
- `model`: PyTorch model
- `train_loader`/`val_loader`: DataLoader instances
- `optimizer`: PyTorch optimizer (e.g., Adam)
- `criterion`: Loss function (e.g., CrossEntropyLoss)
- `device`: torch.device ('cuda', 'cpu', or 'mps')
- `epoch`: Current epoch number

**Returns (train_one_epoch):**
- `train_loss`: Average training loss
- `train_acc`: Training accuracy percentage

**Returns (validate):**
- `val_loss`: Average validation loss
- `val_acc`: Validation accuracy percentage
- `val_balanced_acc`: Balanced accuracy (for imbalanced classes)
- `cm`: Confusion matrix
- `all_labels`: Ground truth labels

## Inference

### Basic Inference

```python
import torch
from src.models.model_paderborn_3class import MultiScaleAttentionFusion3Class

# Load model
model = MultiScaleAttentionFusion3Class(...)
checkpoint = torch.load('checkpoints/model_3class/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
# ... extract features from signal ...

# Inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

with torch.no_grad():
    logits = model(envelope, fft1, fft2, fft3, stats)
    probs = torch.softmax(logits, dim=1)
    predictions = torch.argmax(logits, dim=1)

# Get class name
class_names = ['Healthy', 'Inner Fault', 'Outer Fault']
predicted_class = class_names[predictions[0].item()]
confidence = probs[0, predictions[0]].item()

print(f"Prediction: {predicted_class} (confidence: {confidence:.2%})")
```

### Batch Inference

```python
def predict_batch(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in dataloader:
            envelope = batch['envelope'].to(device)
            fft1 = batch['fft_scale1'].to(device)
            fft2 = batch['fft_scale2'].to(device)
            fft3 = batch['fft_scale3'].to(device)
            stats = batch['stats'].to(device)

            logits = model(envelope, fft1, fft2, fft3, stats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())

    return all_predictions, all_probabilities
```

## Utilities

### Model Checkpointing

```python
# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_acc': val_acc
}, 'checkpoints/checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

### Visualization

```python
from src.utils.visualize_results import plot_training_curves, plot_confusion_matrix

# Plot training curves
plot_training_curves(
    history,
    save_path='results/training_curves.png'
)

# Plot confusion matrix
plot_confusion_matrix(
    cm,
    class_names=['Healthy', 'Inner', 'Outer'],
    save_path='results/confusion_matrix.png'
)
```

## Configuration

### Recommended Hyperparameters

**For 3-Class Model:**
```python
config = {
    'hidden_dim': 64,
    'fusion_dim': 128,
    'dropout': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 60,
    'weight_decay': 1e-5
}
```

**For 4-Class Model:**
```python
config = {
    'hidden_dim': 64,
    'fusion_dim': 128,
    'dropout': 0.3,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'epochs': 80,
    'weight_decay': 1e-5
}
```

## Error Handling

All functions raise standard Python exceptions:
- `FileNotFoundError`: Dataset files not found
- `ValueError`: Invalid parameters or data format
- `RuntimeError`: Model or training errors

Example error handling:

```python
try:
    dataset = Paderborn3ClassDataset(data_root='data/')
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
except Exception as e:
    print(f"Error loading dataset: {e}")
```
