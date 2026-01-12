# Bearing Fault Detection using Deep Learning

A production-ready machine learning system for automated bearing fault diagnosis using advanced signal processing and deep learning techniques. Achieves **99.26% accuracy** on industrial bearing datasets.

## Overview

This project implements a multi-scale attention fusion neural network for detecting and classifying bearing faults from vibration signals. The system processes raw accelerometer data and classifies bearing conditions into multiple fault categories with high precision.

### Key Features

- **Multi-Scale Feature Extraction**: Combines envelope spectrum, multi-scale FFT, and time-domain statistics
- **Attention Mechanism**: Adaptive feature fusion using attention layers
- **High Accuracy**: 99.26% test accuracy on Paderborn dataset
- **Domain Adaptation**: Experimental cross-dataset generalization (CWRU ↔ Paderborn)
- **Comprehensive Validation**: Data leakage verification, cross-validation, and ablation studies
- **Production Ready**: Optimized models with checkpoints and inference pipelines

## Performance

| Model | Dataset | Accuracy | Balanced Accuracy |
|-------|---------|----------|-------------------|
| MultiScaleAttentionFusion-3Class | Paderborn | 99.26% | 99.31% |
| MultiScaleAttentionFusion-4Class | Paderborn | ~98% | ~98% |
| DANN (Cross-Dataset) | CWRU→Paderborn | 85.71% (domain) | 35.71% (class) |

## Architecture

The core architecture employs a multi-branch feature extraction approach:

```
Input Signal (Vibration Data)
    ├── Envelope Spectrum Branch (FFT of Hilbert Envelope)
    ├── Multi-Scale FFT Branch (3 scales: 2048, 8192, 32768)
    └── Time-Domain Statistics Branch (8 features)
              ↓
    Attention Fusion Layer
              ↓
    Classification Head
```

### Signal Processing Pipeline

1. **Envelope Spectrum**: Butterworth bandpass filter (1-5 kHz) → Hilbert transform → FFT
2. **Multi-Scale FFT**: Three scales with Hann windowing for multi-resolution analysis
3. **Time Statistics**: Mean, STD, RMS, crest factor, skewness, kurtosis, max, min
4. **Windowing**: Overlapping windows (2048 samples, stride 512)
5. **Normalization**: Z-score normalization per window

## Project Structure

```
├── 01_MAIN_RESULTS/                    # Production models and results
│   ├── Paderborn_3Class_Supervised/    # Best performing 3-class model
│   ├── Paderborn_4Class_Supervised/    # 4-class model with cage detection
│   └── create_visualizations.py        # Result visualization tools
│
├── 02_VALIDATION_EXPERIMENTS/          # Model validation and testing
│   ├── cross_validate_models.py        # Multi-seed cross-validation
│   ├── verify_data_leakage.py          # Data integrity checks
│   └── test_reverse_domain.py          # Reverse domain adaptation tests
│
├── 03_EARLIER_CWRU_EXPERIMENTS/        # CWRU dataset experiments
│   ├── CWRU_Baseline/                  # Baseline implementations
│   └── CWRU_Improved/                  # Enhanced models with augmentation
│
├── 04_PADERBORN_BASELINE_ATTEMPTS/     # Initial Paderborn experiments
│   ├── dataset_paderborn_supervised.py
│   ├── model_paderborn.py
│   └── train_paderborn.py
│
├── 05_HYBRID_COMPLEX_MODELS/           # Advanced hybrid architectures
│   ├── model_hybrid.py                 # ECA + dilated convolutions
│   └── train_hybrid.py
│
└── 06_DOMAIN_ADAPTATION_ATTEMPTS/      # Cross-dataset generalization
    ├── model_dann.py                   # Domain-adversarial neural network
    ├── train_dann.py
    └── dataset_dann.py
```

## Datasets

### Paderborn University Bearing Dataset
- **Sampling Rate**: 25,600 Hz
- **Classes**:
  - **3-Class**: Healthy, Inner Race Fault, Outer Race Fault
  - **4-Class**: Healthy, Inner Race, Outer Race, Cage Fault
- **Variants**: 27 different bearing damage codes
- **Channels**: Vibration (Module 1)

### CWRU (Case Western Reserve University) Dataset
- **Sampling Rate**: 12,000 Hz
- **Classes**: Healthy, Inner Race, Outer Race
- **Signal**: Drive-end accelerometer
- **Format**: `.mat` files

## Installation

### Requirements

```bash
Python 3.8+
PyTorch 1.10+
NumPy
SciPy
scikit-learn
matplotlib
seaborn
tqdm
```

### Setup

```bash
# Clone the repository
git clone https://github.com/Shawon559/Bearing-fault-detection.git
cd Bearing-fault-detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```python
# Train 3-class Paderborn model
cd 01_MAIN_RESULTS/Paderborn_3Class_Supervised
python train_paderborn_3class_improved.py

# Train 4-class Paderborn model
cd 01_MAIN_RESULTS/Paderborn_4Class_Supervised
python train_paderborn_4class_improved.py
```

### Inference

```python
import torch
from model_paderborn_3class import MultiScaleAttentionFusion3Class

# Load trained model
model = MultiScaleAttentionFusion3Class(...)
checkpoint = torch.load('checkpoints_3class_improved/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(features)
```

### Validation & Testing

```python
# Verify data integrity
cd 02_VALIDATION_EXPERIMENTS
python verify_data_leakage.py

# Run cross-validation
python cross_validate_models.py

# Test reverse domain adaptation
python test_reverse_domain.py
```

## Model Details

### MultiScaleAttentionFusion3Class

**Architecture:**
- 3 independent FFT encoders (BatchNorm + ReLU)
- Attention-based fusion layer
- Dropout: 0.3
- Hidden dim: 64, Fusion dim: 128

**Training:**
- Optimizer: Adam
- Loss: Cross-Entropy
- Epochs: 60
- Batch size: 32
- Gradient clipping: max norm 1.0

### DANN (Domain-Adversarial Neural Network)

**Components:**
- Gradient reversal layer
- Label classifier (fault prediction)
- Domain discriminator (source vs target)
- Lambda scheduling for adversarial training

## Results & Visualizations

All results and visualizations are available in `01_MAIN_RESULTS/visualizations/`:

- Training/validation curves
- Confusion matrices
- Precision/Recall/F1 metrics
- Cross-validation results
- Model comparison dashboards

## Validation Strategy

1. **File-Level Separation**: Train/val/test splits at file level to prevent data leakage
2. **Window-Level Verification**: Confirmed no overlapping windows across splits
3. **Multi-Seed Testing**: Validated with 5 different random seeds (42, 123, 456, 789, 1011)
4. **Balanced Accuracy**: Accounts for class imbalance in evaluation

## Key Innovations

1. **Multi-Scale Feature Fusion**: Captures fault signatures at multiple frequency resolutions
2. **Attention Mechanism**: Adaptively weights different feature branches
3. **Advanced Signal Processing**: Combines envelope analysis with FFT for comprehensive feature extraction
4. **Domain Adaptation**: Experimental transfer learning between bearing datasets

## Future Work

- Real-time inference optimization
- Additional bearing datasets integration
- Improved domain adaptation techniques
- Deployment to edge devices
- Web API for cloud inference

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{bearing-fault-detection-2024,
  author = {Shawon},
  title = {Bearing Fault Detection using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Shawon559/Bearing-fault-detection}
}
```

## License

MIT License - See LICENSE file for details

## Contact

For questions or collaboration, open an issue on GitHub.

## Acknowledgments

- Paderborn University for the bearing dataset
- Case Western Reserve University for the CWRU dataset
- PyTorch community for the deep learning framework
