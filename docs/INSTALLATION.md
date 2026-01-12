# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM
- 10GB+ disk space for datasets

## Step 1: Clone the Repository

```bash
git clone https://github.com/Shawon559/Bearing-fault-detection.git
cd Bearing-fault-detection
```

## Step 2: Create Virtual Environment

### Using venv (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Using conda

```bash
conda create -n bearing-fault python=3.9
conda activate bearing-fault
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### For GPU Support (CUDA)

If you have an NVIDIA GPU with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### For Apple Silicon (M1/M2/M3)

PyTorch with Metal Performance Shaders (MPS) support:

```bash
pip install torch torchvision
```

## Step 4: Download Datasets

### Paderborn Dataset

1. Visit [Paderborn Bearing Dataset](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/data-sets-and-download)
2. Download the vibration data files
3. Extract to `data/` directory in the project root

Expected structure:
```
data/
├── K001/
├── K002/
├── KA01/
├── KI04/
└── ...
```

### CWRU Dataset (Optional)

1. Visit [CWRU Bearing Dataset](https://engineering.case.edu/bearingdatacenter)
2. Download Drive End bearing data
3. Extract to `CWRU_Dataset/` directory

## Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Step 6: Test with Pre-trained Models

Download pre-trained checkpoints (if available) or train from scratch:

```bash
# Test 3-class model
cd src/training
python train_3class.py --test-only --checkpoint ../../checkpoints/model_3class/best_model.pth
```

## Troubleshooting

### Issue: CUDA out of memory

Solution: Reduce batch size in training scripts
```python
batch_size = 16  # or even 8
```

### Issue: Module not found errors

Solution: Ensure you're in the project root directory and virtual environment is activated
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: Dataset loading errors

Solution: Verify dataset paths in the dataset configuration files
```python
# Check data_root path in src/datasets/dataset_paderborn_3class.py
```

## Development Installation

For development with testing and linting tools:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode with development dependencies.

## Next Steps

After installation:
1. Review [README.md](../README.md) for usage examples
2. Check [docs/TRAINING.md](TRAINING.md) for training instructions
3. Explore [docs/API.md](API.md) for API documentation

## System Requirements

### Minimum
- CPU: 4 cores
- RAM: 8GB
- Storage: 10GB

### Recommended
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA RTX 3060 or better (6GB+ VRAM)
- Storage: 20GB SSD

## Common Issues

### macOS ARM (M1/M2/M3)

If you encounter issues with SciPy:
```bash
brew install openblas
pip install numpy scipy --no-binary :all:
```

### Windows

Ensure Visual C++ Build Tools are installed for compiling certain dependencies.

## Support

For installation issues, please open an issue on GitHub with:
- Operating system and version
- Python version (`python --version`)
- Error message (full traceback)
