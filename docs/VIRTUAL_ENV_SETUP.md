# Virtual Environment Setup Guide

## Quick Setup Commands

### 1. Create Virtual Environment
```bash
# Navigate to your project directory
cd /home/degan/msk

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install pandas numpy matplotlib seaborn torch torch-geometric

# Deactivate when done (optional)
deactivate
```

### 2. Using the Environment

#### Activate Environment (do this each time you want to use the scripts)
```bash
cd /home/degan/msk
source venv/bin/activate
```

#### Run Analysis Scripts
```bash
# Quick analysis (no plotting dependencies needed)
python analyze_results.py training_results_20250812_143022.csv

# Full plotting analysis (requires matplotlib)
python plot_results.py training_results_20250812_143022.csv
```

#### Deactivate Environment (when done)
```bash
deactivate
```

## Package Requirements

### Minimal Requirements (for analyze_results.py only)
- pandas
- numpy

### Full Requirements (for plotting)
- pandas
- numpy  
- matplotlib
- seaborn

### Training Requirements (for main.py)
- torch
- torch-geometric
- pandas
- numpy
- tqdm

## Complete Setup Script

Save this as `setup_env.sh` and run with `bash setup_env.sh`:

```bash
#!/bin/bash
echo "Setting up virtual environment for MSK analysis..."

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install minimal requirements for analysis
echo "Installing minimal requirements..."
pip install pandas numpy

# Install plotting requirements
echo "Installing plotting requirements..."
pip install matplotlib seaborn

# Install PyTorch (CPU version - change if you need GPU)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install additional utilities
pip install tqdm

# Save requirements to file
pip freeze > requirements.txt

echo "Setup complete!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"
```

## Checking Your Setup

### Verify Installation
```bash
source venv/bin/activate
python -c "import pandas, numpy; print('Basic requirements OK')"
python -c "import matplotlib, seaborn; print('Plotting requirements OK')"
python -c "import torch; print('PyTorch OK')"
```

## Troubleshooting

### If you get permission errors:
```bash
# Make sure you're in the right directory
pwd  # Should show /home/degan/msk

# Check Python version
python3 --version

# Try with specific Python version
python3.8 -m venv venv  # or python3.9, python3.10, etc.
```

### If packages fail to install:
```bash
# Update pip first
source venv/bin/activate
pip install --upgrade pip

# Install packages one by one
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
```

### If torch installation fails:
```bash
# CPU-only version (lighter)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Or for GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Directory Structure
After setup, your directory should look like:
```
/home/degan/msk/
├── venv/                    # Virtual environment (created)
├── analyze_results.py       # Analysis script
├── plot_results.py          # Plotting script  
├── requirements.txt         # Package list (created)
├── setup_env.sh            # Setup script (optional)
└── training_results_*.csv   # Your training results
```
