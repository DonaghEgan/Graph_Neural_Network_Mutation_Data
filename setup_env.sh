#!/bin/bash
echo "Setting up virtual environment for Graph Neural Network Mutation Data project..."

# Check if we're in the right directory
if [[ ! -f "main.py" || ! -f "model.py" ]]; then
    echo "Error: main.py or model.py not found. Make sure you're in the Graph_Neural_Network_Mutation_Data directory."
    echo "Current directory: $(pwd)"
    echo "Expected directory: /home/degan/Graph_Neural_Network_Mutation_Data"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install minimal requirements for analysis
echo "Installing basic analysis requirements..."
pip install pandas numpy

# Install plotting requirements
echo "Installing plotting requirements..."
pip install matplotlib seaborn

# Install PyTorch (CPU version by default)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install additional utilities
echo "Installing additional utilities..."
pip install tqdm

# Save requirements to file
echo "Saving requirements..."
pip freeze > requirements_generated.txt

# Test the installation
echo ""
echo "Testing installation..."
python -c "import pandas, numpy; print('✅ Basic requirements OK')" || echo "❌ Basic requirements failed"
python -c "import matplotlib, seaborn; print('✅ Plotting requirements OK')" || echo "❌ Plotting requirements failed"
python -c "import torch; print('✅ PyTorch OK')" || echo "❌ PyTorch failed"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To use the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run training:"
echo "  python main.py"
echo ""
echo "To analyze results:"
echo "  python analyze_results.py training_results_*.csv"
echo ""
echo "To deactivate:"
echo "  deactivate"
