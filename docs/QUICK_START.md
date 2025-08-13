# Quick Start Guide: Graph Neural Network Mutation Data Project

## 🚀 One-Command Setup

```bash
cd /home/degan/Graph_Neural_Network_Mutation_Data
bash setup_env.sh
```

This will automatically:
- Create virtual environment `venv/`
- Install all required packages  
- Test the installation
- Show you next steps

## 📋 Manual Step-by-Step (if automatic setup fails)

### 1. Create and Activate Environment
```bash
cd /home/degan/Graph_Neural_Network_Mutation_Data
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Packages
```bash
# Essential for analysis
pip install pandas numpy

# For plotting
pip install matplotlib seaborn

# For training
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric tqdm
```

### 3. Test Installation
```bash
python test_environment.py
```

## 🔧 Daily Workflow

### Start Working (every time you open terminal)
```bash
cd /home/degan/Graph_Neural_Network_Mutation_Data
source venv/bin/activate
```

### Train Your Model
```bash
python main.py
```
**Output**: Automatically creates timestamped CSV files:
- `training_results_YYYYMMDD_HHMMSS.csv`
- `training_summary_YYYYMMDD_HHMMSS.csv`
- `best_model.pt` and `best_cindex_model.pt`

### Analyze Results
```bash
# Quick analysis without plotting
python analyze_results.py training_results_20250812_143022.csv

# Create publication-quality plots  
python plot_results.py training_results_20250812_143022.csv

# Compare multiple experiments
python analyze_results.py results1.csv results2.csv results3.csv
```

### Stop Working
```bash
deactivate
```

## 🎯 What Each Script Does

### `main.py` (Your training script)
- ✅ **Enhanced model architecture** with attention and residual connections
- ✅ **Better optimizer** (AdamW + scheduling)
- ✅ **Early stopping** to prevent overfitting
- ✅ **Automatic CSV saving** of all metrics
- ✅ **Test set evaluation** at completion

### `analyze_results.py` (Analysis without plotting)
- ✅ Performance metrics summary
- ✅ Overfitting detection  
- ✅ Training stability analysis
- ✅ Actionable recommendations
- ✅ Multi-run comparison

### `plot_results.py` (Comprehensive plotting)
- 📊 Training/validation curves
- 📈 C-index progression
- 🔍 Overfitting indicators
- �� High-resolution plot export

## 📁 Project Structure After Setup
```
/home/degan/Graph_Neural_Network_Mutation_Data/
├── venv/                           # Virtual environment (created)
├── main.py                         # Your training script (enhanced)
├── model.py                        # Your model (improved architecture)
├── cox_loss.py                     # Cox loss functions
├── analyze_results.py              # Analysis script
├── plot_results.py                 # Plotting script  
├── test_environment.py             # Environment test
├── setup_env.sh                    # Setup script
├── requirements.txt                # Package dependencies
├── training_results_*.csv          # Training results (generated)
├── best_model.pt                   # Best models (generated)
└── *.md                           # Documentation
```

## ✅ Success Indicators

After running `python test_environment.py`, you should see:
```
✅ pandas               - Basic data analysis
✅ numpy                - Numerical computing  
✅ matplotlib.pyplot    - Basic plotting
✅ seaborn              - Advanced plotting
✅ torch                - PyTorch deep learning
🎉 All packages imported successfully!
✅ All analysis functionality works!
🚀 Your environment is fully ready!
```

## 🎉 Enhanced Training Features

Your `main.py` now includes:
- **AdamW Optimizer** with learning rate scheduling
- **Dropout & Regularization** for better generalization
- **Early Stopping** with patience (automatically stops when done)
- **Gradient Clipping** for training stability
- **Automatic Model Saving** (best loss + best C-index models)
- **CSV Results Export** with timestamps
- **Comprehensive Test Evaluation** 

## 🚀 Quick Example Workflow

```bash
# 1. Setup (one time only)
bash setup_env.sh

# 2. Start working
source venv/bin/activate

# 3. Train your model (runs automatically with improvements)
python main.py

# 4. Analyze the results
python analyze_results.py training_results_20250812_143022.csv

# 5. Create plots for publication
python plot_results.py training_results_20250812_143022.csv

# 6. Done working
deactivate
```

Your Graph Neural Network survival prediction model is now production-ready with:
- 🧠 **Enhanced Architecture** 
- 📊 **Automatic Results Tracking**
- 🔍 **Comprehensive Analysis Tools**
- 📈 **Publication-Quality Plots**

Happy training! 🚀🧬
