# Training Results Management Guide

## Overview
Your training script now automatically saves results to CSV files that you can analyze and plot later. This allows you to:
- Track training progress over time
- Compare different experiments
- Create publication-quality plots
- Analyze model performance in detail

## Files Created During Training

### 1. **Training Results CSV** 📊
**Filename**: `training_results_YYYYMMDD_HHMMSS.csv`

**Contents**:
- `epoch`: Epoch number (0, 1, 2, ...)
- `train_loss`: Training Cox loss per epoch
- `val_loss`: Validation Cox loss per epoch
- `train_ci`: Training C-index per epoch
- `val_ci`: Validation C-index per epoch

### 2. **Training Summary CSV** 📋
**Filename**: `training_summary_YYYYMMDD_HHMMSS.csv`

**Contents**:
- Key metrics summary (best validation C-index, final test results, etc.)
- Single row with all important numbers for quick comparison

## Analysis Tools

### 1. **Quick Analysis Script** 🔍
**File**: `analyze_results.py`

**Usage**:
```bash
# Analyze single training run
python analyze_results.py training_results_20250812_143022.csv

# Compare multiple training runs
python analyze_results.py training_results_run1.csv training_results_run2.csv training_results_run3.csv
```

**Features**:
- ✅ Performance metrics summary
- ✅ Convergence analysis (is model still improving?)
- ✅ Overfitting detection
- ✅ Model stability assessment
- ✅ Actionable recommendations
- ✅ Multi-run comparison

**Example Output**:
```
TRAINING RESULTS ANALYSIS
============================================================
Best validation C-index:         0.7234 (epoch 87)
Final validation C-index:        0.7156
Validation loss trend:           Decreasing (-0.001234)
Average loss gap (val-train):    0.0456
✅ Good generalization (reasonable loss gap)
✅ Stable training
```

### 2. **Plotting Script** 📈
**File**: `plot_results.py`

**Usage**:
```bash
python plot_results.py training_results_20250812_143022.csv
```

**Features**:
- 📊 Four-panel overview plot (loss, C-index, overfitting indicators)
- 📈 Individual detailed plots
- 💾 Automatic saving of high-resolution plots
- 📊 Summary statistics

**Plots Created**:
1. **Training and Validation Loss**: Monitor convergence
2. **Training and Validation C-index**: Track performance
3. **Overfitting Indicator**: Val loss - Train loss
4. **Generalization Gap**: Train CI - Val CI

## Example Workflow

### 1. **Run Training**
```bash
python main.py
```
**Output**: 
- `training_results_20250812_143022.csv`
- `training_summary_20250812_143022.csv`
- `best_model.pt`
- `best_cindex_model.pt`

### 2. **Quick Analysis**
```bash
python analyze_results.py training_results_20250812_143022.csv
```

### 3. **Create Plots**
```bash
python plot_results.py training_results_20250812_143022.csv
```
**Output**: 
- `training_plots_20250812_144530.png`
- `loss_curves_20250812_144530.png`
- `cindex_curves_20250812_144530.png`

### 4. **Compare Experiments**
```bash
# After running multiple experiments
python analyze_results.py training_results_exp1.csv training_results_exp2.csv training_results_exp3.csv
```

## CSV File Structure

### Training Results CSV
```csv
epoch,train_loss,val_loss,train_ci,val_ci
0,2.1234,2.3456,0.5123,0.5234
1,1.9876,2.1234,0.5345,0.5456
2,1.8765,2.0123,0.5567,0.5678
...
```

### Training Summary CSV
```csv
metric,value
best_val_ci,0.7234
best_val_loss,1.2345
final_test_ci,0.7156
final_test_loss,1.3456
total_epochs,150
test_ci_best_model,0.7189
test_loss_best_model,1.3234
```

## Custom Analysis

### Load and Analyze in Python
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('training_results_20250812_143022.csv')

# Custom analysis
best_epoch = df.loc[df['val_ci'].idxmax(), 'epoch']
print(f"Best validation C-index: {df['val_ci'].max():.4f} at epoch {best_epoch}")

# Custom plots
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['val_ci'], label='Validation C-index')
plt.xlabel('Epoch')
plt.ylabel('C-index')
plt.title('Model Performance Over Time')
plt.legend()
plt.show()
```

### Load and Analyze in R
```r
library(ggplot2)
library(dplyr)

# Load results
df <- read.csv('training_results_20250812_143022.csv')

# Custom analysis
best_epoch <- df[which.max(df$val_ci), 'epoch']
cat(sprintf("Best validation C-index: %.4f at epoch %d\n", max(df$val_ci), best_epoch))

# Custom plots
ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = train_ci, color = "Training")) +
  geom_line(aes(y = val_ci, color = "Validation")) +
  labs(title = "C-index Over Time", y = "C-index", color = "Set") +
  theme_minimal()
```

## Tips for Experiment Management

### 1. **Organize Results by Experiment**
```bash
mkdir experiments
mkdir experiments/baseline
mkdir experiments/improved_architecture
mkdir experiments/hyperparameter_tuning

# Move results to appropriate folders
mv training_results_*.csv experiments/baseline/
```

### 2. **Compare Different Architectures**
```bash
# Run multiple experiments
python main.py  # baseline
# ... modify model.py ...
python main.py  # improved architecture

# Compare results
python analyze_results.py experiments/*/training_results_*.csv
```

### 3. **Track Hyperparameters**
Create a simple experiment log:
```csv
experiment,csv_file,architecture,lr,dropout,best_val_ci,notes
baseline,training_results_20250812_143022.csv,Net_omics,1e-3,0.3,0.7234,Original model
improved,training_results_20250812_150000.csv,MultiScaleGNN,1e-3,0.2,0.7456,Added attention
```

Your training results are now automatically saved and ready for comprehensive analysis! 🎉
