"""
Plotting utility for saved training results
Usage: python plot_results.py training_results_YYYYMMDD_HHMMSS.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import glob
from datetime import datetime

def find_most_recent_training_file():
    """
    Find the most recent training results CSV file in multiple possible locations
    
    Returns:
        str: Path to the most recent training results file, or None if not found
    """
    print("🔍 Searching for training files...")
    
    # Search in multiple possible locations
    search_dirs = [
        ".",                           # Current directory
        "../..",                       # Project root
        "../../results/training_outputs",  # Expected results directory
        "../../results",               # Results directory
        "../../src/core",             # Where main.py is located
        "../../outputs",              # Alternative outputs directory
        "/home/degan/Graph_Neural_Network_Mutation_Data",  # Absolute path to project
        "/home/degan/Graph_Neural_Network_Mutation_Data/results",
        "/home/degan/Graph_Neural_Network_Mutation_Data/src/core"
    ]
    
    training_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"  📂 Searching in: {search_dir}")
            
            # Look for training results files with various patterns
            # Prioritize actual training results files over summary files
            patterns = [
                "training_results_*.csv",
                "*training_results*.csv",
                "*results*.csv"
            ]
            
            for pattern in patterns:
                files = glob.glob(os.path.join(search_dir, pattern))
                # Exclude summary files
                files = [f for f in files if "summary" not in os.path.basename(f).lower()]
                training_files.extend(files)
            
            # List all CSV files found
            all_csvs = glob.glob(os.path.join(search_dir, "*.csv"))
            if all_csvs:
                print(f"    📄 Found CSV files: {', '.join([os.path.basename(f) for f in all_csvs])}")
    
    # Remove duplicates
    training_files = list(set(training_files))
    
    if not training_files:
        print("❌ No training files found!")
        return None
    
    print(f"📊 Found potential training files: {', '.join([os.path.basename(f) for f in training_files])}")
    
    # Filter files that actually contain required columns
    valid_files = []
    for file_path in training_files:
        try:
            # Read first few rows to check structure
            test_df = pd.read_csv(file_path, nrows=5)
            required_cols = ['epoch', 'train_loss', 'val_loss']
            if all(col in test_df.columns for col in required_cols):
                valid_files.append(file_path)
                print(f"  ✅ Valid training file: {os.path.basename(file_path)}")
            else:
                print(f"  ❌ Invalid file (missing columns): {os.path.basename(file_path)}")
                print(f"      Has columns: {', '.join(test_df.columns)}")
        except Exception as e:
            print(f"  ❌ Error reading file {os.path.basename(file_path)}: {e}")
    
    if not valid_files:
        print("❌ No valid training files found!")
        return None
    
    # Return the most recent file based on modification time
    most_recent = max(valid_files, key=os.path.getmtime)
    print(f"📊 Selected most recent file: {os.path.basename(most_recent)}")
    return most_recent

def plot_training_results(csv_file, save_plots=True):
    """
    Plot training results from saved CSV file
    
    Args:
        csv_file (str): Path to the training results CSV file
        save_plots (bool): Whether to save plots as image files
    """
    
    # Read the data
    df = pd.read_csv(csv_file)
    required_cols = ['epoch', 'train_loss', 'val_loss']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns in CSV: {missing}")
        sys.exit(1)

    # Optional columns
    has_train_ci = 'train_ci' in df.columns
    has_val_ci = 'val_ci' in df.columns

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Results Overview', fontsize=16, fontweight='bold')
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # Plot 1: Loss curves
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cox Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: C-index curves (if available)
    if has_train_ci and has_val_ci:
        ax2.plot(df['epoch'], df['train_ci'], label='Train C-index', linewidth=2)
        ax2.plot(df['epoch'], df['val_ci'], label='Validation C-index', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('C-index')
        ax2.set_title('Training and Validation C-index')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.4, 1.0)
    else:
        ax2.text(0.5, 0.5, 'C-index columns missing', ha='center', va='center', fontsize=14)
        ax2.axis('off')

    # Plot 3: Loss difference (overfitting indicator)
    loss_diff = df['val_loss'] - df['train_loss']
    ax3.plot(df['epoch'], loss_diff, label='Val Loss - Train Loss', color='red', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.set_title('Overfitting Indicator (Val Loss - Train Loss)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: C-index difference (if available)
    if has_train_ci and has_val_ci:
        ci_diff = df['val_ci'] - df['train_ci']
        ax4.plot(df['epoch'], ci_diff, label='Val CI - Train CI', color='green', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('C-index Difference')
        ax4.set_title('Generalization Gap (Val CI - Train CI)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'C-index columns missing', ha='center', va='center', fontsize=14)
        ax4.axis('off')

    plt.tight_layout()

    if save_plots:
        # Create results/figures directory if it doesn't exist
        results_dir = "/home/degan/Graph_Neural_Network_Mutation_Data/results/figures"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save the combined plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(results_dir, f"training_plots_{timestamp}.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plots saved as: {plot_filename}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)

    print(f"Total epochs: {len(df)}")
    if has_val_ci:
        print(f"Best validation C-index: {df['val_ci'].max():.4f} (epoch {df.loc[df['val_ci'].idxmax(), 'epoch']})")
        print(f"Final validation C-index: {df['val_ci'].iloc[-1]:.4f}")
    print(f"Lowest validation loss: {df['val_loss'].min():.4f} (epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
    print(f"Final validation loss: {df['val_loss'].iloc[-1]:.4f}")

    # Check for overfitting
    last_10_epochs = df.tail(10)
    if len(last_10_epochs) >= 10:
        avg_val_loss_last_10 = last_10_epochs['val_loss'].mean()
        avg_train_loss_last_10 = last_10_epochs['train_loss'].mean()
        overfitting_score = avg_val_loss_last_10 - avg_train_loss_last_10

        print(f"\nOverfitting Analysis (last 10 epochs):")
        print(f"Average validation loss: {avg_val_loss_last_10:.4f}")
        print(f"Average training loss: {avg_train_loss_last_10:.4f}")
        print(f"Overfitting score: {overfitting_score:.4f}")

        if overfitting_score > 0.1:
            print("⚠️  Warning: Model may be overfitting")
        elif overfitting_score < -0.05:
            print("✅ Model appears to be generalizing well")
        else:
            print("ℹ️  Model performance seems reasonable")

def create_detailed_plots(csv_file):
    """Create more detailed individual plots"""
    
    df = pd.read_csv(csv_file)
    required_cols = ['epoch', 'train_loss', 'val_loss']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"Error: Missing required columns in CSV: {missing}")
        return

    has_train_ci = 'train_ci' in df.columns
    has_val_ci = 'val_ci' in df.columns

    # Individual loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2, alpha=0.8)
    plt.fill_between(df['epoch'], df['train_loss'], alpha=0.3)
    plt.fill_between(df['epoch'], df['val_loss'], alpha=0.3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Cox Loss', fontsize=12)
    plt.title('Training Progress: Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Create results/figures directory if it doesn't exist
    results_dir = "/home/degan/Graph_Neural_Network_Mutation_Data/results/figures"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, f"loss_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Individual C-index plot (if available)
    if has_train_ci and has_val_ci:
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_ci'], label='Train C-index', linewidth=2, alpha=0.8)
        plt.plot(df['epoch'], df['val_ci'], label='Validation C-index', linewidth=2, alpha=0.8)
        plt.fill_between(df['epoch'], df['train_ci'], alpha=0.3)
        plt.fill_between(df['epoch'], df['val_ci'], alpha=0.3)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('C-index', fontsize=12)
        plt.title('Training Progress: C-index Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.0)
        plt.tight_layout()

        plt.savefig(os.path.join(results_dir, f"cindex_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("C-index columns missing, skipping C-index plot.")

if __name__ == "__main__":
    # Check if a specific file was provided as command line argument
    if len(sys.argv) == 2:
        csv_file = sys.argv[1]
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found!")
            sys.exit(1)
        print(f"Using specified file: {csv_file}")
    else:
        # Automatically find the most recent training file
        csv_file = find_most_recent_training_file()
        if csv_file is None:
            print("No training results file found!")
            print("Please either:")
            print("1. Run the training script to generate results, or")
            print("2. Specify a file: python plot_results.py <training_results.csv>")
            sys.exit(1)
    
    print(f"📖 Loading training results from: {os.path.basename(csv_file)}")
    
    # Create overview plots
    plot_training_results(csv_file, save_plots=True)
    
    # Create detailed individual plots
    create_detailed_plots(csv_file)
    
    print("\n✅ Plotting completed!")
