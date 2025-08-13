"""
Plotting utility for saved training results
Usage: python plot_results.py training_results_YYYYMMDD_HHMMSS.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from datetime import datetime

def plot_training_results(csv_file, save_plots=True):
    """
    Plot training results from saved CSV file
    
    Args:
        csv_file (str): Path to the training results CSV file
        save_plots (bool): Whether to save plots as image files
    """
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Results Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curves
    ax1.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cox Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: C-index curves
    ax2.plot(df['epoch'], df['train_ci'], label='Train C-index', linewidth=2)
    ax2.plot(df['epoch'], df['val_ci'], label='Validation C-index', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('C-index')
    ax2.set_title('Training and Validation C-index')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.0)  # C-index typically ranges from 0.5 to 1.0
    
    # Plot 3: Loss difference (overfitting indicator)
    loss_diff = df['val_loss'] - df['train_loss']
    ax3.plot(df['epoch'], loss_diff, label='Val Loss - Train Loss', color='red', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Difference')
    ax3.set_title('Overfitting Indicator (Val Loss - Train Loss)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: C-index difference
    ci_diff = df['val_ci'] - df['train_ci']
    ax4.plot(df['epoch'], ci_diff, label='Val CI - Train CI', color='green', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('C-index Difference')
    ax4.set_title('Generalization Gap (Val CI - Train CI)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        # Save the combined plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"training_plots_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Combined plots saved as: {plot_filename}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*50)
    print("TRAINING SUMMARY STATISTICS")
    print("="*50)
    
    print(f"Total epochs: {len(df)}")
    print(f"Best validation C-index: {df['val_ci'].max():.4f} (epoch {df.loc[df['val_ci'].idxmax(), 'epoch']})")
    print(f"Lowest validation loss: {df['val_loss'].min():.4f} (epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
    print(f"Final validation C-index: {df['val_ci'].iloc[-1]:.4f}")
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"loss_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Individual C-index plot
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
    
    plt.savefig(f"cindex_curves_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_results.py <training_results.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    print(f"Loading training results from: {csv_file}")
    
    # Create overview plots
    plot_training_results(csv_file, save_plots=True)
    
    # Create detailed individual plots
    create_detailed_plots(csv_file)
    
    print("\nPlotting completed!")
