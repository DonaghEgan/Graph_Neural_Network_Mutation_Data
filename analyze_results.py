"""
Quick analysis of training results
Simple script to analyze training results without plotting dependencies
"""

import pandas as pd
import numpy as np
import sys
import os

def analyze_training_results(csv_file):
    """
    Analyze training results from CSV file
    
    Args:
        csv_file (str): Path to the training results CSV file
    """
    
    # Read the data
    df = pd.read_csv(csv_file)
    
    print("="*60)
    print("TRAINING RESULTS ANALYSIS")
    print("="*60)
    
    # Basic info
    print(f"Dataset: {csv_file}")
    print(f"Total epochs trained: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Performance metrics
    print(f"\n{'PERFORMANCE METRICS':<30}")
    print("-" * 60)
    print(f"{'Best validation C-index:':<30} {df['val_ci'].max():.4f} (epoch {df.loc[df['val_ci'].idxmax(), 'epoch']})")
    print(f"{'Lowest validation loss:':<30} {df['val_loss'].min():.4f} (epoch {df.loc[df['val_loss'].idxmin(), 'epoch']})")
    print(f"{'Final validation C-index:':<30} {df['val_ci'].iloc[-1]:.4f}")
    print(f"{'Final validation loss:':<30} {df['val_loss'].iloc[-1]:.4f}")
    print(f"{'Final training C-index:':<30} {df['train_ci'].iloc[-1]:.4f}")
    print(f"{'Final training loss:':<30} {df['train_loss'].iloc[-1]:.4f}")
    
    # Convergence analysis
    print(f"\n{'CONVERGENCE ANALYSIS':<30}")
    print("-" * 60)
    
    # Check if validation loss is still decreasing
    last_20_epochs = df.tail(20) if len(df) >= 20 else df.tail(10)
    if len(last_20_epochs) >= 5:
        val_loss_trend = np.polyfit(range(len(last_20_epochs)), last_20_epochs['val_loss'], 1)[0]
        val_ci_trend = np.polyfit(range(len(last_20_epochs)), last_20_epochs['val_ci'], 1)[0]
        
        print(f"{'Validation loss trend:':<30} {'Decreasing' if val_loss_trend < 0 else 'Increasing'} ({val_loss_trend:.6f})")
        print(f"{'Validation C-index trend:':<30} {'Increasing' if val_ci_trend > 0 else 'Decreasing'} ({val_ci_trend:.6f})")
        
        if val_loss_trend < -0.001:
            print("✅ Model is still improving (loss decreasing)")
        elif abs(val_loss_trend) < 0.001:
            print("⚖️  Model has converged (loss stable)")
        else:
            print("❌ Model may be degrading (loss increasing)")
    
    # Overfitting analysis
    print(f"\n{'OVERFITTING ANALYSIS':<30}")
    print("-" * 60)
    
    # Calculate average gap in last epochs
    last_10_epochs = df.tail(10) if len(df) >= 10 else df
    avg_loss_gap = (last_10_epochs['val_loss'] - last_10_epochs['train_loss']).mean()
    avg_ci_gap = (last_10_epochs['train_ci'] - last_10_epochs['val_ci']).mean()
    
    print(f"{'Average loss gap (val-train):':<30} {avg_loss_gap:.4f}")
    print(f"{'Average C-index gap (train-val):':<30} {avg_ci_gap:.4f}")
    
    if avg_loss_gap > 0.2:
        print("⚠️  High overfitting detected (large loss gap)")
    elif avg_loss_gap > 0.1:
        print("⚠️  Moderate overfitting detected")
    elif avg_loss_gap < 0:
        print("ℹ️  Underfitting may be occurring (val loss < train loss)")
    else:
        print("✅ Good generalization (reasonable loss gap)")
    
    # Model stability
    print(f"\n{'MODEL STABILITY':<30}")
    print("-" * 60)
    
    val_ci_std = df['val_ci'].tail(20).std() if len(df) >= 20 else df['val_ci'].std()
    val_loss_std = df['val_loss'].tail(20).std() if len(df) >= 20 else df['val_loss'].std()
    
    print(f"{'Validation C-index std (last 20):':<30} {val_ci_std:.4f}")
    print(f"{'Validation loss std (last 20):':<30} {val_loss_std:.4f}")
    
    if val_ci_std < 0.01:
        print("✅ Very stable training")
    elif val_ci_std < 0.02:
        print("✅ Stable training")
    else:
        print("⚠️  Unstable training (high variance)")
    
    # Recommendations
    print(f"\n{'RECOMMENDATIONS':<30}")
    print("-" * 60)
    
    best_ci_epoch = df.loc[df['val_ci'].idxmax(), 'epoch']
    total_epochs = len(df)
    
    if best_ci_epoch < total_epochs * 0.5:
        print("• Consider longer training - best performance was early")
    elif best_ci_epoch > total_epochs * 0.9:
        print("• Training stopped at good time - model was still improving")
    else:
        print("• Training duration seems appropriate")
    
    if avg_loss_gap > 0.15:
        print("• Consider stronger regularization (higher dropout, weight decay)")
        print("• Consider reducing model complexity")
    
    if val_ci_std > 0.02:
        print("• Consider lower learning rate for more stable training")
        print("• Consider different optimizer settings")
    
    final_ci = df['val_ci'].iloc[-1]
    if final_ci < 0.6:
        print("• Model performance is poor - consider architecture changes")
    elif final_ci < 0.7:
        print("• Model performance is moderate - room for improvement")
    else:
        print("• Model performance is good!")

def compare_multiple_results(*csv_files):
    """Compare results from multiple training runs"""
    
    if len(csv_files) < 2:
        print("Need at least 2 CSV files to compare")
        return
    
    print("="*80)
    print("COMPARING MULTIPLE TRAINING RUNS")
    print("="*80)
    
    results = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        results.append({
            'file': os.path.basename(csv_file),
            'best_val_ci': df['val_ci'].max(),
            'final_val_ci': df['val_ci'].iloc[-1],
            'min_val_loss': df['val_loss'].min(),
            'final_val_loss': df['val_loss'].iloc[-1],
            'epochs': len(df)
        })
    
    # Print comparison table
    print(f"{'File':<25} {'Best CI':<10} {'Final CI':<10} {'Min Loss':<10} {'Final Loss':<10} {'Epochs':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['file']:<25} {result['best_val_ci']:<10.4f} {result['final_val_ci']:<10.4f} "
              f"{result['min_val_loss']:<10.4f} {result['final_val_loss']:<10.4f} {result['epochs']:<8}")
    
    # Find best model
    best_ci_idx = max(range(len(results)), key=lambda i: results[i]['best_val_ci'])
    print(f"\n🏆 Best model by C-index: {results[best_ci_idx]['file']} (C-index: {results[best_ci_idx]['best_val_ci']:.4f})")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <training_results.csv> [additional_results.csv ...]")
        sys.exit(1)
    
    csv_files = sys.argv[1:]
    
    # Check if all files exist
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file}' not found!")
            sys.exit(1)
    
    # Analyze first file in detail
    analyze_training_results(csv_files[0])
    
    # Compare multiple files if provided
    if len(csv_files) > 1:
        print("\n")
        compare_multiple_results(*csv_files)
