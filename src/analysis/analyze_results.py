"""
Quick analysis of training results
Simple script to analyze training results without plotting dependencies
Automatically looks for training results in the organized directory structure
"""

import pandas as pd
import numpy as np
import sys
import os
import glob
from pathlib import Path

def find_training_results(base_dir=None):
    """
    Find training result files in the organized directory structure
    
    Args:
        base_dir (str): Base directory to search from. If None, searches from script location
        
    Returns:
        tuple: (training_results_files, training_summary_files)
    """
    if base_dir is None:
        # Get the script directory and navigate to project root
        script_dir = Path(__file__).parent
        # Assuming script is in src/analysis/, go up two levels to project root
        project_root = script_dir.parent.parent
    else:
        project_root = Path(base_dir)
    
    # Define the training outputs directory
    training_outputs_dir = project_root / "results" / "training_outputs"
    
    print(f"🔍 Searching for training results in: {training_outputs_dir}")
    
    if not training_outputs_dir.exists():
        # Fallback: search in project root
        print(f"⚠️  Training outputs directory not found, searching in project root: {project_root}")
        training_outputs_dir = project_root
    
    # Find training results files
    training_results_pattern = str(training_outputs_dir / "training_results_*.csv")
    training_summary_pattern = str(training_outputs_dir / "training_summary_*.csv")
    
    training_results_files = sorted(glob.glob(training_results_pattern))
    training_summary_files = sorted(glob.glob(training_summary_pattern))
    
    print(f"📊 Found {len(training_results_files)} training results files")
    print(f"📋 Found {len(training_summary_files)} training summary files")
    
    if training_results_files:
        print("\nTraining Results Files:")
        for i, file in enumerate(training_results_files, 1):
            print(f"  {i}. {os.path.basename(file)}")
    
    if training_summary_files:
        print("\nTraining Summary Files:")
        for i, file in enumerate(training_summary_files, 1):
            print(f"  {i}. {os.path.basename(file)}")
    
    return training_results_files, training_summary_files

def get_latest_training_results(base_dir=None):
    """
    Get the most recent training results file
    
    Args:
        base_dir (str): Base directory to search from
        
    Returns:
        str: Path to the most recent training results file
    """
    training_results_files, _ = find_training_results(base_dir)
    
    if not training_results_files:
        return None
    
    # Get the most recent file (assuming timestamp in filename)
    latest_file = training_results_files[-1]
    print(f"📈 Using latest training results: {os.path.basename(latest_file)}")
    return latest_file

def analyze_training_results(csv_file):
    """
    Analyze training results from CSV file
    
    Args:
        csv_file (str): Path to the training results CSV file
    """
    
    # Read the data and ensure numeric columns are properly converted
    df = pd.read_csv(csv_file)
    
    # Convert numeric columns to proper data types
    numeric_columns = ['epoch', 'train_loss', 'val_loss', 'train_ci', 'val_ci']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Check for any conversion issues
    if df[numeric_columns].isnull().any().any():
        print("⚠️  Warning: Some data could not be converted to numeric format")
        print("   This might indicate formatting issues in the CSV file")
    
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
    
    # Safely get the best validation C-index and its epoch
    val_ci_max = df['val_ci'].max()
    val_ci_max_idx = df['val_ci'].idxmax()
    best_val_ci_epoch = df.loc[val_ci_max_idx, 'epoch']
    
    # Safely get the lowest validation loss and its epoch  
    val_loss_min = df['val_loss'].min()
    val_loss_min_idx = df['val_loss'].idxmin()
    best_val_loss_epoch = df.loc[val_loss_min_idx, 'epoch']
    
    print(f"{'Best validation C-index:':<30} {val_ci_max:.4f} (epoch {best_val_ci_epoch:.0f})")
    print(f"{'Lowest validation loss:':<30} {val_loss_min:.4f} (epoch {best_val_loss_epoch:.0f})")
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
        
        # Convert numeric columns to proper data types
        numeric_columns = ['epoch', 'train_loss', 'val_loss', 'train_ci', 'val_ci']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
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

def print_usage():
    """Print usage information"""
    print("📊 Training Results Analyzer")
    print("="*50)
    print("Analyzes training results from your GNN survival prediction model")
    print("")
    print("USAGE:")
    print("  python analyze_results.py                           # Auto-find latest results")
    print("  python analyze_results.py <file.csv>                # Analyze specific file") 
    print("  python analyze_results.py <file1.csv> <file2.csv>   # Compare multiple files")
    print("")
    print("EXPECTED DIRECTORY STRUCTURE:")
    print("  Graph_Neural_Network_Mutation_Data/")
    print("  ├── results/")
    print("  │   └── training_outputs/")
    print("  │       ├── training_results_*.csv")
    print("  │       └── training_summary_*.csv") 
    print("  └── src/")
    print("      └── analysis/")
    print("          └── analyze_results.py")
    print("")
    print("FEATURES:")
    print("  • Performance metrics analysis")
    print("  • Convergence analysis") 
    print("  • Overfitting detection")
    print("  • Training stability assessment")
    print("  • Automated recommendations")
    print("  • Multi-file comparison")

if __name__ == "__main__":
    # Handle help request
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Define the expected training outputs directory
    training_outputs_dir = "/home/degan/Graph_Neural_Network_Mutation_Data/results/training_outputs"
    
    if len(sys.argv) < 2:
        print("🔍 No specific files provided, searching for training results...")
        print("="*80)
        
        # Try to find training results automatically
        training_results_files, training_summary_files = find_training_results()
        
        if not training_results_files:
            print("❌ No training results files found!")
            print("\n💡 Usage options:")
            print("   1. python analyze_results.py <training_results.csv> [additional_results.csv ...]")
            print("   2. Place training results in: results/training_outputs/")
            print("   3. Run training first to generate results")
            sys.exit(1)
        
        # Use the latest training results file
        latest_file = training_results_files[-1]
        csv_files = [latest_file]
        print(f"\n📊 Analyzing latest training results: {os.path.basename(latest_file)}")
        print("="*80)
        
    else:
        csv_files = sys.argv[1:]
        
        # Check if files exist, and if not, try to find them in training outputs directory
        resolved_files = []
        for csv_file in csv_files:
            if os.path.exists(csv_file):
                resolved_files.append(csv_file)
            else:
                # Try to find the file in training outputs directory
                potential_path = os.path.join(training_outputs_dir, csv_file)
                if os.path.exists(potential_path):
                    resolved_files.append(potential_path)
                    print(f"📁 Found file in training outputs: {csv_file}")
                else:
                    print(f"❌ Error: File '{csv_file}' not found!")
                    print(f"   Searched in: {csv_file}")
                    print(f"   Searched in: {potential_path}")
                    sys.exit(1)
        
        csv_files = resolved_files
    
    # Analyze first file in detail
    analyze_training_results(csv_files[0])
    
    # Compare multiple files if provided
    if len(csv_files) > 1:
        print("\n")
        compare_multiple_results(*csv_files)
    
    # If we have training summary files, offer to analyze them too
    if len(sys.argv) < 2:  # Only if we auto-discovered files
        _, training_summary_files = find_training_results()
        if training_summary_files:
            print("\n" + "="*80)
            print("📋 TRAINING SUMMARY ANALYSIS")
            print("="*80)
            print(f"Found {len(training_summary_files)} training summary files")
            print("Use plot_training_results.R to visualize summary statistics")
            
            # Show latest summary
            latest_summary = training_summary_files[-1]
            print(f"\nLatest summary file: {os.path.basename(latest_summary)}")
            try:
                summary_df = pd.read_csv(latest_summary)
                print("\nSummary metrics:")
                for _, row in summary_df.iterrows():
                    print(f"  {row['metric']}: {row['value']}")
            except Exception as e:
                print(f"Could not read summary file: {e}")
