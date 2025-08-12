#!/usr/bin/env python3
"""
Quick test script to verify virtual environment setup
"""

def test_imports():
    """Test that all required packages can be imported"""
    
    tests = [
        ("pandas", "Basic data analysis"),
        ("numpy", "Numerical computing"),
        ("matplotlib.pyplot", "Basic plotting"),
        ("seaborn", "Advanced plotting"),
        ("torch", "PyTorch deep learning"),
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    all_passed = True
    
    for package, description in tests:
        try:
            __import__(package)
            print(f"✅ {package:<20} - {description}")
        except ImportError as e:
            print(f"❌ {package:<20} - FAILED: {e}")
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("🎉 All packages imported successfully!")
        print("Your virtual environment is ready to use.")
    else:
        print("⚠️  Some packages failed to import.")
        print("Please check the installation and try again.")
    
    return all_passed

def test_analysis_functionality():
    """Test basic functionality needed for analysis"""
    
    print("\nTesting analysis functionality...")
    print("=" * 50)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create a dummy dataset similar to training results
        dummy_data = pd.DataFrame({
            'epoch': range(10),
            'train_loss': np.random.uniform(1.0, 2.0, 10),
            'val_loss': np.random.uniform(1.2, 2.2, 10),
            'train_ci': np.random.uniform(0.6, 0.8, 10),
            'val_ci': np.random.uniform(0.5, 0.7, 10)
        })
        
        # Test basic operations
        best_ci = dummy_data['val_ci'].max()
        best_epoch = dummy_data.loc[dummy_data['val_ci'].idxmax(), 'epoch']
        
        print(f"✅ DataFrame creation and manipulation works")
        print(f"   Sample analysis: Best C-index {best_ci:.4f} at epoch {best_epoch}")
        
        # Test saving
        test_file = "test_results.csv"
        dummy_data.to_csv(test_file, index=False)
        loaded_data = pd.read_csv(test_file)
        
        print(f"✅ CSV saving and loading works")
        
        # Clean up
        import os
        os.remove(test_file)
        
        print("✅ All analysis functionality works!")
        
    except Exception as e:
        print(f"❌ Analysis functionality test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Virtual Environment Test Script")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test functionality
    if imports_ok:
        functionality_ok = test_analysis_functionality()
        
        if functionality_ok:
            print("\n🚀 Your environment is fully ready!")
            print("\nNext steps:")
            print("1. Run your training: python main.py")
            print("2. Analyze results: python analyze_results.py training_results_*.csv")
            print("3. Create plots: python plot_results.py training_results_*.csv")
        else:
            print("\n⚠️  Environment setup needs attention.")
    else:
        print("\n❌ Please install missing packages and try again.")
