#!/usr/bin/env python3
"""
Verification script to check that all improvements are implemented
"""

def check_main_py_improvements():
    """Check that main.py has all the key improvements"""
    
    print("Checking main.py improvements...")
    print("=" * 50)
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    checks = [
        ("AdamW optimizer", "torch.optim.AdamW" in content),
        ("Learning rate scheduler", "CosineAnnealingLR" in content or "scheduler" in content),
        ("Early stopping", "EarlyStopping" in content),
        ("Gradient clipping", "clip_grad_norm_" in content),
        ("L2 regularization", "l2_reg" in content),
        ("CSV saving", "to_csv" in content),
        ("Dropout parameter", "dropout=0.3" in content),
        ("Model import", "import model as m" in content),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_model_py_improvements():
    """Check that model.py has all the key improvements"""
    
    print("\nChecking model.py improvements...")
    print("=" * 50)
    
    with open('model.py', 'r') as f:
        content = f.read()
    
    checks = [
        ("Enhanced GIN Layer", "dropout=0.2" in content),
        ("Attention mechanism", "self.attention" in content),
        ("Residual connections", "x_residual" in content),
        ("Input projection", "self.input_proj" in content),
        ("Enhanced clinical branch", "self.clin_proj" in content),
        ("Fusion layer", "self.fusion" in content),
        ("Dropout in Net_omics", "dropout=0.3" in content),
        ("BatchNorm layers", "BatchNorm1d" in content),
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    return all_passed

def check_analysis_tools():
    """Check that analysis tools are available"""
    
    print("\nChecking analysis tools...")
    print("=" * 50)
    
    import os
    
    tools = [
        ("analyze_results.py", "Results analysis script"),
        ("plot_results.py", "Plotting script"),
        ("setup_env.sh", "Environment setup"),
        ("test_environment.py", "Environment test"),
        ("requirements.txt", "Package requirements"),
    ]
    
    all_present = True
    for filename, description in tools:
        if os.path.exists(filename):
            print(f"✅ {filename:<20} - {description}")
        else:
            print(f"❌ {filename:<20} - MISSING")
            all_present = False
    
    return all_present

if __name__ == "__main__":
    print("Verifying All Improvements Implementation")
    print("=" * 60)
    
    main_ok = check_main_py_improvements()
    model_ok = check_model_py_improvements()
    tools_ok = check_analysis_tools()
    
    print("\n" + "=" * 60)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 60)
    
    if main_ok and model_ok and tools_ok:
        print("🎉 ALL IMPROVEMENTS SUCCESSFULLY IMPLEMENTED!")
        print("\nYour enhanced Graph Neural Network project includes:")
        print("• Improved model architecture with attention")
        print("• Better training with AdamW + scheduling")
        print("• Early stopping and regularization")
        print("• Automatic CSV result saving")
        print("• Comprehensive analysis tools")
        print("\nReady to train: python main.py")
    else:
        print("⚠️  Some improvements may be missing:")
        if not main_ok:
            print("• Check main.py improvements")
        if not model_ok:
            print("• Check model.py improvements")
        if not tools_ok:
            print("• Check analysis tools")
