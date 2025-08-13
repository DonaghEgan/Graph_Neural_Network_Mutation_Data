#!/bin/bash

# Graph_Neural_Network_Mutation_Data Directory Cleanup Script
# This script organizes and cleans up temporary, cache, and redundant files

echo "🧹 Starting directory cleanup for Graph_Neural_Network_Mutation_Data"
echo "======================================================================"

# Create backup directory for important files before cleanup
echo "📁 Creating backup directory..."
mkdir -p cleanup_backup

# Count files before cleanup
TOTAL_FILES_BEFORE=$(find . -type f | wc -l)
echo "📊 Total files before cleanup: $TOTAL_FILES_BEFORE"

echo ""
echo "🗑️  CLEANING TEMPORARY AND CACHE FILES"
echo "========================================"

# 1. Remove vim swap files
echo "🔧 Removing vim swap files..."
find . -name "*.swp" -o -name "*.swo" -delete
SWAP_COUNT=$(find . -name "*.swp" -o -name "*.swo" 2>/dev/null | wc -l)
echo "   Removed vim swap files"

# 2. Clean Python cache files
echo "🐍 Cleaning Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   Removed Python cache files and __pycache__ directories"

# 3. Remove R temporary files
echo "📊 Removing R temporary files..."
find . -name "Rplots.pdf" -delete 2>/dev/null
find . -name ".RData" -delete 2>/dev/null
find . -name ".Rhistory" -delete 2>/dev/null
echo "   Removed R temporary files"

# 4. Clean up model checkpoint files (keep only the most recent)
echo "🎯 Managing model checkpoint files..."
if ls best_*.pt >/dev/null 2>&1; then
    echo "   Found model checkpoint files:"
    ls -la best_*.pt
    echo "   Keeping all model files (user can manually remove if needed)"
else
    echo "   No model checkpoint files found"
fi

echo ""
echo "📋 ORGANIZING TRAINING RESULTS"
echo "==============================="

# 5. Organize training results
echo "📈 Organizing training result files..."
mkdir -p results/training_outputs
mkdir -p results/figures

# Move training results to organized directory
if ls training_results_*.csv >/dev/null 2>&1; then
    mv training_results_*.csv results/training_outputs/ 2>/dev/null
    echo "   Moved training result CSVs to results/training_outputs/"
fi

if ls training_summary_*.csv >/dev/null 2>&1; then
    mv training_summary_*.csv results/training_outputs/ 2>/dev/null
    echo "   Moved training summary CSVs to results/training_outputs/"
fi

# Move any generated figures
if [ -d "figures" ] && [ "$(ls -A figures)" ]; then
    mv figures/* results/figures/ 2>/dev/null
    rmdir figures 2>/dev/null
    echo "   Moved figures to results/figures/"
fi

echo ""
echo "📚 ORGANIZING DOCUMENTATION"
echo "==========================="

# 6. Organize documentation files
echo "📖 Organizing documentation files..."
mkdir -p docs

# Move markdown documentation files
if ls *.md >/dev/null 2>&1; then
    mv *.md docs/ 2>/dev/null
    echo "   Moved documentation files to docs/"
fi

echo ""
echo "🔧 ORGANIZING SOURCE CODE"
echo "========================="

# 7. Organize source code
echo "💻 Organizing source code files..."
mkdir -p src/core
mkdir -p src/analysis
mkdir -p src/utils
mkdir -p scripts

# Move core model files
if [ -f "model.py" ]; then
    cp model.py src/core/
    echo "   Copied model.py to src/core/"
fi

if [ -f "cox_loss.py" ]; then
    cp cox_loss.py src/core/
    echo "   Copied cox_loss.py to src/core/"
fi

if [ -f "main.py" ]; then
    cp main.py src/core/
    echo "   Copied main.py to src/core/"
fi

# Move utility files
if [ -f "utility_functions.py" ]; then
    cp utility_functions.py src/utils/
    echo "   Copied utility_functions.py to src/utils/"
fi

if [ -f "process_data.py" ]; then
    cp process_data.py src/utils/
    echo "   Copied process_data.py to src/utils/"
fi

if [ -f "read_specific.py" ]; then
    cp read_specific.py src/utils/
    echo "   Copied read_specific.py to src/utils/"
fi

# Move analysis files
if [ -f "analyze_results.py" ]; then
    cp analyze_results.py src/analysis/
    echo "   Copied analyze_results.py to src/analysis/"
fi

if [ -f "plot_results.py" ]; then
    cp plot_results.py src/analysis/
    echo "   Copied plot_results.py to src/analysis/"
fi

if [ -f "plot_training_results.R" ]; then
    cp plot_training_results.R src/analysis/
    echo "   Copied plot_training_results.R to src/analysis/"
fi

# Move script files
if [ -f "setup_env.sh" ]; then
    cp setup_env.sh scripts/
    echo "   Copied setup_env.sh to scripts/"
fi

if [ -f "download_study.py" ]; then
    cp download_study.py scripts/
    echo "   Copied download_study.py to scripts/"
fi

echo ""
echo "⚠️  IDENTIFYING FILES FOR MANUAL REVIEW"
echo "======================================="

# 8. List files that need manual review
echo "🔍 Files that may need manual review:"

# Large data files
if [ -d "data" ]; then
    echo "   📁 data/ directory ($(du -sh data/ | cut -f1))"
fi

if [ -d "temp" ]; then
    echo "   📁 temp/ directory ($(du -sh temp/ | cut -f1))"
fi

# Virtual environment (large)
if [ -d "venv" ]; then
    echo "   📁 venv/ directory ($(du -sh venv/ | cut -f1)) - Virtual environment"
fi

# Old text files that might be obsolete
if [ -f "text_emedding.py" ]; then
    echo "   📄 text_emedding.py - Check if this is still needed"
fi

if [ -f "plotting.R" ]; then
    echo "   📄 plotting.R - Old plotting script (replaced by plot_training_results.R)"
fi

echo ""
echo "📊 CLEANUP SUMMARY"
echo "=================="

# Count files after cleanup
TOTAL_FILES_AFTER=$(find . -type f | wc -l)
FILES_REMOVED=$((TOTAL_FILES_BEFORE - TOTAL_FILES_AFTER))

echo "📈 Cleanup Statistics:"
echo "   • Files before cleanup: $TOTAL_FILES_BEFORE"
echo "   • Files after cleanup: $TOTAL_FILES_AFTER"
echo "   • Files removed/organized: $FILES_REMOVED"

echo ""
echo "📁 New directory structure:"
echo "   • src/core/          - Core model files (model.py, cox_loss.py, main.py)"
echo "   • src/utils/         - Utility functions"
echo "   • src/analysis/      - Analysis and plotting scripts"
echo "   • scripts/           - Setup and utility scripts"
echo "   • results/           - Training results and figures"
echo "   • docs/              - Documentation files"

echo ""
echo "✅ CLEANUP COMPLETE!"
echo "===================="
echo ""
echo "🚀 Next steps:"
echo "   1. Review the organized structure"
echo "   2. Consider removing old/duplicate files from root directory"
echo "   3. Update import paths in scripts if needed"
echo "   4. Consider removing or archiving the 'venv' directory if using conda"
echo "   5. Review files in 'temp' and 'data' directories for cleanup"

echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "   • Original files are kept in root directory for safety"
echo "   • Only temporary/cache files were deleted permanently"
echo "   • Core files were COPIED to organized structure"
echo "   • Review and manually remove duplicates when satisfied with new structure"
