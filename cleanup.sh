#!/bin/bash
# Cleanup script to remove unnecessary files and free up storage

echo "========================================"
echo "Storage Cleanup Script"
echo "========================================"

# Show current size
echo -e "\nCurrent workspace size:"
du -sh /workspaces/HeartBeatIrregularity

# Remove Python cache files
echo -e "\n1. Removing Python cache files..."
find /workspaces/HeartBeatIrregularity -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find /workspaces/HeartBeatIrregularity -type f -name "*.pyc" -delete 2>/dev/null
find /workspaces/HeartBeatIrregularity -type f -name "*.pyo" -delete 2>/dev/null
echo "   ✓ Python cache cleaned"

# Remove pytest cache
echo -e "\n2. Removing pytest cache..."
find /workspaces/HeartBeatIrregularity -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Pytest cache cleaned"

# Remove Jupyter notebook checkpoints
echo -e "\n3. Removing Jupyter checkpoints..."
find /workspaces/HeartBeatIrregularity -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Jupyter checkpoints cleaned"

# Remove old log files (optional - uncomment if needed)
# echo -e "\n4. Removing old log files..."
# find /workspaces/HeartBeatIrregularity -type f -name "*.log" -delete 2>/dev/null
# echo "   ✓ Log files cleaned"

# Remove old saved models (BE CAREFUL - only uncomment if you have backups)
# echo -e "\n5. Removing old saved models..."
# rm -rf /workspaces/HeartBeatIrregularity/saved/*
# echo "   ✓ Old models cleaned"

# Remove temporary files
echo -e "\n4. Removing temporary files..."
find /workspaces/HeartBeatIrregularity -type f -name "*~" -delete 2>/dev/null
find /workspaces/HeartBeatIrregularity -type f -name "*.tmp" -delete 2>/dev/null
find /workspaces/HeartBeatIrregularity -type f -name "*.bak" -delete 2>/dev/null
echo "   ✓ Temporary files cleaned"

# Remove ONNX/TF intermediate files if they exist
echo -e "\n5. Removing conversion intermediate files..."
find /workspaces/HeartBeatIrregularity -type f -name "*.onnx" -delete 2>/dev/null
find /workspaces/HeartBeatIrregularity -type d -name "saved_model" -exec rm -rf {} + 2>/dev/null
echo "   ✓ Conversion artifacts cleaned"

# Clean conda/pip cache (optional - uncomment if needed)
# echo -e "\n6. Cleaning conda/pip cache..."
# conda clean -a -y 2>/dev/null
# pip cache purge 2>/dev/null
# echo "   ✓ Package cache cleaned"

echo -e "\n========================================"
echo "Cleanup Complete!"
echo "========================================"

# Show new size
echo -e "\nNew workspace size:"
du -sh /workspaces/HeartBeatIrregularity

echo -e "\nSpace saved:"
echo "(Run 'du -sh /workspaces/HeartBeatIrregularity' to check current size)"
echo -e "\nDone!"
