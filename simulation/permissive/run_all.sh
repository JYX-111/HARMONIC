#!/bin/bash

# Script to run all visualization scripts in sequence
# Author: Auto-generated
# Date: 2026-03-18

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Running all visualization scripts in sequence"
echo "Working directory: $SCRIPT_DIR"
echo "============================================================"
echo ""

# Array of scripts to run
scripts=(
    "draw_gt.py"
    "draw_ours.py"
    "draw_without_texture.py"
    "box_draw_fnr.py"
)

# Run each script
for script in "${scripts[@]}"; do
    echo "============================================================"
    echo "Running: $script"
    echo "============================================================"
    echo ""

    # Run the script
    python "$script"

    # Check exit code
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $script completed successfully"
        echo ""
    else
        echo ""
        echo "✗ $script failed with exit code $?"
        echo ""
        echo "Stopping execution due to error"
        exit 1
    fi
done

echo "============================================================"
echo "All scripts completed successfully!"
echo "============================================================"
