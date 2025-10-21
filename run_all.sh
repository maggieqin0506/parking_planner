#!/bin/bash

echo "=========================================="
echo "Neural Network-Enhanced Parking Planner"
echo "Comparing Paper's Method vs Neural Network"
echo "=========================================="
echo ""

# Step 1: Generate training data using paper's method
echo "Step 1: Generating training data from paper's scenario planner..."
python data_generator.py

if [ $? -ne 0 ]; then
    echo "Error: Data generation failed"
    exit 1
fi

echo ""
echo "Step 2: Training neural network..."
python train_network.py

if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo ""
echo "Step 3: Running comparison experiments..."
python main.py

if [ $? -ne 0 ]; then
    echo "Error: Experiments failed"
    exit 1
fi

echo ""
echo "Step 4: Generating comparison table..."
python comparison_table.py

if [ $? -ne 0 ]; then
    echo "Warning: Table generation had issues, but continuing..."
fi

echo ""
echo "=========================================="
echo "All tasks completed successfully!"
echo ""
echo "Results:"
echo "  - Paths: results/paths/"
echo "  - Comparisons: results/comparison_*.png"
echo "  - Summary: results/summary.png"
echo "  - Table: results/comparison_table.txt"
echo "  - Data: results/results_summary.json"
echo "=========================================="