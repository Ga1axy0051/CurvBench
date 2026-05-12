#!/bin/bash
set -e

ENV_NAME="curvbench"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Conda not found"; exit 1; }
conda activate $ENV_NAME || { echo "Failed to activate $ENV_NAME"; exit 1; }

MODELS=("cal_curv")
DATASETS=("cora" "citeseer" "PubMed")

echo "Running Curvature Calculation (Test Mode)..."
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "----------------------------------------"
        echo "Testing Curvature: $model | Dataset: $dataset"
        python ../main.py --model "$model" --dataset "$dataset" || echo "Failed $model on $dataset"
    done
done
echo "Curvature Testing Finished!"
