#!/bin/bash
set -e

ENV_NAME="curvbench"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Conda not found"; exit 1; }
conda activate $ENV_NAME || { echo "Failed to activate $ENV_NAME"; exit 1; }

MODELS=("mlp_gcn_gat" "cusp" "hat" "hgcn" "hybonet" "qgcn" "graphmore" "graphsage")
DATASETS=("cora" "citeseer" "PubMed")

echo "Running Classic Baselines (Test Mode)..."
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "----------------------------------------"
        echo "Testing Model: $model | Dataset: $dataset"
        python main.py --model "$model" --dataset "$dataset" --task nc || echo "Failed $model on $dataset"
    done
done
echo "Classic Testing Finished!"
