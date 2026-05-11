#!/bin/bash
set -e

ENV_NAME="curvbench"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Conda not found"; exit 1; }
conda activate $ENV_NAME || { echo "Failed to activate $ENV_NAME"; exit 1; }

MODELS=("gcope" "mdgfm" "mdgpt" "samgpt")
DATASETS=("cora" "citeseer" "PubMed")
SHOTS=(1 5)

echo "Running GFM Baselines (Test Mode)..."
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for shot in "${SHOTS[@]}"; do
            echo "----------------------------------------"
            echo "Testing GFM: $model | Dataset: $dataset | Shots: $shot"
            python main.py --model "$model" --dataset "$dataset" --task nc --shot_num "$shot" || echo "Failed $model on $dataset ($shot-shot)"
        done
    done
done
echo "GFM Testing Finished!"
