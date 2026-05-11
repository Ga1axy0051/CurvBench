#!/bin/bash
set -e

ENV_NAME="curvbench"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Conda not found"; exit 1; }
conda activate $ENV_NAME || { echo "Failed to activate $ENV_NAME"; exit 1; }

MODELS=("gcope" "mdgfm" "mdgpt")
DATASETS=("Actor" "Airport" "Carcinogenesis_data" "citeseer" "cora" "cornell" "cs_phds_lp_ready" "cs_phds_nc_ready" "Disease" "f1" "Hepatitis_std_data" "PTE" "PubMed" "telecom" "Toxicology_data")
SHOTS=(1 5)

echo "Running GFM Baselines (Full Mode)..."
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for shot in "${SHOTS[@]}"; do
            echo "----------------------------------------"
            echo "Running GFM: $model | Dataset: $dataset | Shots: $shot"
            python main.py --model "$model" --dataset "$dataset" --task nc --shot_num "$shot" || echo "Failed $model on $dataset ($shot-shot)"
        done
    done
done
echo "GFM Full Benchmark Finished!"
