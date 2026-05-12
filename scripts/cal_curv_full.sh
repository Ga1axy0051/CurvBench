#!/bin/bash
set -e

ENV_NAME="curvbench"
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Conda not found"; exit 1; }
conda activate $ENV_NAME || { echo "Failed to activate $ENV_NAME"; exit 1; }

MODELS=("cal_curv")
DATASETS=("Actor" "Airport" "Carcinogenesis_data" "citeseer" "cora" "cornell" "cs_phds_lp_ready" "cs_phds_nc_ready" "Disease" "f1" "Hepatitis_std_data" "PTE" "PubMed" "telecom" "Toxicology_data")

echo "Running Curvature Calculation (Full Mode)..."
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "----------------------------------------"
        echo "Running Curvature: $model | Dataset: $dataset"
        python ../main.py --model "$model" --dataset "$dataset" || echo "Failed $model on $dataset"
    done
done
echo "Curvature Full Benchmark Finished!"
