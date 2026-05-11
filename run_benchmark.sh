#!/bin/bash

# ==============================================================================
# CurvBench Automated Setup and Benchmark Script
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "========================================"
echo "1. Checking/Creating Conda Environment "
echo "========================================"

# Name of the conda environment
ENV_NAME="curvbench"

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Initialize conda for bash
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if environment exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating unified environment '$ENV_NAME'..."
    conda env create -f environment.yml
fi

echo "Activating environment '$ENV_NAME'..."
conda activate $ENV_NAME

echo "========================================"
echo "2. Downloading Datasets from Hugging Face"
echo "========================================"
DATASET_DIR="datasets"
if [ ! -d "$DATASET_DIR" ]; then
    mkdir -p $DATASET_DIR
fi

echo "Downloading datasets to $DATASET_DIR..."
# Setting huggingface mirror to avoid network issues
export HF_ENDPOINT="https://hf-mirror.com"

# Using huggingface python snippet to download datasets
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Ga1axy/CurvBench', repo_type='dataset', local_dir='$DATASET_DIR')"

echo "========================================"
echo "3. Running Benchmark Examples"
echo "========================================"
# The main.py should be the unified entry point for everything.
# Usage: python main.py --model {baseline_name} --task {nc/lp} --dataset {dataset_name}

echo "Testing a baseline NC task..."
# Ensure you implement main.py properly before running this
# python main.py --model mlp --task nc --dataset cora

echo "Done! The environment is set up and datasets are ready."
echo "You can now run 'conda activate $ENV_NAME' and start benchmarking."
