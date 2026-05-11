#!/bin/bash

# ================= Configuration =================
# Set the environment name defined in your environment.yml
ENV_NAME="carte_env"
# Directory path containing the scripts (quotes handle the '&' character)
SCRIPT_DIR="graphsage&pcnet_script"
YML_FILE="${SCRIPT_DIR}/environment.yml"
# =================================================

echo "Checking for Conda environment: ${ENV_NAME}..."

# Check if the conda environment exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment '${ENV_NAME}' already exists."
else
    echo "Environment '${ENV_NAME}' not found. Starting installation..."
    if [ ! -f "${YML_FILE}" ]; then
        echo "Error: ${YML_FILE} not found."
        exit 1
    fi
    conda env create -f "${YML_FILE}"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create environment from ${YML_FILE}."
        exit 1
    fi
fi

echo ""
echo "Step 1/2: Running benchmark calculation script: cal_cs_phd.py..."
conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/cal_cs_phd.py"

if [ $? -ne 0 ]; then
    echo "Error: cal_cs_phd.py failed. Aborting."
    exit 1
fi

echo ""
echo "Step 2/2: Running benchmark script: run_benchmark.py..."
conda run -n "${ENV_NAME}" python "${SCRIPT_DIR}/run_benchmark.py"

if [ $? -ne 0 ]; then
    echo "Error: run_benchmark.py failed."
    exit 1
fi

echo ""
echo "Process completed successfully."