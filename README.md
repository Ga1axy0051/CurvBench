# CurvBench (Curvature Graph Benchmark Framework)

CurvBench is a unified and comprehensive benchmark framework dedicated to evaluating graph neural networks (GNNs) and graph foundation models (GFMs) on various graph datasets, with a special emphasis on their geometric properties (such as sectional curvature) and downstream efficacy. This project encompasses standard model evaluations ranging from classical GNN architectures to cutting-edge pre-trained foundation models, providing complete automated pipelines to measure both model performance and dataset properties.

## 📁 Project Structure

The project architecture is highly modular, clearly separating the dependencies of classical baseline evaluations from those of graph foundation models.

```text
CurvBench/
├── Cal_curv/               # Module for automatic graph curvature distribution calculation (supports Fast GPU / Strict computation).
├── classic_baselines/      # Classical graph neural networks (Traditional and improved GNN variants).
│   ├── cusp_script/        # Cusp (CVPR 2023)
│   ├── graphmore_script/   # GraphMoRe
│   ├── graphsage&pcnet_script/ # GraphSAGE and PCNet baselines
│   ├── hat_script/         # HAT
│   ├── hnn_hgcn_script/    # Hyperbolic baselines: HNN and HGCN
│   ├── hybonet_script/     # HyboNet
│   ├── mlp_gcn_gat_script/ # Core classical baselines (MLP, GCN, GAT)
│   └── qgcn_script/        # QGCN (Quantum-state GCN)
├── gfm_baselines/          # Graph Foundation Models (GFMs) and few-shot inference models.
│   ├── GCOPE-main/         # GCOPE (Pre-trained structure)
│   ├── mdgfm/              # MDGFM
│   ├── mdgpt/              # MDGPT
│   ├── SAMGPT/             # SAMGPT (Newly integrated)
│   ├── GraphGlue/          # GraphGlue (Newly integrated)
│   └── SA2GFM/             # SA2GFM (Newly integrated)
├── datasets/               # Unified mounted parquet format downstream datasets (downloaded from HF Hub).
├── scripts/                # Bash scripts for automated batch testing and evaluations.
├── main.py                 # The unified central dispatcher and entry point for all models.
├── parquet_loader.py       # Seamlessly loads and transforms the mounted Parquet datasets for all model architectures.
└── environment.yml         # Base Conda Python environment configuration for CurvBench.
```

## 📊 Supported Datasets

A total of 15 standard graph datasets are integrated, classified by their underlying geometric curvature properties:

- **Non-zero Curvature:** `cora`, `citeseer`, `pubmed`
- **Positive Curvature:** `cornell`, `airport`, `actor`
- **Negative Curvature:** `disease`, `telecom`, `cs_phds` (including `cs_phds_lp_ready` / `cs_phds_nc_ready`)
- **Table-derived:** `Carcinogenesis_data`, `f1`, `Hepatitis_std_data`, `PTE`, `Toxicology_data`

## 🚀 Unified Entry Point & Execution

The design pattern of this framework ensures that **all executions are routed through `main.py`**. You can evaluate any specific model (or a specific GFM) on any compatible dataset using a unified standard interface:

```bash
# Clone the repository (if you haven't already)
git clone git@github.com:Ga1axy0051/CurvBench.git
cd CurvBench

# Create the conda environment from the provided yaml file
conda env create -f environment.yml

# Activate the environment
conda activate curvbench

# Install HF CLI
pip install -U "huggingface_hub[cli]"

# Download datasets to the project root (using mirror)
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset Ga1axy/CurvBench --local-dir datasets/

# Calculate the sectional geometric curvature of a graph dataset
python main.py --model cal_curv --dataset cora

# Evaluate classical baselines (e.g., GCN, GraphSAGE, PCNet)
python main.py --model graphsage --dataset Actor --task nc
python main.py --model pcnet --dataset cora --task nc
python main.py --model hat --dataset citeseer

# Evaluate Graph Foundation Models (GFMs) with few-shot configurations (e.g., 5-shot)
python main.py --model gcope --dataset PubMed --shot_num 5
python main.py --model mdgpt --dataset telecom --shot_num 1
python main.py --model samgpt --dataset f1 --shot_num 1
python main.py --model graphglue --dataset cora --shot_num 5
python main.py --model sa2gfm --dataset citeseer --shot_num 1
```

**Available model identifiers (`--model`) include:**
- **Classical Baselines:** `mlp_gcn_gat`, `cusp`, `hat`, `hnn`, `hgcn`, `hybonet`, `qgcn`, `graphmore`, `graphsage`, `pcnet`
- **Graph Foundation Models:** `gcope`, `mdgfm`, `mdgpt`, `samgpt`, `graphglue`, `sa2gfm`
- **Specialized Modules:** `cal_curv`

## ⚙️ Automated Benchmarking Scripts

We provide comprehensive bash scripts targeting different modules. These include `*_test.sh` for rapid validation (evaluating a small subset) and `*_full.sh` for exhaustive benchmarking across all datasets:

```bash
cd scripts/

# Calculate curvature distributions for all 15 datasets (generates bar chart visualizations in logs)
bash cal_curv_full.sh

# Evaluate all classical baseline models
bash classic_test.sh           # Test on a small subset (e.g., core citation networks)
nohup bash classic_full.sh &   # Run exhaustive evaluation on all datasets in the background

# Evaluate all Graph Foundation Models (GFM scripts internally cover both 1-shot and 5-shot configurations)
bash gfm_test.sh
```
