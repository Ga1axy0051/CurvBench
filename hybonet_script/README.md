# Fully Hyperbolic NN (HyboNet / hyperbolic GCN)

**Node classification (nc)** and **link prediction (lp)** on graphs. Main code lives in **`gcn/`**.

## Setup

```bash
cd gcn
pip install -r requirements.txt
```

Put datasets under **`data/`** at the repo root (layout and formats: `gcn/README.md`, `gcn/utils/data_utils.py`).

## Training

From the **repository root** (scripts `cd` into `gcn` for you):

```bash
bash scripts/run.cora.nc.sh
```

Or from inside `gcn`:

```bash
python train.py --task nc --dataset cora --model HyboNet ...
```

Optional: `source scripts/set_env.sh` from the repo root to set `DATAPATH`, `LOG_DIR`, and `PYTHONPATH`.

## What is in `scripts/`

| File | Purpose |
|------|---------|
| `run.*.sh` | Example commands for single datasets |
| `run_benchmark_15datasets.sh` | Multi-dataset benchmark (NC then LP) |
| `set_env.sh` | Example environment variables |

Logs and aggregates default to **`logs/`** at the repo root.

## Repository layout

- **`gcn/`** — `train.py`, models, data loading  
- **`scripts/`** — Shell helpers; safe to run from any working directory  
- **`telecom/`** — Optional: convert raw graphs to `telecom_*` inputs 

