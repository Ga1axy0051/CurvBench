# 1. Install dependencies

```bash
conda create -n gcn_baseline python=3.11 -y
conda activate gcn_baseline
pip install -r requirements.txt
```

If `torch` fails, install PyTorch first, then `pip install -r requirements.txt` again.

# 2. Run all 15 datasets (NC, then LP)

From the repo root (Linux / macOS / WSL):

```bash
bash scripts/run_15datasets_nc_lp.sh
```

Single dataset example:

```bash
python main.py --dataset cora --task nc
python main.py --dataset cs_phds --task lp
```

# 3. Check outputs

Metrics JSON files are written under `data/` by default:

```bash
ls data/baseline_*_metrics.json
```

Optional: capture a full run log:

```bash
mkdir -p log
bash scripts/run_15datasets_nc_lp.sh 2>&1 | tee log/run_all.txt
```
