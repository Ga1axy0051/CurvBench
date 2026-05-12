# CUSP


```bash
# 1. Install dependencies
conda create -n hat python=3.11 -y
conda activate hat
pip install -r requirements.txt

# 2. Run all benchmark datasets (NC then LP, seeds 0–4, --config config.json)
bash script/run_nc_lp_15.sh

# 3. Check outputs
ls results/
```

Metrics JSON and summaries are under `results/nc_<dataset>` and `results/lp_<dataset>`.


