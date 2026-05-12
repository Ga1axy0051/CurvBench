#!/usr/bin/env bash
# Run node_classification + link_prediction for 15 benchmark datasets (incl. cs_phds).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
# Explicit --config value for train.py (default: repo root config.json)
CONFIG="${CONFIG:-config.json}"

SEEDS=(0 1 2 3 4)
DS=(
  Cora Citeseer PubMed Chameleon Actor Squirrel Texas Cornell
  AirportUSA AirportBrazil AirportEurope AirportLocal Disease Telecom cs_phds
)

echo "[run_nc_lp_15] train.py --config ${CONFIG}"
echo "[run_nc_lp_15] cwd: $(pwd)"

for d in "${DS[@]}"; do
  echo "=== NC  ${d}  (--config ${CONFIG}) ==="
  "$PY" train.py \
    --config "${CONFIG}" \
    --dataset "${d}" \
    --task node_classification \
    --export_dir "results/nc_${d}" \
    --seeds "${SEEDS[@]}"

  echo "=== LP  ${d}  (--config ${CONFIG}) ==="
  "$PY" train.py \
    --config "${CONFIG}" \
    --dataset "${d}" \
    --task link_prediction \
    --export_dir "results/lp_${d}" \
    --seeds "${SEEDS[@]}"
done
