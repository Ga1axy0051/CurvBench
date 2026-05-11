#!/usr/bin/env bash
# 15 datasets = config datasets.exptable2graph_keys (6) + base_benchmark (9). All NC, then all LP.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
CFG="${CONFIG:-config.json}"

DATASETS=(
  carcinogenesis hockey hepatitis pte toxicology f1
  cora disease citeseer airport cornell actor pubmed telecom cs_phds
)

run_one() {
  local ds="$1" task="$2"
  echo "======== ${ds}  ${task} ========"
  "$PY" main.py --config "$CFG" --dataset "$ds" --task "$task"
}

for task in nc lp; do
  for ds in "${DATASETS[@]}"; do
    run_one "$ds" "$task"
  done
done

echo "done"
