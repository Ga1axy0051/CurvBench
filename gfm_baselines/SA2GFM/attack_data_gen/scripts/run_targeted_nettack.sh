#!/usr/bin/env bash
# Run full targeted pipeline: train GCN (optional) -> Nettack reports -> assemble final .pt
set -euo pipefail

AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AG_ROOT"

# Default: use ../data when SA2GFM is nested in the main repo (…/test_pretrain/SA2GFM/attack_data_gen)
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  export SA2GFM_DATA_ROOT="$(cd "$AG_ROOT/../.." && pwd)/data"
fi

DATASET="${1:?usage: $0 <dataset> [p]}"
P="${2:-1}"

echo "SA2GFM_DATA_ROOT=$SA2GFM_DATA_ROOT"
echo "dataset=$DATASET p=$P"

if [[ "${SKIP_SURROGATE_TRAIN:-0}" != "1" ]]; then
  python pipeline/01_train_gcn_surrogate.py --datasets "$DATASET" --device cuda
else
  echo "SKIP_SURROGATE_TRAIN=1 -> reuse checkpoints/gcn_${DATASET}.pth"
fi
python pipeline/02_nettack_reports.py --dataset "$DATASET" --p "$P"
python pipeline/03_assemble_final.py --dataset "$DATASET" --p-values "$P"

echo "Final graphs under: $AG_ROOT/outputs/attack_post/${DATASET}_p${P}_final/"
