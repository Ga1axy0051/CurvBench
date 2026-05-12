#!/usr/bin/env bash
set -euo pipefail
AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AG_ROOT"
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  export SA2GFM_DATA_ROOT="$(cd "$AG_ROOT/../.." && pwd)/data"
fi
DATASET="${1:?usage: $0 <dataset>}"
python pipeline/04_random_perturb.py --dataset "$DATASET" --type both
echo "Saved under $AG_ROOT/outputs/attacked_data_random/"
