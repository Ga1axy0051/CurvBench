#!/usr/bin/env bash
# MoE few-shot node classification; forwards extra args to train_downstream.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  export SA2GFM_DATA_ROOT="$(cd "$ROOT/../.." && pwd)/data"
fi
exec python "$ROOT/downstream/pipeline/train_downstream.py" "$@"
