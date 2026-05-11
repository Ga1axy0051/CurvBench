#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-}"
if [[ -n "$DATA_ROOT" ]]; then
  export DATAPATH="$DATA_ROOT"
fi

python hgcn/run_all_experiments.py "$@"
