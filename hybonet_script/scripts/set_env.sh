#!/usr/bin/env bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HGCN_HOME="$ROOT"
export LOG_DIR="$HGCN_HOME/logs"
export PYTHONPATH="$HGCN_HOME/gcn:${PYTHONPATH:-}"
export DATAPATH="$HGCN_HOME/data"
