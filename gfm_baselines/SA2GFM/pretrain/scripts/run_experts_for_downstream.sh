#!/usr/bin/env bash
# Train source experts for downstream.
# Usage:
#   bash pretrain/scripts/run_experts_for_downstream.sh <target_dataset> [extra pretrain args...]
#   bash pretrain/scripts/run_experts_for_downstream.sh --sources src1 src2 ... -- [extra pretrain args...]
#   bash pretrain/scripts/run_experts_for_downstream.sh <target_dataset> --sources src1 src2 ...
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TARGET=""
declare -a SOURCES=()
declare -a EXTRA_ARGS=()

if [[ $# -eq 0 ]]; then
  echo "usage: $0 <downstream_dataset> [extra args...]"
  echo "   or: $0 --sources src1 src2 ... -- [extra args...]"
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sources)
      shift
      while [[ $# -gt 0 && "$1" != "--" && "$1" != --* ]]; do
        SOURCES+=("$1")
        shift
      done
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    --*)
      EXTRA_ARGS+=("$1")
      shift
      if [[ $# -gt 0 && "$1" != --* ]]; then
        EXTRA_ARGS+=("$1")
        shift
      fi
      ;;
    *)
      if [[ -z "$TARGET" ]]; then
        TARGET="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ ${#SOURCES[@]} -eq 0 ]]; then
  if [[ -z "$TARGET" ]]; then
    echo "Either provide <target_dataset> or pass --sources explicitly."
    exit 1
  fi
  case "$TARGET" in
    cora)           SOURCES=(citeseer pubmed P-home wikics) ;;
    citeseer)       SOURCES=(cora pubmed P-home wikics) ;;
    pubmed)         SOURCES=(cora citeseer P-home wikics) ;;
    P-tech)         SOURCES=(cora citeseer pubmed P-home wikics) ;;
    P-home)         SOURCES=(cora citeseer pubmed wikics) ;;
    wikics)         SOURCES=(cora citeseer pubmed P-home) ;;
    arxiv)          SOURCES=(P-home P-tech wikics) ;;
    *) echo "Unknown target and no --sources provided: $TARGET"; exit 1 ;;
  esac
fi

for e in "${SOURCES[@]}"; do
  if [[ -n "$TARGET" ]]; then
    echo "=== expert dataset: $e (for downstream $TARGET) ==="
  else
    echo "=== expert dataset: $e ==="
  fi
  bash "$ROOT/pretrain/scripts/run_pretrain.sh" "$e" "${EXTRA_ARGS[@]}"
done
