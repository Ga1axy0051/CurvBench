#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/gcn"
exec python train.py --task nc --dataset Actor --model HyboNet --dim 32 --lr 0.005 --num-layers 2 --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold Lorentz --log-freq 10 --cuda -1 --seed 1234 --margin 1 --split-seed 1234 --patience 200 --min-epochs 50 --epochs 3000
