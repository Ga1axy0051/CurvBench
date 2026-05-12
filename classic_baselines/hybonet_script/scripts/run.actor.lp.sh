#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/gcn"
exec python train.py --task lp --dataset Actor --model HyboNet --dim 16 --lr 0.02 --num-layers 2 --bias 1 --dropout 0.7 --weight-decay 0.001 --manifold Lorentz --log-freq 5 --cuda -1 --patience 500 --margin 0.1 --seed 1234
