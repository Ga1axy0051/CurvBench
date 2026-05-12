#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/gcn"
exec python train.py --patience 500 --task nc --dataset cora --model HyboNet --dim 16 --lr 0.02 --num-layers 3 --bias 1 --dropout 0.9 --weight-decay 0.01 --manifold Lorentz --log-freq 5 --cuda -1 --seed 1234 --margin 1 --grad-clip 1
