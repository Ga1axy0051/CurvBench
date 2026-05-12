#!/usr/bin/env bash
# 15 datasets x 5 seeds (default 0-4): NC then LP via run_standard_benchmark_suite.py.
# Edit NC_ARGS / LP_ARGS to change hyperparameters. Suite overrides --dataset, --seed, --split-seed.
#
#   bash gcn/run_benchmark_15datasets.sh
#   SUITE_SEEDS=0,1,2 bash gcn/run_benchmark_15datasets.sh
#
# Outputs: logs/standard_benchmark_aggregate/{nc,lp}/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}/gcn"

SUITE_SEEDS="${SUITE_SEEDS:-0,1,2,3,4}"

LIST15_NC='cora,pubmed,citeseer,airport,Actor,cornell,wisconsin,texas,disease_nc,telecom_nc,cs_phds_nc,exptable_carcinogenesis,exptable_hepatitis_std,exptable_hockey,exptable_pte'
LIST15_LP='cora,pubmed,citeseer,airport,Actor,cornell,wisconsin,texas,disease_lp,telecom_lp,cs_phds_lp,exptable_carcinogenesis,exptable_hepatitis_std,exptable_hockey,exptable_pte'

# NC
NC_ARGS=(
  --task nc
  --model HyboNet
  --dim 32
  --lr 0.005
  --num-layers 2
  --bias 1
  --dropout 0.5
  --weight-decay 0.001
  --manifold Lorentz
  --c 1.0
  --margin 1.0
  --epochs 3000
  --patience 200
  --min-epochs 50
  --optimizer radam
  --momentum 0.999
  --gamma 0.5
  --lr-reduce-freq 3000
  --log-freq 10
  --eval-freq 1
  --grad-clip 1.0
  --cuda -1
  --double-precision 0
  --use-att 0
  --local-agg 0
  --n-heads 4
  --alpha 0.2
  --act None
  --use-feats 1
  --normalize-feats 1
  --normalize-adj 1
  --save 0
  --pos-weight 0
  --pretrained-embeddings none
  --r 2.0
  --t 1.0
  --suite-datasets "${LIST15_NC}"
  --suite-seeds "${SUITE_SEEDS}"
)

# LP
LP_ARGS=(
  --task lp
  --model HyboNet
  --dim 16
  --lr 0.02
  --num-layers 2
  --bias 1
  --dropout 0.7
  --weight-decay 0.001
  --manifold Lorentz
  --c 1.0
  --margin 0.1
  --r 2.0
  --t 1.0
  --epochs 5000
  --patience 500
  --min-epochs 100
  --optimizer radam
  --momentum 0.999
  --gamma 0.5
  --lr-reduce-freq 5000
  --log-freq 5
  --eval-freq 1
  --grad-clip 1.0
  --cuda -1
  --double-precision 0
  --use-att 0
  --local-agg 0
  --n-heads 4
  --alpha 0.2
  --act None
  --use-feats 1
  --normalize-feats 1
  --normalize-adj 1
  --val-prop 0.05
  --test-prop 0.1
  --save 0
  --pos-weight 0
  --pretrained-embeddings none
  --suite-datasets "${LIST15_LP}"
  --suite-seeds "${SUITE_SEEDS}"
)

echo "[15datasets] NC suite (HyboNet dim=32 ...)" >&2
python run_standard_benchmark_suite.py "${NC_ARGS[@]}"

echo "[15datasets] LP suite (HyboNet dim=16 ...)" >&2
python run_standard_benchmark_suite.py "${LP_ARGS[@]}"

echo "[15datasets] done -> logs/standard_benchmark_aggregate/{nc,lp}/" >&2
