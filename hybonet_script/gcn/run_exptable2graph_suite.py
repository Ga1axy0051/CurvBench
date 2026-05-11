#!/usr/bin/env python
"""
Multi-seed suite for ``exptable_*`` graphs (default six keys, seeds 0-4).

From ``gcn/``::

  python run_exptable2graph_suite.py --task nc --model HyboNet --dim 32 --lr 0.005 \\
      --num-layers 2 --epochs 500 --patience 100

Artifacts per run: ``logs/experiments/<dataset>/<task>/seed_<s>/`` (curve + config forced on).
Aggregate: ``logs/exptable2graph_aggregate/<task>/summary.json`` plus ``mean_std_summary.*``.
"""
from __future__ import print_function

import argparse
import json
import os
import sys

_GCN_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_GCN_ROOT, os.pardir))
if _GCN_ROOT not in sys.path:
    sys.path.insert(0, _GCN_ROOT)

from config import parser
from train import train
from utils.benchmark_aggregate import build_summary_payload, write_mean_std_summary_files
from utils.benchmark_suites import default_exptable_dataset_keys

for _args in (
    ('--suite-seeds', {'type': str, 'default': '0,1,2,3,4',
                       'help': 'comma-separated seeds, default 0-4'}),
    ('--suite-datasets', {'type': str, 'default': None,
                          'help': 'comma-separated exptable_* names; default all six'}),
):
    try:
        parser.add_argument(_args[0], **_args[1])
    except argparse.ArgumentError:
        pass


def _ensure_suite_artifacts(args):
    args.save_curve = max(int(getattr(args, 'save_curve', 0)), 1)
    args.always_save_config = max(int(getattr(args, 'always_save_config', 0)), 1)


def main():
    args = parser.parse_args()
    _ensure_suite_artifacts(args)
    seeds = [int(x.strip()) for x in args.suite_seeds.split(',') if x.strip()]
    if args.suite_datasets:
        dsets = [x.strip() for x in args.suite_datasets.split(',') if x.strip()]
    else:
        dsets = default_exptable_dataset_keys()

    n_total = len(dsets) * len(seeds)
    print(
        '[suite] {} datasets x {} seeds = {} runs'.format(
            len(dsets), len(seeds), n_total),
        flush=True)

    all_rows = []
    task = args.task
    for ds in dsets:
        for sd in seeds:
            args.dataset = ds
            args.seed = sd
            args.split_seed = sd
            print('=== {} seed {} ==='.format(ds, sd), flush=True)
            row = train(args)
            row['dataset'] = ds
            all_rows.append(row)

    payload = build_summary_payload(all_rows, dsets, task)
    out_root = os.path.join(_REPO, 'logs', 'exptable2graph_aggregate', task)
    os.makedirs(out_root, exist_ok=True)
    summary_path = os.path.join(out_root, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print('Wrote', summary_path)
    j2, t2 = write_mean_std_summary_files(out_root, task, payload)
    print('Wrote', j2)
    print('Wrote', t2)


if __name__ == '__main__':
    main()
