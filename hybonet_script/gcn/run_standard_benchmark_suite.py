#!/usr/bin/env python
"""
Multi-seed benchmark driver. Default dataset list: ``utils/benchmark_suites`` (8 sets;
``disease_nc``/``telecom_nc`` vs ``disease_lp``/``telecom_lp`` by task).

``telecom_*`` expects ``data/telecom`` with ``edges.csv``, ``feats.npz``, ``labels.npy``
(see ``telecom/convert_telecom_to_gcn.py``).

Seeds default to 0-4 (``--suite-seeds``). Each run writes under
``logs/experiments/<dataset>/<task>/seed_<s>/`` (``train.py`` forces save-curve + always-save-config).

Example single dataset::

  python run_standard_benchmark_suite.py --task nc --suite-datasets cs_phds_nc

Aggregates (overwritten each full run) under ``logs/standard_benchmark_aggregate/<task>/``:
``summary.json`` (per_run + per_dataset mean/std), ``mean_std_summary.{json,txt}``.
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
from utils.benchmark_suites import default_standard_dataset_names

for _args in (
    ('--suite-seeds', {'type': str, 'default': '0,1,2,3,4',
                       'help': 'comma-separated seeds, default 0-4'}),
    ('--suite-datasets', {'type': str, 'default': None,
                          'help': 'comma-separated names under data/; default: eight benchmarks '
                                   '(NC: …disease_nc,telecom_nc / LP: …disease_lp,telecom_lp)'}),
):
    try:
        parser.add_argument(_args[0], **_args[1])
    except argparse.ArgumentError:
        pass


def _ensure_suite_artifacts(args):
    """Enable curve + config dumps without forcing full checkpoint save."""
    args.save_curve = max(int(getattr(args, 'save_curve', 0)), 1)
    args.always_save_config = max(int(getattr(args, 'always_save_config', 0)), 1)


def main():
    args = parser.parse_args()
    _ensure_suite_artifacts(args)
    seeds = [int(x.strip()) for x in args.suite_seeds.split(',') if x.strip()]
    if args.suite_datasets:
        dsets = [x.strip() for x in args.suite_datasets.split(',') if x.strip()]
    else:
        dsets = default_standard_dataset_names(args.task)

    task = args.task
    print(
        '[suite] {} datasets x {} seeds = {} runs'.format(
            len(dsets), len(seeds), len(dsets) * len(seeds)),
        flush=True)
    all_rows = []
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
    out_root = os.path.join(_REPO, 'logs', 'standard_benchmark_aggregate', task)
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
