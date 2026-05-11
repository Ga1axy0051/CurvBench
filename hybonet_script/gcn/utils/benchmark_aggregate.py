"""Aggregate multi-run metrics: per-dataset mean and population std (ddof=0)."""
from __future__ import print_function

import json
import os

import numpy as np


def _metric_keys_for_task(task):
    if task == 'nc':
        eff = ['test_acc', 'test_macro_f1', 'test_micro_f1']
    else:
        eff = [
            'test_roc', 'test_ap', 'test_acc', 'test_macro_f1', 'test_micro_f1',
            'test_pair_acc',
        ]
    timing = ['total_train_time_sec', 'time_per_epoch_sec', 'cum_test_time_sec']
    return eff + timing


def aggregate_by_dataset(all_rows, datasets, task):
    """dataset -> {metric: {mean, std}, ...}."""
    metric_keys = _metric_keys_for_task(task)
    by_ds = {}
    for ds in datasets:
        rows = [r for r in all_rows if r.get('dataset') == ds]
        if not rows:
            continue
        agg = {
            'dataset': ds,
            'task': task,
            'n_runs': len(rows),
            'seeds': sorted({int(r['seed']) for r in rows if 'seed' in r}),
        }
        for k in metric_keys:
            vals = []
            for r in rows:
                v = r.get(k)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if np.isnan(fv):
                        continue
                    vals.append(fv)
                except (TypeError, ValueError):
                    continue
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                agg[k] = {'mean': float(arr.mean()), 'std': float(arr.std(ddof=0))}
        by_ds[ds] = agg
    return by_ds


def build_summary_payload(all_rows, datasets, task):
    return {
        'per_run': all_rows,
        'per_dataset': aggregate_by_dataset(all_rows, datasets, task),
    }


def write_mean_std_summary_files(out_dir, task, payload):
    """Write mean_std_summary.json and .txt (per_dataset only, no per_run)."""
    os.makedirs(out_dir, exist_ok=True)
    per_ds = payload.get('per_dataset') or {}
    slim = {
        'task': task,
        'std_note': 'population std (ddof=0), same as per_dataset in summary.json',
        'per_dataset': per_ds,
    }
    json_path = os.path.join(out_dir, 'mean_std_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)

    lines = [
        'task: {}'.format(task),
        'std: population (ddof=0)',
        '',
    ]
    for ds in sorted(per_ds.keys()):
        agg = per_ds[ds]
        lines.append('== {} =='.format(ds))
        lines.append('  n_runs: {}  seeds: {}'.format(
            agg.get('n_runs', '?'), agg.get('seeds', [])))
        for k in sorted(agg.keys()):
            if k in ('dataset', 'task', 'n_runs', 'seeds'):
                continue
            v = agg[k]
            if isinstance(v, dict) and 'mean' in v and 'std' in v:
                lines.append('  {:<26} mean={:.8f}  std={:.8f}'.format(
                    k, float(v['mean']), float(v['std'])))
        lines.append('')
    txt_path = os.path.join(out_dir, 'mean_std_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return json_path, txt_path
