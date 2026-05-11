import argparse
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from config import parser as config_parser
from models.base_models import NCModel
from utils.data_utils import load_data


def ensure_runtime_env(project_root):
    os.environ.setdefault('DATAPATH', os.path.join(project_root, 'data'))
    os.environ.setdefault('LOG_DIR', os.path.join(project_root, 'logs'))


def is_seed_dir(path):
    return os.path.isdir(path) and os.path.basename(path).startswith('seed_')


def iter_seed_dirs(path):
    seed_dirs = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if is_seed_dir(os.path.join(path, name))
    ]
    return sorted(seed_dirs, key=lambda p: int(os.path.basename(p).split('_', 1)[1]))


def parse_task_model_dataset(dir_name):
    parts = dir_name.split('_', 2)
    if len(parts) != 3:
        return None
    return parts[0].lower(), parts[1], parts[2]


def build_args(config_dict):
    args = config_parser.parse_args([])
    for key, value in config_dict.items():
        setattr(args, key, value)
    args.cuda = -1
    args.device = 'cpu'
    args.save = 0
    return args


def extract_model_state_dict(payload):
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict']
    return payload



def prepare_data_and_model(project_root, seed_dir):
    config_path = os.path.join(seed_dir, 'config.json')
    model_path = os.path.join(seed_dir, 'model.pth')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    args = build_args(config)

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)

    ensure_runtime_env(project_root)
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    args.n_classes = int(data['labels'].max().item() + 1)

    model = NCModel(args)
    payload = torch.load(model_path, map_location='cpu')
    state_dict = extract_model_state_dict(payload)
    model.load_state_dict(state_dict)
    model.eval()
    return args, data, model


def evaluate_split(model, data, split):
    idx = data[f'idx_{split}']
    with torch.no_grad():
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        output = model.decode(embeddings, data['adj_train_norm'], idx)
        preds = output.max(1)[1].cpu().numpy()
        labels = data['labels'][idx].cpu().numpy()

    return {
        f'{split}_acc': float(accuracy_score(labels, preds)),
        f'{split}_micro_f1': float(f1_score(labels, preds, average='micro', zero_division=0)),
        f'{split}_weighted_f1': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        f'{split}_macro_f1': float(f1_score(labels, preds, average='macro', zero_division=0)),
        f'{split}_wf1': float(f1_score(labels, preds, average='weighted', zero_division=0)),
        f'{split}_mf1': float(f1_score(labels, preds, average='macro', zero_division=0)),
    }


def aggregate_metric_dicts(metric_dicts):
    summary = {}
    if not metric_dicts:
        return summary

    metric_names = sorted(metric_dicts[0].keys())
    for metric_name in metric_names:
        values = [metrics[metric_name] for metrics in metric_dicts]
        summary[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'n_runs': len(values),
        }
    return summary


def format_stat(stat):
    return f"{stat['mean']:.6f} +/- {stat['std']:.6f}"


def reevaluate_nc_results(base_dir, project_root):
    aggregated = {}

    for dir_name in sorted(os.listdir(base_dir)):
        exp_dir = os.path.join(base_dir, dir_name)
        if not os.path.isdir(exp_dir):
            continue

        parsed = parse_task_model_dataset(dir_name)
        if parsed is None:
            continue
        task, model_name, dataset = parsed
        if task != 'nc':
            continue

        seed_metrics = []
        print(f'\n=== Reevaluating {dir_name} ===')
        for seed_dir in iter_seed_dirs(exp_dir):
            config_path = os.path.join(seed_dir, 'config.json')
            model_path = os.path.join(seed_dir, 'model.pth')
            if not os.path.exists(config_path) or not os.path.exists(model_path):
                print(f'Skip {seed_dir}: missing config.json or model.pth')
                continue

            try:
                _, data, model = prepare_data_and_model(project_root, seed_dir)
                metrics = {}
                metrics.update(evaluate_split(model, data, 'val'))
                metrics.update(evaluate_split(model, data, 'test'))
                seed_metrics.append({
                    'seed_dir': seed_dir,
                    'metrics': metrics,
                })
                print(f"{os.path.basename(seed_dir)}: "
                      f"val_acc={metrics['val_acc']:.6f}, val_micro_f1={metrics['val_micro_f1']:.6f}, "
                      f"val_weighted_f1={metrics['val_weighted_f1']:.6f}, val_macro_f1={metrics['val_macro_f1']:.6f}, "
                      f"test_acc={metrics['test_acc']:.6f}, test_micro_f1={metrics['test_micro_f1']:.6f}, "
                      f"test_weighted_f1={metrics['test_weighted_f1']:.6f}, test_macro_f1={metrics['test_macro_f1']:.6f}")
            except Exception as exc:
                print(f'Skip {seed_dir}: {exc}')

        summary = aggregate_metric_dicts([item['metrics'] for item in seed_metrics])
        aggregated[f'{model_name}:{dataset}'] = {
            'experiment_dir': exp_dir,
            'seed_results': seed_metrics,
            'summary': summary,
            'note': 'Metrics are recomputed from saved model.pth checkpoints. model.pth may be either a raw state_dict or a checkpoint payload with model_state_dict; in both cases it reflects the final saved state, not guaranteed to be the original best-val checkpoint.',
        }

        if summary:
            print('--- Aggregate ---')
            for metric_name in [
                'val_acc',
                'val_micro_f1',
                'val_weighted_f1',
                'val_macro_f1',
                'test_acc',
                'test_micro_f1',
                'test_weighted_f1',
                'test_macro_f1',
            ]:
                stat = summary.get(metric_name)
                if stat is not None:
                    print(f'{metric_name}: {format_stat(stat)}')

    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='Result')
    parser.add_argument('--summary_path', default=None)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    results = reevaluate_nc_results(args.base_dir, project_root)

    if args.summary_path:
        with open(args.summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f'Saved summary to: {os.path.abspath(args.summary_path)}')


if __name__ == '__main__':
    main()
