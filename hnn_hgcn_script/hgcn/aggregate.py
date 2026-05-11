import argparse
import ast
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from reevaluate_nc_metrics import prepare_data_and_model, evaluate_split, aggregate_metric_dicts
except Exception:
    prepare_data_and_model = None
    evaluate_split = None
    aggregate_metric_dicts = None


LOG_BASE_DIR = 'Result'
DEFAULT_EXPORT_DIR = 'Best_Results'
FINAL_SUMMARY_PATTERN = re.compile(r'FINAL_SUMMARY\s+(.*)')
COLON_PAIR_PATTERN = re.compile(r'(\w+):\s*([^\s]+)')
PAIR_PATTERN = re.compile(r'(\w+)=([^\s]+)')
EPOCH_PATTERN = re.compile(r'Epoch:\s*(\d+)')
TIME_PATTERN = re.compile(r'time:\s*([0-9.eE±]+)s')
TOTAL_TIME_PATTERN = re.compile(r'Total time elapsed:\s*([0-9.eE±]+)s')
ELAPSED_PATTERN = re.compile(r'INFO\s*:\s*root\s*:\s*.*?(\d+:\d{2}:\d{2})')
BEST_EPOCH_PATTERN = re.compile(r'best_epoch=(\d+)')

CURVE_LABELS = {
    'lp_train_loss': 'LP Train Loss',
    'lp_val_loss': 'LP Val Loss',
    'nc_train_loss': 'NC Train Loss',
    'nc_val_loss': 'NC Val Loss',
}

TASK_CONFIG = {
    'nc': {
        'objective_metric': 'test_macro_f1',
        'summary_metrics': [
            'val_acc',
            'val_micro_f1',
            'val_macro_f1',
            'val_weighted_f1',
            'test_acc',
            'test_micro_f1',
            'test_macro_f1',
            'test_weighted_f1',
        ],
        'curve_order': ['nc_train_loss', 'nc_val_loss'],
    },
    'lp': {
        'objective_metric': 'test_auc',
        'summary_metrics': ['val_loss', 'val_auc', 'val_ap', 'test_loss', 'test_auc', 'test_ap'],
        'curve_order': ['lp_train_loss', 'lp_val_loss'],
    },
}

LEGACY_MODELS = {'HGCN', 'HNN'}
EXPORT_FILES = [
    'log.txt',
    'config.json',
    'runner_log.txt',
    'trial_output.txt',
    'model.pth',
    'checkpoint_last.pt',
    'predictions.npy',
    'labels.npy',
    'embeddings.npy',
]


def is_seed_dir(path):
    return os.path.isdir(path) and os.path.basename(path).startswith('seed_')


def iter_seed_dirs(path):
    if not os.path.isdir(path):
        return []
    seed_dirs = [
        os.path.join(path, name)
        for name in os.listdir(path)
        if is_seed_dir(os.path.join(path, name))
    ]
    return sorted(seed_dirs, key=lambda p: int(os.path.basename(p).split('_', 1)[1]))


def parse_standard_task_model_dataset(dir_name):
    parts = dir_name.split('_', 2)
    if len(parts) != 3:
        return None
    return parts[0].lower(), parts[1], parts[2]


def parse_legacy_model_dataset(dir_name):
    parts = dir_name.split('_', 1)
    if len(parts) != 2 or parts[0] not in LEGACY_MODELS:
        return None
    return parts[0], parts[1]


def try_parse_value(raw):
    lowered = raw.lower()
    if lowered == 'none':
        return None
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    try:
        if raw.startswith('0') and raw != '0' and not raw.startswith('0.'):
            return raw
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def parse_summary_line(line):
    pairs = dict(PAIR_PATTERN.findall(line))
    return {key: try_parse_value(value) for key, value in pairs.items()}


def parse_colon_metrics(line):
    pairs = COLON_PAIR_PATTERN.findall(line)
    return {key: try_parse_value(value) for key, value in pairs}


def normalize_summary(task, summary):
    if summary is None:
        return None
    normalized = dict(summary)
    if task == 'lp':
        if 'val_roc' in normalized and 'val_auc' not in normalized:
            normalized['val_auc'] = normalized['val_roc']
        if 'test_roc' in normalized and 'test_auc' not in normalized:
            normalized['test_auc'] = normalized['test_roc']
        return normalized

    if 'val_mf1' in normalized and 'val_macro_f1' not in normalized:
        normalized['val_macro_f1'] = normalized['val_mf1']
    if 'test_mf1' in normalized and 'test_macro_f1' not in normalized:
        normalized['test_macro_f1'] = normalized['test_mf1']
    if 'val_wf1' in normalized and 'val_weighted_f1' not in normalized:
        normalized['val_weighted_f1'] = normalized['val_wf1']
    if 'test_wf1' in normalized and 'test_weighted_f1' not in normalized:
        normalized['test_weighted_f1'] = normalized['test_wf1']
    if 'val_acc' in normalized and 'val_micro_f1' not in normalized:
        normalized['val_micro_f1'] = normalized['val_acc']
    if 'test_acc' in normalized and 'test_micro_f1' not in normalized:
        normalized['test_micro_f1'] = normalized['test_acc']
    return normalized


def read_log_lines(log_path):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            return f.readlines()
    except FileNotFoundError:
        return []


def extract_last_run_lines(log_path):
    lines = read_log_lines(log_path)
    if not lines:
        return []

    run_starts = [
        idx
        for idx, line in enumerate(lines)
        if 'Namespace(' in line or 'Using seed' in line
    ]
    if not run_starts:
        return lines

    blocks = []
    for idx, start in enumerate(run_starts):
        end = run_starts[idx + 1] if idx + 1 < len(run_starts) else len(lines)
        blocks.append(lines[start:end])

    completed_blocks = [
        block
        for block in blocks
        if any('FINAL_SUMMARY' in line or 'Test set results:' in line for line in block)
    ]
    if completed_blocks:
        return completed_blocks[-1]
    return blocks[-1]


def extract_final_summary(log_path, task):
    run_lines = extract_last_run_lines(log_path)
    if not run_lines:
        return None

    content = ''.join(run_lines)
    matches = list(FINAL_SUMMARY_PATTERN.finditer(content))
    if matches:
        return normalize_summary(task, parse_summary_line(matches[-1].group(1)))

    val_summary = None
    test_summary = None
    for line in run_lines:
        if 'Val set results:' in line:
            val_summary = parse_colon_metrics(line)
        elif 'Test set results:' in line:
            test_summary = parse_colon_metrics(line)

    if val_summary is None and test_summary is None:
        return None

    merged = {}
    if val_summary is not None:
        merged.update(val_summary)
    if test_summary is not None:
        merged.update(test_summary)
    return normalize_summary(task, merged)


def parse_elapsed_to_seconds(raw):
    hours, minutes, seconds = raw.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def extract_total_runtime_seconds(log_path):
    max_elapsed = None
    final_total = None
    for line in extract_last_run_lines(log_path):
        total_match = TOTAL_TIME_PATTERN.search(line)
        if total_match:
            final_total = float(total_match.group(1))
        elapsed_match = ELAPSED_PATTERN.search(line)
        if elapsed_match:
            current = parse_elapsed_to_seconds(elapsed_match.group(1))
            max_elapsed = current if max_elapsed is None else max(max_elapsed, current)
    return final_total if final_total is not None else max_elapsed


def extract_best_epoch(log_path):
    best_epoch = None
    for line in extract_last_run_lines(log_path):
        match = BEST_EPOCH_PATTERN.search(line)
        if match:
            best_epoch = int(match.group(1))
    return best_epoch


def extract_epoch_times(log_path):
    times = []
    for line in extract_last_run_lines(log_path):
        match = TIME_PATTERN.search(line)
        if match:
            times.append(float(match.group(1)))
    return times


def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None


def get_group_params(group_path, exp_path, task, model, dataset):
    seed_dirs = iter_seed_dirs(group_path)
    if seed_dirs:
        config = load_json(os.path.join(seed_dirs[0], 'config.json'))
        if config is not None:
            return config

    for path in [
        os.path.join(exp_path, 'optuna_best.txt'),
        os.path.join(exp_path, f'optuna_{dataset}_{task}_{model}_best.txt'),
        os.path.join(exp_path, f'optuna_{dataset}_{task}_best.txt'),
    ]:
        lines = read_log_lines(path)
        if not lines:
            continue
        try:
            parsed = ast.literal_eval(lines[0].strip())
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    return None


def iter_param_dirs(exp_path):
    root_seed_dirs = iter_seed_dirs(exp_path)
    if root_seed_dirs:
        return [('default', exp_path)]

    result = []
    for name in sorted(os.listdir(exp_path)):
        candidate = os.path.join(exp_path, name)
        if not os.path.isdir(candidate) or name == 'optuna_runs':
            continue
        if is_seed_dir(candidate):
            continue
        if iter_seed_dirs(candidate):
            result.append((name, candidate))

    if result:
        return result

    optuna_root = os.path.join(exp_path, 'optuna_runs')
    if os.path.isdir(optuna_root):
        for name in sorted(os.listdir(optuna_root)):
            trial_path = os.path.join(optuna_root, name)
            if iter_seed_dirs(trial_path):
                result.append((name, trial_path))
    return result


def iter_experiments(base_dir, task):
    for dir_name in sorted(os.listdir(base_dir)):
        exp_path = os.path.join(base_dir, dir_name)
        if not os.path.isdir(exp_path):
            continue

        parsed = parse_standard_task_model_dataset(dir_name)
        if parsed is not None:
            dir_task, model, dataset = parsed
            if dir_task == task:
                yield model, dataset, exp_path, True
            continue

        legacy = parse_legacy_model_dataset(dir_name)
        if legacy is not None:
            model, dataset = legacy
            yield model, dataset, exp_path, False


def fresh_curves(task):
    if task == 'lp':
        return {
            'lp_train_loss': [],
            'lp_val_loss': [],
        }
    return {
        'nc_train_loss': [],
        'nc_val_loss': [],
    }


def append_curve_point(curves, curve_key, epoch, metrics_key, metrics):
    value = metrics.get(metrics_key)
    if value is not None:
        curves[curve_key].append((epoch, float(value)))


def extract_curves(log_path, task):
    curves = fresh_curves(task)
    for line in extract_last_run_lines(log_path):
        if 'Epoch:' not in line:
            continue
        epoch_match = EPOCH_PATTERN.search(line)
        if not epoch_match:
            continue
        epoch = int(epoch_match.group(1))
        metrics = parse_colon_metrics(line)

        if task == 'lp':
            if 'train_loss' in metrics:
                append_curve_point(curves, 'lp_train_loss', epoch, 'train_loss', metrics)
            if 'val_loss' in metrics:
                append_curve_point(curves, 'lp_val_loss', epoch, 'val_loss', metrics)
        else:
            if 'train_loss' in metrics:
                append_curve_point(curves, 'nc_train_loss', epoch, 'train_loss', metrics)
            if 'val_loss' in metrics:
                append_curve_point(curves, 'nc_val_loss', epoch, 'val_loss', metrics)
    return curves


def align_curves(curves):
    valid_curves = [curve for curve in curves if curve]
    if not valid_curves:
        return None

    min_len = min(len(curve) for curve in valid_curves)
    if min_len == 0:
        return None

    aligned_epochs = np.array([valid_curves[0][idx][0] for idx in range(min_len)])
    aligned_values = np.array([[curve[idx][1] for idx in range(min_len)] for curve in valid_curves])
    return aligned_epochs, aligned_values.mean(axis=0)


def format_duration(seconds):
    if seconds is None:
        return 'N/A'
    total = int(round(seconds))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f'{hours}:{minutes:02d}:{secs:02d}'


def format_metric_stats(metric_stats, metric_name):
    stats = metric_stats.get(metric_name)
    if stats is None:
        return None
    return f'{stats["mean"]:.6f} +/- {stats["std"]:.6f}'


def curve_points_to_json(points):
    return [[int(epoch), float(value)] for epoch, value in points]


def build_curve_export(task, group_path):
    seed_curves = {}
    for seed_path in iter_seed_dirs(group_path):
        seed_name = os.path.basename(seed_path)
        seed_curves[seed_name] = extract_curves(os.path.join(seed_path, 'log.txt'), task)

    export = {}
    for curve_key in TASK_CONFIG[task]['curve_order']:
        per_seed = {}
        for seed_name, curves in seed_curves.items():
            points = curves.get(curve_key, [])
            if points:
                per_seed[seed_name] = curve_points_to_json(points)
        aligned = align_curves([curves.get(curve_key, []) for curves in seed_curves.values()])
        mean_curve = []
        if aligned is not None:
            epochs, mean_values = aligned
            mean_curve = [[int(epoch), float(value)] for epoch, value in zip(epochs.tolist(), mean_values.tolist())]
        if per_seed or mean_curve:
            export[curve_key] = {
                'label': CURVE_LABELS.get(curve_key, curve_key),
                'per_seed': per_seed,
                'mean_curve': mean_curve,
            }
    return export


def maybe_reevaluate_nc(group_path):
    if prepare_data_and_model is None or evaluate_split is None or aggregate_metric_dicts is None:
        return None

    seed_results = []
    for seed_path in iter_seed_dirs(group_path):
        config_path = os.path.join(seed_path, 'config.json')
        model_path = os.path.join(seed_path, 'model.pth')
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            continue
        try:
            _, data, model = prepare_data_and_model(Path(seed_path))
            metrics = {}
            metrics.update(evaluate_split(model, data, 'val'))
            metrics.update(evaluate_split(model, data, 'test'))
            seed_results.append({'seed': os.path.basename(seed_path), 'metrics': metrics})
        except Exception:
            continue

    if not seed_results:
        return None

    return {
        'seed_results': seed_results,
        'summary': aggregate_metric_dicts([item['metrics'] for item in seed_results]),
        'note': 'Metrics are recomputed from saved model.pth checkpoints. model.pth is the final checkpoint, not guaranteed to be the best-val checkpoint used in the original log summary.',
    }


def plot_curve_bundle(curve_export, dest_root, task, model, dataset, plot_per_seed=False):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib is not available, skip plotting.')
        return

    plot_dir = os.path.join(dest_root, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for curve_key, payload in curve_export.items():
        mean_curve = payload.get('mean_curve', [])
        if mean_curve:
            epochs = [point[0] for point in mean_curve]
            values = [point[1] for point in mean_curve]
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, values)
            plt.title(f'{task.upper()} {model} {dataset} - {payload["label"]} (mean over seeds)')
            plt.xlabel('Epoch')
            plt.ylabel(payload['label'])
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{task}_{model}_{dataset}_{curve_key}.png'))
            plt.close()

        if plot_per_seed:
            for seed_name, points in payload.get('per_seed', {}).items():
                epochs = [point[0] for point in points]
                values = [point[1] for point in points]
                plt.figure(figsize=(10, 5))
                plt.plot(epochs, values)
                plt.title(f'{task.upper()} {model} {dataset} - {payload["label"]} ({seed_name})')
                plt.xlabel('Epoch')
                plt.ylabel(payload['label'])
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, f'{task}_{model}_{dataset}_{curve_key}_{seed_name}.png'))
                plt.close()


def summarize_results(base_dir, task, *, skip_reevaluate_nc=False):
    config = TASK_CONFIG[task]
    objective_metric = config['objective_metric']
    raw_results = defaultdict(
        lambda: {
            'summary_metrics': defaultdict(list),
            'epoch_times': [],
            'total_runtimes': [],
            'best_epochs': [],
            'params': None,
            'group_path': None,
            'seed_details': [],
            'source_dir': None,
            'layout': None,
        }
    )
    best_overall_configs = {}

    print(f'--- Step 1: Collecting results from {base_dir} (Metric: {objective_metric.upper()}) ---')

    for model, dataset, exp_path, is_standard in iter_experiments(base_dir, task):
        for param_id, group_path in iter_param_dirs(exp_path):
            key = (model, dataset, param_id)
            if raw_results[key]['params'] is None:
                raw_results[key]['params'] = get_group_params(group_path, exp_path, task, model, dataset)
            raw_results[key]['group_path'] = group_path
            raw_results[key]['source_dir'] = exp_path
            raw_results[key]['layout'] = 'standard' if is_standard else 'legacy'

            for seed_path in iter_seed_dirs(group_path):
                log_path = os.path.join(seed_path, 'log.txt')
                summary = extract_final_summary(log_path, task)
                if summary is None:
                    continue

                epoch_times = extract_epoch_times(log_path)
                total_runtime = extract_total_runtime_seconds(log_path)
                best_epoch = extract_best_epoch(log_path)
                seed_detail = {
                    'seed': os.path.basename(seed_path),
                    'metrics': summary,
                    'avg_epoch_time_seconds': float(np.mean(epoch_times)) if epoch_times else None,
                    'total_runtime_seconds': float(total_runtime) if total_runtime is not None else None,
                    'best_epoch': int(best_epoch) if best_epoch is not None else None,
                    'log_path': log_path,
                    'config_path': os.path.join(seed_path, 'config.json'),
                }
                raw_results[key]['seed_details'].append(seed_detail)

                for metric_name in config['summary_metrics']:
                    value = summary.get(metric_name)
                    if value is not None:
                        raw_results[key]['summary_metrics'][metric_name].append(float(value))

                if epoch_times:
                    raw_results[key]['epoch_times'].append(float(np.mean(epoch_times)))
                if total_runtime is not None:
                    raw_results[key]['total_runtimes'].append(float(total_runtime))
                if best_epoch is not None:
                    raw_results[key]['best_epochs'].append(float(best_epoch))

    print('\n--- Step 2: Calculating Mean/Std and Identifying Best Configs ---')

    for key, data in raw_results.items():
        model, dataset, param_id = key

        metric_stats = {}
        for metric_name in config['summary_metrics']:
            values = data['summary_metrics'].get(metric_name, [])
            if not values:
                continue
            metric_stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'n_runs': len(values),
            }

        reevaluated_nc = None
        if task == 'nc' and not skip_reevaluate_nc:
            reevaluated_nc = maybe_reevaluate_nc(data['group_path'])
        export_metric_stats = dict(metric_stats)
        if task == 'nc' and reevaluated_nc and reevaluated_nc.get('summary'):
            for metric_name in [
                'val_acc',
                'val_micro_f1',
                'val_macro_f1',
                'val_weighted_f1',
                'test_acc',
                'test_micro_f1',
                'test_macro_f1',
                'test_weighted_f1',
            ]:
                if metric_name in reevaluated_nc['summary']:
                    export_metric_stats[metric_name] = reevaluated_nc['summary'][metric_name]

        objective_source = export_metric_stats if objective_metric in export_metric_stats else metric_stats
        objective_stats = objective_source.get(objective_metric)
        if objective_stats is None:
            continue

        current_result = {
            'params': data['params'],
            'param_id': param_id,
            'group_path': data['group_path'],
            'source_dir': data['source_dir'],
            'layout': data['layout'],
            'metric_stats': metric_stats,
            'export_metric_stats': export_metric_stats,
            'mean': objective_stats['mean'],
            'std': objective_stats['std'],
            'n_runs': objective_stats['n_runs'],
            'mean_epoch_time': float(np.mean(data['epoch_times'])) if data['epoch_times'] else 0.0,
            'mean_total_runtime': float(np.mean(data['total_runtimes'])) if data['total_runtimes'] else 0.0,
            'sum_total_runtime': float(np.sum(data['total_runtimes'])) if data['total_runtimes'] else 0.0,
            'mean_best_epoch': float(np.mean(data['best_epochs'])) if data['best_epochs'] else None,
            'seed_details': data['seed_details'],
            'reevaluated_nc': reevaluated_nc,
        }

        result_key = (model, dataset)
        previous = best_overall_configs.get(result_key)
        if previous is None or current_result['mean'] > previous['mean']:
            best_overall_configs[result_key] = current_result

    print('\n=======================================================')
    print(f'Final Best Hyperparameter Results for {task.upper()} (Metric: {objective_metric.upper()})')
    print('=======================================================')

    for model, dataset in sorted(best_overall_configs.keys()):
        result = best_overall_configs[(model, dataset)]
        print(f'\n--- Best for {model} on {dataset} ---')
        print(f'  > Config Group: {result["param_id"]}')
        print(f'  > Layout: {result["layout"]}')
        print(f'  > Runs: {result["n_runs"]}')

        for metric_name in config['summary_metrics']:
            metric_text = format_metric_stats(result['export_metric_stats'], metric_name)
            if metric_text is not None:
                print(f'  > {metric_name.upper()}: {metric_text}')

        if result['reevaluated_nc'] and result['reevaluated_nc'].get('summary'):
            print('  > Reevaluated NC metrics:')
            for metric_name in [
                'val_acc',
                'val_micro_f1',
                'val_macro_f1',
                'val_weighted_f1',
                'test_acc',
                'test_micro_f1',
                'test_macro_f1',
                'test_weighted_f1',
            ]:
                stats = result['reevaluated_nc']['summary'].get(metric_name)
                if stats is not None:
                    print(f'    - {metric_name}: {stats["mean"]:.6f} +/- {stats["std"]:.6f}')

        if result['mean_best_epoch'] is not None:
            print(f'  > Mean BEST_EPOCH: {result["mean_best_epoch"]:.2f}')
        print(f'  > Mean Total Runtime: {format_duration(result["mean_total_runtime"])} ({result["mean_total_runtime"]:.1f}s)')
        print(f'  > Summed Total Runtime: {format_duration(result["sum_total_runtime"])} ({result["sum_total_runtime"]:.1f}s)')
        print(f'  > Avg Time per Epoch: {result["mean_epoch_time"]:.4f}s')
        if result['params'] is not None:
            print('  > Parameters:')
            print(json.dumps(result['params'], indent=2, ensure_ascii=False))

    return best_overall_configs


def export_best_results(best_configs, task, export_base_dir, plot_per_seed=False):
    print(f"\n--- Step 3: Exporting original files and summaries to '{export_base_dir}' ---")
    os.makedirs(export_base_dir, exist_ok=True)

    for (model, dataset), result in best_configs.items():
        dest_root = os.path.join(export_base_dir, f'{task}_{model}_{dataset}')
        os.makedirs(dest_root, exist_ok=True)

        reevaluated_by_seed = {}
        if result['reevaluated_nc']:
            reevaluated_by_seed = {
                item['seed']: item['metrics']
                for item in result['reevaluated_nc'].get('seed_results', [])
            }

        seed_metric_rows = []
        for seed_path in iter_seed_dirs(result['group_path']):
            seed_name = os.path.basename(seed_path)
            target_seed_path = os.path.join(dest_root, seed_name)
            os.makedirs(target_seed_path, exist_ok=True)

            for file_name in EXPORT_FILES:
                src_file = os.path.join(seed_path, file_name)
                if os.path.exists(src_file):
                    shutil.copy2(src_file, target_seed_path)

        for seed_detail in result['seed_details']:
            row = dict(seed_detail)
            if seed_detail['seed'] in reevaluated_by_seed:
                row['reevaluated_metrics'] = reevaluated_by_seed[seed_detail['seed']]
            seed_metric_rows.append(row)

        curve_export = build_curve_export(task, result['group_path'])
        plot_curve_bundle(curve_export, dest_root, task, model, dataset, plot_per_seed=plot_per_seed)

        aggregate_summary = {
            'task': task,
            'model': model,
            'dataset': dataset,
            'param_id': result['param_id'],
            'layout': result['layout'],
            'source_dir': result['source_dir'],
            'params': result['params'],
            'metric_stats': result['metric_stats'],
            'export_metric_stats': result['export_metric_stats'],
            'mean_total_runtime_seconds': result['mean_total_runtime'],
            'sum_total_runtime_seconds': result['sum_total_runtime'],
            'avg_epoch_time_seconds': result['mean_epoch_time'],
            'mean_best_epoch': result['mean_best_epoch'],
            'n_runs': result['n_runs'],
            'reevaluated_nc': result['reevaluated_nc'],
        }

        with open(os.path.join(dest_root, 'aggregate_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(aggregate_summary, f, indent=2, ensure_ascii=False)

        with open(os.path.join(dest_root, 'seed_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(seed_metric_rows, f, indent=2, ensure_ascii=False)

        with open(os.path.join(dest_root, 'curve_data.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'task': task,
                'model': model,
                'dataset': dataset,
                'curves': curve_export,
            }, f, indent=2, ensure_ascii=False)

    print(f'All best configuration files have been exported to: {os.path.abspath(export_base_dir)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=LOG_BASE_DIR)
    parser.add_argument('--task', default='nc', choices=['nc', 'lp'])
    parser.add_argument('--export_dir', default=DEFAULT_EXPORT_DIR)
    parser.add_argument('--plot_per_seed', action='store_true')
    parser.add_argument(
        '--skip_reevaluate_nc',
        action='store_true',
        help='Skip expensive NC reevaluation even if dependencies are available.',
    )
    args = parser.parse_args()

    best_configs = summarize_results(args.base_dir, args.task, skip_reevaluate_nc=args.skip_reevaluate_nc)
    export_best_results(best_configs, args.task, args.export_dir, plot_per_seed=args.plot_per_seed)


if __name__ == '__main__':
    main()
