import argparse
import json
import re
from datetime import datetime
from pathlib import Path

try:
    import torch
except Exception:
    torch = None

EXPECTED_SEEDS = [0, 1, 2, 3, 4]
TARGET_EPOCHS = 500
TARGET_PATIENCE = 100
LEGACY_HGCN_MODELS = {'HGCN', 'HNN'}

ROOT = Path(__file__).resolve().parent

FINAL_SUMMARY_PATTERN = re.compile(r'FINAL_SUMMARY\s+(.*)')
COLON_PAIR_PATTERN = re.compile(r'(\w+):\s*([^\s]+)')
PAIR_PATTERN = re.compile(r'(\w+)=([^\s]+)')
TIME_COLON_PATTERN = re.compile(r'time:\s*([0-9.eE±]+)s')
TIME_EQUAL_PATTERN = re.compile(r'time=([0-9.eE±]+)')
TOTAL_TIME_PATTERN = re.compile(r'Total time elapsed:\s*([0-9.eE±]+)s')
ELAPSED_PATTERN = re.compile(r'INFO\s*-\s*.*?\s*-\s*(\d+:\d{2}:\d{2})\s*-')
BEST_EPOCH_PATTERN = re.compile(r'best_epoch=(\d+)')
TRIAL_DONE_PATTERN = re.compile(r'Trial Done', re.IGNORECASE)
TRIAL_HEADER_PATTERN = re.compile(r'\btrial[_\s-]?(\d+)\b', re.IGNORECASE)
OPTUNA_CMD_TRIALS_PATTERN = re.compile(r'--n[-_]trials\s+(\d+)', re.IGNORECASE)
OPTUNA_CMD_NUM_SEEDS_PATTERN = re.compile(r'--num[-_]seeds\s+(\d+)', re.IGNORECASE)
OPTUNA_START_TRIALS_PATTERN = re.compile(r'(\d+)\s+trial\(s\)', re.IGNORECASE)
OPTUNA_START_NUM_SEEDS_PATTERN = re.compile(r'(\d+)\s+seed(?:\(s\)|s)\b', re.IGNORECASE)
TRAIN_CMD_EPOCHS_PATTERN = re.compile(r'--epochs\s+(\d+)', re.IGNORECASE)
TRAIN_CMD_PATIENCE_PATTERN = re.compile(r'--patience\s+(\d+)', re.IGNORECASE)
TRAIN_CMD_EPOCHS_NC_PATTERN = re.compile(r'--epochs(?:[-_](?:cls|nc))\s+(\d+)', re.IGNORECASE)
TRAIN_CMD_PATIENCE_NC_PATTERN = re.compile(r'--patience(?:[-_](?:cls|nc))\s+(\d+)', re.IGNORECASE)
TRAIN_CMD_EPOCHS_LP_PATTERN = re.compile(r'--epochs(?:[-_]lp)\s+(\d+)', re.IGNORECASE)
TRAIN_CMD_PATIENCE_LP_PATTERN = re.compile(r'--patience(?:[-_]lp)\s+(\d+)', re.IGNORECASE)

PROJECTS = [
    {
        'name': 'hgcn',
        'result_dir': ROOT / 'hgcn' / 'Result',
        'task_case': 'lower',
        'requires_model_file': True,
        'optuna_kind': 'txt',
    },
    {
        'name': 'QGCN-main',
        'result_dir': ROOT / 'QGCN-main' / 'Result',
        'task_case': 'lower',
        'requires_model_file': True,
        'optuna_kind': 'txt',
    },
    {
        'name': 'GraphMoRE-main',
        'result_dir': ROOT / 'GraphMoRE-main' / 'Result',
        'task_case': 'upper',
        'requires_model_file': False,
        'optuna_kind': 'json',
    },
]

EXPECTED_EXPERIMENTS = {
    'hgcn': [
        *[(baseline, 'lp', dataset) for baseline in ['HGCN', 'HNN'] for dataset in ['disease_lp', 'telecom', 'actor', 'cornell', 'cs_phds']],
        *[(baseline, 'nc', dataset)
          for baseline in ['HGCN', 'HNN']
          for dataset in ['cora', 'citeseer', 'pubmed', 'airport', 'disease_nc', 'telecom', 'actor', 'cornell', 'cs_phds', 'Carcinogenesis_data', 'Hepatitis_std_data', 'Hockey_data', 'PTE', 'Toxicology_data']],
    ],
    'QGCN-main': [
        *[('QGCN', 'lp', dataset) for dataset in ['disease_lp', 'telecom', 'actor', 'cornell', 'cs_phds']],
        *[('QGCN', 'nc', dataset) for dataset in ['cora', 'citeseer', 'pubmed', 'airport', 'disease_nc', 'telecom', 'actor', 'cornell', 'cs_phds', 'Carcinogenesis_data', 'Hepatitis_std_data', 'Hockey_data', 'PTE', 'Toxicology_data']],
    ],
    'GraphMoRE-main': [
        *[('GraphMoRE', 'LP', dataset) for dataset in ['telecom', 'Actor', 'cornell', 'cs_phds', 'disease_lp']],
        *[('GraphMoRE', 'NC', dataset) for dataset in ['cora', 'citeseer', 'pubmed', 'airport', 'telecom', 'Actor', 'cornell', 'cs_phds', 'disease_nc', 'Carcinogenesis_data', 'Hepatitis_std_data', 'Hockey_data', 'PTE', 'Toxicology_data']],
    ],
}


def build_expected_dataset_aliases():
    aliases = {}
    for project_name, items in EXPECTED_EXPERIMENTS.items():
        dataset_aliases = {}
        for _baseline, _task, dataset in items:
            dataset_aliases.setdefault(str(dataset).lower(), dataset)
        aliases[project_name] = dataset_aliases
    return aliases


EXPECTED_DATASET_ALIASES = build_expected_dataset_aliases()


def display_path(path: Path):
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        try:
            return str(path.relative_to(ROOT))
        except Exception:
            return str(path)



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


def normalize_summary(task: str | None, summary):
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



def load_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception:
        return None



def load_torch_payload(path: Path):
    if torch is None:
        return None
    try:
        return torch.load(path, map_location='cpu')
    except FileNotFoundError:
        return None
    except Exception:
        return None



def read_lines(path: Path):
    try:
        with path.open('r', encoding='utf-8', errors='replace') as f:
            return f.readlines()
    except FileNotFoundError:
        return []



def parse_task_model_dataset(project_name: str, dir_name: str):
    parts = dir_name.split('_', 2)

    if project_name == 'GraphMoRE-main':
        if len(parts) == 3 and parts[0] in {'LP', 'NC'}:
            return {
                'task': parts[0],
                'baseline': parts[1],
                'dataset': parts[2],
                'layout': 'standard',
            }
        return None

    if len(parts) == 3 and parts[0].lower() in {'lp', 'nc'}:
        return {
            'task': parts[0],
            'baseline': parts[1],
            'dataset': parts[2],
            'layout': 'standard',
        }

    if project_name == 'hgcn':
        legacy_parts = dir_name.split('_', 1)
        if len(legacy_parts) == 2 and legacy_parts[0] in LEGACY_HGCN_MODELS:
            return {
                'task': None,
                'baseline': legacy_parts[0],
                'dataset': legacy_parts[1],
                'layout': 'legacy',
            }

    return None


def canonicalize_dataset_name(project_name: str, dataset: str):
    dataset_str = str(dataset)
    return EXPECTED_DATASET_ALIASES.get(project_name, {}).get(dataset_str.lower(), dataset_str)


def canonical_task_key(task: str):
    return str(task).lower()


def canonical_dataset_key(project_name: str, dataset: str):
    return canonicalize_dataset_name(project_name, dataset).lower()


def canonical_experiment_key(project_name: str, baseline: str, task: str, dataset: str):
    return (
        str(project_name),
        str(baseline),
        canonical_task_key(task),
        canonical_dataset_key(project_name, dataset),
    )


def choose_preferred_experiment(existing: dict, candidate: dict):
    def sort_key(exp: dict):
        completed = len(exp.get('completed_seeds', []))
        failed = len(exp.get('failed_seeds', []))
        rerun = len(exp.get('rerun_seeds', []))
        optuna_best = 1 if exp.get('optuna', {}).get('best_exists') else 0
        fully_complete = 1 if rerun == 0 else 0
        preferred_spelling = 1 if exp.get('raw_dataset', exp.get('dataset')) == exp.get('dataset') else 0
        return (fully_complete, completed, optuna_best, preferred_spelling, -failed, -rerun)

    return max(existing, candidate, key=sort_key)


def determine_recovery_action(exp: dict):
    if exp.get('optuna', {}).get('needs_rerun', False):
        return 'rerun_optuna_and_train'
    if exp.get('rerun_seeds'):
        return 'rerun_seeds_only'
    return 'none'


def has_formally_complete_seed_set(group_payload: dict):
    if set(group_payload.get('completed_seeds', [])) != set(EXPECTED_SEEDS):
        return False
    if group_payload.get('rerun_seeds') or group_payload.get('failed_seeds') or group_payload.get('missing_seeds'):
        return False
    for seed in group_payload.get('seeds', []):
        if seed.get('status') != 'complete':
            return False
        if seed.get('meets_epoch_budget') is False:
            return False
    return True


def downgrade_optuna_rerun_for_complete_results(optuna_info: dict, group_payload: dict):
    if not optuna_info.get('needs_rerun'):
        return optuna_info
    if not has_formally_complete_seed_set(group_payload):
        return optuna_info

    reasons = list(optuna_info.get('rerun_reasons', []))
    advisory_only_reasons = {
        'missing_optuna_best',
        'optuna_smoke_trials',
        'optuna_epoch_budget_below_target',
        'optuna_patience_below_target',
    }
    if not reasons or not set(reasons).issubset(advisory_only_reasons):
        return optuna_info

    downgraded = dict(optuna_info)
    downgraded['needs_rerun'] = False
    downgraded['rerun_reasons'] = []
    downgraded['advisory_only'] = True
    downgraded['advisory_reasons'] = reasons
    return downgraded



def is_seed_dir(path: Path):
    return path.is_dir() and path.name.startswith('seed_')



def iter_seed_dirs(path: Path):
    if not path.exists() or not path.is_dir():
        return []
    seed_dirs = [child for child in path.iterdir() if is_seed_dir(child)]
    return sorted(seed_dirs, key=lambda p: int(p.name.split('_', 1)[1]))



def iter_param_groups(exp_dir: Path, layout: str):
    if layout == 'standard':
        return [('default', exp_dir)]

    groups = []
    for child in sorted(exp_dir.iterdir()):
        if child.is_dir() and iter_seed_dirs(child):
            groups.append((child.name, child))
    return groups



def extract_last_run_lines(log_path: Path):
    lines = read_lines(log_path)
    if not lines:
        return []

    run_starts = [
        idx
        for idx, line in enumerate(lines)
        if 'Namespace(' in line or 'INFO:root:Using seed' in line
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



def extract_final_summary(log_path: Path, task: str | None = None):
    run_lines = extract_last_run_lines(log_path)
    if not run_lines:
        return None

    content = ''.join(run_lines)
    matches = list(FINAL_SUMMARY_PATTERN.finditer(content))
    if matches:
        return normalize_summary(task, parse_summary_line(matches[-1].group(1)))

    test_summary = None
    val_summary = None
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



def infer_task_from_group(group_path: Path):
    seed_dirs = iter_seed_dirs(group_path)
    if not seed_dirs:
        return None

    config = load_json(seed_dirs[0] / 'config.json')
    if isinstance(config, dict) and config.get('task'):
        return str(config['task']).lower()

    summary = extract_final_summary(seed_dirs[0] / 'log.txt')
    if summary is None:
        return None
    if any(key in summary for key in ['test_roc', 'val_roc', 'test_ap', 'val_ap']):
        return 'lp'
    return 'nc'



def parse_elapsed_to_seconds(raw: str):
    hours, minutes, seconds = raw.split(':')
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)



def extract_total_runtime_seconds(log_path: Path):
    max_elapsed = None
    final_total = None
    for line in extract_last_run_lines(log_path):
        match = ELAPSED_PATTERN.search(line)
        if match:
            current = parse_elapsed_to_seconds(match.group(1))
            max_elapsed = current if max_elapsed is None else max(max_elapsed, current)
        total_match = TOTAL_TIME_PATTERN.search(line)
        if total_match:
            final_total = float(total_match.group(1))
    return final_total if final_total is not None else max_elapsed



def extract_best_epoch(log_path: Path):
    best_epoch = None
    for line in extract_last_run_lines(log_path):
        match = BEST_EPOCH_PATTERN.search(line)
        if match:
            best_epoch = int(match.group(1))
    return best_epoch



def extract_epoch_times(log_path: Path):
    times = []
    for line in extract_last_run_lines(log_path):
        match = TIME_COLON_PATTERN.search(line) or TIME_EQUAL_PATTERN.search(line)
        if match:
            times.append(float(match.group(1)))
    return times



def detect_available_curve_metrics(log_path: Path):
    content = ''.join(extract_last_run_lines(log_path))
    metrics = []
    for marker in [
        'train_loss', 'val_loss', 'val_acc', 'val_f1', 'val_micro_f1', 'val_weighted_f1', 'val_macro_f1', 'val_wf1', 'val_mf1',
        'val_roc', 'val_auc', 'val_ap', 'train_accuracy', 'train_roc', 'train_AUC', 'train_AP',
    ]:
        if marker in content:
            metrics.append(marker)
    return metrics



def detect_has_early_stopping(log_path: Path):
    content = ''.join(extract_last_run_lines(log_path))
    return 'Early stopping' in content or 'early stop' in content.lower()



def read_config_epoch_budget(config: dict | None, task: str):
    if not config:
        return None
    if 'epochs' in config:
        return try_parse_value(str(config.get('epochs'))) if config.get('epochs') is not None else None
    if task.lower() == 'nc' and 'epochs_cls' in config:
        return try_parse_value(str(config.get('epochs_cls')))
    if task.lower() == 'lp' and 'epochs_lp' in config:
        return try_parse_value(str(config.get('epochs_lp')))
    return None



def read_config_patience(config: dict | None, task: str):
    if not config:
        return None
    if 'patience' in config:
        return try_parse_value(str(config.get('patience'))) if config.get('patience') is not None else None
    if task.lower() == 'nc' and 'patience_cls' in config:
        return try_parse_value(str(config.get('patience_cls')))
    if task.lower() == 'lp' and 'patience_lp' in config:
        return try_parse_value(str(config.get('patience_lp')))
    return None



def detect_optuna_info(project: dict, exp_dir: Path):
    status_payload = load_json(exp_dir / 'status.json')
    status_optuna = status_payload.get('optuna') if isinstance(status_payload, dict) else None
    if not isinstance(status_optuna, dict):
        status_optuna = {}

    if project['optuna_kind'] == 'json':
        best_path = exp_dir / 'optuna_best.json'
        payload = load_json(best_path) if best_path.exists() else None
        observed_trials = payload.get('n_trials') if isinstance(payload, dict) else None
    else:
        best_path = exp_dir / 'optuna_best.txt'
        payload = None
        observed_trials = None

    optuna_runs_dir = exp_dir / 'optuna_runs'
    trial_dir_count = 0
    if optuna_runs_dir.exists() and optuna_runs_dir.is_dir():
        trial_dir_count = sum(1 for child in optuna_runs_dir.iterdir() if child.is_dir() and child.name.startswith('trial_'))

    log_path = exp_dir / 'optuna_log.txt'
    log_lines = read_lines(log_path)
    log_text = ''.join(log_lines)
    trial_done_count = sum(1 for line in log_lines if TRIAL_DONE_PATTERN.search(line))
    trial_numbers = set()
    for line in log_lines:
        match = TRIAL_HEADER_PATTERN.search(line)
        if match:
            trial_numbers.add(int(match.group(1)))

    inferred_requested_trials = None
    inferred_requested_num_seeds = None
    trial_cmd_matches = OPTUNA_CMD_TRIALS_PATTERN.findall(log_text)
    if trial_cmd_matches:
        inferred_requested_trials = int(trial_cmd_matches[-1])
    else:
        start_match = OPTUNA_START_TRIALS_PATTERN.search(log_text)
        if start_match:
            inferred_requested_trials = int(start_match.group(1))

    seed_cmd_matches = OPTUNA_CMD_NUM_SEEDS_PATTERN.findall(log_text)
    if seed_cmd_matches:
        inferred_requested_num_seeds = int(seed_cmd_matches[-1])
    else:
        start_seed_match = OPTUNA_START_NUM_SEEDS_PATTERN.search(log_text)
        if start_seed_match:
            inferred_requested_num_seeds = int(start_seed_match.group(1))

    inferred_requested_epochs = None
    inferred_requested_patience = None
    inferred_requested_epochs_nc = None
    inferred_requested_patience_nc = None
    inferred_requested_epochs_lp = None
    inferred_requested_patience_lp = None

    epoch_matches = [int(value) for value in TRAIN_CMD_EPOCHS_PATTERN.findall(log_text)]
    patience_matches = [int(value) for value in TRAIN_CMD_PATIENCE_PATTERN.findall(log_text)]
    epoch_nc_matches = [int(value) for value in TRAIN_CMD_EPOCHS_NC_PATTERN.findall(log_text)]
    patience_nc_matches = [int(value) for value in TRAIN_CMD_PATIENCE_NC_PATTERN.findall(log_text)]
    epoch_lp_matches = [int(value) for value in TRAIN_CMD_EPOCHS_LP_PATTERN.findall(log_text)]
    patience_lp_matches = [int(value) for value in TRAIN_CMD_PATIENCE_LP_PATTERN.findall(log_text)]

    if epoch_matches:
        inferred_requested_epochs = min(epoch_matches)
    if patience_matches:
        inferred_requested_patience = min(patience_matches)
    if epoch_nc_matches:
        inferred_requested_epochs_nc = min(epoch_nc_matches)
    if patience_nc_matches:
        inferred_requested_patience_nc = min(patience_nc_matches)
    if epoch_lp_matches:
        inferred_requested_epochs_lp = min(epoch_lp_matches)
    if patience_lp_matches:
        inferred_requested_patience_lp = min(patience_lp_matches)

    candidates = [observed_trials, trial_dir_count, trial_done_count, len(trial_numbers)]
    observed_trials = max(value for value in candidates if value is not None) if any(value is not None for value in candidates) else None

    return_code = status_optuna.get('return_code')
    oom = bool(status_optuna.get('oom', False))
    skipped = bool(status_optuna.get('skipped', False))
    payload_requested_trials = payload.get('n_trials') if isinstance(payload, dict) else None
    payload_requested_num_seeds = payload.get('num_seeds') if isinstance(payload, dict) else None
    payload_requested_epochs_nc = payload.get('epochs_nc') if isinstance(payload, dict) else None
    payload_requested_patience_nc = payload.get('patience_nc') if isinstance(payload, dict) else None
    payload_requested_epochs_lp = payload.get('epochs_lp') if isinstance(payload, dict) else None
    payload_requested_patience_lp = payload.get('patience_lp') if isinstance(payload, dict) else None

    requested_trials = status_optuna.get('requested_trials', inferred_requested_trials if inferred_requested_trials is not None else payload_requested_trials)
    requested_num_seeds = status_optuna.get('requested_num_seeds', inferred_requested_num_seeds if inferred_requested_num_seeds is not None else payload_requested_num_seeds)
    requested_epochs = status_optuna.get('requested_epochs', inferred_requested_epochs)
    requested_patience = status_optuna.get('requested_patience', inferred_requested_patience)
    requested_epochs_nc = status_optuna.get('requested_epochs_nc', inferred_requested_epochs_nc if inferred_requested_epochs_nc is not None else payload_requested_epochs_nc)
    requested_patience_nc = status_optuna.get('requested_patience_nc', inferred_requested_patience_nc if inferred_requested_patience_nc is not None else payload_requested_patience_nc)
    requested_epochs_lp = status_optuna.get('requested_epochs_lp', inferred_requested_epochs_lp if inferred_requested_epochs_lp is not None else payload_requested_epochs_lp)
    requested_patience_lp = status_optuna.get('requested_patience_lp', inferred_requested_patience_lp if inferred_requested_patience_lp is not None else payload_requested_patience_lp)
    smoke_like_trials = False
    if isinstance(requested_trials, int):
        smoke_like_trials = requested_trials <= 1
    elif observed_trials is not None:
        smoke_like_trials = observed_trials <= 1

    requested_epoch_budgets = [
        requested_epochs,
        requested_epochs_nc,
        requested_epochs_lp,
    ]
    requested_patience_budgets = [
        requested_patience,
        requested_patience_nc,
        requested_patience_lp,
    ]
    low_epoch_budget = any(isinstance(value, int) and value < TARGET_EPOCHS for value in requested_epoch_budgets)
    low_patience_budget = any(isinstance(value, int) and value < TARGET_PATIENCE for value in requested_patience_budgets)

    reasons = []
    if not best_path.exists():
        reasons.append('missing_optuna_best')
    if return_code not in (None, 0):
        reasons.append('optuna_nonzero_return_code')
    if oom:
        reasons.append('optuna_oom')
    if smoke_like_trials:
        reasons.append('optuna_smoke_trials')
    if low_epoch_budget:
        reasons.append('optuna_epoch_budget_below_target')
    if low_patience_budget:
        reasons.append('optuna_patience_below_target')

    needs_rerun = bool(reasons)

    return {
        'best_exists': best_path.exists(),
        'best_path': display_path(best_path),
        'log_exists': log_path.exists(),
        'log_path': display_path(log_path),
        'trial_dir_count': trial_dir_count,
        'observed_trials': observed_trials,
        'meets_target_trials': observed_trials is not None and observed_trials >= 30,
        'requested_trials': requested_trials,
        'requested_num_seeds': requested_num_seeds,
        'requested_epochs': requested_epochs,
        'requested_patience': requested_patience,
        'requested_epochs_nc': requested_epochs_nc,
        'requested_patience_nc': requested_patience_nc,
        'requested_epochs_lp': requested_epochs_lp,
        'requested_patience_lp': requested_patience_lp,
        'return_code': return_code,
        'oom': oom,
        'skipped': skipped,
        'smoke_like_trials': smoke_like_trials,
        'low_epoch_budget': low_epoch_budget,
        'low_patience_budget': low_patience_budget,
        'needs_rerun': needs_rerun,
        'rerun_reasons': reasons,
        'payload': payload,
    }



def evaluate_seed_dir(project: dict, seed_dir: Path, seed: int, task: str):
    result = {
        'seed': seed,
        'path': display_path(seed_dir),
        'exists': seed_dir.exists(),
        'status': 'missing',
        'issues': [],
        'metrics': None,
        'curve_metrics': [],
        'runtime_seconds': None,
        'avg_epoch_time_seconds': None,
        'best_epoch': None,
        'config_epochs': None,
        'config_patience': None,
        'meets_epoch_budget': None,
        'return_code': None,
        'has_early_stopping': False,
        'checkpoint_exists': False,
        'checkpoint_completed': None,
        'checkpoint_epoch': None,
    }
    if not seed_dir.exists():
        result['issues'].append('missing_seed_dir')
        return result

    checkpoint_path = seed_dir / 'checkpoint_last.pt'
    checkpoint_payload = None
    if checkpoint_path.exists():
        result['checkpoint_exists'] = True
        checkpoint_payload = load_torch_payload(checkpoint_path)
        if checkpoint_payload is None:
            result['issues'].append('checkpoint_unreadable')
    config_path = seed_dir / 'config.json'
    log_path = seed_dir / 'log.txt'
    runner_log_path = seed_dir / 'runner_log.txt'
    model_path = seed_dir / 'model.pth'

    config = load_json(config_path) if config_path.exists() else None
    if not config_path.exists():
        result['issues'].append('missing_config')
    if isinstance(checkpoint_payload, dict):
        result['checkpoint_completed'] = checkpoint_payload.get('completed')
        checkpoint_epoch = checkpoint_payload.get('epoch')
        if checkpoint_epoch is not None:
            try:
                result['checkpoint_epoch'] = int(checkpoint_epoch)
            except Exception:
                result['checkpoint_epoch'] = checkpoint_epoch
    if not log_path.exists():
        result['issues'].append('missing_log')
    if project['requires_model_file'] and not model_path.exists():
        result['issues'].append('missing_model')

    if runner_log_path.exists():
        for line in reversed(read_lines(runner_log_path)):
            if line.startswith('RETURN_CODE:'):
                try:
                    result['return_code'] = int(line.split(':', 1)[1].strip())
                except Exception:
                    result['return_code'] = None
                break

    result['config_epochs'] = read_config_epoch_budget(config, task)
    result['config_patience'] = read_config_patience(config, task)

    if log_path.exists():
        summary = extract_final_summary(log_path, task)
        result['metrics'] = summary
        result['curve_metrics'] = detect_available_curve_metrics(log_path)
        result['runtime_seconds'] = extract_total_runtime_seconds(log_path)
        epoch_times = extract_epoch_times(log_path)
        if epoch_times:
            result['avg_epoch_time_seconds'] = float(sum(epoch_times) / len(epoch_times))
        result['best_epoch'] = extract_best_epoch(log_path)
        result['has_early_stopping'] = detect_has_early_stopping(log_path)
        if summary is None:
            result['issues'].append('missing_final_summary')

    if result['checkpoint_completed'] is True and result['metrics'] is None:
        result['issues'].append('checkpoint_marked_complete_without_final_summary')

    if result['config_epochs'] is not None:
        result['meets_epoch_budget'] = result['config_epochs'] >= TARGET_EPOCHS
        if not result['meets_epoch_budget']:
            result['issues'].append('epoch_budget_below_target')

    if result['return_code'] not in (None, 0):
        result['issues'].append('nonzero_return_code')

    if result['issues']:
        critical = ['missing_seed_dir', 'missing_config', 'missing_log', 'missing_model', 'missing_final_summary', 'nonzero_return_code']
        if any(issue in result['issues'] for issue in critical):
            result['status'] = 'failed'
        else:
            result['status'] = 'needs_rerun'
    else:
        result['status'] = 'complete'

    return result



def evaluate_group(project: dict, group_path: Path, task: str):
    return [evaluate_seed_dir(project, group_path / f'seed_{seed}', seed, task) for seed in EXPECTED_SEEDS]



def choose_best_legacy_group(group_summaries):
    if not group_summaries:
        return None

    def sort_key(item):
        completed = len(item['completed_seeds'])
        failed = len(item['failed_seeds'])
        rerun = len(item['rerun_seeds'])
        preferred = 1 if item['param_group'] == 'best' else 0
        return (preferred, completed, -failed, -rerun)

    return max(group_summaries, key=sort_key)



def summarize_group_payload(project: dict, exp_dir: Path, group_name: str, group_path: Path, task: str):
    seed_results = evaluate_group(project, group_path, task)
    completed_seeds = [item['seed'] for item in seed_results if item['status'] == 'complete']
    failed_seeds = [item['seed'] for item in seed_results if item['status'] == 'failed']
    rerun_seeds = [item['seed'] for item in seed_results if item['status'] != 'complete']
    missing_seeds = [item['seed'] for item in seed_results if not item['exists']]
    all_curve_metrics = sorted({metric for item in seed_results for metric in item['curve_metrics']})
    all_summary_metric_keys = sorted({key for item in seed_results if item['metrics'] for key in item['metrics'].keys()})

    return {
        'param_group': group_name,
        'group_dir': display_path(group_path),
        'task': task,
        'completed_seeds': completed_seeds,
        'failed_seeds': failed_seeds,
        'rerun_seeds': rerun_seeds,
        'missing_seeds': missing_seeds,
        'available_curve_metrics': all_curve_metrics,
        'available_summary_metrics': all_summary_metric_keys,
        'seeds': seed_results,
    }



def summarize_experiment(project: dict, exp_dir: Path):
    parsed = parse_task_model_dataset(project['name'], exp_dir.name)
    if parsed is None:
        return None

    baseline = parsed['baseline']
    raw_dataset = parsed['dataset']
    dataset = canonicalize_dataset_name(project['name'], raw_dataset)
    layout = parsed['layout']
    status_payload = load_json(exp_dir / 'status.json')
    optuna_info = detect_optuna_info(project, exp_dir)

    if layout == 'standard':
        task = parsed['task']
        task_lower = task.lower()
        group_payload = summarize_group_payload(project, exp_dir, 'default', exp_dir, task_lower)
    else:
        group_payloads = []
        for group_name, group_path in iter_param_groups(exp_dir, layout):
            inferred_task = infer_task_from_group(group_path)
            if inferred_task is None:
                continue
            group_payloads.append(summarize_group_payload(project, exp_dir, group_name, group_path, inferred_task))
        group_payload = choose_best_legacy_group(group_payloads)
        if group_payload is None:
            return None
        task = group_payload['task']

    optuna_info = downgrade_optuna_rerun_for_complete_results(optuna_info, group_payload)
    overall_status = 'needs_rerun' if group_payload['rerun_seeds'] else 'complete'
    if optuna_info.get('needs_rerun'):
        overall_status = 'needs_rerun'

    experiment = {
        'project': project['name'],
        'baseline': baseline,
        'task': task.upper() if project['name'] == 'GraphMoRE-main' else task,
        'dataset': dataset,
        'raw_dataset': raw_dataset,
        'layout': layout,
        'selected_group': group_payload['param_group'],
        'selected_group_dir': group_payload['group_dir'],
        'result_dir': display_path(exp_dir),
        'result_dir_exists': True,
        'overall_status': overall_status,
        'optuna': optuna_info,
        'status_json_exists': (exp_dir / 'status.json').exists(),
        'status_json': status_payload,
        'expected_seeds': EXPECTED_SEEDS,
        'completed_seeds': group_payload['completed_seeds'],
        'failed_seeds': group_payload['failed_seeds'],
        'missing_seeds': group_payload['missing_seeds'],
        'rerun_seeds': group_payload['rerun_seeds'],
        'available_curve_metrics': group_payload['available_curve_metrics'],
        'available_summary_metrics': group_payload['available_summary_metrics'],
        'seeds': group_payload['seeds'],
    }
    return experiment


def build_missing_experiment(project: dict, baseline: str, task: str, dataset: str):
    dataset = canonicalize_dataset_name(project['name'], dataset)
    task_dir = task if project['name'] == 'GraphMoRE-main' else task.lower()
    exp_name = f'{task_dir}_{baseline}_{dataset}'
    exp_dir = project['result_dir'] / exp_name
    seeds = [
        {
            'seed': seed,
            'path': display_path(exp_dir / f'seed_{seed}'),
            'exists': False,
            'status': 'missing',
            'issues': ['missing_seed_dir'],
            'metrics': None,
            'curve_metrics': [],
            'runtime_seconds': None,
            'avg_epoch_time_seconds': None,
            'best_epoch': None,
            'config_epochs': None,
            'config_patience': None,
            'meets_epoch_budget': None,
            'return_code': None,
            'has_early_stopping': False,
            'checkpoint_exists': False,
            'checkpoint_completed': None,
            'checkpoint_epoch': None,
        }
        for seed in EXPECTED_SEEDS
    ]
    return {
        'project': project['name'],
        'baseline': baseline,
        'task': task,
        'dataset': dataset,
        'raw_dataset': dataset,
        'layout': 'standard',
        'selected_group': 'default',
        'selected_group_dir': display_path(exp_dir),
        'result_dir': display_path(exp_dir),
        'result_dir_exists': False,
        'overall_status': 'needs_rerun',
        'optuna': {
            'best_exists': False,
            'best_path': display_path(exp_dir / ('optuna_best.json' if project['optuna_kind'] == 'json' else 'optuna_best.txt')),
            'log_exists': False,
            'log_path': display_path(exp_dir / 'optuna_log.txt'),
            'trial_dir_count': 0,
            'observed_trials': 0,
            'meets_target_trials': False,
            'payload': None,
        },
        'status_json_exists': False,
        'status_json': None,
        'expected_seeds': EXPECTED_SEEDS,
        'completed_seeds': [],
        'failed_seeds': [],
        'missing_seeds': list(EXPECTED_SEEDS),
        'rerun_seeds': list(EXPECTED_SEEDS),
        'available_curve_metrics': [],
        'available_summary_metrics': [],
        'seeds': seeds,
    }



def build_report(summary):
    lines = []
    lines.append('# Experiment Audit Report')
    lines.append('')
    lines.append(f'- Generated at: {summary["generated_at"]}')
    lines.append(f'- Target seeds: {summary["expected_seeds"]}')
    lines.append(f'- Target epochs: {summary["target_epochs"]}')
    lines.append(f'- Target patience: {summary["target_patience"]}')
    lines.append('')

    for project_name in summary['projects']:
        lines.append(f'## {project_name}')
        lines.append('')
        project_experiments = [item for item in summary['experiments'] if item['project'] == project_name]
        if not project_experiments:
            lines.append('- No experiments found.')
            lines.append('')
            continue

        for exp in project_experiments:
            lines.append(f'### {exp["task"]}_{exp["baseline"]}_{exp["dataset"]}')
            lines.append(f'- Status: {exp["overall_status"]}')
            if exp.get('raw_dataset') and exp['raw_dataset'] != exp['dataset']:
                lines.append(f'- Raw dataset label: {exp["raw_dataset"]}')
            lines.append(f'- Layout: {exp["layout"]}')
            lines.append(f'- Selected group: {exp["selected_group"]}')
            lines.append(f'- Completed seeds: {exp["completed_seeds"]}')
            lines.append(f'- Rerun seeds: {exp["rerun_seeds"]}')
            lines.append(f'- Missing seeds: {exp["missing_seeds"]}')
            lines.append(f'- Failed seeds: {exp["failed_seeds"]}')
            lines.append(
                f'- Optuna: exists={exp["optuna"]["best_exists"]}, observed_trials={exp["optuna"]["observed_trials"]}, meets_30={exp["optuna"]["meets_target_trials"]}'
            )
            if exp['optuna'].get('requested_trials') is not None:
                lines.append(
                    f'- Optuna requested trials: {exp["optuna"]["requested_trials"]}, num_seeds={exp["optuna"].get("requested_num_seeds")}'
                )
            optuna_budgets = []
            if exp['optuna'].get('requested_epochs') is not None:
                optuna_budgets.append(f'epochs={exp["optuna"]["requested_epochs"]}')
            if exp['optuna'].get('requested_patience') is not None:
                optuna_budgets.append(f'patience={exp["optuna"]["requested_patience"]}')
            if exp['optuna'].get('requested_epochs_nc') is not None:
                optuna_budgets.append(f'epochs_nc={exp["optuna"]["requested_epochs_nc"]}')
            if exp['optuna'].get('requested_patience_nc') is not None:
                optuna_budgets.append(f'patience_nc={exp["optuna"]["requested_patience_nc"]}')
            if exp['optuna'].get('requested_epochs_lp') is not None:
                optuna_budgets.append(f'epochs_lp={exp["optuna"]["requested_epochs_lp"]}')
            if exp['optuna'].get('requested_patience_lp') is not None:
                optuna_budgets.append(f'patience_lp={exp["optuna"]["requested_patience_lp"]}')
            if optuna_budgets:
                lines.append(f'- Optuna requested budgets: {", ".join(optuna_budgets)}')
            if exp['optuna'].get('needs_rerun'):
                lines.append(f'- Optuna rerun reasons: {exp["optuna"]["rerun_reasons"]}')
            if exp['optuna'].get('advisory_only'):
                lines.append(f'- Optuna advisory only: {exp["optuna"].get("advisory_reasons", [])}')
            if exp['available_summary_metrics']:
                lines.append(f'- Summary metrics: {exp["available_summary_metrics"]}')
            if exp['available_curve_metrics']:
                lines.append(f'- Curve metrics: {exp["available_curve_metrics"]}')
            lines.append('')
            for seed in exp['seeds']:
                issues = ', '.join(seed['issues']) if seed['issues'] else 'none'
                lines.append(
                    f'  - seed_{seed["seed"]}: status={seed["status"]}, epochs={seed["config_epochs"]}, '
                    f'best_epoch={seed["best_epoch"]}, checkpoint_exists={seed["checkpoint_exists"]}, '
                    f'checkpoint_completed={seed["checkpoint_completed"]}, checkpoint_epoch={seed["checkpoint_epoch"]}, '
                    f'return_code={seed["return_code"]}, issues={issues}'
                )
            lines.append('')

    return '\n'.join(lines) + '\n'



def audit_all(output_json: Path, output_md: Path):
    experiments_by_key = {}

    for project in PROJECTS:
        result_dir = project['result_dir']
        if result_dir.exists():
            for child in sorted(result_dir.iterdir()):
                if not child.is_dir():
                    continue
                exp = summarize_experiment(project, child)
                if exp is not None:
                    key = canonical_experiment_key(exp['project'], exp['baseline'], exp['task'], exp['dataset'])
                    existing = experiments_by_key.get(key)
                    experiments_by_key[key] = exp if existing is None else choose_preferred_experiment(existing, exp)

        for baseline, task, dataset in EXPECTED_EXPERIMENTS.get(project['name'], []):
            key = canonical_experiment_key(project['name'], baseline, task, dataset)
            if key in experiments_by_key:
                continue
            experiments_by_key[key] = build_missing_experiment(project, baseline, task, dataset)

    experiments = list(experiments_by_key.values())

    experiments.sort(key=lambda exp: (exp['project'], str(exp['task']), exp['baseline'], exp['dataset']))

    summary = {
        'generated_at': datetime.now().isoformat(),
        'root': str(ROOT),
        'projects': [project['name'] for project in PROJECTS],
        'expected_seeds': EXPECTED_SEEDS,
        'target_epochs': TARGET_EPOCHS,
        'target_patience': TARGET_PATIENCE,
        'experiments': experiments,
        'rerun_targets': [
            {
                'project': exp['project'],
                'task': exp['task'],
                'baseline': exp['baseline'],
                'dataset': exp['dataset'],
                'recovery_action': determine_recovery_action(exp),
                'selected_group': exp['selected_group'],
                'result_dir_exists': exp.get('result_dir_exists', True),
                'optuna_best_exists': exp['optuna']['best_exists'],
                'optuna_needs_rerun': exp['optuna'].get('needs_rerun', False),
                'optuna_rerun_reasons': exp['optuna'].get('rerun_reasons', []),
                'rerun_seeds': exp['rerun_seeds'],
                'checkpoint_seed_states': [
                    {
                        'seed': seed['seed'],
                        'checkpoint_exists': seed['checkpoint_exists'],
                        'checkpoint_completed': seed['checkpoint_completed'],
                        'checkpoint_epoch': seed['checkpoint_epoch'],
                    }
                    for seed in exp['seeds']
                ],
            }
            for exp in experiments
            if exp['rerun_seeds'] or exp['optuna'].get('needs_rerun', False)
        ],
    }
    summary['rerun_target_counts'] = {
        'rerun_optuna_and_train': sum(1 for item in summary['rerun_targets'] if item['recovery_action'] == 'rerun_optuna_and_train'),
        'rerun_seeds_only': sum(1 for item in summary['rerun_targets'] if item['recovery_action'] == 'rerun_seeds_only'),
    }

    with output_json.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    output_md.write_text(build_report(summary), encoding='utf-8')
    return summary



def main():
    parser = argparse.ArgumentParser(description='Audit existing experiment results and identify rerun targets.')
    parser.add_argument('--output-json', default='audit_summary.json')
    parser.add_argument('--output-md', default='audit_report.md')
    args = parser.parse_args()

    output_json = ROOT / args.output_json
    output_md = ROOT / args.output_md
    summary = audit_all(output_json, output_md)
    print(f'Saved audit JSON to: {output_json}')
    print(f'Saved audit report to: {output_md}')
    print(f'Experiments scanned: {len(summary["experiments"])}')
    print(f'Experiments needing rerun: {len(summary["rerun_targets"])}')


if __name__ == '__main__':
    main()
