import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_ROOT.parent
MAIN_PY = PROJECT_ROOT / 'main.py'
OPTUNA_PY = PROJECT_ROOT / 'optuna_train.py'
AUDIT_SCRIPT = ROOT_DIR / 'audit_experiments.py'
PROJECT_NAME = 'GraphMoRE-main'
DEFAULT_ROOT_PATH = PROJECT_ROOT / 'datasets'

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gpu_profile import build_gpu_profile

SAFE_LARGE_GRAPH = True
GPU_ID = 0
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 3047
ROOT_PATH = DEFAULT_ROOT_PATH
SAVE_ROOT = PROJECT_ROOT / 'Result'
OPTUNA_TRIALS = 30
OPTUNA_NUM_SEEDS = 1
EXP_ITERS = 1

TRAIN_PROGRESS_PATTERNS = [
    re.compile(r'Epoch\s+1:', re.IGNORECASE),
    re.compile(r'Epoch\s+0001', re.IGNORECASE),
]

OOM_PATTERNS = [
    r"cuda out of memory",
    r"out of memory",
    r"cublas.*status.*alloc",
    r"cudnn.*status.*alloc",
    r"unable to allocate",
    r"arraymemoryerror",
]

TASK_DATASETS = {
    'NC': ['cora', 'citeseer', 'pubmed', 'airport', 'telecom', 'Actor', 'cornell', 'cs_phds', 'disease_nc', 'Carcinogenesis_data', 'Hepatitis_std_data', 'Hockey_data', 'PTE', 'Toxicology_data'],
    'LP': ['telecom', 'Actor', 'cornell', 'cs_phds', 'disease_lp'],
}
DEFAULT_NC_DATASETS = list(TASK_DATASETS['NC'])
DEFAULT_LP_DATASETS = list(TASK_DATASETS['LP'])


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_env():
    env = os.environ.copy()
    env.setdefault('CUDA_LAUNCH_BLOCKING', '0')
    env.setdefault('TORCH_SHOW_CPP_STACKTRACES', '1')
    env.setdefault('TORCH_DISABLE_ADDR2LINE', '1')
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('PYTHONPATH', str(PROJECT_ROOT))
    return env


def write_json(path, payload):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def run_command(cmd, log_path, smoke_stop_after_first_epoch=False):
    ensure_dir(os.path.dirname(log_path))
    smoke_reached_epoch = False
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('CMD: ' + ' '.join(str(x) for x in cmd) + '\n')
        if smoke_stop_after_first_epoch:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=build_env())
            assert process.stdout is not None
            for line in process.stdout:
                f.write(line)
                f.flush()
                if any(pattern.search(line) for pattern in TRAIN_PROGRESS_PATTERNS):
                    smoke_reached_epoch = True
                    process.terminate()
                    break
            process.wait()
            result_code = 0 if smoke_reached_epoch else int(process.returncode or 0)
            if smoke_reached_epoch:
                f.write('\n[SmokeTest] Reached first epoch; stopped early and marked as success.\n')
        else:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, cwd=PROJECT_ROOT, env=build_env())
            result_code = result.returncode
        f.write('\nRETURN_CODE: ' + str(result_code) + '\n')
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    oom_hit = any(re.search(pattern, content, re.IGNORECASE) for pattern in OOM_PATTERNS)
    return result_code, oom_hit, smoke_reached_epoch


def load_best_params(best_path):
    if not os.path.exists(best_path):
        return None
    with open(best_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload.get('best_params')


def sanitize_train_params(dataset, task, best_params):
    params = dict(best_params or {})
    if not params:
        return params

    if dataset.lower() == 'telecom':
        params['backbone'] = 'gcn'
        params['sample_hop'] = [1, 2]
        params['w_decay_gating'] = min(float(params.get('w_decay_gating', 1e-5)), 1e-5)
        params['lr_gating'] = min(float(params.get('lr_gating', 1e-3)), 1e-3)
        if task == 'LP':
            params['coef_dis'] = min(float(params.get('coef_dis', 1e-3)), 1e-3)
            params['t'] = max(float(params.get('t', 1.0)), 1.0)
            params['r'] = min(float(params.get('r', 2.0)), 2.0)
    elif dataset.lower() == 'f1_ultimate_hetero_graph':
        # f1 is extremely large. We intentionally keep seed training in a
        # conservative configuration to avoid catastrophic subgraph `torch.cat`
        # OOM spikes even on 80GB GPUs.
        params.setdefault('backbone', 'gcn')
        params.setdefault('hidden_features', 16)
        params.setdefault('embed_features', 16)
        params.setdefault('n_layers', 1)
        params['sample_hop'] = [1]
        params.setdefault('lr_gating', 1e-3)
        params.setdefault('w_decay_gating', 1e-5)
        if task == 'NC':
            params.setdefault('hidden_features_cls', 16)
            params.setdefault('lr_cls', 1e-3)
            params.setdefault('w_decay_cls', 1e-4)
            params.setdefault('drop_edge_cls', 0.0)
    return params


def get_seeds_for_dataset(dataset):
    return SEEDS


def build_train_cmd(
    dataset,
    task,
    seed,
    save_dir,
    best_params,
    gpu_id,
    root_path,
    split_seed,
    exp_iters,
    epochs_nc,
    patience_nc,
    min_epoch_nc,
    epochs_lp,
    patience_lp,
    min_epoch_lp,
    resume=False,
):
    cmd = [
        sys.executable,
        str(MAIN_PY),
        '--dataset', dataset,
        '--downstream_task', task,
        '--root_path', str(root_path),
        '--gpu', str(gpu_id),
        '--seed', str(seed),
        '--split_seed', str(split_seed),
        '--exp_iters', str(exp_iters),
        '--save_dir', str(save_dir),
        '--resume', str(int(resume)),
    ]

    if task == 'NC':
        cmd.extend([
            '--epochs_cls', str(epochs_nc),
            '--patience_cls', str(patience_nc),
            '--min_epoch_cls', str(min_epoch_nc),
            '--epochs_lp', str(epochs_lp),
            '--patience_lp', str(patience_lp),
            '--min_epoch_lp', str(min_epoch_lp),
        ])
    else:
        cmd.extend([
            '--epochs_lp', str(epochs_lp),
            '--patience_lp', str(patience_lp),
            '--min_epoch_lp', str(min_epoch_lp),
        ])

    for key, value in (best_params or {}).items():
        flag = '--' + key
        if isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def run_optuna_search(
    dataset,
    task,
    exp_root,
    gpu_id,
    split_seed,
    root_path,
    exp_iters,
    safe_large_graph,
    optuna_trials,
    optuna_num_seeds,
    optuna_base_seed,
    epochs_nc,
    patience_nc,
    min_epoch_nc,
    epochs_lp,
    patience_lp,
    min_epoch_lp,
):
    optuna_log = os.path.join(exp_root, 'optuna_log.txt')
    best_path = os.path.join(exp_root, 'optuna_best.json')
    n_trials = optuna_trials if optuna_trials > 0 else OPTUNA_TRIALS
    cmd = [
        sys.executable,
        str(OPTUNA_PY),
        '--dataset', dataset,
        '--task', task,
        '--gpu', str(gpu_id),
        '--n_trials', str(n_trials),
        '--num_seeds', str(optuna_num_seeds),
        '--base_seed', str(optuna_base_seed),
        '--split_seed', str(split_seed),
        '--root_path', str(root_path),
        '--exp_iters', str(exp_iters),
        '--epochs_nc', str(epochs_nc),
        '--patience_nc', str(patience_nc),
        '--min_epoch_nc', str(min_epoch_nc),
        '--epochs_lp', str(epochs_lp),
        '--patience_lp', str(patience_lp),
        '--min_epoch_lp', str(min_epoch_lp),
        '--work_dir', os.path.join(exp_root, 'optuna_runs'),
        '--best_path', best_path,
    ]
    if safe_large_graph:
        cmd.append('--safe_large_graph')
    return run_command(cmd, optuna_log), best_path, optuna_log


def load_audit_module():
    if not AUDIT_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location('audit_experiments', AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(args):
    module = load_audit_module()
    if module is None:
        print(f'Audit script not found: {AUDIT_SCRIPT}')
        return 1
    output_json = ROOT_DIR / args.audit_json
    output_md = ROOT_DIR / args.audit_md
    summary = module.audit_all(output_json, output_md)
    print(f'Saved audit JSON to: {output_json}')
    print(f'Saved audit report to: {output_md}')
    print(f'Experiments scanned: {len(summary["experiments"])}')
    print(f'Experiments needing rerun: {len(summary["rerun_targets"])}')
    return 0


def get_resume_experiment(module, exp_root):
    if module is None:
        return None
    project = next((item for item in module.PROJECTS if item['name'] == PROJECT_NAME), None)
    if project is None:
        return None
    return module.summarize_experiment(project, Path(exp_root))


def resolve_seed_plan(args, dataset, exp_root):
    requested = get_seeds_for_dataset(dataset)
    if args.stage != 'resume':
        return requested, None, {}

    audit_module = load_audit_module()
    exp = get_resume_experiment(audit_module, exp_root) if os.path.isdir(exp_root) else None
    if exp is None:
        return requested, None, {seed: {'seed': seed, 'resume_mode': 'fresh_no_audit'} for seed in requested}

    rerun_set = set(exp.get('rerun_seeds', []))
    seed_records = {item.get('seed'): item for item in exp.get('seeds', [])}
    seeds_to_run = [seed for seed in requested if seed in rerun_set]
    seed_plan = {}
    for seed in requested:
        record = dict(seed_records.get(seed) or {'seed': seed})
        if seed not in rerun_set:
            record['resume_mode'] = 'skip_complete'
        elif record.get('checkpoint_exists') and record.get('checkpoint_completed') is not True:
            record['resume_mode'] = 'resume_from_checkpoint'
        else:
            record['resume_mode'] = 'rerun_without_checkpoint'
        seed_plan[seed] = record
    return seeds_to_run, exp, seed_plan


def should_run_optuna(stage, best_path, exp_audit):
    if stage in {'stage2', 'resume'}:
        return False
    return stage == 'all'


def run_experiment(dataset, task, args):
    exp_name = f'{task}_GraphMoRE_{dataset}'
    exp_root = os.path.join(args.save_root, exp_name)
    ensure_dir(exp_root)

    seeds_to_run, exp_audit, seed_plan = resolve_seed_plan(args, dataset, exp_root)

    status = {
        'dataset': dataset,
        'task': task,
        'stage': args.stage,
        'requested_seeds': get_seeds_for_dataset(dataset),
        'planned_seeds': seeds_to_run,
        'skipped_existing_seeds': [seed for seed in get_seeds_for_dataset(dataset) if seed not in seeds_to_run],
        'seed_plan': seed_plan,
        'oom_skip': False,
        'optuna': {},
        'seeds': [],
        'resume_audit': exp_audit,
    }

    best_path = os.path.join(exp_root, 'optuna_best.json')
    if should_run_optuna(args.stage, best_path, exp_audit):
        optuna_status, best_path, optuna_log = run_optuna_search(
            dataset,
            task,
            exp_root,
            args.gpu_id,
            args.split_seed,
            args.root_path,
            args.exp_iters,
            args.safe_large_graph,
            args.optuna_trials,
            args.optuna_num_seeds,
            args.optuna_base_seed,
            args.epochs_nc,
            args.patience_nc,
            args.min_epoch_nc,
            args.epochs_lp,
            args.patience_lp,
            args.min_epoch_lp,
        )
        optuna_code, optuna_oom, _ = optuna_status
        status['optuna'] = {
            'return_code': optuna_code,
            'oom': optuna_oom,
            'best_path': best_path,
            'log': optuna_log,
            'requested_trials': int(args.optuna_trials),
            'requested_num_seeds': int(args.optuna_num_seeds),
            'requested_epochs_nc': int(args.epochs_nc),
            'requested_patience_nc': int(args.patience_nc),
            'requested_epochs_lp': int(args.epochs_lp),
            'requested_patience_lp': int(args.patience_lp),
        }
        if optuna_oom:
            status['oom_skip'] = True
        if optuna_code != 0:
            write_json(os.path.join(exp_root, 'status.json'), status)
            return status
    else:
        status['optuna'] = {
            'skipped': True,
            'best_path': best_path,
            'reason': 'existing_best' if os.path.exists(best_path) else 'stage_without_optuna',
            'requested_trials': int(args.optuna_trials),
            'requested_num_seeds': int(args.optuna_num_seeds),
            'requested_epochs_nc': int(args.epochs_nc),
            'requested_patience_nc': int(args.patience_nc),
            'requested_epochs_lp': int(args.epochs_lp),
            'requested_patience_lp': int(args.patience_lp),
        }

    best_params = sanitize_train_params(dataset, task, load_best_params(best_path))
    if args.stage == 'resume' and not os.path.exists(best_path):
        status['skipped_no_best_params'] = True
        status['skip_reason'] = 'resume_requires_existing_optuna_best'
        write_json(os.path.join(exp_root, 'status.json'), status)
        return status

    for seed in seeds_to_run:
        seed_dir = os.path.join(exp_root, f'seed_{seed}')
        ensure_dir(seed_dir)
        runner_log = os.path.join(seed_dir, 'runner_log.txt')
        seed_info = dict(seed_plan.get(seed) or {'seed': seed, 'resume_mode': 'fresh'})
        resume_flag = args.stage == 'resume' and seed_info.get('resume_mode') == 'resume_from_checkpoint'
        cmd = build_train_cmd(
            dataset,
            task,
            seed,
            seed_dir,
            best_params,
            args.gpu_id,
            args.root_path,
            args.split_seed,
            args.exp_iters,
            args.epochs_nc,
            args.patience_nc,
            args.min_epoch_nc,
            args.epochs_lp,
            args.patience_lp,
            args.min_epoch_lp,
            resume=resume_flag,
        )
        return_code, oom_hit, smoke_reached_epoch = run_command(cmd, runner_log, smoke_stop_after_first_epoch=bool(args.smoke_stop_after_first_epoch))
        status['seeds'].append({
            'seed': seed,
            'return_code': return_code,
            'oom': oom_hit,
            'runner_log': runner_log,
            'resume_mode': seed_info.get('resume_mode'),
            'checkpoint_exists': seed_info.get('checkpoint_exists'),
            'checkpoint_completed': seed_info.get('checkpoint_completed'),
            'checkpoint_epoch': seed_info.get('checkpoint_epoch'),
            'resume_flag': int(resume_flag),
            'smoke_reached_epoch': smoke_reached_epoch,
        })
        if oom_hit:
            status['oom_skip'] = True
            break
        if return_code != 0:
            break

    write_json(os.path.join(exp_root, 'status.json'), status)
    return status


def main():
    global ROOT_PATH, SAVE_ROOT, SAFE_LARGE_GRAPH

    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all', choices=['all', 'stage2', 'resume', 'audit'])
    parser.add_argument('--root_path', default=str(DEFAULT_ROOT_PATH))
    parser.add_argument('--save_root', default=str(PROJECT_ROOT / 'Result'))
    parser.add_argument('--safe_large_graph', dest='safe_large_graph', action='store_true')
    parser.add_argument('--no_safe_large_graph', dest='safe_large_graph', action='store_false')
    parser.add_argument('--gpu-id', type=int, default=GPU_ID)
    parser.add_argument('--split-seed', type=int, default=SPLIT_SEED)
    parser.add_argument('--exp-iters', type=int, default=EXP_ITERS)
    parser.add_argument('--optuna-trials', type=int, default=OPTUNA_TRIALS)
    parser.add_argument('--optuna-num-seeds', type=int, default=OPTUNA_NUM_SEEDS)
    parser.add_argument('--optuna-base-seed', type=int, default=0)
    parser.add_argument('--epochs-nc', type=int, default=500)
    parser.add_argument('--patience-nc', type=int, default=100)
    parser.add_argument('--min-epoch-nc', type=int, default=100)
    parser.add_argument('--epochs-lp', type=int, default=500)
    parser.add_argument('--patience-lp', type=int, default=100)
    parser.add_argument('--min-epoch-lp', type=int, default=100)
    parser.add_argument('--smoke-stop-after-first-epoch', action='store_true')
    parser.add_argument('--nc-datasets', nargs='*', default=DEFAULT_NC_DATASETS)
    parser.add_argument('--lp-datasets', nargs='*', default=DEFAULT_LP_DATASETS)
    parser.add_argument('--audit-json', default='audit_summary.json')
    parser.add_argument('--audit-md', default='audit_report.md')
    parser.set_defaults(safe_large_graph=SAFE_LARGE_GRAPH)
    args = parser.parse_args()

    # Use absolute paths so subprocesses with a different cwd (GraphMoRE-main)
    # still write artifacts into the intended baseline `serial_runs/...` tree.
    args.root_path = Path(args.root_path).resolve()
    args.save_root = Path(args.save_root).resolve()
    args.gpu_profile = build_gpu_profile(args.gpu_id)
    ROOT_PATH = args.root_path
    SAVE_ROOT = args.save_root
    SAFE_LARGE_GRAPH = args.safe_large_graph

    if args.stage == 'audit':
        raise SystemExit(run_audit(args))

    ensure_dir(args.save_root)

    summary = {
        'gpu_id': args.gpu_id,
        'seeds': SEEDS,
        'split_seed': args.split_seed,
        'stage': args.stage,
        'gpu_profile': args.gpu_profile,
        'safe_large_graph': SAFE_LARGE_GRAPH,
        'results': [],
    }

    for dataset in args.nc_datasets:
        summary['results'].append(run_experiment(dataset, 'NC', args))

    for dataset in args.lp_datasets:
        summary['results'].append(run_experiment(dataset, 'LP', args))

    write_json(os.path.join(args.save_root, 'summary.json'), summary)


if __name__ == '__main__':
    main()
