"""Auto-run experiments: Optuna search then multi-seed training."""

import argparse
import ast
import datetime
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_ROOT.parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
OPTUNA_SCRIPT = PROJECT_ROOT / "optuna_train.py"
AUDIT_SCRIPT = ROOT_DIR / "audit_experiments.py"
PROJECT_NAME = "hgcn"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gpu_profile import build_gpu_profile

try:
    from reevaluate_nc_metrics import prepare_data_and_model, evaluate_split, aggregate_metric_dicts
except Exception:
    prepare_data_and_model = None
    evaluate_split = None
    aggregate_metric_dicts = None

GPU_ID = 0
SEEDS = [0, 1, 2, 3, 4]
SPLIT_SEED = 1234
SAVE_ROOT = PROJECT_ROOT / "Result"
MODELS = ["HNN", "HGCN"]

OPTUNA_TRIALS = 30
OPTUNA_NUM_SEEDS = 1
OPTUNA_BASE_SEED = 0
MANIFOLD = "PoincareBall"
OPTIMIZER = "RiemannianAdam"
USE_FEATS = 1
NORMALIZE_FEATS = 1
NORMALIZE_ADJ = 1
DIM = 16
NUM_LAYERS = 2
BIAS = 1
EPOCHS = 500
PATIENCE = 100
DOUBLE_PRECISION = 0

TASK_DATASETS = {
    "lp": ["disease_lp"],
    "nc": [
        "cora",
        "citeseer",
        "pubmed",
        "airport",
        "disease_nc",
        "Carcinogenesis_data",
        "Hepatitis_std_data",
        "Hockey_data",
        "PTE",
        "Toxicology_data",
    ],
    "lp_and_nc": [
        "telecom",
        "actor",
        "cornell",
        "cs_phds",
    ],
}
DEFAULT_LP_DATASETS = list(TASK_DATASETS["lp"])
DEFAULT_NC_DATASETS = list(TASK_DATASETS["nc"])
DEFAULT_BOTH_DATASETS = list(TASK_DATASETS["lp_and_nc"])

OOM_PATTERNS = [
    r"cuda out of memory",
    r"out of memory",
    r"cublas.*status.*alloc",
    r"cudnn.*status.*alloc",
    r"unable to allocate",
    r"arraymemoryerror",
]

TRAIN_PROGRESS_PATTERNS = [
    re.compile(r"epoch:\s*0*1\b", re.IGNORECASE),
    re.compile(r"epoch\s+1:\s*", re.IGNORECASE),
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def build_env():
    env = os.environ.copy()
    env.setdefault("DATAPATH", str(PROJECT_ROOT / "data"))
    env.setdefault("LOG_DIR", str(PROJECT_ROOT / "logs"))
    env.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    env.setdefault("TORCH_DISABLE_ADDR2LINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    return env


def run_command(cmd, log_path, smoke_stop_after_first_epoch=False):
    ensure_dir(os.path.dirname(log_path))
    smoke_reached_epoch = False
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(str(x) for x in cmd) + "\n")
        f.write("START: " + datetime.datetime.now().isoformat() + "\n\n")
        if smoke_stop_after_first_epoch:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=build_env(),
                cwd=PROJECT_ROOT,
            )
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
                f.write("\n[SmokeTest] Reached first epoch; stopped early and marked as success.\n")
        else:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                env=build_env(),
                cwd=PROJECT_ROOT,
            )
            result_code = result.returncode
        if result_code != 0:
            f.write("\n[RunnerError] Non-zero exit code.\n")
            f.write("[RunnerError] Return code: " + str(result_code) + "\n")
            f.write("[RunnerError] Command: " + " ".join(str(x) for x in cmd) + "\n")
        f.write("\nEND: " + datetime.datetime.now().isoformat() + "\n")
        f.write("RETURN_CODE: " + str(result_code) + "\n")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    oom_hit = any(re.search(pattern, content, re.IGNORECASE) for pattern in OOM_PATTERNS)
    return result_code, oom_hit, smoke_reached_epoch


def _to_cli_flag(key):
    return "--" + key.replace("_", "-")


def get_base_train_params():
    return {
        "manifold": MANIFOLD,
        "optimizer": OPTIMIZER,
        "dim": DIM,
        "num_layers": NUM_LAYERS,
        "bias": BIAS,
        "normalize_feats": NORMALIZE_FEATS,
        "normalize_adj": NORMALIZE_ADJ,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "double_precision": DOUBLE_PRECISION,
        "use_feats": USE_FEATS,
    }


def build_cmd(dataset, task, model, seed, save_dir, best_params=None, gpu_id=None, split_seed=None, resume=0):
    merged_params = get_base_train_params()
    if best_params:
        merged_params.update(best_params)

    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        dataset,
        "--task",
        task,
        "--model",
        model,
        "--seed",
        str(seed),
        "--split-seed",
        str(SPLIT_SEED if split_seed is None else split_seed),
        "--cuda",
        str(GPU_ID if gpu_id is None else gpu_id),
        "--save",
        "1",
        "--save-dir",
        str(save_dir),
        "--resume",
        str(int(resume)),
    ]

    for key, value in merged_params.items():
        flag = _to_cli_flag(key)
        cmd.extend([flag, "None" if value is None else str(value)])

    return cmd


def run_optuna_search(
    dataset,
    task,
    model,
    exp_root,
    gpu_id,
    split_seed,
    optuna_trials,
    optuna_num_seeds,
    optuna_base_seed,
    optuna_trial_timeout_seconds,
):
    optuna_log = os.path.join(exp_root, "optuna_log.txt")
    best_path = os.path.join(exp_root, "optuna_best.txt")
    cmd = [
        sys.executable,
        str(OPTUNA_SCRIPT),
        "--dataset",
        dataset,
        "--task",
        task,
        "--model",
        model,
        "--cuda",
        str(gpu_id),
        "--n-trials",
        str(optuna_trials),
        "--num-seeds",
        str(optuna_num_seeds),
        "--base-seed",
        str(optuna_base_seed),
        "--split-seed",
        str(split_seed),
        "--best-path",
        best_path,
    ]
    if int(optuna_trial_timeout_seconds) > 0:
        cmd.extend(["--trial-timeout-seconds", str(int(optuna_trial_timeout_seconds))])
    return_code, oom_hit, _ = run_command(cmd, optuna_log)
    return return_code, oom_hit, optuna_log, best_path


def load_optuna_best(best_path):
    if not os.path.exists(best_path):
        return None
    with open(best_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    if not lines:
        return None
    try:
        params = ast.literal_eval(lines[0])
    except Exception:
        return None
    return params


def sanitize_best_params(dataset, task, model, best_params):
    if not best_params:
        best_params = get_base_train_params()
    else:
        best_params = dict(best_params)
    if dataset.lower() == "telecom":
        best_params["use_feats"] = 1
        best_params["normalize_feats"] = 1
        if task == "nc":
            best_params["use_att"] = 0
            best_params["local_agg"] = 0
    return best_params


def reevaluate_nc_seed_metrics(seed_dir):
    if prepare_data_and_model is None or evaluate_split is None:
        return None

    config_path = os.path.join(seed_dir, "config.json")
    model_path = os.path.join(seed_dir, "model.pth")
    if not os.path.exists(config_path) or not os.path.exists(model_path):
        return None

    try:
        _, data, model_ref = prepare_data_and_model(Path(seed_dir))
    except Exception:
        return None

    metrics = {}
    metrics.update(evaluate_split(model_ref, data, "val"))
    metrics.update(evaluate_split(model_ref, data, "test"))
    return metrics
def load_audit_module():
    if not AUDIT_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("audit_experiments", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(args):
    module = load_audit_module()
    if module is None:
        print(f"Audit script not found: {AUDIT_SCRIPT}")
        return 1
    output_json = ROOT_DIR / args.audit_json
    output_md = ROOT_DIR / args.audit_md
    summary = module.audit_all(output_json, output_md)
    print(f"Saved audit JSON to: {output_json}")
    print(f"Saved audit report to: {output_md}")
    print(f"Experiments scanned: {len(summary['experiments'])}")
    print(f"Experiments needing rerun: {len(summary['rerun_targets'])}")
    return 0


def get_resume_experiment(module, exp_root):
    if module is None:
        return None
    project = next((item for item in module.PROJECTS if item['name'] == PROJECT_NAME), None)
    if project is None:
        return None
    return module.summarize_experiment(project, Path(exp_root))


def resolve_seed_plan(args, exp_root):
    requested = list(args.seeds)
    if args.stage != "resume":
        return requested, None, {}

    audit_module = load_audit_module()
    exp = get_resume_experiment(audit_module, exp_root) if os.path.isdir(exp_root) else None
    if exp is None:
        return requested, None, {seed: {'seed': seed, 'resume_mode': 'fresh_no_audit'} for seed in requested}

    rerun_set = set(exp.get("rerun_seeds", []))
    seed_records = {item.get('seed'): item for item in exp.get('seeds', [])}
    seeds_to_run = [seed for seed in requested if seed in rerun_set]
    seed_plan = {}
    for seed in requested:
        record = seed_records.get(seed) or {'seed': seed}
        if seed not in rerun_set:
            record = dict(record)
            record['resume_mode'] = 'skip_complete'
            seed_plan[seed] = record
            continue
        record = dict(record)
        if record.get('checkpoint_exists') and record.get('checkpoint_completed') is not True:
            record['resume_mode'] = 'resume_from_checkpoint'
        else:
            record['resume_mode'] = 'rerun_without_checkpoint'
        seed_plan[seed] = record
    return seeds_to_run, exp, seed_plan


def should_run_optuna(args, best_path, exp_audit):
    if args.stage in {"stage2", "resume"}:
        return False
    return args.stage == "all"


def run_experiment(args, dataset, task, model):
    exp_name = f"{task}_{model}_{dataset}"
    exp_root = os.path.join(args.save_root, exp_name)
    ensure_dir(exp_root)

    seeds_to_run, exp_audit, seed_plan = resolve_seed_plan(args, exp_root)

    status = {
        "dataset": dataset,
        "task": task,
        "model": model,
        "stage": args.stage,
        "requested_seeds": list(args.seeds),
        "planned_seeds": seeds_to_run,
        "skipped_existing_seeds": [seed for seed in args.seeds if seed not in seeds_to_run],
        "seed_plan": seed_plan,
        "seeds": [],
        "oom_skip": False,
        "optuna": {},
        "resume_audit": exp_audit,
    }

    best_path = os.path.join(exp_root, "optuna_best.txt")
    run_optuna = should_run_optuna(args, best_path, exp_audit)
    if run_optuna:
        optuna_code, optuna_oom, optuna_log, best_path = run_optuna_search(
            dataset,
            task,
            model,
            exp_root,
            args.gpu_id,
            args.split_seed,
            args.optuna_trials,
            args.optuna_num_seeds,
            args.optuna_base_seed,
            args.optuna_trial_timeout_seconds,
        )
        status["optuna"] = {
            "return_code": optuna_code,
            "oom": optuna_oom,
            "log": optuna_log,
            "best_path": best_path,
            "requested_trials": int(args.optuna_trials),
            "requested_num_seeds": int(args.optuna_num_seeds),
            "requested_epochs": int(args.epochs),
            "requested_patience": int(args.patience),
        }
        if optuna_oom:
            status["oom_skip"] = True
            write_json(os.path.join(exp_root, "status.json"), status)
            return status
        if optuna_code != 0:
            write_json(os.path.join(exp_root, "status.json"), status)
            return status
    else:
        status["optuna"] = {
            "skipped": True,
            "best_path": best_path,
            "reason": "existing_best" if os.path.exists(best_path) else "stage_without_optuna",
            "requested_trials": int(args.optuna_trials),
            "requested_num_seeds": int(args.optuna_num_seeds),
            "requested_epochs": int(args.epochs),
            "requested_patience": int(args.patience),
        }

    best_params = sanitize_best_params(dataset, task, model, load_optuna_best(best_path))
    if args.stage == "resume" and not os.path.exists(best_path):
        status["skipped_no_best_params"] = True
        status["skip_reason"] = "resume_requires_existing_optuna_best"
        write_json(os.path.join(exp_root, "status.json"), status)
        return status

    if task == "nc":
        reevaluated_existing = []
        for seed in status["skipped_existing_seeds"]:
            seed_dir = os.path.join(exp_root, f"seed_{seed}")
            metrics = reevaluate_nc_seed_metrics(seed_dir)
            if metrics is not None:
                reevaluated_existing.append({"seed": seed, "metrics": metrics})
        if reevaluated_existing:
            status["reevaluated_existing_seeds"] = reevaluated_existing

    for seed in seeds_to_run:
        seed_dir = os.path.join(exp_root, f"seed_{seed}")
        ensure_dir(seed_dir)
        runner_log = os.path.join(seed_dir, "runner_log.txt")
        seed_info = dict(seed_plan.get(seed) or {'seed': seed, 'resume_mode': 'fresh'})
        resume_flag = args.stage == "resume" and seed_info.get('resume_mode') == 'resume_from_checkpoint'
        cmd = build_cmd(
            dataset,
            task,
            model,
            seed,
            seed_dir,
            best_params,
            gpu_id=args.gpu_id,
            split_seed=args.split_seed,
            resume=resume_flag,
        )

        smoke_stop = bool(args.smoke_stop_after_first_epoch)
        return_code, oom_hit, smoke_reached_epoch = run_command(cmd, runner_log, smoke_stop_after_first_epoch=smoke_stop)
        status["seeds"].append(
            {
                "seed": seed,
                "return_code": return_code,
                "oom": oom_hit,
                "runner_log": runner_log,
                "resume_mode": seed_info.get('resume_mode'),
                "checkpoint_exists": seed_info.get('checkpoint_exists'),
                "checkpoint_completed": seed_info.get('checkpoint_completed'),
                "checkpoint_epoch": seed_info.get('checkpoint_epoch'),
                "resume_flag": int(resume_flag),
                "reevaluated_metrics": None,
                "smoke_reached_epoch": smoke_reached_epoch,
            }
        )

        if return_code == 0 and task == "nc":
            reevaluated_metrics = reevaluate_nc_seed_metrics(seed_dir)
            if reevaluated_metrics is not None:
                status["seeds"][-1]["reevaluated_metrics"] = reevaluated_metrics

        if oom_hit:
            status["oom_skip"] = True
            break

    write_json(os.path.join(exp_root, "status.json"), status)
    return status


def rename_legacy_results(args):
    result_root = Path(args.save_root)
    renamed = []
    skipped = []

    for child in sorted(result_root.iterdir()):
        if not child.is_dir():
            continue
        parts = child.name.split('_', 1)
        if len(parts) != 2 or parts[0] not in {"HGCN", "HNN"}:
            continue

        model, dataset = parts
        target = result_root / f"nc_{model}_{dataset}"
        if target.exists():
            skipped.append({"source": str(child), "target": str(target), "reason": "target_exists"})
            continue

        child.rename(target)
        renamed.append({"source": str(child), "target": str(target)})

    payload = {"renamed": renamed, "skipped": skipped}
    write_json(result_root / "rename_legacy_summary.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "stage2", "resume", "audit", "rename-legacy"],
        help="all=optuna+train, stage2=skip optuna, resume=only rerun incomplete seeds, audit=generate audit report",
    )
    parser.add_argument("--save-root", default=str(SAVE_ROOT))
    parser.add_argument("--gpu-id", type=int, default=GPU_ID)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--lp-datasets", nargs="*", default=DEFAULT_LP_DATASETS)
    parser.add_argument("--nc-datasets", nargs="*", default=DEFAULT_NC_DATASETS)
    parser.add_argument("--both-datasets", nargs="*", default=DEFAULT_BOTH_DATASETS)
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_TRIALS)
    parser.add_argument("--optuna-num-seeds", type=int, default=OPTUNA_NUM_SEEDS)
    parser.add_argument("--optuna-base-seed", type=int, default=OPTUNA_BASE_SEED)
    parser.add_argument("--optuna-trial-timeout-seconds", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--smoke-stop-after-first-epoch", action="store_true")
    parser.add_argument("--audit-json", default="audit_summary.json")
    parser.add_argument("--audit-md", default="audit_report.md")
    args = parser.parse_args()

    if args.stage == "audit":
        raise SystemExit(run_audit(args))
    if args.stage == "rename-legacy":
        raise SystemExit(rename_legacy_results(args))

    args.gpu_profile = build_gpu_profile(args.gpu_id)

    ensure_dir(args.save_root)
    summary = {
        "gpu_id": args.gpu_id,
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "results": [],
        "stage": args.stage,
        "gpu_profile": args.gpu_profile,
    }

    for dataset in args.lp_datasets:
        for model in args.models:
            summary["results"].append(run_experiment(args, dataset, "lp", model))

    for dataset in args.nc_datasets:
        for model in args.models:
            summary["results"].append(run_experiment(args, dataset, "nc", model))

    for dataset in args.both_datasets:
        for model in args.models:
            summary["results"].append(run_experiment(args, dataset, "lp", model))
            summary["results"].append(run_experiment(args, dataset, "nc", model))

    write_json(os.path.join(args.save_root, "summary.json"), summary)


if __name__ == "__main__":
    main()
