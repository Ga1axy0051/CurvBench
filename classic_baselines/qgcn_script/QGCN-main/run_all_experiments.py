"""Run Optuna search and multi-seed training for multiple datasets/models."""

from __future__ import annotations

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
PROJECT_NAME = "QGCN-main"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gpu_profile import build_gpu_profile

DEFAULT_LP_DATASETS = ["disease_lp", "telecom", "actor", "cornell", "cs_phds"]
DEFAULT_NC_DATASETS = ["cora", "citeseer", "pubmed", "airport", "disease_nc", "telecom", "actor", "cornell", "cs_phds", "Carcinogenesis_data", "Hepatitis_std_data", "Hockey_data", "PTE", "Toxicology_data"]
DEFAULT_BOTH_DATASETS = []
DEFAULT_MODELS = ["QGCN"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

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


def minimum_supported_num_layers(model: str) -> int:
    return 2 if model in {"HGCN", "HNN", "QGCN"} else 0


def sanitize_num_layers(model: str, value: int) -> int:
    return max(int(value), minimum_supported_num_layers(model))


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DATAPATH", str(PROJECT_ROOT / "data"))
    env.setdefault("LOG_DIR", str(PROJECT_ROOT / "logs"))
    env.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    env.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")
    env.setdefault("TORCH_DISABLE_ADDR2LINE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    return env


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_command(cmd: list[str], log_path: Path, smoke_stop_after_first_epoch: bool = False) -> tuple[int, bool, bool]:
    ensure_dir(log_path.parent)
    smoke_reached_epoch = False
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(str(x) for x in cmd) + "\n")
        handle.write("START: " + datetime.datetime.now().isoformat() + "\n\n")
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
                handle.write(line)
                handle.flush()
                if any(pattern.search(line) for pattern in TRAIN_PROGRESS_PATTERNS):
                    smoke_reached_epoch = True
                    process.terminate()
                    break
            process.wait()
            result_code = 0 if smoke_reached_epoch else int(process.returncode or 0)
            if smoke_reached_epoch:
                handle.write("\n[SmokeTest] Reached first epoch; stopped early and marked as success.\n")
        else:
            result = subprocess.run(
                cmd,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=build_env(),
                cwd=PROJECT_ROOT,
            )
            result_code = result.returncode
        if result_code != 0:
            handle.write("\n[RunnerError] Non-zero exit code.\n")
            handle.write("[RunnerError] Return code: " + str(result_code) + "\n")
            handle.write("[RunnerError] Command: " + " ".join(str(x) for x in cmd) + "\n")
        handle.write("\nEND: " + datetime.datetime.now().isoformat() + "\n")
        handle.write("RETURN_CODE: " + str(result_code) + "\n")

    content = log_path.read_text(encoding="utf-8", errors="replace")
    oom_hit = any(re.search(pattern, content, re.IGNORECASE) for pattern in OOM_PATTERNS)
    return result_code, oom_hit, smoke_reached_epoch


def to_cli_flag(key: str) -> str:
    if key in {"space_dim", "time_dim"}:
        return "--" + key
    return "--" + key.replace("_", "-")


def get_base_train_params(args, model: str) -> dict:
    if model in {"GCN", "GAT", "MLP", "Shallow"}:
        manifold = "Euclidean"
        optimizer = "Adam"
    elif model == "QGCN":
        manifold = "PseudoHyperboloid"
        optimizer = "RiemannianAdam"
    else:
        manifold = args.manifold
        optimizer = args.optimizer
    params = {
        "manifold": manifold,
        "optimizer": optimizer,
        "dim": args.dim,
        "num_layers": args.num_layers,
        "bias": args.bias,
        "normalize_feats": args.normalize_feats,
        "normalize_adj": args.normalize_adj,
        "epochs": args.epochs,
        "patience": args.patience,
        "double_precision": args.double_precision,
        "use_feats": args.use_feats,
    }
    params["num_layers"] = sanitize_num_layers(model, params["num_layers"])
    if model == "QGCN":
        params["space_dim"] = args.space_dim
        params["time_dim"] = args.time_dim
        params["c"] = None
    return params


def build_train_cmd(args, dataset: str, task: str, model: str, seed: int, save_dir: Path, best_params: dict | None, resume: bool = False) -> list[str]:
    merged_params = get_base_train_params(args, model)
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
        str(args.split_seed),
        "--cuda",
        str(args.gpu_id),
        "--save",
        "1",
        "--save-dir",
        str(save_dir),
        "--resume",
        str(int(resume)),
    ]

    for key, value in merged_params.items():
        cmd.extend([to_cli_flag(key), "None" if value is None else str(value)])
    return cmd


def build_optuna_cmd(args, dataset: str, task: str, model: str, best_path: Path) -> list[str]:
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
        str(args.gpu_id),
        "--n-trials",
        str(args.optuna_trials),
        "--num-seeds",
        str(args.optuna_num_seeds),
        "--base-seed",
        str(args.optuna_base_seed),
        "--split-seed",
        str(args.split_seed),
        "--dim",
        str(args.dim),
        "--space-dim",
        str(args.space_dim),
        "--time-dim",
        str(args.time_dim),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--best-path",
        str(best_path),
    ]
    if int(args.optuna_trial_timeout_seconds) > 0:
        cmd.extend(["--trial-timeout-seconds", str(int(args.optuna_trial_timeout_seconds))])
    return cmd


def load_optuna_best(best_path: Path) -> dict | None:
    if not best_path.exists():
        return None
    lines = best_path.read_text(encoding="utf-8").strip().splitlines()
    if not lines:
        return None
    try:
        params = ast.literal_eval(lines[0])
    except Exception:
        return None
    return params if isinstance(params, dict) else None


def sanitize_best_params(args, dataset: str, task: str, model: str, best_params: dict | None) -> dict | None:
    if not best_params:
        best_params = get_base_train_params(args, model)
    else:
        best_params = dict(best_params)
    if model in {"GCN", "GAT", "MLP", "Shallow"}:
        best_params.setdefault("manifold", "Euclidean")
        best_params.setdefault("optimizer", "Adam")
    elif model == "QGCN":
        best_params.setdefault("manifold", "PseudoHyperboloid")
        best_params.setdefault("optimizer", "RiemannianAdam")
        best_params.setdefault("space_dim", args.space_dim)
        best_params.setdefault("time_dim", args.time_dim)
        best_params.setdefault("c", None)
    else:
        best_params.setdefault("manifold", "PoincareBall")
        best_params.setdefault("optimizer", "RiemannianAdam")
    best_params["num_layers"] = sanitize_num_layers(model, best_params.get("num_layers", args.num_layers))
    dataset_lower = dataset.lower()
    if model == "QGCN" and dataset_lower in {"cora", "citeseer", "pubmed"}:
        # Citation graphs should use real node features; identity (use_feats=0) makes features NxN and can
        # severely hurt both speed and accuracy.
        best_params["use_feats"] = 1
        best_params["normalize_feats"] = 1
    if dataset.lower() == "telecom":
        best_params["use_feats"] = 1
        best_params["normalize_feats"] = 1
        if task == "nc":
            best_params["use_att"] = 0
            best_params["local_agg"] = 0
    if model == "QGCN" and task == "nc" and dataset_lower in {"disease_nc", "cs_phds"}:
        best_params["use_att"] = 0
        best_params["local_agg"] = 0
    return best_params


def load_audit_module():
    if not AUDIT_SCRIPT.exists():
        return None
    spec = importlib.util.spec_from_file_location("audit_experiments", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_audit(args) -> int:
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


def get_resume_experiment(module, exp_root: Path):
    if module is None:
        return None
    project = next((item for item in module.PROJECTS if item['name'] == PROJECT_NAME), None)
    if project is None:
        return None
    return module.summarize_experiment(project, exp_root)


def resolve_seed_plan(args, exp_root: Path, dataset: str, task: str, model: str) -> tuple[list[int], dict | None, dict[int, dict]]:
    requested = list(args.seeds)
    if args.stage != "resume":
        return requested, None, {}

    audit_module = load_audit_module()
    exp = get_resume_experiment(audit_module, exp_root) if exp_root.exists() else None
    if exp is None:
        return requested, None, {seed: {'seed': seed, 'resume_mode': 'fresh_no_audit'} for seed in requested}

    rerun_set = set(exp.get("rerun_seeds", []))
    seed_records = {item.get('seed'): item for item in exp.get('seeds', [])}
    seeds_to_run = [seed for seed in requested if seed in rerun_set]
    seed_plan: dict[int, dict] = {}
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


def should_run_optuna(args, best_path: Path, exp_audit: dict | None) -> bool:
    if args.stage in {"stage2", "resume"}:
        return False
    return args.stage == "all"


def run_experiment(args, dataset: str, task: str, model: str) -> dict:
    exp_name = f"{task}_{model}_{dataset}"
    exp_root = args.save_root / exp_name
    ensure_dir(exp_root)

    seeds_to_run, exp_audit, seed_plan = resolve_seed_plan(args, exp_root, dataset, task, model)

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

    best_path = exp_root / "optuna_best.txt"
    run_optuna = should_run_optuna(args, best_path, exp_audit)
    if run_optuna:
        optuna_log = exp_root / "optuna_log.txt"
        optuna_cmd = build_optuna_cmd(args, dataset, task, model, best_path)
        optuna_code, optuna_oom, _ = run_command(optuna_cmd, optuna_log)
        status["optuna"] = {
            "return_code": optuna_code,
            "oom": optuna_oom,
            "log": str(optuna_log),
            "best_path": str(best_path),
            "requested_trials": int(args.optuna_trials),
            "requested_num_seeds": int(args.optuna_num_seeds),
            "requested_epochs": int(args.epochs),
            "requested_patience": int(args.patience),
        }
        if optuna_oom:
            status["oom_skip"] = True
            write_json(exp_root / "status.json", status)
            return status
        if optuna_code != 0:
            write_json(exp_root / "status.json", status)
            return status
    else:
        status["optuna"] = {
            "skipped": True,
            "best_path": str(best_path),
            "reason": "existing_best" if best_path.exists() else "stage_without_optuna",
            "requested_trials": int(args.optuna_trials),
            "requested_num_seeds": int(args.optuna_num_seeds),
            "requested_epochs": int(args.epochs),
            "requested_patience": int(args.patience),
        }

    best_params = sanitize_best_params(args, dataset, task, model, load_optuna_best(best_path))
    if args.stage == "resume" and not best_path.exists():
        status["skipped_no_best_params"] = True
        status["skip_reason"] = "resume_requires_existing_optuna_best"
        write_json(exp_root / "status.json", status)
        return status

    for seed in seeds_to_run:
        seed_dir = exp_root / f"seed_{seed}"
        ensure_dir(seed_dir)
        runner_log = seed_dir / "runner_log.txt"
        seed_info = dict(seed_plan.get(seed) or {'seed': seed, 'resume_mode': 'fresh'})
        resume_flag = args.stage == "resume" and seed_info.get('resume_mode') == 'resume_from_checkpoint'
        cmd = build_train_cmd(args, dataset, task, model, seed, seed_dir, best_params, resume=resume_flag)
        smoke_stop = bool(args.smoke_stop_after_first_epoch)
        return_code, oom_hit, smoke_reached_epoch = run_command(cmd, runner_log, smoke_stop_after_first_epoch=smoke_stop)
        status["seeds"].append(
            {
                "seed": seed,
                "return_code": return_code,
                "oom": oom_hit,
                "runner_log": str(runner_log),
                "resume_mode": seed_info.get('resume_mode'),
                "checkpoint_exists": seed_info.get('checkpoint_exists'),
                "checkpoint_completed": seed_info.get('checkpoint_completed'),
                "checkpoint_epoch": seed_info.get('checkpoint_epoch'),
                "resume_flag": int(resume_flag),
                "smoke_reached_epoch": smoke_reached_epoch,
            }
        )
        if oom_hit:
            status["oom_skip"] = True
            break

    write_json(exp_root / "status.json", status)
    return status


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch Optuna + training runner")
    parser.add_argument("--stage", default="all", choices=["all", "stage2", "resume", "audit"])
    parser.add_argument("--save-root", default="Result")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--lp-datasets", nargs="*", default=DEFAULT_LP_DATASETS)
    parser.add_argument("--nc-datasets", nargs="*", default=DEFAULT_NC_DATASETS)
    parser.add_argument("--both-datasets", nargs="*", default=DEFAULT_BOTH_DATASETS)
    parser.add_argument("--optuna-trials", type=int, default=30)
    parser.add_argument("--optuna-num-seeds", type=int, default=1)
    parser.add_argument("--optuna-base-seed", type=int, default=0)
    parser.add_argument("--optuna-trial-timeout-seconds", type=int, default=900)
    parser.add_argument("--manifold", default="PoincareBall")
    parser.add_argument("--optimizer", default="RiemannianAdam")
    parser.add_argument("--use-feats", type=int, default=1)
    parser.add_argument("--normalize-feats", type=int, default=1)
    parser.add_argument("--normalize-adj", type=int, default=1)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--bias", type=int, default=1)
    parser.add_argument("--space-dim", type=int, default=15)
    parser.add_argument("--time-dim", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--smoke-stop-after-first-epoch", action="store_true")
    parser.add_argument("--double-precision", type=int, default=0)
    parser.add_argument("--audit-json", default="audit_summary.json")
    parser.add_argument("--audit-md", default="audit_report.md")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.save_root = Path(args.save_root)
    args.gpu_profile = build_gpu_profile(args.gpu_id)

    if args.stage == "audit":
        return run_audit(args)

    ensure_dir(args.save_root)

    summary = {
        "gpu_id": args.gpu_id,
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "stage": args.stage,
        "gpu_profile": args.gpu_profile,
        "results": [],
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

    write_json(args.save_root / "summary.json", summary)
    print(f"Saved summary to: {args.save_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
