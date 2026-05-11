import argparse
import os
import subprocess
import sys

# Base directory for the repository
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Map of baselines to their respective directories and main scripts
BASELINE_MAP = {
    "mlp_gcn_gat": {
        "dir": "mlp_gcn_gat_script",
        "script": "main.py"
    },
    "cusp": {
        "dir": "cusp_script",
        "script": "train.py"
    },
    "hat": {
        "dir": "hat_script",
        "script": "hat_new.py"
    },
    "hgcn": {
        "dir": "hnn_hgcn_script/hgcn",
        "script": "train.py"
    },
    "hybonet": {
        "dir": "hybonet_script/gcn",
        "script": "train.py"
    },
    "qgcn": {
        "dir": "qgcn_script/QGCN-main",
        "script": "train.py"
    },
    "graphmore": {
        "dir": "graphmore_script/GraphMoRE-main",
        "script": "main.py"
    },
    "gcope": {
        "dir": "GCOPE-main",
        "script": "run_benchmark.py"  
    },
    "graphsage": {
        "dir": "graphsage&pcnet_script",
        "script": "run_benchmark.py"
    },
    "cal_curv": {
        "dir": "Cal_curv",
        "script": "run_benchmark.py"
    },
    "mdgfm": {
        "dir": "mdgfm",
        "script": "runexp.py"
    },
    "mdgpt": {
        "dir": "mdgpt",
        "script": "execute.py"
    },
    "samgpt": {
        "dir": "SAMGPT/src",
        "script": "execute.py"
    }
}

def main():
    parser = argparse.ArgumentParser(description="CurvBench Unified Runner")
    parser.add_argument('--model', type=str, required=True, choices=list(BASELINE_MAP.keys()),
                        help="Baseline model to run")
    parser.add_argument('--task', type=str, required=False, default="nc",
                        help="Task type: nc or lp (Passed to the baseline if supported)")
    parser.add_argument('--dataset', type=str, required=True, 
                        help="Dataset name to evaluate on")
    parser.add_argument('--shot_num', type=int, required=False, default=1,
                        help="Number of shots for few-shot learning baselines")
    
    # Accept any extra arguments to pass down to the baseline
    args, unknown = parser.parse_known_args()

    model_info = BASELINE_MAP[args.model]
    target_dir = os.path.join(BASE_DIR, model_info["dir"])
    target_script = os.path.join(target_dir, model_info["script"])

    if not os.path.exists(target_script):
        print(f"Error: Could not find script {target_script}")
        sys.exit(1)

    # Build the command
    data_root = os.path.join(BASE_DIR, "datasets")
    if args.model == "hat":
        # HAT parses -dataset instead of --dataset
        command = [
            sys.executable,  
            target_script,
            "-dataset", args.dataset
        ]
        # HAT doesn't take --data-root or --task by default
        command.extend(unknown)
    else:
        command = [
            sys.executable,  
            target_script,
            "--dataset", args.dataset
        ]
        
        if args.model in ["mlp_gcn_gat"]:
            command.extend(["--data-root", data_root])
        elif args.model in ["cusp"]:
            command.extend(["--data_root", data_root])
            
        if args.task:
            if args.model == "cusp" and args.task == "nc":
                command.extend(["--task", "node_classification"])
            elif args.model == "cusp" and args.task == "lp":
                command.extend(["--task", "link_prediction"])
            elif args.model == "graphmore" and args.task == "nc":
                command.extend(["--downstream_task", "NC"])
            elif args.model == "graphmore" and args.task == "lp":
                command.extend(["--downstream_task", "LP"])
            else:
                command.extend(["--task", args.task])
                
        # Forward shot_num to baselines that might support it (mdgfm, gcope, mdgpt, samgpt)
        if args.model in ["gcope", "mdgfm", "mdgpt", "samgpt"]:
            command.extend(["--shot_num", str(args.shot_num)])
            
        command.extend(unknown)

    # hgcn/qgcn/hybonet uses DATAPATH env variable
    os.environ['DATAPATH'] = data_root

    print("==================================================")
    print(f"Running CurvBench - Model: {args.model} | Dataset: {args.dataset} | Task: {args.task}")
    print(f"Target Directory: {target_dir}")
    print(f"Command: {' '.join(command)}")
    print("==================================================")

    # Execute the baseline in its respective directory so relative paths and configs work unmodified
    os.environ['PYTHONPATH'] = target_dir + ":" + os.environ.get('PYTHONPATH', '')
    
    try:
        subprocess.run(command, cwd=target_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nExecution failed with error code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
