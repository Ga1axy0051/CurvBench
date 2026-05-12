import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN = PROJECT_ROOT / "main.py"
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--task", type=str, choices=["nc", "lp"], default="nc")
    p.add_argument("--datasets", type=str, default="")
    p.add_argument("--extra-args", type=str, default="")
    args = p.parse_args()

    import json

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if args.datasets.strip():
        names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    else:
        ds = cfg.get("datasets") or {}
        names = list(ds.get("exptable2graph_keys", [])) + list(ds.get("base_benchmark", []))

    for name in names:
        cmd = [sys.executable, str(MAIN), "--config", args.config, "--dataset", name, "--task", args.task]
        if args.extra_args:
            cmd.extend(args.extra_args.split())
        print("RUN:", " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        if r.returncode != 0:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
