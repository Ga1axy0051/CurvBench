import argparse
from pathlib import Path

import torch

from configs import parse_adaption_config, parse_pretrain_config
from utils.runtime import print_final_config, set_data_root_env, set_seed


def _resolve_data_root(root: str) -> Path:
    base = Path(__file__).resolve().parent
    path = Path(root).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def pretrain_main(remaining_argv=None):
    config = parse_pretrain_config(remaining_argv)
    set_data_root_env(_resolve_data_root(config.root))
    from runners.pretrain_runner import PretrainRunner

    set_seed(config.seed)
    print_final_config(config)
    runner = PretrainRunner(config)
    runner.run()
    torch.cuda.empty_cache()


def adapt_main(remaining_argv=None):
    config = parse_adaption_config(remaining_argv)
    set_data_root_env(_resolve_data_root(config.root))
    from runners.adapt_runner import AdaptRunner

    set_seed(config.seed)
    print_final_config(config)
    runner = AdaptRunner(config)
    runner.run()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA2GFM unified pretrain/adapt entry")
    parser.add_argument("--run_type", type=str, default="pretrain", choices=["pretrain", "adapt"])
    args, remaining_argv = parser.parse_known_args()
    if args.run_type == "pretrain":
        pretrain_main(remaining_argv)
    elif args.run_type == "adapt":
        adapt_main(remaining_argv)
    else:
        raise ValueError(f"Invalid run_type: {args.run_type}")
