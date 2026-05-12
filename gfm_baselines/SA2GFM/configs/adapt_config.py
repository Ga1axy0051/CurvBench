from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from configs.base_config import BaseConfig, add_base_args, dataclass_kwargs, load_config_with_overrides


@dataclass
class AdaptConfig(BaseConfig):
    data_name: str = ""
    k_shot: int = 5
    pretrained_checkpoint: str = ""
    expert_paths: list[str] | None = None
    attack_type: str = "none"
    attack_ratio: float = 0.1
    p: int = 1
    random_attack_type: str = "feature"
    split_id: int = -1


def get_adapt_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SA2GFM unified downstream adaptation configuration")
    add_base_args(parser)
    parser.add_argument("--data_name", type=str, default="")
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--expert_paths", "--expert-paths", nargs="*", type=str, default=None)
    parser.add_argument("--attack_type", type=str, default="none")
    parser.add_argument("--attack_ratio", type=float, default=0.1)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--random_attack_type", type=str, default="feature")
    parser.add_argument("--split_id", type=int, default=-1)
    return parser


def parse_adaption_config(remaining_argv=None) -> AdaptConfig:
    parser = get_adapt_parser()
    args, _raw, config_path = load_config_with_overrides(parser, remaining_argv)
    config = AdaptConfig(**dataclass_kwargs(AdaptConfig, args))
    if not config.fold_name:
        if config_path is not None:
            config.fold_name = Path(config_path).stem
        elif config.source_datasets:
            config.fold_name = "_".join(config.source_datasets)
        else:
            config.fold_name = "manual_adapt"
    if not config.data_name and config.target_datasets:
        config.data_name = config.target_datasets[0]
    return config
