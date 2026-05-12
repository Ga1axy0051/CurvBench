from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

from configs.base_config import BaseConfig, add_base_args, dataclass_kwargs, load_config_with_overrides


@dataclass
class PretrainConfig(BaseConfig):
    checkpoint_root: str = "checkpoints/pretrain"
    manifest_name: str = "pretrain_manifest.yaml"


def get_pretrain_parser() -> ArgumentParser:
    parser = ArgumentParser(description="SA2GFM unified pretrain configuration")
    add_base_args(parser)
    parser.add_argument("--checkpoint_root", type=str, default="checkpoints/pretrain")
    parser.add_argument("--manifest_name", type=str, default="pretrain_manifest.yaml")
    return parser


def parse_pretrain_config(remaining_argv=None) -> PretrainConfig:
    parser = get_pretrain_parser()
    args, _raw, config_path = load_config_with_overrides(parser, remaining_argv)
    config = PretrainConfig(**dataclass_kwargs(PretrainConfig, args))
    if not config.fold_name:
        if config_path is not None:
            config.fold_name = Path(config_path).stem
        elif config.source_datasets:
            config.fold_name = "_".join(config.source_datasets)
        else:
            config.fold_name = "manual_pretrain"
    return config
