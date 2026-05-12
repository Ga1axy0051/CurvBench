from __future__ import annotations

import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from dataclasses import fields as dataclass_fields

from utils.runtime import load_yaml, resolve_path


@dataclass
class BaseConfig:
    root: str = "./datasets"
    seed: int = 42
    gpu: int = 0
    fold_name: str = ""
    source_datasets: list[str] = field(default_factory=list)
    target_datasets: list[str] = field(default_factory=list)
    auto_prepare: bool = True
    num_splits: int = 20
    num_val: float = 0.1
    task_type: str = "node_cls"
    metric: str = "acc"
    no_swanlab: bool = False

    # community detection
    community_method: str = "louvain"
    community_resolution: float = 1.0

    # feature enhancement
    svd_dim: int = 32
    text_svd_dim: int = 32
    bert: str = "bert-base-uncased"
    bert_batch_size: int = 16
    max_neighbors_in_text: int = 48

    # pretrain shared
    pretrain_hid_units: int = 256
    pretrain_out_channels: int = 64
    pretrain_num_layers: int = 2
    pretrain_dropout: float = 0.2
    pretrain_lr: float = 1e-5
    pretrain_l2_coef: float = 0.0
    pretrain_nb_epochs: int = 1000
    pretrain_patience: int = 100
    pretrain_eval_steps: int = 10
    pretrain_neg_samples: int = 50
    pretrain_kl_weight: float = 0.0

    # downstream shared
    adapt_epochs: int = 100
    adapt_lr: float = 1e-3
    unify_dim: int = 64
    hidden_dim: int = 64
    num_heads: int = 4
    head_dim: int = 16
    gamma: float = 5.0
    tau: float = 0.0
    lambda_var: float = 1.0
    alpha: float = 0.5
    gcn_layers: int = 2
    dropout: float = 0.5
    out_channels: int = 64
    hid_units: int = 256
    moe_weight: float = 0.1
    structure_weight: float = 0.1
    bucket_boundaries: list[int] = field(default_factory=lambda: [30, 100])
    inter_cluster_optimizer: bool = True
    appnp_alpha: float = 0.1
    appnp_k: int = 10
    inter_cluster_threshold: float = 0.5
    inter_cluster_temperature: float = 10.0
    moe_embedding_weight: float = 0.1
    multi_embedding_weight: float = 0.9


def add_base_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--config_load_path", type=str, default=None)
    parser.add_argument("--root", type=str, default="./datasets")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fold_name", type=str, default="")
    parser.add_argument("--source_datasets", "--source-datasets", nargs="*", type=str, default=None)
    parser.add_argument("--target_datasets", "--target-datasets", nargs="*", type=str, default=None)
    parser.add_argument("--auto_prepare", dest="auto_prepare", action="store_true")
    parser.add_argument("--no_auto_prepare", dest="auto_prepare", action="store_false")
    parser.set_defaults(auto_prepare=True)
    parser.add_argument("--num_splits", type=int, default=20)
    parser.add_argument("--num_val", type=float, default=0.1)
    parser.add_argument("--task_type", type=str, default="node_cls")
    parser.add_argument("--metric", type=str, default="acc")
    parser.add_argument("--no_swanlab", "--no-swanlab", action="store_true")

    parser.add_argument("--community_method", type=str, default="louvain")
    parser.add_argument("--community_resolution", type=float, default=1.0)

    parser.add_argument("--svd_dim", type=int, default=32)
    parser.add_argument("--text_svd_dim", type=int, default=32)
    parser.add_argument("--bert", type=str, default="bert-base-uncased")
    parser.add_argument("--bert_batch_size", type=int, default=16)
    parser.add_argument("--max_neighbors_in_text", type=int, default=48)

    parser.add_argument("--pretrain_hid_units", type=int, default=256)
    parser.add_argument("--pretrain_out_channels", type=int, default=64)
    parser.add_argument("--pretrain_num_layers", type=int, default=2)
    parser.add_argument("--pretrain_dropout", type=float, default=0.2)
    parser.add_argument("--pretrain_lr", type=float, default=1e-5)
    parser.add_argument("--pretrain_l2_coef", type=float, default=0.0)
    parser.add_argument("--pretrain_nb_epochs", type=int, default=1000)
    parser.add_argument("--pretrain_patience", type=int, default=100)
    parser.add_argument("--pretrain_eval_steps", type=int, default=10)
    parser.add_argument("--pretrain_neg_samples", type=int, default=50)
    parser.add_argument("--pretrain_kl_weight", type=float, default=0.0)

    parser.add_argument("--adapt_epochs", type=int, default=100)
    parser.add_argument("--adapt_lr", type=float, default=1e-3)
    parser.add_argument("--unify_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--hid_units", type=int, default=256)
    parser.add_argument("--moe_weight", type=float, default=0.1)
    parser.add_argument("--structure_weight", type=float, default=0.1)
    parser.add_argument("--bucket_boundaries", nargs="+", type=int, default=[30, 100])
    parser.add_argument(
        "--inter_cluster_optimizer",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        default=True,
    )
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--appnp_k", type=int, default=10)
    parser.add_argument("--inter_cluster_threshold", type=float, default=0.5)
    parser.add_argument("--inter_cluster_temperature", type=float, default=10.0)
    parser.add_argument("--moe_embedding_weight", type=float, default=0.1)
    parser.add_argument("--multi_embedding_weight", type=float, default=0.9)
    return parser


def _cli_provided_keys(remaining_argv: list[str] | None) -> set[str]:
    cli_args = remaining_argv if remaining_argv is not None else sys.argv[1:]
    provided = set()
    for arg in cli_args:
        if arg.startswith("--"):
            provided.add(arg[2:].replace("-", "_"))
    return provided


def load_config_with_overrides(
    parser: ArgumentParser,
    remaining_argv: list[str] | None,
) -> tuple[Any, dict[str, Any], Path | None]:
    args = parser.parse_args(remaining_argv)
    config_path = None
    if getattr(args, "config_load_path", None):
        config_path = resolve_path(args.config_load_path, Path.cwd())
        yaml_config = load_yaml(config_path)
        provided = _cli_provided_keys(remaining_argv)
        for key, value in yaml_config.items():
            if hasattr(args, key) and key not in provided:
                setattr(args, key, value)
    return args, vars(args).copy(), config_path


def dataclass_kwargs(dataclass_type, args_namespace: Any) -> dict[str, Any]:
    allowed = {field.name for field in dataclass_fields(dataclass_type)}
    data = vars(args_namespace)
    return {key: value for key, value in data.items() if key in allowed}
