"""Path-aware argparse for downstream MoE finetuning (cross-domain friendly)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

_SA2GFM = Path(__file__).resolve().parents[2]
_AG = _SA2GFM / "attack_data_gen"
sys.path.insert(0, str(_AG))
from lib.data_utils import canonical_graph_path, get_num_classes, get_num_nodes, load_graph, normalize_dataset_name  # noqa: E402
from lib.paths import paths  # noqa: E402


def get_pretrain_datasets(dataset: str) -> list:
    data_name = normalize_dataset_name(dataset)
    datasets = []
    if data_name == "cora":
        datasets = ["citeseer", "pubmed", "P-home", "wikics"]
    elif data_name == "citeseer":
        datasets = ["cora", "pubmed", "P-home", "wikics"]
    elif data_name == "pubmed":
        datasets = ["cora", "citeseer", "P-home", "wikics"]
    elif data_name == "P-tech":
        datasets = ["cora", "citeseer", "pubmed", "P-home", "wikics"]
    elif data_name == "P-home":
        datasets = ["cora", "citeseer", "pubmed", "wikics"]
    elif data_name == "wikics":
        datasets = ["cora", "citeseer", "pubmed", "P-home"]
    elif data_name == "arxiv":
        datasets = ["P-home", "P-tech", "wikics"]
    return datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _num_nodes_and_classes_from_path(data_path: str):
    data = _torch_load(data_path)
    return get_num_nodes(data), get_num_classes(data, ignore_negative=True)


def get_args():
    parser = argparse.ArgumentParser("SA2GFM_downstream")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument(
        "--source-datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit source expert list for cross-domain downstream. If omitted, fall back to the legacy built-in mapping.",
    )
    parser.add_argument(
        "--expert-paths",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit expert checkpoint paths. If provided, these take priority over source-dataset name lookup.",
    )
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
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Train-only optimization steps per split; test accuracy computed once after all epochs",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=-1,
        help="If >= 0, only run this few-shot split index; -1 = all splits up to --num-splits",
    )
    parser.add_argument("--moe_weight", type=float, default=0.1)
    parser.add_argument("--structure_weight", type=float, default=0.1)
    parser.add_argument("--bucket_boundaries", type=int, nargs="+", default=[30, 100])
    parser.add_argument("--inter_cluster_optimizer", default=True)
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--appnp_k", type=int, default=10)
    parser.add_argument("--inter_cluster_threshold", type=float, default=0.5)
    parser.add_argument("--inter_cluster_temperature", type=float, default=10.0)
    parser.add_argument("--moe_embedding_weight", type=float, default=0.1)
    parser.add_argument("--multi_embedding_weight", type=float, default=0.9)
    parser.add_argument(
        "--attack_type",
        default="none",
        choices=["none", "random", "targeted_poisoning", "targeted_evasion"],
    )
    parser.add_argument("--attack_ratio", type=float, default=0.1)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--random_attack_type", default="feature", choices=["feature", "structure"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num-splits",
        type=int,
        default=-1,
        help="Number of few-shot splits to run (-1: 20 for all datasets)",
    )
    parser.add_argument("--no-swanlab", action="store_true")
    args = parser.parse_args()

    dr = paths.data_root
    ds = normalize_dataset_name(args.dataset)
    args.dataset = ds

    if args.attack_type == "none":
        args.data_path = str(canonical_graph_path(ds))
    elif args.attack_type == "targeted_poisoning":
        sub = paths.attack_post_dir / f"{ds}_p{args.p}_final"
        args.data_path = str(sub / f"{ds}_poisoning_final.pt")
    elif args.attack_type == "targeted_evasion":
        sub = paths.attack_post_dir / f"{ds}_p{args.p}_final"
        args.data_path = str(sub / f"{ds}_evasion_final.pt")
    elif args.attack_type == "random":
        if args.random_attack_type == "feature":
            args.data_path = str(paths.attack_random_dir / f"{ds}_feature_p{args.attack_ratio}.pt")
        else:
            args.data_path = str(paths.attack_random_dir / f"{ds}_structure_p{args.attack_ratio}.pt")
    else:
        raise ValueError(args.attack_type)

    args.pre_train_model_dir_single = str(paths.save_model_dir)
    args.pre_train_model_dir_many = str(dr / "save_model_many")
    args.communitys_dir = str(paths.communities_dir)
    args.community_file = str(paths.communities_dir / f"{ds}_communities.pt")
    args.down_data_dir = str(paths.few_shot_dir / ds / f"{args.shot_num}shot")

    emb_path = dr / "reduced_embeddings" / f"{ds}_reduced_embeddings.pt"
    if emb_path.is_file():
        args.txt_features = _torch_load(str(emb_path))["embeddings"]
    else:
        args.txt_features = None

    set_seed(args.seed)
    if args.attack_type == "none":
        clean_data = load_graph(ds)
        args.num_nodes, args.num_classes = get_num_nodes(clean_data), get_num_classes(clean_data, ignore_negative=True)
    else:
        args.num_nodes, args.num_classes = _num_nodes_and_classes_from_path(args.data_path)
    if args.source_datasets is not None:
        args.source_datasets = [normalize_dataset_name(name) for name in args.source_datasets]
    else:
        args.source_datasets = get_pretrain_datasets(ds)
        if not args.source_datasets:
            raise ValueError(
                f"No built-in source expert list is defined for target dataset {ds!r}. "
                "For custom cross-domain experiments, pass --source-datasets explicitly."
            )
    if args.expert_paths is not None:
        args.expert_paths = list(args.expert_paths)

    if args.num_splits < 0:
        args.num_splits = 20
    return args
