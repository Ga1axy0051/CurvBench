#!/usr/bin/env python3
"""
MoE downstream node classification — simplified protocol:
  For each few-shot split: train for --epochs using train nodes only (no test in the loop),
  then evaluate once on the fixed test band and print test accuracy.
  No Top-K selection, no checkpoint saving, no re-test pipeline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree
from sklearn.metrics import f1_score

_DOWN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_DOWN))
sys.path.insert(0, str(_DOWN / "lib"))
_AG = Path(__file__).resolve().parents[2] / "attack_data_gen"
sys.path.insert(0, str(_AG))
_PRETRAIN_PIPELINE = Path(__file__).resolve().parents[2] / "pretrain" / "pipeline"
sys.path.insert(0, str(_PRETRAIN_PIPELINE))

from config import get_args  # noqa: E402
from lib.data_utils import load_graph, normalize_dataset_name  # noqa: E402
from models.down_model import SparseLookup, downprompt  # noqa: E402
from model import JointContrastiveModel  # noqa: E402


def _torch_load(path: str, map_location=None):
    kw = {}
    if map_location is not None:
        kw["map_location"] = map_location
    try:
        return torch.load(path, weights_only=False, **kw)
    except TypeError:
        return torch.load(path, **kw)


def _load_expert_model(path: str, device):
    payload = _torch_load(path, map_location=device)
    if isinstance(payload, dict) and payload.get("checkpoint_type") == "joint_contrastive_expert":
        model = JointContrastiveModel(**payload["model_kwargs"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model
    if hasattr(payload, "to"):
        return payload.to(device)
    raise ValueError(f"Unsupported expert checkpoint format: {path}")


def build_node_to_cluster_map(all_communities):
    return {int(node): i for i, comm in enumerate(all_communities) for node in comm}


def precompute_appnp_matrix(edge_index, num_nodes, alpha=0.1, k=10, device="cpu", batch_size=512):
    with torch.no_grad():
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_with_loops
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        T = torch.sparse_coo_tensor(edge_index_with_loops, norm, (num_nodes, num_nodes)).to(device).coalesce()

    all_S_cols = []
    for i in range(0, num_nodes, batch_size):
        batch_indices = torch.arange(i, min(i + batch_size, num_nodes), device=device)
        H_0_batch = torch.sparse_coo_tensor(
            torch.stack([batch_indices, batch_indices]),
            torch.ones(len(batch_indices), device=device),
            (num_nodes, num_nodes),
        ).coalesce()
        H_batch = H_0_batch.clone()
        for _ in range(k):
            H_batch = (torch.sparse.mm(T, H_batch).coalesce() * (1 - alpha) + H_0_batch * alpha).coalesce()
        all_S_cols.append(H_batch.cpu())

    indices = torch.cat([s.indices() for s in all_S_cols], dim=1)
    values = torch.cat([s.values() for s in all_S_cols], dim=0)
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device).coalesce()


def load_communities(file_path):
    return _torch_load(file_path)["communities"]


def load_few_shot_data(data_dir, split_id=0):
    data = _torch_load(f"{data_dir}/split_{split_id}.pt")
    return data


def accuracy(pred, labels):
    return (torch.argmax(pred, dim=1) == labels).float().mean().item()


def macro_f1(pred, labels):
    pred_labels = torch.argmax(pred, dim=1).detach().cpu().numpy()
    true_labels = labels.detach().cpu().numpy()
    return float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))


def train_one_split(
    args,
    features,
    edge_index,
    num_nodes,
    pretrained_models,
    pretrain_model_multi,
    train_idx,
    train_labels,
    test_idx,
    test_labels,
    all_communities,
    S_lookup,
    split_id: int,
):
    device = features.device
    train_labels = (
        torch.tensor(train_labels, dtype=torch.long).to(device)
        if not isinstance(train_labels, torch.Tensor)
        else train_labels.to(device)
    )
    test_labels = (
        torch.tensor(test_labels, dtype=torch.long).to(device)
        if not isinstance(test_labels, torch.Tensor)
        else test_labels.to(device)
    )
    for model_expert in pretrained_models:
        model_expert.eval()

    model = downprompt(args=args, pretrained_models=pretrained_models, pretrain_model_multi=pretrain_model_multi).to(
        device
    )
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    valid_communities = [c for c in all_communities if len(c) > 1]
    size_one_communities = [c for c in all_communities if len(c) == 1]
    node_to_cluster_id = build_node_to_cluster_map(all_communities) if args.inter_cluster_optimizer else None

    for _epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out, moe_loss, struct_loss = model(
            x=features,
            edge_index=edge_index,
            num_nodes=num_nodes,
            idx=train_idx,
            labels=train_labels,
            is_train=True,
            valid_communities=valid_communities,
            size_one_communities=size_one_communities,
            S_lookup=S_lookup,
            node_to_cluster_id=node_to_cluster_id,
        )
        loss = F.cross_entropy(out, train_labels) + args.moe_weight * moe_loss + args.structure_weight * struct_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out, _, _ = model(
            x=features,
            edge_index=edge_index,
            num_nodes=num_nodes,
            idx=test_idx,
            labels=test_labels,
            is_train=False,
            valid_communities=valid_communities,
            size_one_communities=size_one_communities,
            S_lookup=S_lookup,
            node_to_cluster_id=node_to_cluster_id,
        )
    return accuracy(out, test_labels), macro_f1(out, test_labels)


def run_downstream(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.attack_type == "none":
        data = load_graph(
            args.dataset,
            require_enhanced=True,
            build_node2vec_if_missing=False,
        )
    else:
        data = _torch_load(args.data_path)
    if hasattr(data, "enhanced_x_64") and data.enhanced_x_64 is not None:
        feat = data.enhanced_x_64
    elif hasattr(data, "enhanced_x") and data.enhanced_x is not None:
        feat = data.enhanced_x
    else:
        raise ValueError(f"{args.dataset} needs enhanced_x_64 or enhanced_x (run node_feature_enhance).")
    features, edge_index = feat.to(device), data.edge_index.to(device)
    num_nodes = features.shape[0]
    all_communities = load_communities(args.community_file)

    S_lookup = None
    if args.inter_cluster_optimizer:
        S_precomputed = precompute_appnp_matrix(edge_index, num_nodes, args.appnp_alpha, args.appnp_k, device)
        S_lookup = SparseLookup(S_precomputed, num_nodes)
        del S_precomputed
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pretrained_models = []
    if getattr(args, "expert_paths", None):
        expert_paths = args.expert_paths
    else:
        pretrain_datasets = [normalize_dataset_name(name) for name in args.source_datasets]
        expert_paths = [f"{args.pre_train_model_dir_single}/{name}.pt" for name in pretrain_datasets if os.path.exists(f"{args.pre_train_model_dir_single}/{name}.pt")]

    for path in expert_paths:
        if os.path.exists(path):
            pretrained_models.append(_load_expert_model(path, device))

    if not pretrained_models:
        raise FileNotFoundError(
            "No expert checkpoints could be loaded. "
            "Pass --expert-paths explicitly or ensure --source-datasets experts exist under save_model."
        )

    multi_model = None

    if not args.no_swanlab:
        import swanlab

        swanlab.init(
            project="sa2gfm_downstream",
            config=vars(args),
            requirements_collect=False,
        )

    if args.split_id >= 0:
        split_range = [args.split_id]
    else:
        split_range = range(args.num_splits)

    accs = []
    macro_f1s = []
    for i in split_range:
        split_data = load_few_shot_data(args.down_data_dir, i)
        if "num_classes" in split_data:
            args.num_classes = int(split_data["num_classes"])
        train_idx = split_data["indices"]
        train_labels = split_data["labels"]
        if "test_indices" in split_data and "test_labels" in split_data:
            test_idx = split_data["test_indices"]
            test_labels_tensor = torch.tensor(split_data["test_labels"], dtype=torch.long, device=device)
        else:
            test_idx = list(range(num_nodes - 1000, num_nodes))
            test_labels_tensor = data.y[test_idx].to(device)

        acc, macro_f1_value = train_one_split(
            args,
            features,
            edge_index,
            num_nodes,
            pretrained_models,
            multi_model,
            train_idx,
            train_labels,
            test_idx,
            test_labels_tensor,
            all_communities,
            S_lookup,
            split_id=i,
        )
        accs.append(acc)
        macro_f1s.append(macro_f1_value)
        print(
            f"split {i:4d} | test_acc = {acc:.4f} | macro_f1 = {macro_f1_value:.4f}  "
            f"(train_epochs={args.epochs}, test evaluated once)"
        )
        if not args.no_swanlab:
            import swanlab

            swanlab.log({"split": i, "test_acc": acc, "test_macro_f1": macro_f1_value})

    arr = np.array(accs, dtype=np.float64)
    macro_arr = np.array(macro_f1s, dtype=np.float64)
    print(
        f"\n--- summary over {len(accs)} split(s) ---\n"
        f"mean test_acc = {arr.mean():.4f}  std = {arr.std():.4f}  min = {arr.min():.4f}  max = {arr.max():.4f}\n"
        f"mean macro_f1 = {macro_arr.mean():.4f}  std = {macro_arr.std():.4f}  min = {macro_arr.min():.4f}  max = {macro_arr.max():.4f}"
    )
    if not args.no_swanlab:
        import swanlab

        swanlab.log(
            {
                "summary_mean": float(arr.mean()),
                "summary_std": float(arr.std()),
                "summary_macro_f1_mean": float(macro_arr.mean()),
                "summary_macro_f1_std": float(macro_arr.std()),
                "num_splits_ran": len(accs),
            }
        )
    return {
        "mean_test_acc": float(arr.mean()),
        "std_test_acc": float(arr.std()),
        "mean_macro_f1": float(macro_arr.mean()),
        "std_macro_f1": float(macro_arr.std()),
        "num_splits_ran": len(accs),
    }


def main():
    args = get_args()
    run_downstream(args)


if __name__ == "__main__":
    main()
