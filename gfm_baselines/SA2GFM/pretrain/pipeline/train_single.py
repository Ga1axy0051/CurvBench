#!/usr/bin/env python3
"""
Single-dataset contrastive pretraining (JointContrastiveModel).
Does NOT implement multi-graph joint pretrain (--joint_pretrain); that path is intentionally omitted.

MoE downstream loads experts by name list get_pretrain_datasets(target)—usually *other* graphs, not target.
Prefer scripts/run_experts_for_downstream.sh <target> to train all needed experts.

Reads: graph data resolved by the unified loader, requiring enhanced_x / enhanced_x_64
Writes: {SA2GFM_DATA_ROOT}/save_model/{dataset}.pt  (state_dict + metadata checkpoint)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

_PIPELINE = Path(__file__).resolve().parent  # .../pretrain/pipeline
sys.path.insert(0, str(_PIPELINE))
_SA2GFM = _PIPELINE.parents[2]  # .../SA2GFM
_AG = _SA2GFM / "attack_data_gen"
sys.path.insert(0, str(_AG))

from data_utils import get_negative_samples, load_dataset_pt, sparse_mx_to_torch_sparse_tensor
from lib.data_utils import normalize_dataset_name
from lib.paths import paths
from model import JointContrastiveModel


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="SA2GFM single-dataset pretrain (no joint merge)")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seed", type=int, default=39)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--hid_units", type=int, default=256)
    p.add_argument("--out_channels", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--l2_coef", type=float, default=0.0)
    p.add_argument("--nb_epochs", type=int, default=3000)
    p.add_argument("--patience", type=int, default=200)
    p.add_argument("--eval_steps", type=int, default=10)
    p.add_argument("--neg_samples", type=int, default=50)
    p.add_argument("--kl_weight", type=float, default=0.0)
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Default: {save_model_dir}/{dataset}.pt",
    )
    p.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    return p.parse_args()


def _build_checkpoint_payload(
    model: JointContrastiveModel,
    dataset: str,
    in_channels: int,
    hid_units: int,
    out_channels: int,
    num_layers: int,
    dropout: float,
    seed: int,
    best_epoch: int | None = None,
    best_loss: float | None = None,
):
    return {
        "checkpoint_type": "joint_contrastive_expert",
        "dataset": dataset,
        "model_class": "JointContrastiveModel",
        "model_kwargs": {
            "in_channels": int(in_channels),
            "hidden_channels": int(hid_units),
            "out_channels": int(out_channels),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        },
        "state_dict": model.state_dict(),
        "seed": int(seed),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_loss": None if best_loss is None else float(best_loss),
    }


def train_expert(
    dataset: str,
    seed: int = 39,
    gpu: int = 0,
    hid_units: int = 256,
    out_channels: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-5,
    l2_coef: float = 0.0,
    nb_epochs: int = 3000,
    patience: int = 200,
    eval_steps: int = 10,
    neg_samples: int = 50,
    kl_weight: float = 0.0,
    output: str = "",
    no_swanlab: bool = False,
):
    dataset = normalize_dataset_name(dataset)
    set_seed(seed)
    paths.save_model_dir.mkdir(parents=True, exist_ok=True)

    features_np, adj_sp, num_nodes = load_dataset_pt(dataset)
    print(f"Loaded {dataset}: nodes={num_nodes}, feat_dim={features_np.shape[1]}, edges~={adj_sp.nnz // 2}")

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    neg = torch.from_numpy(get_negative_samples(adj_sp, num_nodes, neg_samples)).to(device)

    features = torch.FloatTensor(features_np).to(device)
    adj = sparse_mx_to_torch_sparse_tensor(adj_sp).to(device)

    model = JointContrastiveModel(
        in_channels=features.shape[1],
        hidden_channels=hid_units,
        out_channels=out_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    out_path = Path(output) if output else paths.save_model_dir / f"{dataset}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _build_checkpoint_payload(
            model=model,
            dataset=dataset,
            in_channels=features.shape[1],
            hid_units=hid_units,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            seed=seed,
        ),
        out_path,
    )
    print(f"Initial checkpoint -> {out_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    if not no_swanlab:
        import swanlab

        swanlab.init(
            project="sa2gfm_pretrain",
            config={
                "dataset": dataset,
                "seed": seed,
                "gpu": gpu,
                "hid_units": hid_units,
                "out_channels": out_channels,
                "num_layers": num_layers,
                "dropout": dropout,
                "lr": lr,
                "l2_coef": l2_coef,
                "nb_epochs": nb_epochs,
                "patience": patience,
                "eval_steps": eval_steps,
                "neg_samples": neg_samples,
                "kl_weight": kl_weight,
                "output": str(out_path),
            },
            requirements_collect=False,
        )

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in tqdm(range(1, nb_epochs + 1), desc=f"pretrain:{dataset}"):
        model.train()
        optimizer.zero_grad()
        contrastive_loss, kl_loss = model([features], [adj], neg)
        total_loss = contrastive_loss + kl_weight * kl_loss
        if torch.isnan(total_loss):
            continue
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if epoch % eval_steps != 0:
            continue

        print(
            f"epoch {epoch:5d} | L_con={contrastive_loss.item():.4f} L_kl={kl_loss.item():.4f} L={total_loss.item():.4f}"
        )
        if not no_swanlab:
            import swanlab

            swanlab.log(
                {
                    "epoch": epoch,
                    "contrastive_loss": contrastive_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "total_loss": total_loss.item(),
                }
            )

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                _build_checkpoint_payload(
                    model=model,
                    dataset=dataset,
                    in_channels=features.shape[1],
                    hid_units=hid_units,
                    out_channels=out_channels,
                    num_layers=num_layers,
                    dropout=dropout,
                    seed=seed,
                    best_epoch=best_epoch,
                    best_loss=best_loss,
                ),
                out_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= max(1, patience // max(1, eval_steps)):
            print(f"Early stop at epoch {epoch} (best {best_epoch}, loss {best_loss:.4f})")
            break

    print(f"Done. Best loss {best_loss:.4f} @ epoch {best_epoch}. Saved: {out_path}")
    return out_path


def train():
    args = parse_args()
    train_expert(
        dataset=args.dataset,
        seed=args.seed,
        gpu=args.gpu,
        hid_units=args.hid_units,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        l2_coef=args.l2_coef,
        nb_epochs=args.nb_epochs,
        patience=args.patience,
        eval_steps=args.eval_steps,
        neg_samples=args.neg_samples,
        kl_weight=args.kl_weight,
        output=args.output,
        no_swanlab=args.no_swanlab,
    )


if __name__ == "__main__":
    train()
