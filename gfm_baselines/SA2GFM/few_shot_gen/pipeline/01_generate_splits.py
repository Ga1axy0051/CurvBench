#!/usr/bin/env python3
"""
Generate GraphGlue-style few-shot splits for downstream node classification.

Writes: {SA2GFM_DATA_ROOT}/few_shot/{dataset}/{k}shot/split_{i}.pt
Each file contains:
  - support indices/labels
  - validation indices/labels
  - test indices/labels
  - num_classes and split metadata

Sampling is restricted to nodes with non-negative labels. Valid labels are remapped to
contiguous integers from 0 before any split is generated.
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_PIPELINE = Path(__file__).resolve().parent
_SA2GFM = _PIPELINE.parents[2]
_AG = _SA2GFM / "attack_data_gen"
sys.path.insert(0, str(_AG))

from lib.data_utils import flatten_node_labels, load_graph, normalize_dataset_name, prepare_few_shot_labels
from lib.paths import paths


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_label_indices(remapped_labels: torch.Tensor, n_way: int):
    label_to_positions: defaultdict = defaultdict(list)
    for pos, label in enumerate(remapped_labels.tolist()):
        label_to_positions[int(label)].append(pos)

    unique_labels = sorted(label_to_positions.keys())
    if n_way > 0 and n_way < len(unique_labels):
        selected_labels = unique_labels[:n_way]
        label_to_positions = {k: v for k, v in label_to_positions.items() if k in selected_labels}
    else:
        selected_labels = unique_labels

    return label_to_positions, selected_labels


def generate_few_shot_split(label_to_positions, selected_labels, k_shot: int, num_val: float):
    support_positions = []
    remaining_positions = []

    for label in selected_labels:
        pool = np.asarray(label_to_positions[label], dtype=np.int64)
        if len(pool) < k_shot:
            raise ValueError(f"Class {label} has only {len(pool)} samples, but k_shot={k_shot}.")

        np.random.shuffle(pool)
        support_positions.extend(pool[:k_shot].tolist())
        remaining_positions.extend(pool[k_shot:].tolist())

    remaining_positions = np.asarray(remaining_positions, dtype=np.int64)
    np.random.shuffle(remaining_positions)
    val_size = int(len(remaining_positions) * num_val)
    val_positions = remaining_positions[:val_size].tolist()
    test_positions = remaining_positions[val_size:].tolist()
    return support_positions, val_positions, test_positions


def parse_args():
    p = argparse.ArgumentParser(description="Generate few-shot splits under data/few_shot/")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--k-shot", type=int, dest="k_shot", choices=[1, 5], required=True)
    p.add_argument("--n-splits", type=int, default=20, help="Number of split_*.pt files (>= downstream --num-splits)")
    p.add_argument("--n-way", type=int, default=0, help="0 = all classes; else first n-way labels only")
    p.add_argument("--num-val", type=float, default=0.1, help="Validation ratio within the remaining pool after k-shot support sampling")
    p.add_argument("--seed", type=int, default=39)
    p.add_argument(
        "--write-example",
        action="store_true",
        help="Write example_structure.txt under dataset few_shot folder",
    )
    return p.parse_args()


def generate_splits_for_dataset(
    dataset: str,
    k_shot: int,
    n_splits: int = 20,
    n_way: int = 0,
    num_val: float = 0.1,
    seed: int = 39,
    write_example: bool = False,
) -> Path:
    set_seed(seed)
    dataset = normalize_dataset_name(dataset)

    data = load_graph(dataset)
    labels = flatten_node_labels(data.y)
    valid_node_idx, remapped_labels = prepare_few_shot_labels(data, dataset)
    valid_orig_labels = labels[valid_node_idx]

    label_to_positions, selected_labels = get_label_indices(remapped_labels, n_way=n_way)
    selected_orig_labels = sorted(
        {int(valid_orig_labels[pos].item()) for label in selected_labels for pos in label_to_positions[label]}
    )

    print(f"Dataset: {dataset}")
    print(f"Total nodes: {int(data.num_nodes)}  valid labeled nodes: {int(valid_node_idx.numel())}")
    print(f"Ignored unlabeled / negative-label nodes: {int(data.num_nodes) - int(valid_node_idx.numel())}")
    print(f"Num classes: {len(selected_labels)}  remapped_labels={selected_labels}  original_labels={selected_orig_labels}")
    for lb in selected_labels:
        print(f"  - label {lb}: {len(label_to_positions[lb])} samples")

    out_root = paths.few_shot_dir / dataset
    shot_dir = out_root / f"{k_shot}shot"
    shot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating {n_splits} {k_shot}-shot splits -> {shot_dir}")

    for split_idx in tqdm(range(n_splits)):
        support_positions, val_positions, test_positions = generate_few_shot_split(
            label_to_positions, selected_labels, k_shot, num_val
        )
        support_indices = valid_node_idx[support_positions].cpu().tolist()
        support_labels = remapped_labels[support_positions].cpu().tolist()
        val_indices = valid_node_idx[val_positions].cpu().tolist()
        val_labels = remapped_labels[val_positions].cpu().tolist()
        test_indices = valid_node_idx[test_positions].cpu().tolist()
        test_labels = remapped_labels[test_positions].cpu().tolist()
        split_data = {
            "indices": support_indices,
            "labels": support_labels,
            "val_indices": val_indices,
            "val_labels": val_labels,
            "test_indices": test_indices,
            "test_labels": test_labels,
            "num_classes": len(selected_labels),
            "class_values": selected_orig_labels,
            "k_shot": k_shot,
            "split_id": split_idx,
        }
        torch.save(split_data, shot_dir / f"split_{split_idx}.pt")

    print(f"Done: {n_splits} files")

    if write_example:
        example_file = out_root / "example_structure.txt"
        example_file.write_text(
            f"Dataset: {dataset}\n"
            f"Few-shot: {k_shot}-shot, {len(selected_labels)}-way\n\n"
            "Each split_*.pt:\n"
            "  'indices': support node indices\n"
            "  'labels': support labels aligned with indices\n"
            "  'val_indices' / 'val_labels': validation nodes\n"
            "  'test_indices' / 'test_labels': test nodes\n"
            "  'num_classes': output dimension for downstream classifier\n\n"
            f"Example load: torch.load('{shot_dir}/split_0.pt')\n",
            encoding="utf-8",
        )
        print(f"Wrote: {example_file}")
    return shot_dir


def main():
    args = parse_args()
    generate_splits_for_dataset(
        dataset=args.dataset,
        k_shot=args.k_shot,
        n_splits=args.n_splits,
        n_way=args.n_way,
        num_val=args.num_val,
        seed=args.seed,
        write_example=args.write_example,
    )


if __name__ == "__main__":
    main()
