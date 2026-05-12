#!/usr/bin/env python3
"""
Detect disjoint communities on an undirected graph and save
`{SA2GFM_DATA_ROOT}/communities/{dataset}_communities.pt`
compatible with `down_all_sparse_multi` (key `communities`: list of node-id lists).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import torch

# Reuse SA2GFM path + graph loader from attack_data_gen bundle
_AG = Path(__file__).resolve().parents[2] / "attack_data_gen"
sys.path.insert(0, str(_AG))
from lib.data_utils import load_graph, normalize_dataset_name
from lib.paths import paths


def edge_index_to_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    ei = edge_index.cpu().numpy()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        if u != v:
            g.add_edge(u, v)
    return g


def partition_to_communities(partition: list) -> list[list[int]]:
    """Stable list-of-lists: each community sorted, communities ordered by min node id."""
    comms = [sorted(int(x) for x in block) for block in partition]
    comms.sort(key=lambda c: c[0] if c else 0)
    return comms


def detect_communities(g: nx.Graph, method: str, seed: int, resolution: float):
    if method == "louvain":
        try:
            part = nx.community.louvain_communities(g, resolution=resolution, seed=seed)
        except AttributeError as e:
            raise RuntimeError(
                "Louvain requires networkx>=3.2 with nx.community.louvain_communities. "
                "Upgrade networkx or choose --method greedy_modularity."
            ) from e
        return partition_to_communities(part)

    if method == "greedy_modularity":
        part = nx.community.greedy_modularity_communities(g, resolution=resolution)
        return partition_to_communities(part)

    if method == "label_propagation":
        part = nx.community.label_propagation_communities(g)
        return partition_to_communities(part)

    raise ValueError(f"unknown method: {method}")


def validate_partition(comms: list[list[int]], num_nodes: int) -> None:
    seen = []
    for c in comms:
        seen.extend(c)
    if len(seen) != len(set(seen)):
        raise ValueError("partition has duplicate node ids")
    if set(seen) != set(range(num_nodes)):
        missing = set(range(num_nodes)) - set(seen)
        extra = set(seen) - set(range(num_nodes))
        raise ValueError(f"bad cover: missing={len(missing)} extra={len(extra)}")


def run_detection(dataset: str, method: str = "louvain", seed: int = 42, resolution: float = 1.0) -> Path:
    dataset = normalize_dataset_name(dataset)
    data = load_graph(dataset)
    n = int(data.num_nodes) if hasattr(data, "num_nodes") else data.x.shape[0]
    g = edge_index_to_graph(data.edge_index, n)
    comms = detect_communities(g, method, seed, resolution)
    validate_partition(comms, n)

    paths.communities_dir.mkdir(parents=True, exist_ok=True)
    out = paths.communities_dir / f"{dataset}_communities.pt"
    payload = {
        "communities": comms,
        "meta": {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "resolution": resolution,
            "num_nodes": n,
            "num_communities": len(comms),
        },
    }
    torch.save(payload, out)
    print(f"Saved {out} ({len(comms)} communities)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="louvain",
        choices=["louvain", "greedy_modularity", "label_propagation"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Louvain / greedy_modularity resolution (NetworkX semantics).",
    )
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    run_detection(args.dataset, args.method, args.seed, args.resolution)


if __name__ == "__main__":
    main()
