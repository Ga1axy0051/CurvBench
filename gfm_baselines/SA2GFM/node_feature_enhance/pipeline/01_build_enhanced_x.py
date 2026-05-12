#!/usr/bin/env python3
"""
Build enhanced_x_64 = concat( SVD(x, 32), SVD(BERT(template(node, community, neighbors)), 32) ).

Input graph loading is unified with GraphGlue-style local dataset parsing, so the dataset can come
from a canonical `.pt`, a local imported directory (`data.pt`, `x/y/edge_index.npy`), or supported
raw files such as Actor/Cornell/Disease custom formats.

Prerequisites:
  - graph data with at least `.x` and `.edge_index`
  - {SA2GFM_DATA_ROOT}/communities/{dataset}_communities.pt : dict with key 'communities'

Output:
  - torch.save updated Data to --output (default: canonical ori/{dataset}.pt)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Reuse SA2GFM path resolution
_AG = Path(__file__).resolve().parents[2] / "attack_data_gen"
sys.path.insert(0, str(_AG))
from lib.data_utils import load_graph, normalize_dataset_name  # noqa: E402
from lib.paths import paths  # noqa: E402


def _load_pt(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def communities_to_cluster_id(communities: list[list[int]], num_nodes: int) -> list[int]:
    """Each node's community index in the partition (0..C-1)."""
    cid = [-1] * num_nodes
    for c_idx, nodes in enumerate(communities):
        for n in nodes:
            n = int(n)
            if 0 <= n < num_nodes:
                cid[n] = c_idx
    return cid


def build_neighbor_lists(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True) -> list[list[int]]:
    ei = edge_index.cpu()
    if undirected:
        ei = to_undirected(ei, num_nodes=num_nodes)
    src, dst = ei[0].tolist(), ei[1].tolist()
    nbrs: list[set[int]] = [set() for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        u, v = int(u), int(v)
        if u != v:
            nbrs[u].add(v)
    return [sorted(s) for s in nbrs]


def default_node_text_template(
    node_id: int,
    cluster_id: int,
    neighbor_ids: list[int],
    max_neighbors: int,
) -> str:
    """English template: cluster id + truncated neighbor list for BERT."""
    if cluster_id < 0:
        cluster_str = "unknown"
    else:
        cluster_str = str(cluster_id)
    nb = neighbor_ids[:max_neighbors]
    if len(neighbor_ids) > max_neighbors:
        tail = f", {len(neighbor_ids)} neighbors in total"
    else:
        tail = ""
    nb_str = ", ".join(str(j) for j in nb) if nb else "none"
    return f"Node {node_id} belongs to cluster {cluster_str}; neighbors: {nb_str}{tail}."


def svd_project(mat: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
    """TruncatedSVD on dense float matrix; shape (N, n_components)."""
    n = min(n_components, mat.shape[1], max(1, mat.shape[0] - 1))
    if n < n_components:
        out = np.zeros((mat.shape[0], n_components), dtype=np.float32)
        if n > 0:
            svd = TruncatedSVD(n_components=n, random_state=random_state)
            out[:, :n] = svd.fit_transform(mat.astype(np.float32))
        return out
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    return svd.fit_transform(mat.astype(np.float32)).astype(np.float32)


@torch.inference_mode()
def bert_encode_texts(
    texts: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_vecs: list[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BERT encode"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc).last_hidden_state  # (B, L, H)
        mask = enc["attention_mask"].unsqueeze(-1)
        summed = (out * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        pooled = (summed / counts).float().cpu().numpy()
        all_vecs.append(pooled)
    return np.concatenate(all_vecs, axis=0)


def build_enhanced_x(
    dataset: str,
    output: str = "",
    svd_dim: int = 32,
    text_svd_dim: int = 32,
    bert: str = "bert-base-uncased",
    batch_size: int = 16,
    device: str = "cuda",
    seed: int = 42,
    max_neighbors_in_text: int = 48,
) -> Path:
    dataset = normalize_dataset_name(dataset)
    comm_path = paths.communities_dir / f"{dataset}_communities.pt"
    if not comm_path.is_file():
        raise FileNotFoundError(
            f"{comm_path}\nGenerate communities first (e.g. SA2GFM/community_detection)."
        )

    data = load_graph(dataset, require_features=True, build_node2vec_if_missing=False)
    if isinstance(data, dict):
        data = Data.from_dict(data)

    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Data must have attribute x (raw node features).")
    x = data.x.float()
    num_nodes = int(x.shape[0])
    edge_index = data.edge_index

    comm_blob = _load_pt(comm_path)
    communities = comm_blob["communities"]
    cluster_ids = communities_to_cluster_id(communities, num_nodes)
    nbr_lists = build_neighbor_lists(edge_index, num_nodes, undirected=True)

    texts = [
        default_node_text_template(
            i, cluster_ids[i], nbr_lists[i], max_neighbors_in_text
        )
        for i in range(num_nodes)
    ]

    x_np = x.cpu().numpy()
    struct32 = svd_project(x_np, svd_dim, seed)

    use_cuda = torch.cuda.is_available() and device != "cpu"
    dev = torch.device("cuda" if use_cuda else "cpu")
    bert_h = bert_encode_texts(texts, bert, dev, batch_size)
    text32 = svd_project(bert_h, text_svd_dim, seed + 1)

    if struct32.shape[1] != svd_dim or text32.shape[1] != text_svd_dim:
        raise RuntimeError("SVD dimension mismatch after projection.")

    enhanced = np.concatenate([struct32, text32], axis=1)
    enhanced_t = torch.from_numpy(enhanced).to(dtype=torch.float32)

    out = data.clone()
    out.svd_x = torch.from_numpy(struct32).float()
    out.text_svd_embedding = torch.from_numpy(text32).float()
    out.enhanced_x_64 = enhanced_t
    out.enhanced_x = enhanced_t.clone()

    out_path = Path(output) if output else paths.graph_ori_dir / f"{dataset}.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"Saved {out_path} | enhanced_x_64 shape = {tuple(out.enhanced_x_64.shape)}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="If empty: overwrite/save canonical {ori}/{dataset}.pt",
    )
    parser.add_argument("--svd-dim", type=int, default=32)
    parser.add_argument("--text-svd-dim", type=int, default=32)
    parser.add_argument(
        "--bert",
        type=str,
        default="bert-base-uncased",
        help="Hugging Face model id (must match template language; default English)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-neighbors-in-text", type=int, default=48)
    args = parser.parse_args()
    args.dataset = normalize_dataset_name(args.dataset)

    build_enhanced_x(
        dataset=args.dataset,
        output=args.output,
        svd_dim=args.svd_dim,
        text_svd_dim=args.text_svd_dim,
        bert=args.bert,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        max_neighbors_in_text=args.max_neighbors_in_text,
    )


if __name__ == "__main__":
    main()
