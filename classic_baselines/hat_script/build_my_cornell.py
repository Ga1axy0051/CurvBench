"""Build my_cornell from Geom-GCN WebKB Cornell txts (float feature vectors).

Offline: put out1_node_feature_label.txt and out1_graph_edges.txt under a candidate raw dir.
Split 60/20/20 seed=42. PyG: HAT_ALLOW_PYG_CORNELL=1. Output: data/other/my_cornell/.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer


def offline_raw_dir_candidates(root: str):
    return [
        os.path.join(root, "data", "cornell_geom_gcn", "raw"),
        os.path.join(root, "data", "cornell_pyg", "cornell", "raw"),
        os.path.join(root, "data", "cornell_pyg", "Cornell", "raw"),
        os.path.join(root, "data", "cornell", "raw"),
    ]


def load_geom_gcn_offline(raw_dir: str):
    """Parse two Geom-GCN txts; skip header row."""
    feat_path = os.path.join(raw_dir, "out1_node_feature_label.txt")
    edge_path = os.path.join(raw_dir, "out1_graph_edges.txt")
    if not (os.path.isfile(feat_path) and os.path.isfile(edge_path)):
        return None

    node_feats = {}
    labels = {}
    with open(feat_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            feat_str = parts[1]
            lab = int(parts[2])
            feat = np.asarray(
                [float(x) for x in feat_str.split(",")], dtype=np.float32
            )
            node_feats[nid] = feat
            labels[nid] = lab

    edges = []
    with open(edge_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = line.split()[:2]
            edges.append((int(a), int(b)))

    all_ids = set(node_feats.keys())
    for u, v in edges:
        all_ids.add(u)
        all_ids.add(v)

    sorted_ids = sorted(all_ids)
    id2i = {nid: i for i, nid in enumerate(sorted_ids)}
    n = len(sorted_ids)

    dim = next(iter(node_feats.values())).shape[0]
    x = np.zeros((n, dim), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for nid in sorted_ids:
        idx = id2i[nid]
        if nid in node_feats:
            x[idx] = node_feats[nid]
        if nid in labels:
            y[idx] = labels[nid]

    src = np.array([id2i[u] for u, v in edges], dtype=np.int64)
    dst = np.array([id2i[v] for u, v in edges], dtype=np.int64)
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    adj = adj.tocsr()

    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    n_tr = int(0.6 * n)
    n_va = int(0.2 * n)
    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[perm[:n_tr]] = True
    val_mask[perm[n_tr : n_tr + n_va]] = True
    test_mask[perm[n_tr + n_va :]] = True

    return x, y, adj, train_mask, val_mask, test_mask


def try_load_from_candidates(root: str):
    for raw_dir in offline_raw_dir_candidates(root):
        pack = load_geom_gcn_offline(raw_dir)
        if pack is not None:
            return raw_dir, pack
    return None, None


def load_via_pyg(root: str):
    try:
        from torch_geometric.datasets import WebKB
    except ImportError:
        return None

    raw_root = os.path.join(root, "data", "cornell_pyg")
    dataset = WebKB(root=raw_root, name="Cornell")
    data = dataset[0]

    def to_np(t):
        if hasattr(t, "detach"):
            return t.detach().cpu().numpy()
        return np.asarray(t)

    x = to_np(data.x).astype(np.float32)
    y = to_np(data.y).astype(np.int64).ravel()
    ei = to_np(data.edge_index).astype(np.int64)
    n = x.shape[0]
    src, dst = ei[0], ei[1]
    vals = np.ones(src.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((vals, (src, dst)), shape=(n, n), dtype=np.float32)
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    adj = adj.tocsr()

    def pick_split(m, split_idx=0):
        if m is None:
            return None
        m = to_np(m)
        if m.ndim == 2:
            return m[:, split_idx].astype(bool)
        return m.astype(bool)

    train_mask = pick_split(data.train_mask, 0)
    val_mask = pick_split(data.val_mask, 0)
    test_mask = pick_split(data.test_mask, 0)

    if train_mask is None or val_mask is None or test_mask is None:
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        n_tr = int(0.6 * n)
        n_va = int(0.2 * n)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[perm[:n_tr]] = True
        val_mask[perm[n_tr : n_tr + n_va]] = True
        test_mask[perm[n_tr + n_va :]] = True

    return x, y, adj, train_mask, val_mask, test_mask


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    default_hint = os.path.join(root, "data", "cornell_geom_gcn", "raw")

    used_dir, pack = try_load_from_candidates(root)
    source = "offline (geom-gcn txt)"
    if used_dir is not None:
        source = f"offline (geom-gcn txt) — {used_dir}"

    allow_pyg = os.environ.get("HAT_ALLOW_PYG_CORNELL", "").strip() in (
        "1", "true", "True", "yes", "YES",
    )
    if pack is None and allow_pyg:
        print("No local txt; HAT_ALLOW_PYG_CORNELL=1, trying PyG download...")
        pack = load_via_pyg(root)
        source = "torch_geometric WebKB"

    if pack is None:
        print(
            "\nFailed: missing both txts in any of:\n"
            + "\n".join(f"  - {d}" for d in offline_raw_dir_candidates(root))
        )
        print(
            f"\nTry dir: {default_hint}\n"
            "Or PyG: set HAT_ALLOW_PYG_CORNELL=1\n"
        )
        sys.exit(1)

    x, y, adj, train_mask, val_mask, test_mask = pack
    lb = LabelBinarizer()
    label_matrix = lb.fit_transform(y).astype(np.float32)
    if label_matrix.shape[1] == 1:
        label_matrix = np.hstack([1.0 - label_matrix, label_matrix])

    feats = sp.csr_matrix(x)
    dataset_name = "my_cornell"
    out_dir = os.path.join(root, "data", "other", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    sp.save_npz(os.path.join(out_dir, f"{dataset_name}.adj_matrix.npz"), adj)
    sp.save_npz(os.path.join(out_dir, f"{dataset_name}.attr_matrix.npz"), feats)
    np.save(os.path.join(out_dir, f"{dataset_name}.label_matrix.npy"), label_matrix)
    np.save(os.path.join(out_dir, f"{dataset_name}.train_mask.npy"), train_mask)
    np.save(os.path.join(out_dir, f"{dataset_name}.val_mask.npy"), val_mask)
    np.save(os.path.join(out_dir, f"{dataset_name}.test_mask.npy"), test_mask)

    n = x.shape[0]
    print("Source:", source)
    print("Wrote:", out_dir)
    print("nodes:", n, "undirected edges:", adj.nnz // 2, "classes:", label_matrix.shape[1])
    print(
        "split — train:",
        int(train_mask.sum()),
        "val:",
        int(val_mask.sum()),
        "test:",
        int(test_mask.sum()),
    )


if __name__ == "__main__":
    main()
