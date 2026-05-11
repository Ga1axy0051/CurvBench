"""Build my_twitter from data/twitter/higgs-retweet_network.edgelist (source target [weight]).

Edges only: random sparse feats, degree-tertile labels for a runnable NC baseline.
Remap node ids to 0..N-1. Run: python build_my_twitter.py then hat_new.py -dataset my_twitter.
"""
import os
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(root, "data", "twitter", "higgs-retweet_network.edgelist")
    dataset = "my_twitter"
    out_dir = os.path.join(root, "data", "other", dataset)
    os.makedirs(out_dir, exist_ok=True)

    feat_dim = 32
    seed = 42
    rng = np.random.RandomState(seed)

    edges_raw = []
    nodes_set = set()
    with open(in_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue
            edges_raw.append((u, v))
            nodes_set.add(u)
            nodes_set.add(v)

    node_list = sorted(nodes_set)
    id2i = {nid: i for i, nid in enumerate(node_list)}
    n = len(node_list)
    print("Nodes:", n, "edges (rows):", len(edges_raw))

    src = np.array([id2i[u] for u, v in edges_raw], dtype=np.int64)
    dst = np.array([id2i[v] for u, v in edges_raw], dtype=np.int64)
    data = np.ones(len(src), dtype=np.float32)
    adj = sp.coo_matrix((data, (src, dst)), shape=(n, n), dtype=np.float32)
    # undirected + unweighted
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj.eliminate_zeros()
    adj = adj.tocsr()

    # random sparse features
    feats = sp.random(n, feat_dim, density=0.05, format="csr", dtype=np.float32, random_state=rng)
    feats.data[:] = rng.randn(feats.nnz).astype(np.float32)

    deg = np.asarray(adj.sum(axis=1)).astype(np.float64).ravel()
    q0, q1 = np.quantile(deg, [1.0 / 3.0, 2.0 / 3.0])
    y_int = np.zeros(n, dtype=np.int64)
    y_int[deg <= q0] = 0
    y_int[(deg > q0) & (deg <= q1)] = 1
    y_int[deg > q1] = 2
    lb = LabelBinarizer()
    label_matrix = lb.fit_transform(y_int).astype(np.float32)
    if label_matrix.shape[1] == 1:
        label_matrix = np.hstack([1.0 - label_matrix, label_matrix])

    idx = rng.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    sp.save_npz(os.path.join(out_dir, f"{dataset}.adj_matrix.npz"), adj)
    sp.save_npz(os.path.join(out_dir, f"{dataset}.attr_matrix.npz"), feats)
    np.save(os.path.join(out_dir, f"{dataset}.label_matrix.npy"), label_matrix)
    np.save(os.path.join(out_dir, f"{dataset}.train_mask.npy"), train_mask)
    np.save(os.path.join(out_dir, f"{dataset}.val_mask.npy"), val_mask)
    np.save(os.path.join(out_dir, f"{dataset}.test_mask.npy"), test_mask)

    print("Wrote:", out_dir)
    print("adj:", adj.shape, "nnz:", adj.nnz)
    print("features:", feats.shape, "labels:", label_matrix.shape)
    print("split:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))


if __name__ == "__main__":
    main()
