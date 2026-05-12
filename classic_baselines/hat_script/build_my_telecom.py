import os
import numpy as np
import scipy.sparse as sp


def main() -> None:
    """
    Convert data/telecom/telecom_graph.pt (PyG Data) into utils/process.py `my_*` format:
      data/other/my_telecom/
        my_telecom.adj_matrix.npz
        my_telecom.attr_matrix.npz
        my_telecom.label_matrix.npy
        my_telecom.train_mask.npy
        my_telecom.val_mask.npy
        my_telecom.test_mask.npy

    Random split: 60/20/20 with seed=42.
    """

    root = os.path.dirname(os.path.abspath(__file__))
    in_path = os.path.join(root, "data", "telecom", "telecom_graph.pt")
    dataset = "my_telecom"
    out_dir = os.path.join(root, "data", "other", dataset)
    os.makedirs(out_dir, exist_ok=True)

    import torch  # type: ignore

    try:
        from torch_geometric.data import Data  # type: ignore

        try:
            torch.serialization.add_safe_globals([Data])
        except Exception:
            pass
    except Exception:
        pass

    data = torch.load(in_path, map_location="cpu", weights_only=False)

    # ---- extract ----
    x = data.x
    y = data.y
    edge_index = data.edge_index

    if x is None or y is None or edge_index is None:
        raise ValueError("telecom_graph.pt missing one of x/y/edge_index")

    x = x.detach().cpu()
    y = y.detach().cpu()
    edge_index = edge_index.detach().cpu()

    n = int(x.shape[0])
    f = int(x.shape[1])
    c = int(y.shape[1])

    # ---- adjacency ----
    # edge_index: [2, E]
    src = edge_index[0].numpy().astype(np.int64, copy=False)
    dst = edge_index[1].numpy().astype(np.int64, copy=False)
    vals = np.ones(src.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((vals, (src, dst)), shape=(n, n), dtype=np.float32)

    # Make undirected and binarize (safe for GAT-style bias preprocessing)
    adj = adj + adj.T
    adj.data[:] = 1.0
    adj = adj.tocsr()

    # ---- features ----
    # process.preprocess_features expects a scipy sparse matrix
    feats = sp.csr_matrix(x.numpy().astype(np.float32, copy=False))

    # ---- labels ----
    label_matrix = y.numpy().astype(np.float32, copy=False)

    # ---- random masks ----
    rng = np.random.RandomState(42)
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

    # ---- save ----
    sp.save_npz(os.path.join(out_dir, f"{dataset}.adj_matrix.npz"), adj)
    sp.save_npz(os.path.join(out_dir, f"{dataset}.attr_matrix.npz"), feats)
    np.save(os.path.join(out_dir, f"{dataset}.label_matrix.npy"), label_matrix)
    np.save(os.path.join(out_dir, f"{dataset}.train_mask.npy"), train_mask)
    np.save(os.path.join(out_dir, f"{dataset}.val_mask.npy"), val_mask)
    np.save(os.path.join(out_dir, f"{dataset}.test_mask.npy"), test_mask)

    print("Wrote:", out_dir)
    print("adj:", adj.shape, "nnz:", adj.nnz)
    print("features:", feats.shape)
    print("labels:", label_matrix.shape, "classes:", c)
    print("split:", int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum()))


if __name__ == "__main__":
    main()

