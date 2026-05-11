
from __future__ import print_function

import argparse
import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import networkx as nx

try:
    import torch
except ImportError:
    torch = None


def _as_numpy(x):
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _edge_index_to_pairs(ei):
    ei = np.asarray(ei)
    if ei.ndim != 2 or ei.shape[0] not in (1, 2):
        raise ValueError("edge_index must be [2, E] or [E, 2]")
    if ei.shape[0] == 2:
        return ei[0], ei[1]
    return ei[:, 0], ei[:, 1]


def align_y_to_nodes(y, n_nodes):
    """Align y to length-n_nodes int labels (vector, one-hot, or flattened N*C)."""
    if y is None:
        return None
    y = _as_numpy(y)
    if y.ndim == 2:
        if y.shape[0] != n_nodes:
            raise ValueError(
                "y rows {} != n_nodes {}".format(y.shape[0], n_nodes)
            )
        if y.shape[1] == 1:
            return y[:, 0].astype(np.int64)
        return np.argmax(y, axis=1).astype(np.int64)
    y = y.reshape(-1)
    if y.shape[0] == n_nodes:
        return y.astype(np.int64)
    if y.shape[0] % n_nodes == 0:
        k = y.shape[0] // n_nodes
        y2 = y.reshape(n_nodes, k)
        if k == 1:
            return y2.ravel().astype(np.int64)
        print(
            "note: treating y as (N, {}), taking argmax per row".format(k),
            file=sys.stderr,
        )
        return np.argmax(y2, axis=1).astype(np.int64)
    raise ValueError(
        "label length {} cannot align to n_nodes {} (not divisible)".format(y.shape[0], n_nodes)
    )


def from_pt(path):
    if torch is None:
        raise RuntimeError("torch required to read .pt")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    # PyG Data
    if hasattr(obj, "edge_index") and hasattr(obj, "x"):
        x = _as_numpy(obj.x)
        ei = _as_numpy(obj.edge_index)
        y = getattr(obj, "y", None)
        if y is not None:
            y = _as_numpy(y)
        row, col = _edge_index_to_pairs(ei)
        return x, row, col, y
    if isinstance(obj, dict):
        if "x" in obj and "edge_index" in obj:
            x = _as_numpy(obj["x"])
            ei = _as_numpy(obj["edge_index"])
            y = obj.get("y")
            if y is not None:
                y = _as_numpy(y)
            row, col = _edge_index_to_pairs(ei)
            return x, row, col, y
    raise ValueError(
        ".pt must be PyG Data or dict with x and edge_index"
    )


def from_gpickle(path):
    try:
        G = nx.read_gpickle(path)
    except Exception:
        with open(path, "rb") as f:
            G = pickle.load(f)
    if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        raise ValueError("gpickle payload is not a NetworkX graph")

    nodes = list(G.nodes())
    try:
        nodes = sorted(nodes, key=lambda n: int(n))
    except Exception:
        nodes = sorted(nodes, key=str)
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)

    feat_keys = ("feat", "feature", "x", "features", "attr")
    X = None
    for key in feat_keys:
        vals = []
        ok = True
        for node in nodes:
            d = G.nodes[node]
            if key not in d:
                ok = False
                break
            vals.append(np.asarray(d[key]).ravel())
        if ok and len(vals) == n:
            X = np.stack(vals, axis=0)
            break
    if X is None:
        print("warn: no node features; using identity I_N", file=sys.stderr)
        X = sp.eye(n).toarray()

    label_keys = ("label", "y", "labels", "class")
    y = None
    for key in label_keys:
        vals = []
        ok = True
        for node in nodes:
            d = G.nodes[node]
            if key not in d:
                ok = False
                break
            vals.append(int(d[key]))
        if ok and len(vals) == n:
            y = np.asarray(vals, dtype=np.int64)
            break
    if y is None:
        print("warn: no node labels; labels.npy all zeros", file=sys.stderr)
        y = np.zeros(n, dtype=np.int64)

    rows, cols = [], []
    for u, v in G.edges():
        if u in idx and v in idx:
            rows.append(idx[u])
            cols.append(idx[v])
    row, col = np.asarray(rows), np.asarray(cols)
    return X, row, col, y


def save_gcn_format(out_dir, x, row, col, y):
    os.makedirs(out_dir, exist_ok=True)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("labels len {} != n {}".format(y.shape[0], n))

    edges_path = os.path.join(out_dir, "edges.csv")
    seen = set()
    with open(edges_path, "w") as f:
        for i in range(len(row)):
            u, v = int(row[i]), int(col[i])
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            f.write("{},{}\n".format(a, b))

    if sp.issparse(x):
        feats = x.tocsr()
    else:
        feats = sp.csr_matrix(np.asarray(x, dtype=np.float64))
    sp.save_npz(os.path.join(out_dir, "feats.npz"), feats)
    np.save(os.path.join(out_dir, "labels.npy"), y)
    print("wrote", edges_path)
    print("wrote", os.path.join(out_dir, "feats.npz"), "shape=", feats.shape)
    print("wrote", os.path.join(out_dir, "labels.npy"), "shape=", y.shape)


def _default_telecom_data_dir():
    _here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(_here)
    return os.path.join(repo, "data", "telecom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default=_default_telecom_data_dir(),
        help="input/output dir with .pt or .gpickle (default: repo data/telecom/)",
    )
    args = ap.parse_args()
    d = args.dir

    pt_names = ["telecom_graph.pt", "telcom_graph.pt", "Telecom_graph.pt"]
    gpickle_names = ["TeleGraph.gpickle", "telegraph.gpickle"]

    pt_path = None
    for name in pt_names:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            pt_path = p
            break

    gp_path = None
    for name in gpickle_names:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            gp_path = p
            break

    if pt_path:
        print("read", pt_path)
        x, row, col, y = from_pt(pt_path)
        n_nodes = int(np.asarray(x).shape[0])
        if y is not None:
            y = align_y_to_nodes(y, n_nodes)
        if y is None:
            y = np.zeros(x.shape[0], dtype=np.int64)
            print("warn: .pt has no labels; zeros", file=sys.stderr)
    elif gp_path:
        print("read", gp_path)
        x, row, col, y = from_gpickle(gp_path)
    else:
        print(
            "missing telecom_graph.pt / telcom_graph.pt / TeleGraph.gpickle in:",
            d,
            file=sys.stderr,
        )
        sys.exit(1)

    if len(row) == 0:
        print("error: no edges", file=sys.stderr)
        sys.exit(1)

    save_gcn_format(d, x, row, col, y)
    print("done; use gcn --dataset telecom_nc or telecom_lp")


if __name__ == "__main__":
    main()
