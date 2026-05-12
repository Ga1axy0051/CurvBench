"""
Disease NC dataset under data/disease/.

  Required: disease_nc.edges.csv (two ints per line, comma-separated, no header).

  Optional labels (same node order as sorted node ids):
    y.npy / disease_y.npy, or disease_nc.labels.csv / labels.csv
  Optional features: x.npy / disease_x.npy; else identity.

  If no label files: shortest-path first-hop heuristic from min node id (WARNING).
"""
import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def _load_edges_csv(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(" ", "").split(",")
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges.append((u, v))
    return edges


def _try_load_y_numpy(root, n_nodes, node_ids_sorted):
    for name in ("y.npy", "disease_y.npy", "labels.npy"):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        y = np.load(p)
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != n_nodes:
            raise ValueError(
                "%s length %d != num_nodes %d" % (name, y.shape[0], n_nodes)
            )
        y = y.astype(np.int64)
        uniq = np.unique(y)
        if y.min() < 0:
            m = {v: i for i, v in enumerate(uniq)}
            y = np.array([m[v] for v in y], dtype=np.int64)
        return torch.tensor(y, dtype=torch.long), "file:%s" % name
    return None, None


def _try_load_y_csv(root, node_ids_sorted):
    import csv

    for name in ("disease_nc.labels.csv", "labels.csv"):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        rows = []
        with open(p, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                rows.append(row)
        if not rows:
            continue
        ncol = len(rows[0])
        if ncol >= 2:
            m = {}
            for row in rows:
                if len(row) < 2:
                    continue
                m[int(row[0])] = int(row[1])
            y_list = []
            for nid in node_ids_sorted:
                if nid not in m:
                    raise ValueError("labels CSV missing node_id=%s" % nid)
                y_list.append(m[nid])
            y = np.array(y_list, dtype=np.int64)
        else:
            if len(rows) != len(node_ids_sorted):
                raise ValueError(
                    "single-column labels rows %d != num_nodes %d" % (len(rows), len(node_ids_sorted))
                )
            y = np.array([int(r[0]) for r in rows], dtype=np.int64)
        uniq = np.unique(y)
        m = {v: i for i, v in enumerate(sorted(uniq))}
        y = np.array([m[v] for v in y], dtype=np.int64)
        return torch.tensor(y, dtype=torch.long), "file:%s" % name
    return None, None


def _try_load_x_numpy(root, n_nodes):
    for name in ("x.npy", "disease_x.npy", "features.npy"):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        x = np.load(p)
        if x.shape[0] != n_nodes:
            raise ValueError("%s dim0 %d != num_nodes %d" % (name, x.shape[0], n_nodes))
        return torch.tensor(x, dtype=torch.float32), "file:%s" % name
    return None, None


def _labels_from_root_branch(G_idx, root_idx, n_nodes):
    """G_idx: networkx graph on remapped indices 0..n-1. root_idx in 0..n-1."""
    y = np.zeros(n_nodes, dtype=np.int64)
    if root_idx not in G_idx:
        return None
    for v in range(n_nodes):
        if v == root_idx:
            y[v] = 0
            continue
        try:
            path = nx.shortest_path(G_idx, root_idx, v)
        except nx.NetworkXNoPath:
            y[v] = 0
            continue
        if len(path) < 2:
            y[v] = 0
        else:
            y[v] = path[1]
    uniq = np.unique(y)
    m = {v: i for i, v in enumerate(uniq)}
    y = np.array([m[int(v)] for v in y], dtype=np.int64)
    return torch.tensor(y, dtype=torch.long)


def build_disease_data(root, verbose=True):
    edge_path = os.path.join(root, "disease_nc.edges.csv")
    if not os.path.isfile(edge_path):
        raise FileNotFoundError("need %s" % edge_path)

    edges_raw = _load_edges_csv(edge_path)
    if not edges_raw:
        raise ValueError("edge list is empty")

    nodes_set = set()
    for u, v in edges_raw:
        nodes_set.add(u)
        nodes_set.add(v)
    node_ids_sorted = sorted(nodes_set)
    n = len(node_ids_sorted)
    id2i = {nid: i for i, nid in enumerate(node_ids_sorted)}

    edge_list = []
    for u, v in edges_raw:
        iu, iv = id2i[u], id2i[v]
        edge_list.append([iu, iv])
        edge_list.append([iv, iu])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    _co = coalesce(edge_index, num_nodes=n)
    edge_index = _co[0] if isinstance(_co, tuple) else _co

    G_idx = nx.Graph()
    G_idx.add_edges_from([(id2i[u], id2i[v]) for u, v in edges_raw])

    y, y_src = _try_load_y_numpy(root, n, node_ids_sorted)
    if y is None:
        y, y_src = _try_load_y_csv(root, node_ids_sorted)
    root_original = min(node_ids_sorted)
    root_idx = id2i[root_original]
    if y is None:
        y = _labels_from_root_branch(G_idx, root_idx, n)
        y_src = "shortest-path branch from root id=%s (heuristic, no y file)" % root_original
        if verbose:
            print(
                "[Disease] WARNING: no y.npy / disease_nc.labels.csv; using root-branch labels. Add y.npy for fixed labels."
            )
    if y is None:
        raise RuntimeError("cannot build y")

    x, x_src = _try_load_x_numpy(root, n)
    if x is None:
        x = torch.eye(n, dtype=torch.float32)
        x_src = "identity (no x.npy)"
    else:
        x = torch.nan_to_num(x)

    if verbose:
        print(
            "[Disease] nodes=%d | edges(undir)=%d | classes=%d | y=%s | x=%s"
            % (n, edge_index.shape[1] // 2, int(y.max().item()) + 1, y_src, x_src)
        )

    return Data(x=x, edge_index=edge_index, y=y.view(-1).long())


class DiseaseDataset:
    def __init__(self, root="data/disease", verbose=True):
        self.root = root
        self._data = build_disease_data(root, verbose=verbose)
        self.num_classes = int(self._data.y.max().item()) + 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._data
