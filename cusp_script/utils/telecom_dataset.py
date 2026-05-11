"""
Telecom graph under data/telecom/ (or custom root).

  Load order:
    1) telecom_graph.pt — torch.load Data or dict with x, edge_index, y
    2) *.gpickle — networkx.Graph with optional node attrs (label/y, feat/x)
    3) edge list CSV (see _EDGE_CANDIDATES)

  If y is missing: try y.npy, else shortest-path-from-root heuristic (prints WARNING).
"""
import os
import pickle

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce

_EDGE_CANDIDATES = (
    "telecom_nc.edges.csv",
    "telecom_edges.csv",
    "edges.csv",
    "graph_edges.csv",
)

_GPICKLE_CANDIDATES = ("telecom_graph.gpickle", "telegraph.gpickle")


def _try_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _coerce_y_to_num_nodes(data):
    """Align y to num_nodes: 1D labels, (N,1), (N,K) argmax, or flattened N*K."""
    if data.y is None:
        return None
    n = data.num_nodes
    y = data.y
    if y.dtype == torch.float32 or y.dtype == torch.float64:
        y = y.float()
    else:
        y = y.long()

    if y.dim() == 2:
        if y.shape[0] != n:
            raise ValueError(
                "y 2D first dim must be num_nodes=%d, got %s" % (n, tuple(y.shape))
            )
        if y.shape[1] == 1:
            data.y = y.squeeze(1).long()
        else:
            data.y = y.argmax(dim=1).long()
        return "from pt (N x K -> argmax)"

    if y.dim() != 1:
        raise ValueError("y must be 1D or 2D, got dim=%d shape=%s" % (y.dim(), tuple(y.shape)))

    if y.shape[0] == n:
        data.y = y.long()
        return "from pt"

    if y.numel() % n == 0 and y.numel() != n:
        k = y.numel() // n
        y2 = y.view(n, k)
        if k == 1:
            data.y = y2.squeeze(1).long()
        else:
            data.y = y2.argmax(dim=1).long()
        return "from pt (flattened N x %d -> argmax)" % k

    raise ValueError(
        "y length %d != num_nodes %d; cannot reshape as N x K. Check saved y in .pt."
        % (y.shape[0], n)
    )


def _normalize_pyg_data(raw, source_name, verbose):
    """Accept Data or dict -> Data with x, edge_index, y (y may be filled later)."""
    if isinstance(raw, Data):
        data = raw
    elif isinstance(raw, dict):
        if isinstance(raw.get("data"), Data):
            data = raw["data"]
        elif "x" in raw and "edge_index" in raw:
            data = Data(
                x=raw["x"],
                edge_index=raw["edge_index"],
                y=raw.get("y"),
            )
        else:
            raise TypeError(
                "telecom_graph.pt dict has no Data or x/edge_index; keys: %s" % list(raw.keys())
            )
    else:
        raise TypeError("telecom_graph.pt unsupported type: %s" % type(raw))

    nn = data.num_nodes
    _co = coalesce(data.edge_index, num_nodes=nn)
    data.edge_index = _co[0] if isinstance(_co, tuple) else _co

    if data.x is None:
        n = data.num_nodes
        data.x = torch.eye(n, dtype=torch.float32)
        x_src = "identity (no x in pt)"
    else:
        data.x = torch.nan_to_num(data.x.float())
        x_src = "from pt"

    if data.y is None:
        y_src = None
    else:
        y_src = _coerce_y_to_num_nodes(data)

    if verbose:
        print("[Telecom] source=%s | nodes=%d | edges(undir)=%d" % (source_name, data.num_nodes, data.edge_index.shape[1] // 2))
    return data, x_src, y_src


def _nx_graph_to_pyg(G, source_name, verbose):
    if not isinstance(G, nx.Graph):
        raise TypeError("gpickle must be networkx.Graph, got %s" % type(G))

    nodes = sorted(G.nodes())
    n = len(nodes)
    id2i = {nid: i for i, nid in enumerate(nodes)}
    edge_list = []
    for u, v in G.edges():
        iu, iv = id2i[u], id2i[v]
        edge_list.append([iu, iv])
        edge_list.append([iv, iu])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    _co = coalesce(edge_index, num_nodes=n)
    edge_index = _co[0] if isinstance(_co, tuple) else _co

    G_idx = nx.Graph()
    G_idx.add_edges_from([(id2i[u], id2i[v]) for u, v in G.edges()])

    y_list = []
    x_rows = []
    label_keys = ("y", "label", "labels", "class", "community", "attr")
    feat_keys = ("x", "feat", "feature", "features", "h")

    for nid in nodes:
        d = G.nodes[nid]
        if isinstance(d, dict):
            lab = None
            for k in label_keys:
                if k in d and d[k] is not None:
                    lab = d[k]
                    break
            y_list.append(lab)
            feat = None
            for k in feat_keys:
                if k in d and d[k] is not None:
                    feat = d[k]
                    break
            if feat is not None:
                if hasattr(feat, "tolist"):
                    x_rows.append(np.asarray(feat, dtype=np.float32).ravel())
            else:
                x_rows.append(None)
        else:
            y_list.append(None)
            x_rows.append(None)

    has_y = all(v is not None for v in y_list)
    if has_y:
        y_raw = np.array([int(float(v)) if not isinstance(v, (int, np.integer)) else int(v) for v in y_list], dtype=np.int64)
        uniq = np.unique(y_raw)
        m = {v: i for i, v in enumerate(sorted(uniq))}
        y = torch.tensor([m[v] for v in y_raw], dtype=torch.long)
        y_src = "node_attr (%s)" % source_name
    else:
        y = None
        y_src = None

    if all(r is not None for r in x_rows):
        x = torch.tensor(np.stack(x_rows), dtype=torch.float32)
        x = torch.nan_to_num(x)
        x_src = "node_attr (%s)" % source_name
    else:
        x = torch.eye(n, dtype=torch.float32)
        x_src = "identity (no node feats in gpickle)"

    data = Data(x=x, edge_index=edge_index, y=y)
    if verbose:
        print(
            "[Telecom] source=%s | nodes=%d | edges(undir)=%d | gpickle"
            % (source_name, n, edge_index.shape[1] // 2)
        )
    return data, G_idx, id2i, x_src, y_src


def _labels_from_root_branch(G_idx, root_idx, n_nodes):
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
        y[v] = path[1] if len(path) >= 2 else 0
    uniq = np.unique(y)
    m = {v: i for i, v in enumerate(uniq)}
    y = np.array([m[int(v)] for v in y], dtype=np.int64)
    return torch.tensor(y, dtype=torch.long)


def _find_edge_path(root):
    for name in _EDGE_CANDIDATES:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p, name
    return None, None


def _find_gpickle_path(root):
    for name in _GPICKLE_CANDIDATES:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            return p, name
    return None, None


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


def _try_load_y_numpy(root, n_nodes):
    for name in ("y.npy", "telecom_y.npy", "labels.npy"):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        y = np.load(p)
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != n_nodes:
            raise ValueError("%s length %d != num_nodes %d" % (name, y.shape[0], n_nodes))
        y = y.astype(np.int64)
        uniq = np.unique(y)
        if y.min() < 0:
            m = {v: i for i, v in enumerate(uniq)}
            y = np.array([m[v] for v in y], dtype=np.int64)
        return torch.tensor(y, dtype=torch.long), "file:%s" % name
    return None, None


def _try_load_y_csv(root, node_ids_sorted):
    import csv

    for name in ("telecom_nc.labels.csv", "telecom_labels.csv", "labels.csv"):
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
        if len(rows[0]) >= 2:
            m = {}
            for row in rows:
                if len(row) < 2:
                    continue
                m[int(row[0])] = int(row[1])
            y_list = [m[nid] for nid in node_ids_sorted]
            y = np.array(y_list, dtype=np.int64)
        else:
            if len(rows) != len(node_ids_sorted):
                continue
            y = np.array([int(r[0]) for r in rows], dtype=np.int64)
        uniq = np.unique(y)
        m = {v: i for i, v in enumerate(sorted(uniq))}
        y = np.array([m[v] for v in y], dtype=np.int64)
        return torch.tensor(y, dtype=torch.long), "file:%s" % name
    return None, None


def _try_load_x_numpy(root, n_nodes):
    for name in ("x.npy", "telecom_x.npy", "features.npy"):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        x = np.load(p)
        if x.shape[0] != n_nodes:
            raise ValueError("%s dim0 %d != num_nodes %d" % (name, x.shape[0], n_nodes))
        return torch.tensor(x, dtype=torch.float32), "file:%s" % name
    return None, None


def build_telecom_data(root, verbose=True):
    pt_path = os.path.join(root, "telecom_graph.pt")
    if os.path.isfile(pt_path):
        raw = _try_torch_load(pt_path)
        data, x_src, y_src = _normalize_pyg_data(raw, "telecom_graph.pt", verbose)
        if data.y is None:
            y, y_src2 = _try_load_y_numpy(root, data.num_nodes)
            if y is None:
                ei = data.edge_index.cpu().numpy()
                G_idx = nx.Graph()
                G_idx.add_edges_from(list(zip(ei[0], ei[1])))
                n = data.num_nodes
                root_idx = 0
                y = _labels_from_root_branch(G_idx, root_idx, n)
                y_src = "shortest-path branch (no y in pt)"
                if verbose:
                    print("[Telecom] WARNING: no y in pt; using root-branch heuristic or add y.npy")
            else:
                y_src = y_src2
            data.y = y.view(-1).long()
        else:
            y_src = y_src or "from pt"
        if verbose:
            print(
                "[Telecom] classes=%d | y=%s | x=%s"
                % (int(data.y.max().item()) + 1, y_src, x_src)
            )
        return data

    gp_path, gp_name = _find_gpickle_path(root)
    if gp_path:
        with open(gp_path, "rb") as f:
            G = pickle.load(f)
        data, G_idx, id2i, x_src, y_src = _nx_graph_to_pyg(G, gp_name, verbose)
        nodes = sorted(G.nodes())
        if data.y is None:
            y, y_src2 = _try_load_y_numpy(root, data.num_nodes)
            if y is None:
                y, y_src2 = _try_load_y_csv(root, nodes)
            if y is None:
                root_original = min(nodes)
                root_idx = id2i[root_original]
                data.y = _labels_from_root_branch(G_idx, root_idx, data.num_nodes)
                y_src = "shortest-path branch (no labels in gpickle)"
                if verbose:
                    print("[Telecom] WARNING: gpickle nodes lack labels; using root-branch heuristic.")
            else:
                data.y = y.view(-1).long()
                y_src = y_src2
        if verbose:
            print(
                "[Telecom] classes=%d | y=%s | x=%s"
                % (int(data.y.max().item()) + 1, y_src, x_src)
            )
        return data

    edge_path, edge_name = _find_edge_path(root)
    if edge_path is None:
        raise FileNotFoundError(
            "No telecom_graph.pt / gpickle / edge CSV under %s."
            % root
        )

    edges_raw = _load_edges_csv(edge_path)
    if not edges_raw:
        raise ValueError("edge file %s is empty" % edge_name)

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

    y, y_src = _try_load_y_numpy(root, n)
    if y is None:
        y, y_src = _try_load_y_csv(root, node_ids_sorted)
    root_original = min(node_ids_sorted)
    root_idx = id2i[root_original]
    if y is None:
        y = _labels_from_root_branch(G_idx, root_idx, n)
        y_src = "shortest-path branch from root id=%s" % root_original
        if verbose:
            print("[Telecom] WARNING: no y file; using root-branch heuristic.")
    if y is None:
        raise RuntimeError("cannot build labels y")

    x, x_src = _try_load_x_numpy(root, n)
    if x is None:
        x = torch.eye(n, dtype=torch.float32)
        x_src = "identity (no x.npy)"
    else:
        x = torch.nan_to_num(x)

    if verbose:
        print(
            "[Telecom] edge_file=%s | nodes=%d | edges(undir)=%d | classes=%d | y=%s | x=%s"
            % (edge_name, n, edge_index.shape[1] // 2, int(y.max().item()) + 1, y_src, x_src)
        )

    return Data(x=x, edge_index=edge_index, y=y.view(-1).long())


class TelecomDataset:
    def __init__(self, root="data/telecom", verbose=True):
        self.root = root
        self._data = build_telecom_data(root, verbose=verbose)
        self.num_classes = int(self._data.y.max().item()) + 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._data
