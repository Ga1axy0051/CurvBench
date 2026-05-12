"""
AirportLocal: local files only (no synthetic community labels).

  airport_edgelist.txt — undirected edges, two ints per line
  airport_alldata.p — pandas DataFrame; col0 = airport ID (intersect with graph nodes)
  gdp / GDP column — quantile bins -> y; other numeric cols (except ID) -> x
"""
import os
import pickle

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


def _load_edgelist(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u != v:
                edges.append((u, v))
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def _find_gdp_column(df):
    import pandas as pd
    for c in ("gdp", "GDP", "Gdp"):
        if c in df.columns:
            return c
    return None


def _align_df_to_graph_nodes(df, nodes):
    """Keep intersection of graph nodes and first-column IDs in the table."""
    import pandas as pd

    node_set = set(nodes)
    key = df.columns[0]
    sub = df[df[key].isin(node_set)].drop_duplicates(subset=key, keep="first")
    if sub.shape[0] == 0:
        return None, 0, len(node_set)
    sub = sub.set_index(key)
    try:
        sub.index = pd.Index([int(i) for i in sub.index], name=sub.index.name)
    except (TypeError, ValueError):
        pass
    node_set = {int(n) for n in node_set}
    # Normalize index types before intersection
    common = sorted(node_set & {int(i) for i in sub.index})
    if len(common) == 0:
        return None, 0, len(node_set)
    return sub.loc[common], len(common), len(node_set)


def _build_from_airport_alldata(data_dir, G, gdp_bins=10):
    import pandas as pd

    path = os.path.join(data_dir, "airport_alldata.p")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "AirportLocal needs %s next to airport_edgelist.txt (not found)." % path
        )

    with open(path, "rb") as f:
        df = pickle.load(f)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("airport_alldata.p must be pandas.DataFrame, got %s" % type(df))

    gdp_col = _find_gdp_column(df)
    if gdp_col is None:
        raise ValueError(
            "airport_alldata.p has no gdp/GDP column. Columns: %s" % list(df.columns)
        )

    nodes_full = sorted(G.nodes())
    out = _align_df_to_graph_nodes(df, nodes_full)
    if out[0] is None:
        _, n_hit, n_g = out
        raise ValueError(
            "No overlap between edgelist nodes and alldata col0 (table_hits=%d, graph_nodes=%d)."
            % (n_hit, n_g)
        )
    aligned, n_common, n_graph = out
    if n_common < 2:
        raise ValueError(
            "Too few overlapping nodes (%d); check alldata vs edgelist." % n_common
        )

    nodes = sorted(aligned.index)
    aligned = aligned.loc[nodes]
    G_sub = G.subgraph(nodes).copy()

    gdp = pd.to_numeric(aligned[gdp_col], errors="coerce")
    if gdp.isna().all():
        raise ValueError("gdp column is all NaN; cannot bin to labels.")
    gdp = gdp.fillna(gdp.median())

    k = min(int(gdp_bins), max(2, int(gdp.nunique())))
    try:
        y_series = pd.qcut(gdp, q=k, duplicates="drop", labels=False)
    except ValueError:
        try:
            y_series = pd.cut(gdp, bins=min(k, len(gdp)), duplicates="drop", labels=False)
        except Exception as e:
            raise ValueError("GDP binning failed: %s" % e) from e
    y_series = y_series.astype("float64").fillna(0).astype("int64")
    uniq = np.sort(np.unique(y_series.values))
    remap = {v: i for i, v in enumerate(uniq)}
    y_arr = np.array([remap[int(v)] for v in y_series.values], dtype=np.int64)
    y = torch.tensor(y_arr, dtype=torch.long)

    id_name = df.columns[0]
    feat_cols = [c for c in aligned.columns if c != gdp_col and c != id_name]
    num_df = aligned[feat_cols].select_dtypes(include=[np.number])
    if id_name in num_df.columns:
        num_df = num_df.drop(columns=[id_name], errors="ignore")
    if num_df.shape[1] == 0:
        n = len(nodes)
        x = torch.eye(n, dtype=torch.float32)
    else:
        x = torch.tensor(num_df.values, dtype=torch.float32)
        x = torch.nan_to_num(x)

    id2i = {nid: i for i, nid in enumerate(nodes)}
    edge_list = []
    for u, v in G_sub.edges():
        iu, iv = id2i[u], id2i[v]
        edge_list.append([iu, iv])
        edge_list.append([iv, iu])
    n = len(nodes)
    if n == 0:
        raise ValueError("graph empty after alignment.")
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    _co = coalesce(edge_index, num_nodes=n)
    edge_index = _co[0] if isinstance(_co, tuple) else _co

    return x, y, edge_index


def build_local_airport_data(data_dir, gdp_bins=10, verbose=True):
    edge_path = os.path.join(data_dir, "airport_edgelist.txt")
    if not os.path.isfile(edge_path):
        raise FileNotFoundError(
            "need %s (build from routes.dat etc.)" % edge_path
        )

    G = _load_edgelist(edge_path)
    if G.number_of_nodes() == 0:
        raise ValueError("edgelist graph is empty.")

    x, y, edge_index = _build_from_airport_alldata(data_dir, G, gdp_bins=gdp_bins)
    if verbose:
        print(
            "[AirportLocal] y=GDP bins | nodes=%d | classes=%d | gdp_bins=%d | x_dim=%d"
            % (y.shape[0], int(y.max().item()) + 1, gdp_bins, x.shape[1])
        )
    return Data(x=x, edge_index=edge_index, y=y.view(-1).long())


class LocalAirportDataset:
    def __init__(self, root="data/Airport", gdp_bins=10, verbose=True):
        self.root = root
        self._data = build_local_airport_data(root, gdp_bins=gdp_bins, verbose=verbose)
        self.num_classes = int(self._data.y.max().item()) + 1

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._data
