import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.preprocessing import LabelBinarizer
import sys
import random

import pandas as pd, numpy as np, pickle as pkl, networkx as nx
from sklearn.preprocessing import LabelBinarizer
import scipy.sparse as sp, os

TABLE2GRAPH_ROOT = os.environ.get("HAT_TABLE2GRAPH_ROOT", "data/exptable2graph")
LEGACY_TABLE2GRAPH_ROOT = os.environ.get(
    "HAT_TABLE2GRAPH_LEGACY", "data/new_table2graph"
)


def _t2g_resolve_csv(*candidates):
    """First existing CSV path among candidates, else None.
    If HAT_TABLE2GRAPH_LEGACY_DISABLE is 1/true/yes, skip legacy fallbacks.
    """
    skip_legacy = os.environ.get("HAT_TABLE2GRAPH_LEGACY_DISABLE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    for i, p in enumerate(candidates):
        if skip_legacy and i > 0:
            break
        if p and os.path.exists(p):
            return p
    return None


def _t2g_first_existing_path(*paths):
    """First path that exists."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _t2g_pt_path_unified_first(subdir, hetero_filename):
    """Prefer unified_data.pt, then hetero_filename, then legacy path if allowed."""
    cands = [
        os.path.join(TABLE2GRAPH_ROOT, subdir, "unified_data.pt"),
        os.path.join(TABLE2GRAPH_ROOT, subdir, hetero_filename),
    ]
    if os.environ.get("HAT_TABLE2GRAPH_LEGACY_DISABLE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        cands.append(
            os.path.join(LEGACY_TABLE2GRAPH_ROOT, subdir, hetero_filename)
        )
    return _t2g_first_existing_path(*cands)


def _t2g_pt_path_f1():
    """First existing f1 .pt: unified, ultimate, f1_HeteroGraph, then legacy."""
    cands = [
        os.path.join(TABLE2GRAPH_ROOT, "f1", "unified_data.pt"),
        os.path.join(TABLE2GRAPH_ROOT, "f1", "f1_ultimate_hetero_graph.pt"),
        os.path.join(TABLE2GRAPH_ROOT, "f1", "f1_HeteroGraph.pt"),
        os.path.join(TABLE2GRAPH_ROOT, "f1", "F1_HeteroGraph.pt"),
    ]
    if os.environ.get("HAT_TABLE2GRAPH_LEGACY_DISABLE", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        cands.extend(
            [
                os.path.join(LEGACY_TABLE2GRAPH_ROOT, "f1", "f1_HeteroGraph.pt"),
                os.path.join(LEGACY_TABLE2GRAPH_ROOT, "f1", "F1_HeteroGraph.pt"),
            ]
        )
    return _t2g_first_existing_path(*cands)


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    # my load data
    if dataset_str[:3] == 'my_':
        names1 = ['adj_matrix.npz', 'attr_matrix.npz']
        names2 = ['label_matrix.npy', 'train_mask.npy', 'val_mask.npy', 'test_mask.npy']
        objects = []
        for tmp_name in names1:
            tmp_path = 'data/other/{}/{}.{}'.format(dataset_str, dataset_str, tmp_name)
            objects.append(sp.load_npz(tmp_path))
        for tmp_name in names2:
            tmp_path = 'data/other/{}/{}.{}'.format(dataset_str, dataset_str, tmp_name)
            objects.append(np.load(tmp_path))
        adj, features, label_matrix, train_mask, val_mask, test_mask = tuple(objects)

        y_train = np.zeros(label_matrix.shape)
        y_val = np.zeros(label_matrix.shape)
        y_test = np.zeros(label_matrix.shape)
        y_train[train_mask, :] = label_matrix[train_mask, :]
        y_val[val_mask, :] = label_matrix[val_mask, :]
        y_test[test_mask, :] = label_matrix[test_mask, :]

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

    if dataset_str == "carcinogenesis":
        return load_data_carcinogenesis()

    if dataset_str == "hepatitis_std":
        return load_data_hepatitis_std()

    if dataset_str == "hockey":
        return load_data_hockey()

    if dataset_str == "pte":
        return load_data_pte()

    if dataset_str == "toxicology":
        return load_data_toxicology()

    if dataset_str == "f1":
        return load_data_f1()

    if dataset_str == "cs_phds_nc":
        return load_data_cs_phds_nc()

    if dataset_str == "cs_phds_lp":
        return load_data_cs_phds_lp()

    if dataset_str == "airport":
        

    # ---- 1. load raw objects ----
    # airport.p : graph (networkx) or something similar
    # airport_alldata.p : pandas DataFrame with a column for node_id and feature columns (incl. gdp)
        with open("data/airport/airport.p", "rb") as f:
            G_full = pkl.load(f)                       # original full graph (nodes are original IDs)
        with open("data/airport/airport_alldata.p", "rb") as f:
            df = pd.read_pickle(f)                     # DataFrame, expected a column with node id

        # ensure df contains node id column (rename if needed)
        if 0 in df.columns and 'node_id' not in df.columns:
            df = df.rename(columns={0: 'node_id'})

        # ---- 2. choose nodes present in both df and G_full ----
        # keep only nodes that actually exist in the graph
        keep_ids = df['node_id'].astype(int).unique()
        # intersection to avoid nodes missing in graph
        nodes_in_graph = set(G_full.nodes())
        keep_ids = [nid for nid in keep_ids if nid in nodes_in_graph]

        # build subgraph but keep the node ordering stable:
        subG = G_full.subgraph(keep_ids).copy()

        # create a deterministic nodelist (this determines row order for adj/features/labels)
        # use sorted(keep_ids) OR use list(subG.nodes()) to follow graph iteration order
        # I recommend using list(subG.nodes()) to preserve original graph order
        nodelist = list(subG.nodes())

        # ---- 3. adjacency built with explicit nodelist so rows/cols align with nodelist ----
        adj = nx.adjacency_matrix(subG, nodelist=nodelist)  # shape (N,N)

        # ---- 4. construct features in same nodelist order ----
        # select numeric columns as features (except node_id)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        if 'node_id' in numeric_cols:
            numeric_cols.remove('node_id')

        # build a lookup dataframe indexed by node_id
        df_indexed = df.set_index('node_id')
        # Reindex to nodelist order; if some node missing in df_indexed, fill with zeros
        feat_df = df_indexed.reindex(nodelist)[numeric_cols].fillna(0.0).astype(np.float32)
        features = sp.csr_matrix(feat_df.values)   # shape (N, D)

        # ---- 5. labels: use df column 'gdp' split into 3 quantiles as before ----
        # make sure to select values in the same nodelist order
        gdp_series = df_indexed.reindex(nodelist)['gdp']
        # compute labels for those not-NaN, and set 0 for NaN (as before)
        gdp_clean = gdp_series.dropna().astype(np.float32)
        labels_q = pd.qcut(gdp_clean, q=3, labels=[0, 1, 2]).astype(int)
        labels_full = np.zeros(len(nodelist), dtype=int)
        # map index positions for non-NaN entries
        non_nan_idx = np.where(~gdp_series.isna().values)[0]
        labels_full[non_nan_idx] = labels_q.values
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels_full)    # shape (N, C)

        # ---- 6. train/val/test splits ----
        if os.path.exists("data/airport/train_idx.npy"):
            idx_train = np.load("data/airport/train_idx.npy").tolist()
            idx_val   = np.load("data/airport/val_idx.npy").tolist()
            idx_test  = np.load("data/airport/test_idx.npy").tolist()
            # these idx are likely original node ids; convert them to positions in nodelist
            # make mapping from original id -> position
            id2pos = {nid: pos for pos, nid in enumerate(nodelist)}
            # filter only elements that exist in id2pos
            idx_train = [id2pos[i] for i in idx_train if i in id2pos]
            idx_val   = [id2pos[i] for i in idx_val if i in id2pos]
            idx_test  = [id2pos[i] for i in idx_test if i in id2pos]
        else:
            N = len(nodelist)
            idx = np.random.permutation(N)
            idx_train = idx[:int(0.6 * N)].tolist()
            idx_val   = idx[int(0.6 * N):int(0.8 * N)].tolist()
            idx_test  = idx[int(0.8 * N):].tolist()

        train_mask = sample_mask(idx_train, len(nodelist))
        val_mask   = sample_mask(idx_val, len(nodelist))
        test_mask  = sample_mask(idx_test, len(nodelist))

        y_train = np.zeros(labels.shape); y_val = np.zeros(labels.shape); y_test = np.zeros(labels.shape)
        y_train[train_mask] = labels[train_mask]
        y_val[val_mask]     = labels[val_mask]
        y_test[test_mask]   = labels[test_mask]

        # ---- 7. sanity prints ----
        print('N nodes (nodelist):', len(nodelist))
        print('adj shape      :', adj.shape)
        print('features shape :', features.shape)
        print('labels shape   :', labels.shape)

        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    if dataset_str == "disease":
        return load_data_disease(use_feats=True, data_path='data/disease')
    


            #return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

   
    
    
    else:
        # origin load data
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

              

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]
        # print(labels)
        # print("the length of label", len(labels))
        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    # print(train_mask)
    # print(labels)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    print(adj.shape)
    print(features.shape)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def _t2g_unwrap_torch_obj(obj):
    if isinstance(obj, dict):
        for k in ("hetero", "hetero_data", "data", "graph", "HeteroData"):
            if k in obj and obj[k] is not None:
                v = obj[k]
                if hasattr(v, "node_types") or hasattr(v, "collect") or hasattr(v, "x"):
                    return v
        for v in obj.values():
            if hasattr(v, "node_types") or (
                hasattr(v, "collect") and callable(getattr(v, "collect", None))
            ):
                return v
    return obj


def _t2g_get_store_tensor(st):
    for name in ("x", "feat", "feature", "features"):
        if hasattr(st, name):
            t = getattr(st, name)
            if t is not None and hasattr(t, "shape") and len(t.shape) >= 1:
                return t, name
    return None, None


def _t2g_get_store_y(st):
    for name in ("y", "label", "labels", "target", "cls", "class_id", "class"):
        if hasattr(st, name):
            t = getattr(st, name)
            if t is not None and hasattr(t, "shape") and len(t.shape) >= 1:
                return t, name
    return None, None


def _t2g_y_tensor_to_int(y_t):
    y = y_t.detach().cpu().numpy()
    if y.ndim == 2 and y.shape[1] > 1:
        return np.argmax(y, axis=1).astype(np.int64)
    return np.round(y.reshape(-1)).astype(np.int64)


def _t2g_xy_from_pt_only(hetero, preferred):
    """Without CSV: find a node type with both x and y tensors."""
    nts_all = _t2g_iter_node_types(hetero)
    order = [x for x in preferred if x in nts_all] + [
        x for x in nts_all if x not in preferred
    ]
    seen = set()
    for nt in order:
        if nt in seen or nt not in hetero:
            continue
        seen.add(nt)
        st = hetero[nt]
        x_t, _ = _t2g_get_store_tensor(st)
        y_t, _ = _t2g_get_store_y(st)
        if x_t is None or y_t is None:
            continue
        if int(x_t.shape[0]) != int(y_t.shape[0]):
            continue
        X = x_t.detach().cpu().numpy().astype(np.float32)
        y_int = _t2g_y_tensor_to_int(y_t)
        return X, y_int, nt
    return None, None, None


def _t2g_find_y_matching_n(hetero, n_exp):
    """Label tensor of length n_exp on any node type or homogeneous data.y."""
    n_exp = int(n_exp)
    for nt in _t2g_iter_node_types(hetero):
        if nt not in hetero:
            continue
        st = hetero[nt]
        y_t, _ = _t2g_get_store_y(st)
        if y_t is not None and int(y_t.shape[0]) == n_exp:
            return y_t, nt
    if hasattr(hetero, "y") and hetero.y is not None:
        try:
            if int(hetero.y.shape[0]) == n_exp:
                return hetero.y, "data.y"
        except Exception:
            pass
    return None, None


def _t2g_xy_from_pt_relaxed(hetero, preferred):
    """Without CSV: same-type x+y, else x from one type with y from another, else homo x/y."""
    out = _t2g_xy_from_pt_only(hetero, preferred)
    if out[0] is not None:
        return out
    nts_all = _t2g_iter_node_types(hetero)
    order = [x for x in preferred if x in nts_all] + [
        x for x in nts_all if x not in preferred
    ]
    for nt in order:
        if nt not in hetero:
            continue
        st = hetero[nt]
        x_t, _ = _t2g_get_store_tensor(st)
        if x_t is None:
            continue
        n = int(x_t.shape[0])
        y_t, y_src = _t2g_find_y_matching_n(hetero, n)
        if y_t is None:
            continue
        X = x_t.detach().cpu().numpy().astype(np.float32)
        y_int = _t2g_y_tensor_to_int(y_t)
        tag = nt if y_src == nt else "%s_x+%s_y" % (nt, y_src)
        return X, y_int, tag
    if hasattr(hetero, "x") and hasattr(hetero, "y"):
        if hetero.x is not None and hetero.y is not None:
            if int(hetero.x.shape[0]) == int(hetero.y.shape[0]):
                X = hetero.x.detach().cpu().numpy().astype(np.float32)
                y_int = _t2g_y_tensor_to_int(hetero.y)
                return X, y_int, "homogeneous_x_y"
    return None, None, None


def _t2g_store_n_nodes(st):
    nn = getattr(st, "num_nodes", None)
    if nn is not None:
        return int(nn)
    t, _ = _t2g_get_store_tensor(st)
    if t is not None:
        return int(t.shape[0])
    return None


def _t2g_iter_node_types(hetero):
    if hasattr(hetero, "node_types") and hetero.node_types is not None:
        out = list(hetero.node_types)
        if out:
            return out
    if hasattr(hetero, "metadata"):
        try:
            out = list(hetero.metadata()[0])
            if out:
                return out
        except Exception:
            pass
    if hasattr(hetero, "collect") and callable(hetero.collect):
        for key in ("x", "feat", "pos"):
            try:
                cx = hetero.collect(key)
                if isinstance(cx, dict) and cx:
                    return list(cx.keys())
            except Exception:
                pass
    nsd = getattr(hetero, "_node_store_dict", None)
    if isinstance(nsd, dict) and nsd:
        return list(nsd.keys())
    return []


def _t2g_resolve_nt_and_tensor(hetero, n_exp, preferred):
    nts = _t2g_iter_node_types(hetero)
    if not nts:
        return None, None, "cannot list node_types (%s)" % type(hetero).__name__
    rows = []
    for nt in nts:
        if nt not in hetero:
            continue
        st = hetero[nt]
        t, _ = _t2g_get_store_tensor(st)
        nn = _t2g_store_n_nodes(st)
        rows.append((nt, nn, t))
    dbg = "; ".join(
        "%s(num=%s,%s)"
        % (a, b, "tensor" if t is not None else "no_tensor")
        for a, b, t in rows
    )
    candidates = []
    for nt, nn, t in rows:
        if t is None:
            continue
        if nn == n_exp or int(t.shape[0]) == n_exp:
            candidates.append((nt, t))
    if not candidates:
        return None, None, dbg
    for p in preferred:
        for nt, t in candidates:
            if nt == p:
                return nt, t, None
    if len(candidates) == 1:
        return candidates[0][0], candidates[0][1], None
    return None, None, "ambiguous node types for N=%d: %s" % (n_exp, [c[0] for c in candidates])


def _t2g_load_torch_optional(path):
    """torch.load if path exists, else None."""
    if path is None or not os.path.exists(path):
        return None
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _t2g_adj_from_edge_index(edge_index, num_nodes):
    """edge_index (2,E) to undirected CSR + self-loops."""
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError("edge_index must be shape (2, E)")
    row = ei[0].astype(np.int64)
    col = ei[1].astype(np.int64)
    data_vals = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix(
        (data_vals, (row, col)), shape=(int(num_nodes), int(num_nodes))
    )
    adj = adj.maximum(adj.T)
    adj.setdiag(1.0)
    return adj


def _t2g_extract_edge_index_num_nodes(data_obj):
    """Parse edge_index and N from PyG Data or HeteroData."""
    data_obj = _t2g_unwrap_torch_obj(data_obj)
    ei = None
    n = None
    try:
        from torch_geometric.data import HeteroData, Data as TGData

        if isinstance(data_obj, TGData) and data_obj.edge_index is not None:
            ei = data_obj.edge_index
            if hasattr(data_obj, "num_nodes") and data_obj.num_nodes is not None:
                n = int(data_obj.num_nodes)
            elif data_obj.x is not None:
                n = int(data_obj.x.shape[0])
            else:
                et = (
                    ei.detach().cpu().numpy()
                    if hasattr(ei, "detach")
                    else np.asarray(ei)
                )
                n = int(et.max()) + 1
        elif isinstance(data_obj, HeteroData):
            homo = data_obj.to_homogeneous()
            ei = homo.edge_index
            n = int(homo.num_nodes)
    except Exception:
        pass
    if ei is None and hasattr(data_obj, "edge_index") and getattr(
        data_obj, "edge_index", None
    ) is not None:
        ei = data_obj.edge_index
        if n is None:
            if hasattr(data_obj, "num_nodes") and data_obj.num_nodes is not None:
                n = int(data_obj.num_nodes)
            elif hasattr(data_obj, "x") and data_obj.x is not None:
                n = int(data_obj.x.shape[0])
    if ei is None:
        return None, None
    ei = ei.detach().cpu().numpy() if hasattr(ei, "detach") else np.asarray(ei)
    if n is None and ei.size > 0:
        n = int(ei.max()) + 1
    return ei, n


def _t2g_build_adj_from_pt(data_obj, n_expected, tag="table2graph"):
    """Build adjacency from edge_index in .pt (no kNN)."""
    ei, ng = _t2g_extract_edge_index_num_nodes(data_obj)
    if ei is None:
        raise ValueError(
            "%s: cannot parse edge_index from .pt (need torch_geometric Data/HeteroData)."
            % tag
        )
    if ng is None:
        raise ValueError("%s: cannot infer num_nodes" % tag)
    if int(ng) != int(n_expected):
        raise ValueError(
            "%s: graph nodes %d != feature/label rows %d."
            % (tag, int(ng), int(n_expected))
        )
    adj = _t2g_adj_from_edge_index(ei, int(n_expected))
    print(
        "%s: adj from edge_index, N=%d, directed edges ~%d"
        % (tag, int(n_expected), int(ei.shape[1]))
    )
    return adj


def _t2g_legacy_disabled():
    return os.environ.get("HAT_TABLE2GRAPH_LEGACY_DISABLE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _t2g_load_table_pt_prefer_unified(subdir, hetero_filename):
    """Prefer unified_data.pt, else first *_HeteroGraph.pt, else hetero_filename / legacy."""
    base = os.path.join(TABLE2GRAPH_ROOT, subdir)
    pu = os.path.join(base, "unified_data.pt")
    du = _t2g_load_torch_optional(pu)
    if du is None and not _t2g_legacy_disabled():
        du = _t2g_load_torch_optional(
            os.path.join(LEGACY_TABLE2GRAPH_ROOT, subdir, "unified_data.pt")
        )
    if du is not None:
        print("table2graph [%s]: using unified_data.pt only" % subdir)
        return du
    import glob

    gh = sorted(glob.glob(os.path.join(base, "*_HeteroGraph.pt")))
    for p in gh:
        if os.path.isfile(p):
            print(
                "table2graph [%s]: no unified, using %s"
                % (subdir, os.path.basename(p))
            )
            return _t2g_load_torch_optional(p)
    cands = [os.path.join(base, hetero_filename)]
    if not _t2g_legacy_disabled():
        cands.append(
            os.path.join(LEGACY_TABLE2GRAPH_ROOT, subdir, hetero_filename)
        )
    ph = _t2g_first_existing_path(*cands)
    if ph:
        print("table2graph [%s]: using %s" % (subdir, os.path.basename(ph)))
        return _t2g_load_torch_optional(ph)
    return None


def _t2g_load_f1_pt_prefer_unified():
    """f1: unified_data.pt first, else *_HeteroGraph.pt or f1/F1 hetero files."""
    base = os.path.join(TABLE2GRAPH_ROOT, "f1")
    du = _t2g_load_torch_optional(os.path.join(base, "unified_data.pt"))
    if du is None and not _t2g_legacy_disabled():
        du = _t2g_load_torch_optional(
            os.path.join(LEGACY_TABLE2GRAPH_ROOT, "f1", "unified_data.pt")
        )
    if du is not None:
        print("table2graph [f1]: using unified_data.pt only")
        return du
    import glob

    for folder in (base, os.path.join(LEGACY_TABLE2GRAPH_ROOT, "f1")):
        if not os.path.isdir(folder):
            continue
        gh = sorted(glob.glob(os.path.join(folder, "*_HeteroGraph.pt")))
        for p in gh:
            if os.path.isfile(p):
                print(
                    "table2graph [f1]: no unified, using %s"
                    % os.path.basename(p)
                )
                return _t2g_load_torch_optional(p)
        ult = os.path.join(folder, "f1_ultimate_hetero_graph.pt")
        if os.path.isfile(ult):
            print("table2graph [f1]: no unified, using f1_ultimate_hetero_graph.pt")
            return _t2g_load_torch_optional(ult)
        for name in ("f1_HeteroGraph.pt", "F1_HeteroGraph.pt"):
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                print("table2graph [f1]: using %s" % name)
                return _t2g_load_torch_optional(p)
    return None


def _cs_phds_torch_load(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cs_phds_to_numpy(obj):
    import torch

    if torch.is_tensor(obj):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if sp.issparse(obj):
        return obj
    return np.asarray(obj)


def _cs_phds_labels_to_int(y_arr):
    """(N,) or (N,1) ints, or (N,C) one-hot / logits -> argmax."""
    y_arr = _cs_phds_to_numpy(y_arr)
    if y_arr.ndim == 2:
        if y_arr.shape[1] == 1:
            y_int = y_arr.reshape(-1).astype(np.int64)
        else:
            y_int = np.argmax(y_arr, axis=1).astype(np.int64)
    else:
        y_int = y_arr.astype(np.int64).reshape(-1)
    return y_int


def _hat_repo_root():
    """Repo root from utils/process.py location."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _cs_phds_data_root():
    """CS PhDs root: HAT_CS_PHDS_ROOT if set, else <repo>/data/cs_phds/."""
    r = os.environ.get("HAT_CS_PHDS_ROOT", "").strip()
    if r:
        return os.path.abspath(os.path.expanduser(r))
    return os.path.join(_hat_repo_root(), "data", "cs_phds")


def _cs_phds_resolve_pt(base_dir, stem):
    """Resolve stem.pt; case-insensitive fallback on disk."""
    exact = os.path.join(base_dir, "%s.pt" % stem)
    if os.path.isfile(exact):
        return exact
    if not os.path.isdir(base_dir):
        return None
    want = "%s.pt" % stem.lower()
    for fn in os.listdir(base_dir):
        if fn.lower() == want:
            return os.path.join(base_dir, fn)
    return None


def _cs_phds_resolve_first(base_dir, stems):
    """First existing path among stem names (no .pt suffix)."""
    for s in stems:
        p = _cs_phds_resolve_pt(base_dir, s)
        if p is not None and os.path.isfile(p):
            return p
    return None


def _cs_phds_extract_labels_from_splits_dict(d, log_tag):
    """Node labels from splits.pt dict."""
    if not isinstance(d, dict):
        raise ValueError("%s: splits.pt top level must be dict" % log_tag)
    for key in ("labels", "y", "label", "node_labels", "Y", "targets", "node_y"):
        if key in d and d[key] is not None:
            return d[key]
    keys = list(d.keys())
    lp_markers = (
        "train_pos",
        "train_neg",
        "val_pos",
        "val_neg",
        "test_pos",
        "test_neg",
    )
    if any(k in d for k in lp_markers):
        raise ValueError(
            "%s: splits.pt is link-prediction split (keys like %s), no per-node labels.\n"
            "This stack expects node classification; use -dataset cs_phds_nc or add labels.pt / labels in splits.\n"
            "Keys: %s"
            % (log_tag, "/".join(lp_markers[:3]) + "/...", keys[:20])
        )
    raise ValueError(
        "%s: no label key in splits.pt (labels/y/...); keys: %s"
        % (log_tag, keys[:40])
    )


def _cs_phds_indices_from_tensor(x):
    a = _cs_phds_to_numpy(x)
    return np.asarray(a, dtype=np.int64).reshape(-1).tolist()


def _cs_phds_try_masks_from_splits(splits_obj, n, log_tag):
    """If splits dict has fixed split, return train/val/test masks; else None."""
    if splits_obj is None or not isinstance(splits_obj, dict):
        return None
    d = splits_obj

    def _mask_from_bool(name):
        if name not in d or d[name] is None:
            return None
        m = _cs_phds_to_numpy(d[name]).astype(np.float64).reshape(-1)
        if m.size != n:
            return None
        return m > 0.5

    tm = _mask_from_bool("train_mask")
    vm = _mask_from_bool("val_mask")
    testm = _mask_from_bool("test_mask")
    if tm is not None and vm is not None and testm is not None:
        return (
            sample_mask(np.flatnonzero(tm).tolist(), n),
            sample_mask(np.flatnonzero(vm).tolist(), n),
            sample_mask(np.flatnonzero(testm).tolist(), n),
        )

    triples = (
        ("train_idx", "val_idx", "test_idx"),
        ("train_indices", "val_indices", "test_indices"),
        ("idx_train", "idx_val", "idx_test"),
    )
    for a, b, c in triples:
        if a not in d or b not in d or c not in d:
            continue
        if d[a] is None or d[b] is None or d[c] is None:
            continue
        try:
            ia = _cs_phds_indices_from_tensor(d[a])
            ib = _cs_phds_indices_from_tensor(d[b])
            ic = _cs_phds_indices_from_tensor(d[c])
        except Exception:
            continue
        if not ia or not ib or not ic:
            continue
        return (
            sample_mask(ia, n),
            sample_mask(ib, n),
            sample_mask(ic, n),
        )
    return None


def _load_data_cs_phds_pack(subdir, log_tag):
    """Load CS PhDs NC tensors from data/cs_phds/<subdir>/. Optional HAT_CS_PHDS_ROOT."""
    from sklearn.model_selection import train_test_split

    root = _cs_phds_data_root()
    base = os.path.join(root, subdir)
    if not os.path.isdir(base):
        raise FileNotFoundError(
            "%s: missing dir %s (subdir %s or HAT_CS_PHDS_ROOT)"
            % (log_tag, os.path.abspath(base), subdir)
        )

    p_adj = _cs_phds_resolve_first(base, ("adj", "adj_train"))
    p_dists = _cs_phds_resolve_first(base, ("dists", "dists_train"))
    p_feats = _cs_phds_resolve_first(
        base,
        (
            "feats",
            "feats_train",
            "feat",
            "feature",
            "features",
            "x",
            "X",
        ),
    )
    p_splits = _cs_phds_resolve_pt(base, "splits")
    p_labels = _cs_phds_resolve_first(
        base,
        (
            "labels",
            "labels_train",
            "label",
            "y",
            "node_labels",
            "Y",
            "targets",
        ),
    )

    missing = []
    if not p_adj:
        missing.append("adj or adj_train")
    if not p_dists:
        missing.append("dists or dists_train")
    if not p_feats:
        missing.append("feats / feat.pt / x.pt")
    if not p_labels and not p_splits:
        missing.append("labels or labels inside splits.pt")
    if missing:
        hint = "dir listing: %s" % ", ".join(sorted(os.listdir(base))[:50])
        raise FileNotFoundError(
            "%s: missing %s | dir: %s\n  %s"
            % (log_tag, ", ".join(missing), os.path.abspath(base), hint)
        )

    splits_obj = None
    if p_splits:
        splits_obj = _cs_phds_torch_load(p_splits)

    if p_labels:
        labels_obj = _cs_phds_torch_load(p_labels)
        labels_source = os.path.basename(p_labels)
    else:
        labels_obj = _cs_phds_extract_labels_from_splits_dict(splits_obj, log_tag)
        labels_source = "splits.pt"

    print("%s: data dir %s" % (log_tag, os.path.abspath(base)))
    print(
        "%s: files adj=%s dists=%s feats=%s labels=%s"
        % (
            log_tag,
            os.path.basename(p_adj),
            os.path.basename(p_dists),
            os.path.basename(p_feats),
            labels_source,
        )
    )

    adj_obj = _cs_phds_torch_load(p_adj)
    dists_obj = _cs_phds_torch_load(p_dists)
    feats_obj = _cs_phds_torch_load(p_feats)

    feats = _cs_phds_to_numpy(feats_obj).astype(np.float32)
    if feats.ndim != 2:
        raise ValueError("%s: feats must be 2D (N, F), got shape=%s" % (log_tag, feats.shape))
    n = int(feats.shape[0])

    adj_np = _cs_phds_to_numpy(adj_obj)
    if sp.issparse(adj_np):
        adj = adj_np.tocsr()
        if adj.shape[0] != n or adj.shape[1] != n:
            raise ValueError(
                "%s: sparse adj shape %s != N=%d" % (log_tag, adj.shape, n)
            )
    else:
        adj_np = np.asarray(adj_np, dtype=np.float32)
        if adj_np.ndim != 2 or adj_np.shape[0] != n or adj_np.shape[1] != n:
            raise ValueError(
                "%s: adj must be (N,N), N=%d, shape=%s"
                % (log_tag, n, getattr(adj_np, "shape", None))
            )
        adj = sp.csr_matrix(adj_np)
        adj = adj.maximum(adj.T)

    y_int = _cs_phds_labels_to_int(labels_obj)
    if y_int.shape[0] != n:
        raise ValueError(
            "%s: labels len %d != feats rows %d"
            % (log_tag, int(y_int.shape[0]), n)
        )

    dists_np = _cs_phds_to_numpy(dists_obj)
    if dists_np.ndim >= 1 and getattr(dists_np, "shape", None) is not None:
        if dists_np.ndim == 2 and dists_np.shape[0] != n:
            raise ValueError(
                "%s: dists dim0 %d != N=%d" % (log_tag, int(dists_np.shape[0]), n)
            )

    if adj.diagonal().sum() == 0:
        adj = adj + sp.eye(n, dtype=np.float32, format="csr")

    lb = LabelBinarizer()
    labels = lb.fit_transform(y_int)
    if labels.shape[1] == 1:
        labels = np.hstack([1 - labels, labels])

    features = sp.csr_matrix(feats)

    split_from_file = _cs_phds_try_masks_from_splits(splits_obj, n, log_tag)
    if split_from_file is not None:
        train_mask, val_mask, test_mask = split_from_file
        print("%s: using fixed train/val/test from splits.pt" % log_tag)
    else:
        idx = np.arange(n)
        try:
            idx_train, idx_temp = train_test_split(
                idx, train_size=0.6, random_state=42, stratify=y_int
            )
            idx_val, idx_test = train_test_split(
                idx_temp, test_size=0.5, random_state=42, stratify=y_int[idx_temp]
            )
        except ValueError:
            idx_train, idx_temp = train_test_split(idx, train_size=0.6, random_state=42)
            idx_val, idx_test = train_test_split(
                idx_temp, test_size=0.5, random_state=42
            )

        train_mask = sample_mask(idx_train.tolist(), n)
        val_mask = sample_mask(idx_val.tolist(), n)
        test_mask = sample_mask(idx_test.tolist(), n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    print(
        "%s: N=%d, feat=%d, C=%d | %s"
        % (log_tag, n, feats.shape[1], labels.shape[1], os.path.abspath(base))
    )
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_cs_phds_nc():
    """CS PhDs node classification pack."""
    return _load_data_cs_phds_pack("cs_phds_nc_ready", "cs_phds_nc")


def load_data_cs_phds_lp():
    """CS PhDs LP folder; LP-only splits error unless labels provided like NC."""
    return _load_data_cs_phds_pack("cs_phds_lp_ready", "cs_phds_lp")


def load_data_carcinogenesis():
    """Carcinogenesis drug NC: .pt + canc.csv/atom fallback, 60/20/20 split."""
    from sklearn.model_selection import train_test_split

    data = _t2g_load_table_pt_prefer_unified(
        "Carcinogenesis_data", "Carcinogenesis_HeteroGraph.pt"
    )
    if data is None:
        raise FileNotFoundError(
            "Carcinogenesis .pt not found (tried unified_data.pt, Carcinogenesis_HeteroGraph.pt"
            + (
                ""
                if os.environ.get("HAT_TABLE2GRAPH_LEGACY_DISABLE", "").lower()
                in ("1", "true", "yes")
                else ", legacy"
            )
            + ") under %s"
            % os.path.join(TABLE2GRAPH_ROOT, "Carcinogenesis_data")
        )
    print("Carcinogenesis: graph loaded (see table2graph logs above)")

    canc_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "Carcinogenesis_data", "csv", "canc.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "Carcinogenesis_data", "csv", "canc.csv"),
    )
    atom_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "Carcinogenesis_data", "csv", "atom.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "Carcinogenesis_data", "csv", "atom.csv"),
    )
    if canc_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in canc_csv.replace(
        "\\", "/"
    ):
        print(
            "Carcinogenesis: canc.csv not in exptable2graph, using %s"
            % canc_csv
        )

    preferred = ("canc", "Canc", "CANC", "drug", "Drug", "molecule", "Molecule")
    use_csv = canc_csv is not None

    if use_csv:
        canc_df = pd.read_csv(canc_csv)
        n_expected = len(canc_df)
        x_t = None
        nts = _t2g_iter_node_types(data)
        if nts:
            nt_used, x_t, err = _t2g_resolve_nt_and_tensor(data, n_expected, preferred)
            if x_t is not None:
                print(
                    "Carcinogenesis: .pt node type '%s' matches canc.csv rows %d"
                    % (nt_used, n_expected)
                )
            elif err:
                print("Carcinogenesis: no .pt features (%s), trying CSV features." % err)
        if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) == n_expected:
            x_t = data.x
            print("Carcinogenesis: using homogeneous x (N=%d)" % n_expected)

        if x_t is None:
            if not atom_csv or not os.path.exists(atom_csv):
                raise ValueError(
                    "no usable .pt features and atom.csv missing in exptable2graph/legacy."
                )
            drug_ids = canc_df["drug_id"].astype(str).values
            atom = pd.read_csv(atom_csv)
            atom["drug"] = atom["drug"].astype(str)
            feat_wide = atom.groupby(["drug", "atomtype"]).size().unstack(fill_value=0)
            feat_wide = feat_wide.reindex(drug_ids).fillna(0.0)
            X = feat_wide.values.astype(np.float32)
            print(
                "Carcinogenesis: atomtype counts from canc.csv + atom.csv (.pt features skipped)"
            )
        else:
            X = x_t.detach().cpu().numpy().astype(np.float32)
        n = X.shape[0]
        if n != n_expected:
            raise ValueError("feature rows %d != canc.csv rows %d" % (n, n_expected))
        y_int = canc_df["class"].values.astype(np.int64)
    else:
        X, y_int, nt_pt = _t2g_xy_from_pt_relaxed(data, preferred)
        if X is None:
            raise FileNotFoundError(
                "canc.csv not found (checked exptable2graph and %s) and .pt has no y "
                "(often only x). Add canc.csv or store y on nodes. node_types=%s"
                % (LEGACY_TABLE2GRAPH_ROOT, _t2g_iter_node_types(data))
            )
        n_expected = int(X.shape[0])
        n = n_expected
        print(
            "Carcinogenesis: no canc.csv, using .pt x/y (%s, N=%d)" % (nt_pt, n_expected)
        )
    lb = LabelBinarizer()
    labels = lb.fit_transform(y_int)
    if labels.shape[1] == 1:
        labels = np.hstack([1 - labels, labels])

    features = sp.csr_matrix(X)
    adj = _t2g_build_adj_from_pt(data, n, "Carcinogenesis")

    idx = np.arange(n)
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, random_state=42, stratify=y_int
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42, stratify=y_int[idx_temp]
    )

    train_mask = sample_mask(idx_train.tolist(), n)
    val_mask = sample_mask(idx_val.tolist(), n)
    test_mask = sample_mask(idx_test.tolist(), n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    print("Carcinogenesis (canc nodes): N=%d, feat=%d, C=%d" % (n, X.shape[1], labels.shape[1]))
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_hepatitis_std():
    """Hepatitis dispat binary; .pt + dispat.csv or sex/age fallback."""
    from sklearn.model_selection import train_test_split

    data = _t2g_load_table_pt_prefer_unified(
        "Hepatitis_std_data", "Hepatitis_HeteroGraph.pt"
    )
    if data is None:
        raise FileNotFoundError(
            "Hepatitis .pt not found (tried unified_data.pt, Hepatitis_HeteroGraph.pt) under %s"
            % os.path.join(TABLE2GRAPH_ROOT, "Hepatitis_std_data")
        )
    print("Hepatitis_std: graph loaded (see table2graph logs above)")

    dispat_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "Hepatitis_std_data", "csv", "dispat.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "Hepatitis_std_data", "csv", "dispat.csv"),
    )
    if dispat_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in dispat_csv.replace(
        "\\", "/"
    ):
        print("Hepatitis_std: dispat.csv not in exptable2graph, using %s" % dispat_csv)

    preferred = ("dispat", "Dispat", "patient", "Patient")
    use_csv = dispat_csv is not None

    if use_csv:
        dispat_df = pd.read_csv(dispat_csv)
        n_expected = len(dispat_df)
        if "Type" in dispat_df.columns:
            y_int = dispat_df["Type"].values.astype(np.int64)
        elif "type" in dispat_df.columns:
            y_int = dispat_df["type"].values.astype(np.int64)
        else:
            raise ValueError(
                "dispat.csv needs Type or type label column; got: %s"
                % list(dispat_df.columns)
            )

        x_t = None
        nts = _t2g_iter_node_types(data)
        if nts:
            nt_used, x_t, err = _t2g_resolve_nt_and_tensor(data, n_expected, preferred)
            if x_t is not None:
                print(
                    "Hepatitis_std: .pt node type '%s' matches dispat.csv rows %d"
                    % (nt_used, n_expected)
                )
            elif err:
                print("Hepatitis_std: no .pt features (%s), trying CSV." % err)
        if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) == n_expected:
            x_t = data.x
            print("Hepatitis_std: using homogeneous x (N=%d)" % n_expected)

        if x_t is None:
            if "sex" not in dispat_df.columns or "age" not in dispat_df.columns:
                raise ValueError(
                    "no usable .pt features and dispat.csv missing sex/age columns"
                )
            X = np.stack(
                [
                    dispat_df["sex"].astype(np.float32).values,
                    dispat_df["age"].astype(np.float32).values,
                ],
                axis=1,
            )
            print("Hepatitis_std: sex/age from dispat.csv (.pt features missing)")
        else:
            X = x_t.detach().cpu().numpy().astype(np.float32)

        n = X.shape[0]
        if n != n_expected:
            raise ValueError("feature rows %d != dispat.csv rows %d" % (n, n_expected))
    else:
        X, y_int, nt_pt = _t2g_xy_from_pt_relaxed(data, preferred)
        if X is None:
            raise FileNotFoundError(
                "dispat.csv not found (exptable2graph and %s); .pt x/y mismatch. node_types=%s"
                % (LEGACY_TABLE2GRAPH_ROOT, _t2g_iter_node_types(data))
            )
        n_expected = int(X.shape[0])
        n = n_expected
        print(
            "Hepatitis_std: no dispat.csv, using .pt x/y on '%s' (N=%d)"
            % (nt_pt, n_expected)
        )

    lb = LabelBinarizer()
    labels = lb.fit_transform(y_int)
    if labels.shape[1] == 1:
        labels = np.hstack([1 - labels, labels])

    features = sp.csr_matrix(X)
    adj = _t2g_build_adj_from_pt(data, n, "Hepatitis_std")

    idx = np.arange(n)
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, random_state=42, stratify=y_int
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42, stratify=y_int[idx_temp]
    )

    train_mask = sample_mask(idx_train.tolist(), n)
    val_mask = sample_mask(idx_val.tolist(), n)
    test_mask = sample_mask(idx_test.tolist(), n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    print(
        "Hepatitis_std (dispat): N=%d, feat=%d, C=%d"
        % (n, X.shape[1], labels.shape[1])
    )
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_hockey():
    """Hockey Master nodes, pos labels; .pt + Master.csv or numeric fallback."""
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    data = _t2g_load_table_pt_prefer_unified("Hockey_data", "Hockey_HeteroGraph.pt")
    if data is None:
        raise FileNotFoundError(
            "Hockey .pt not found (tried unified_data.pt, Hockey_HeteroGraph.pt) under %s"
            % os.path.join(TABLE2GRAPH_ROOT, "Hockey_data")
        )
    print("Hockey: graph loaded (see table2graph logs above)")

    master_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "Hockey_data", "csv", "Master.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "Hockey_data", "csv", "Master.csv"),
    )
    if master_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in master_csv.replace(
        "\\", "/"
    ):
        print("Hockey: Master.csv not in exptable2graph, using %s" % master_csv)

    preferred = ("Master", "master", "PLAYER", "player")
    use_csv = master_csv is not None

    if use_csv:
        master_df = pd.read_csv(master_csv)
        n_expected = len(master_df)

        if "pos" not in master_df.columns:
            raise ValueError(
                "Master.csv needs pos label column; got: %s" % list(master_df.columns)
            )
        pos_series = master_df["pos"].fillna("NA").astype(str)
        le = LabelEncoder()
        y_code = le.fit_transform(pos_series)
        n_cls = len(le.classes_)
        labels = np.zeros((len(y_code), n_cls), dtype=np.float32)
        labels[np.arange(len(y_code)), y_code] = 1.0

        x_t = None
        nts = _t2g_iter_node_types(data)
        if nts:
            nt_used, x_t, err = _t2g_resolve_nt_and_tensor(data, n_expected, preferred)
            if x_t is not None:
                print(
                    "Hockey: .pt node type '%s' matches Master.csv rows %d"
                    % (nt_used, n_expected)
                )
            elif err:
                print("Hockey: no .pt features (%s), trying CSV." % err)
        if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) == n_expected:
            x_t = data.x
            print("Hockey: using homogeneous x (N=%d)" % n_expected)

        if x_t is None:
            num_cols = [
                c
                for c in (
                    "height",
                    "weight",
                    "birthYear",
                    "birthMon",
                    "birthDay",
                    "firstNHL",
                    "lastNHL",
                )
                if c in master_df.columns
            ]
            if not num_cols:
                raise ValueError(
                    "no usable .pt features and no numeric columns in Master.csv"
                )
            X = master_df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(
                np.float32
            )
            print(
                "Hockey: numeric columns %s from Master.csv (.pt features missing)"
                % num_cols
            )
        else:
            X = x_t.detach().cpu().numpy().astype(np.float32)

        n = X.shape[0]
        if n != n_expected:
            raise ValueError("feature rows %d != Master.csv rows %d" % (n, n_expected))
    else:
        X, y_code, nt_pt = _t2g_xy_from_pt_relaxed(data, preferred)
        if X is None:
            raise FileNotFoundError(
                "Master.csv not found (exptable2graph and %s); .pt x/y mismatch. node_types=%s"
                % (LEGACY_TABLE2GRAPH_ROOT, _t2g_iter_node_types(data))
            )
        y_code = y_code.astype(np.int64)
        n_expected = int(X.shape[0])
        n = n_expected
        if len(y_code) != n:
            raise ValueError("Hockey: y length != x length")

        labeled = y_code >= 0
        n_unlab = int((~labeled).sum())
        if n_unlab:
            print(
                "Hockey: %d negative y (often unlabeled -1); full graph kept, split on y>=0 only."
                % n_unlab
            )
        if labeled.sum() == 0:
            raise ValueError("Hockey: no valid non-negative class indices in .pt y")

        y_sup = y_code[labeled]
        uniq_cls = np.unique(y_sup)
        n_cls = len(uniq_cls)
        y_rem = np.searchsorted(uniq_cls, y_sup)
        labels = np.zeros((n, n_cls), dtype=np.float32)
        labels[np.where(labeled)[0], y_rem] = 1.0
        idx_labeled = np.where(labeled)[0]
        print(
            "Hockey: no Master.csv, using .pt x/y on '%s' (N=%d, C=%d, labeled %d)"
            % (nt_pt, n_expected, n_cls, int(labeled.sum()))
        )

    features = sp.csr_matrix(X)
    adj = _t2g_build_adj_from_pt(data, n, "Hockey")

    if use_csv:
        idx = np.arange(n)
        try:
            idx_train, idx_temp = train_test_split(
                idx, train_size=0.6, random_state=42, stratify=y_code
            )
            idx_val, idx_test = train_test_split(
                idx_temp, test_size=0.5, random_state=42, stratify=y_code[idx_temp]
            )
        except ValueError:
            print(
                "Hockey: stratify failed (rare pos class), using random split; class balance may skew."
            )
            idx_train, idx_temp = train_test_split(
                idx, train_size=0.6, random_state=42
            )
            idx_val, idx_test = train_test_split(
                idx_temp, test_size=0.5, random_state=42
            )
    else:
        nl = idx_labeled.size
        rel = np.arange(nl, dtype=np.int64)
        try:
            r_train, r_temp = train_test_split(
                rel, train_size=0.6, random_state=42, stratify=y_rem
            )
            y_tmp = y_rem[r_temp]
            r_val, r_test = train_test_split(
                r_temp, test_size=0.5, random_state=42, stratify=y_tmp
            )
        except ValueError:
            print(
                "Hockey: labeled subset stratify failed, using random split."
            )
            r_train, r_temp = train_test_split(rel, train_size=0.6, random_state=42)
            r_val, r_test = train_test_split(
                r_temp, test_size=0.5, random_state=42
            )
        idx_train = idx_labeled[r_train].tolist()
        idx_val = idx_labeled[r_val].tolist()
        idx_test = idx_labeled[r_test].tolist()

    train_mask = sample_mask(idx_train, n)
    val_mask = sample_mask(idx_val, n)
    test_mask = sample_mask(idx_test, n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    if use_csv:
        cnt = np.bincount(y_code, minlength=n_cls)
        rare = int((cnt < 2).sum())
        if rare > 0:
            print(
                "Hockey: %d pos classes appear once (C=%d); merge rare labels or filter."
                % (rare, n_cls)
            )
    else:
        cnt = np.bincount(y_rem, minlength=n_cls)
        rare = int((cnt < 2).sum())
        if rare > 0:
            print(
                "Hockey: %d classes with single sample in labeled set (C=%d)."
                % (rare, n_cls)
            )

    print(
        "Hockey (Master / pos): N=%d, feat=%d, C=%d"
        % (n, X.shape[1], labels.shape[1])
    )
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_pte():
    """PTE drug binary from pte_active; semi-supervised if .pt y has negatives."""
    from sklearn.model_selection import train_test_split

    data = _t2g_load_table_pt_prefer_unified("PTE", "PTE_Giant_HeteroGraph.pt")
    if data is None:
        raise FileNotFoundError(
            "PTE .pt not found (tried unified_data.pt, PTE_Giant_HeteroGraph.pt) under %s"
            % os.path.join(TABLE2GRAPH_ROOT, "PTE")
        )
    print("PTE: graph loaded (see table2graph logs above)")

    drug_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "PTE", "csv", "pte_drug.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "PTE", "csv", "pte_drug.csv"),
    )
    active_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "PTE", "csv", "pte_active.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "PTE", "csv", "pte_active.csv"),
    )
    atm_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "PTE", "csv", "pte_atm.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "PTE", "csv", "pte_atm.csv"),
    )
    if (drug_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in drug_csv.replace("\\", "/")) or (
        active_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in active_csv.replace("\\", "/")
    ):
        print(
            "PTE: some CSV missing in exptable2graph, using legacy %s"
            % LEGACY_TABLE2GRAPH_ROOT
        )

    preferred = ("pte_drug", "Pte_drug", "drug", "Drug")
    use_csv = drug_csv is not None and active_csv is not None

    if use_csv:
        drug_df = pd.read_csv(drug_csv)
        n_expected = len(drug_df)
        drug_ids = drug_df["drug_id"].astype(str).str.strip().values

        act_df = pd.read_csv(active_csv)
        act = {}
        for _, row in act_df.iterrows():
            did = str(row["drug_id"]).strip()
            act[did] = str(row["is_active"]).strip()

        labeled_idx = np.array(
            [i for i in range(n_expected) if drug_ids[i] in act], dtype=np.int64
        )
        n_miss = n_expected - len(labeled_idx)
        if n_miss:
            print(
                "PTE: %d drugs not in pte_active (excluded from train/val/test)."
                % n_miss
            )
        if len(labeled_idx) < 10:
            raise ValueError("pte_active: too few labeled drugs to split.")

        labels = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n_expected, 1))
        y_lab_codes = []
        for i in labeled_idx:
            is_pos = act[drug_ids[i]].upper().startswith("T")
            y_lab_codes.append(1 if is_pos else 0)
            if is_pos:
                labels[i, 0] = 0.0
                labels[i, 1] = 1.0
            else:
                labels[i, 0] = 1.0
                labels[i, 1] = 0.0

        x_t = None
        nts = _t2g_iter_node_types(data)
        if nts:
            nt_used, x_t, err = _t2g_resolve_nt_and_tensor(data, n_expected, preferred)
            if x_t is not None:
                print(
                    "PTE: .pt node type '%s' matches pte_drug.csv rows %d"
                    % (nt_used, n_expected)
                )
            elif err:
                print("PTE: no .pt features (%s), trying CSV." % err)
        if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) == n_expected:
            x_t = data.x
            print("PTE: using homogeneous x (N=%d)" % n_expected)

        if x_t is None:
            if not atm_csv or not os.path.exists(atm_csv):
                raise ValueError(
                    "no usable .pt features and pte_atm.csv missing in exptable2graph/legacy."
                )
            atom = pd.read_csv(atm_csv)
            atom["drug_id"] = atom["drug_id"].astype(str).str.strip()
            feat_wide = atom.groupby(["drug_id", "atom_type"]).size().unstack(fill_value=0)
            feat_wide = feat_wide.reindex(drug_ids).fillna(0.0)
            X = feat_wide.values.astype(np.float32)
            print("PTE: atom_type counts from pte_atm (.pt features missing)")
        else:
            X = x_t.detach().cpu().numpy().astype(np.float32)

        n = X.shape[0]
        if n != n_expected:
            raise ValueError("feature rows %d != pte_drug.csv rows %d" % (n, n_expected))
    else:
        X, y_bin, nt_pt = _t2g_xy_from_pt_relaxed(data, preferred)
        if X is None:
            raise FileNotFoundError(
                "pte_drug/pte_active CSV not found (exptable2graph and %s); .pt x/y mismatch. node_types=%s"
                % (LEGACY_TABLE2GRAPH_ROOT, _t2g_iter_node_types(data))
            )
        n_expected = int(X.shape[0])
        n = n_expected
        drug_ids = np.array([str(i) for i in range(n_expected)], dtype=object)
        y_bin = y_bin.astype(np.int64)
        u = np.unique(y_bin)

        if np.any(y_bin < 0):
            labeled = y_bin >= 0
            n_unlab = int((~labeled).sum())
            if int(labeled.sum()) < 10:
                raise ValueError(
                    "PTE: fewer than 10 labeled nodes (y>=0); check .pt or add CSV."
                )
            y_sup = y_bin[labeled]
            u_sup = np.unique(y_sup)
            if u_sup.size < 2:
                raise ValueError(
                    "PTE: labeled y has one class only; add pte_drug.csv and pte_active.csv."
                )
            if u_sup.size > 2:
                raise ValueError(
                    "PTE: labeled y has %d classes (need binary); add CSV."
                    % int(u_sup.size)
                )
            us = np.sort(u_sup)
            y_rem = (y_sup == us[1]).astype(np.int64)
            labels = np.zeros((n_expected, 2), dtype=np.float32)
            li = np.where(labeled)[0]
            labels[li, y_rem] = 1.0
            labeled_idx = li.astype(np.int64)
            y_lab_codes = list(y_rem)
            n0 = int((y_rem == 0).sum())
            n1 = int((y_rem == 1).sum())
            print(
                "PTE: %d negative y (unlabeled); full N=%d, split on %d labeled; class0=%d class1=%d"
                % (n_unlab, n_expected, len(li), n0, n1)
            )
            print(
                "PTE: no CSV, semi-supervised binary on .pt node type '%s'"
                % nt_pt
            )
        else:
            if u.size < 2:
                raise ValueError(
                    "PTE: .pt y has a single value; add CSV or fix data."
                )
            if u.size > 2:
                raise ValueError(
                    "PTE: .pt y has %d distinct non-negative values (multiclass); add CSV."
                    % int(u.size)
                )
            us = np.sort(u)
            y2 = y_bin.copy()
            if int(us[0]) != 0 or int(us[1]) != 1:
                y2 = (y_bin == us[1]).astype(np.int64)
                print(
                    "PTE: mapped labels {%s,%s} -> {0,1}"
                    % (int(us[0]), int(us[1]))
                )
            labels = np.zeros((n_expected, 2), dtype=np.float32)
            labels[np.arange(n_expected), y2] = 1.0
            labeled_idx = np.arange(n_expected, dtype=np.int64)
            y_lab_codes = list(y2)
            n0 = int((y2 == 0).sum())
            n1 = int((y2 == 1).sum())
            print(
                "PTE: no CSV, using .pt x/y on '%s' (N=%d, all labeled; class0=%d class1=%d)"
                % (nt_pt, n_expected, n0, n1)
            )

    features = sp.csr_matrix(X)
    adj = _t2g_build_adj_from_pt(data, n, "PTE")

    y_lab = np.array(y_lab_codes, dtype=np.int64)
    rel = np.arange(len(labeled_idx))
    try:
        r_train, r_temp = train_test_split(
            rel, train_size=0.6, random_state=42, stratify=y_lab
        )
        y_temp = y_lab[r_temp]
        r_val, r_test = train_test_split(
            r_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        r_train, r_temp = train_test_split(rel, train_size=0.6, random_state=42)
        r_val, r_test = train_test_split(
            r_temp, test_size=0.5, random_state=42
        )

    idx_train = labeled_idx[r_train].tolist()
    idx_val = labeled_idx[r_val].tolist()
    idx_test = labeled_idx[r_test].tolist()

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    print(
        "PTE (pte_drug / is_active): N=%d, labeled=%d, feat=%d, C=%d"
        % (n, len(labeled_idx), X.shape[1], labels.shape[1])
    )
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_toxicology():
    """Toxicology molecule binary (+/-); imbalanced metrics are expected sometimes."""
    from sklearn.model_selection import train_test_split

    data = _t2g_load_table_pt_prefer_unified(
        "Toxicology_data", "Toxicology_HeteroGraph.pt"
    )
    if data is None:
        raise FileNotFoundError(
            "Toxicology .pt not found (tried unified_data.pt, Toxicology_HeteroGraph.pt) under %s"
            % os.path.join(TABLE2GRAPH_ROOT, "Toxicology_data")
        )
    print("Toxicology: graph loaded (see table2graph logs above)")

    mol_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "Toxicology_data", "csv", "molecule.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "Toxicology_data", "csv", "molecule.csv"),
    )
    atom_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "Toxicology_data", "csv", "atom.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "Toxicology_data", "csv", "atom.csv"),
    )
    if mol_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in mol_csv.replace(
        "\\", "/"
    ):
        print("Toxicology: molecule.csv not in exptable2graph, using %s" % mol_csv)

    preferred = ("molecule", "Molecule", "mol", "Mol")
    use_csv = mol_csv is not None

    if use_csv:
        mol_df = pd.read_csv(mol_csv)
        n_expected = len(mol_df)
        if "label" not in mol_df.columns:
            raise ValueError(
                "molecule.csv needs label column (+/-); got: %s" % list(mol_df.columns)
            )

        mol_ids = mol_df["molecule_id"].astype(str).str.strip().values
        raw_lab = mol_df["label"].astype(str).str.strip().str.upper()
        y_int = np.array([1 if x == "+" else 0 for x in raw_lab], dtype=np.int64)

        x_t = None
        nts = _t2g_iter_node_types(data)
        if nts:
            nt_used, x_t, err = _t2g_resolve_nt_and_tensor(data, n_expected, preferred)
            if x_t is not None:
                print(
                    "Toxicology: .pt node type '%s' matches molecule.csv rows %d"
                    % (nt_used, n_expected)
                )
            elif err:
                print("Toxicology: no .pt features (%s), trying CSV." % err)
        if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) == n_expected:
            x_t = data.x
            print("Toxicology: using homogeneous x (N=%d)" % n_expected)

        if x_t is None:
            if not atom_csv or not os.path.exists(atom_csv):
                raise ValueError(
                    "no usable .pt features and atom.csv missing in exptable2graph/legacy."
                )
            atom = pd.read_csv(atom_csv)
            atom["molecule_id"] = atom["molecule_id"].astype(str).str.strip()
            feat_wide = atom.groupby(["molecule_id", "element"]).size().unstack(fill_value=0)
            feat_wide = feat_wide.reindex(mol_ids).fillna(0.0)
            X = feat_wide.values.astype(np.float32)
            print("Toxicology: element counts from atom.csv (.pt features missing)")
        else:
            X = x_t.detach().cpu().numpy().astype(np.float32)

        n = X.shape[0]
        if n != n_expected:
            raise ValueError("feature rows %d != molecule.csv rows %d" % (n, n_expected))
        y_bin = y_int.astype(np.int64)
        labeled_idx = np.arange(n, dtype=np.int64)
    else:
        X, y_int, nt_pt = _t2g_xy_from_pt_relaxed(data, preferred)
        if X is None:
            raise FileNotFoundError(
                "molecule.csv not found (exptable2graph and %s); .pt x/y mismatch. node_types=%s"
                % (LEGACY_TABLE2GRAPH_ROOT, _t2g_iter_node_types(data))
            )
        n_expected = int(X.shape[0])
        n = n_expected
        y_int = y_int.astype(np.int64)
        valid = y_int >= 0
        labeled_idx = np.flatnonzero(valid)
        if labeled_idx.size < 2:
            raise ValueError(
                "Toxicology: without molecule.csv, .pt needs at least 2 labeled nodes (y>=0)."
            )
        yv = y_int[labeled_idx]
        u = np.unique(yv)
        if u.size > 2:
            raise ValueError(
                "Toxicology: labeled y has %d values (need binary); add molecule.csv."
                % int(u.size)
            )
        if u.size < 2:
            raise ValueError("Toxicology: labeled y has a single class (need binary).")
        us = np.sort(u)
        y_bin = np.zeros(n, dtype=np.int64)
        if int(us[0]) == 0 and int(us[1]) == 1:
            y_bin[labeled_idx] = yv
        else:
            y_bin[labeled_idx] = (yv == us[1]).astype(np.int64)
            print(
                "Toxicology: mapped labels {%s,%s} -> {0,1}"
                % (int(us[0]), int(us[1]))
            )
        if np.any(y_int < 0):
            print(
                "Toxicology: y<0 treated unlabeled; split on %d labeled nodes only"
                % int(labeled_idx.size)
            )
        print(
            "Toxicology: no molecule.csv, using .pt x/y on '%s' (N=%d)"
            % (nt_pt, n_expected)
        )

    n_neg = int((y_bin[labeled_idx] == 0).sum())
    n_pos = int((y_bin[labeled_idx] == 1).sum())
    print(
        "Toxicology: labeled counts neg=%d pos=%d (pos %.2f%%)"
        % (n_neg, n_pos, 100.0 * n_pos / max(labeled_idx.size, 1))
    )
    if max(n_neg, n_pos) / max(labeled_idx.size, 1) >= 0.9:
        print(
            "Toxicology: heavy class imbalance inflates acc/micro vs macro-F1."
        )

    lb = LabelBinarizer()
    blk = lb.fit_transform(y_bin[labeled_idx])
    if blk.shape[1] == 1:
        blk = np.hstack([1 - blk, blk])
    labels = np.zeros((n, blk.shape[1]), dtype=np.float64)
    labels[labeled_idx] = blk

    features = sp.csr_matrix(X)
    adj = _t2g_build_adj_from_pt(data, n, "Toxicology")

    idx_pool = labeled_idx
    strat = y_bin[labeled_idx]
    try:
        idx_train, idx_temp = train_test_split(
            idx_pool, train_size=0.6, random_state=42, stratify=strat
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.5, random_state=42, stratify=y_bin[idx_temp]
        )
    except ValueError:
        idx_train, idx_temp = train_test_split(
            idx_pool, train_size=0.6, random_state=42
        )
        idx_val, idx_test = train_test_split(
            idx_temp, test_size=0.5, random_state=42
        )

    train_mask = sample_mask(idx_train.tolist(), n)
    val_mask = sample_mask(idx_val.tolist(), n)
    test_mask = sample_mask(idx_test.tolist(), n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    print(
        "Toxicology (molecule / +/-): N=%d, feat=%d, C=%d"
        % (n, X.shape[1], labels.shape[1])
    )
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_f1():
    """f1 table2graph NC; same pattern as hepatitis_std (.pt + dispat.csv)."""
    from sklearn.model_selection import train_test_split

    data = _t2g_load_f1_pt_prefer_unified()
    if data is None:
        raise FileNotFoundError(
            "f1 .pt not found (tried unified, *_HeteroGraph, ultimate, etc.) under %s or legacy"
            % os.path.join(TABLE2GRAPH_ROOT, "f1")
        )
    print("f1: graph loaded (see table2graph logs above)")

    dispat_csv = _t2g_resolve_csv(
        os.path.join(TABLE2GRAPH_ROOT, "f1", "csv", "dispat.csv"),
        os.path.join(LEGACY_TABLE2GRAPH_ROOT, "f1", "csv", "dispat.csv"),
    )
    if dispat_csv and LEGACY_TABLE2GRAPH_ROOT.replace("\\", "/") in dispat_csv.replace(
        "\\", "/"
    ):
        print("f1: dispat.csv not in exptable2graph, using %s" % dispat_csv)

    preferred = ("f1", "dispat", "Dispat", "patient", "Patient", "node")
    use_csv = dispat_csv is not None

    if use_csv:
        dispat_df = pd.read_csv(dispat_csv)
        n_expected = len(dispat_df)
        if "Type" in dispat_df.columns:
            y_int = dispat_df["Type"].values.astype(np.int64)
        elif "type" in dispat_df.columns:
            y_int = dispat_df["type"].values.astype(np.int64)
        else:
            raise ValueError(
                "f1/dispat.csv needs Type or type label column; got: %s"
                % list(dispat_df.columns)
            )

        x_t = None
        nts = _t2g_iter_node_types(data)
        if nts:
            nt_used, x_t, err = _t2g_resolve_nt_and_tensor(data, n_expected, preferred)
            if x_t is not None:
                print(
                    "f1: .pt node type '%s' matches dispat.csv rows %d"
                    % (nt_used, n_expected)
                )
            elif err:
                print("f1: no .pt features (%s), trying CSV." % err)
        if hasattr(data, "x") and data.x is not None and int(data.x.shape[0]) == n_expected:
            x_t = data.x
            print("f1: using homogeneous x (N=%d)" % n_expected)

        if x_t is None:
            if "sex" not in dispat_df.columns or "age" not in dispat_df.columns:
                raise ValueError(
                    "no usable .pt features and dispat.csv missing sex/age columns"
                )
            X = np.stack(
                [
                    dispat_df["sex"].astype(np.float32).values,
                    dispat_df["age"].astype(np.float32).values,
                ],
                axis=1,
            )
            print("f1: sex/age from dispat.csv (.pt features missing)")
        else:
            X = x_t.detach().cpu().numpy().astype(np.float32)

        n = X.shape[0]
        if n != n_expected:
            raise ValueError("feature rows %d != dispat.csv rows %d" % (n, n_expected))
    else:
        X, y_int, nt_pt = _t2g_xy_from_pt_relaxed(data, preferred)
        if X is None:
            raise FileNotFoundError(
                "f1/dispat.csv not found (exptable2graph and %s); .pt x/y mismatch. node_types=%s"
                % (LEGACY_TABLE2GRAPH_ROOT, _t2g_iter_node_types(data))
            )
        n_expected = int(X.shape[0])
        n = n_expected
        print("f1: no dispat.csv, using .pt x/y (%s, N=%d)" % (nt_pt, n_expected))

    lb = LabelBinarizer()
    labels = lb.fit_transform(y_int)
    if labels.shape[1] == 1:
        labels = np.hstack([1 - labels, labels])

    features = sp.csr_matrix(X)
    adj = _t2g_build_adj_from_pt(data, n, "f1")

    idx = np.arange(n)
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, random_state=42, stratify=y_int
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42, stratify=y_int[idx_temp]
    )

    train_mask = sample_mask(idx_train.tolist(), n)
    val_mask = sample_mask(idx_val.tolist(), n)
    test_mask = sample_mask(idx_test.tolist(), n)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]

    print(
        "f1 (dispat): N=%d, feat=%d, C=%d" % (n, X.shape[1], labels.shape[1])
    )
    print("adj:", adj.shape, "features:", features.shape)

    return adj.tocsr(), features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_random_data(size):

    adj = sp.random(size, size, density=0.002) # density similar to cora
    features = sp.random(size, 1000, density=0.015)
    int_labels = np.random.randint(7, size=(size))
    labels = np.zeros((size, 7)) # Nx7
    labels[np.arange(size), int_labels] = 1

    train_mask = np.zeros((size,)).astype(bool)
    train_mask[np.arange(size)[0:int(size/2)]] = 1

    val_mask = np.zeros((size,)).astype(bool)
    val_mask[np.arange(size)[int(size/2):]] = 1

    test_mask = np.zeros((size,)).astype(bool)
    test_mask[np.arange(size)[int(size/2):]] = 1

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
  
    # sparse NxN, sparse NxF, norm NxC, ..., norm Nx1, ...
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features_train_standardize(features, train_mask):
    """Z-score from train rows only; drop zero-variance cols on train."""
    tm = np.asarray(train_mask, dtype=bool).ravel()
    if sp.issparse(features):
        f = features.tocsr().astype(np.float64)
    else:
        f = sp.csr_matrix(np.asarray(features, dtype=np.float64))
    f_tr = f[tm]
    mu = np.asarray(f_tr.mean(axis=0)).ravel()
    if sp.issparse(f_tr):
        sq_mean = np.asarray(f_tr.power(2).mean(axis=0)).ravel()
        sigma = np.sqrt(np.maximum(sq_mean - mu * mu, 0.0))
    else:
        sigma = np.asarray(f_tr.std(axis=0)).ravel()
    keep = sigma > 1e-8
    if not np.any(keep):
        print(
            "preprocess_features_train_standardize: zero train variance; "
            "falling back to row-normalize (preprocess_features)."
        )
        out, _ = preprocess_features(
            features if sp.issparse(features) else sp.csr_matrix(np.asarray(features))
        )
        return np.asarray(out, dtype=np.float32)
    mu = mu[keep]
    sigma = sigma[keep]
    f = f[:, keep]
    out = (f.toarray() - mu) / np.maximum(sigma, 1e-12)
    return np.asarray(out, dtype=np.float32)


def preprocess_features_raw_dense(features):
    """Dense float32 only; no row norm or z-score."""
    if sp.issparse(features):
        X = features.toarray()
    else:
        X = np.asarray(features)
    return np.asarray(X, dtype=np.float32)


def maybe_exptable2graph_sanitize_nonfinite(X, enabled, tag="features"):
    """If enabled and non-finite values exist, nan_to_num; else return as-is."""
    if not enabled:
        return np.asarray(X, dtype=np.float32)
    X = np.asarray(X, dtype=np.float32)
    if np.isfinite(X).all():
        return X
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print("%s: non-finite values -> nan_to_num" % tag)
    return X


def sanitize_dense_features_for_hypnet(X, tag="features"):
    """Legacy alias: nan_to_num on full dense X. Prefer maybe_exptable2graph_sanitize_nonfinite."""
    X = np.asarray(X, dtype=np.float32)
    if np.isfinite(X).all():
        return X
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print("%s: non-finite values -> nan_to_num" % tag)
    return X


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_adj_bias(adj):
    num_nodes = adj.shape[0]
    adj = adj + sp.eye(num_nodes)  # self-loop
    adj[adj > 0.0] = 1.0
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    adj = adj.astype(np.float32)
    indices = np.vstack((adj.col, adj.row)).transpose()  # This is where I made a mistake, I used (adj.row, adj.col) instead
    # return tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
    return indices, adj.data, adj.shape
# ################ MY DATASET: DISEASE ####################################
def load_data_disease(use_feats, data_path, return_label=True):
    """Load disease_nc (airport-style pipeline)."""
    import pandas as pd
    import numpy as np
    import scipy.sparse as sp
    import networkx as nx
    import os
    from sklearn.preprocessing import LabelBinarizer

    edges = pd.read_csv(os.path.join(data_path, "disease_nc.edges.csv"), header=None).values
    G = nx.Graph()
    G.add_edges_from(edges)
    
    all_nodes = sorted(G.nodes())
    n_nodes = len(all_nodes)
    
    adj = nx.adjacency_matrix(G, nodelist=all_nodes)
    
    adj = adj.astype(np.float32)
    
    if not sp.isspmatrix_csr(adj):
        adj = adj.tocsr()

    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "disease_nc.feats.npz"))
        if not sp.isspmatrix_csr(features):
            features = features.tocsr()
        features = features.astype(np.float32)
        if features.shape[0] != n_nodes:
            features = sp.eye(n_nodes, dtype=np.float32).tocsr()
    else:
        features = sp.eye(n_nodes, dtype=np.float32).tocsr()

    labels_raw = np.load(os.path.join(data_path, "disease_nc.labels.npy"))
    
    if len(labels_raw) != n_nodes:
        if len(labels_raw) > n_nodes:
            labels_raw = labels_raw[:n_nodes]
        else:
            most_common = np.bincount(labels_raw).argmax()
            labels_raw = np.pad(labels_raw, (0, n_nodes - len(labels_raw)), constant_values=most_common)
    
    unique_labels = np.unique(labels_raw)
    n_classes = len(unique_labels)
    
    print(f'Label distribution: {np.bincount(labels_raw.astype(int))}')
    print(f'Number of classes: {n_classes}')
    
    if n_classes <= 2:
        if n_classes == 1:
            print('Warning: Only 1 class found! Creating dummy binary labels.')
            labels = np.zeros((n_nodes, 2))
            labels[:, 0] = 1
        else:
            lb = LabelBinarizer()
            labels = lb.fit_transform(labels_raw)
            if labels.shape[1] == 1:
                labels = np.hstack([1 - labels, labels])
    else:
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels_raw)

    num_nodes = labels.shape[0]
    
    np.random.seed(42)
    idx = np.random.permutation(num_nodes)
    
    idx_train = idx[:int(0.6*num_nodes)]
    idx_val = idx[int(0.6*num_nodes):int(0.8*num_nodes)]
    idx_test = idx[int(0.8*num_nodes):]
    
    train_mask = sample_mask(idx_train, num_nodes)
    val_mask = sample_mask(idx_val, num_nodes)
    test_mask = sample_mask(idx_test, num_nodes)

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask] = labels[train_mask]
    y_val[val_mask] = labels[val_mask]
    y_test[test_mask] = labels[test_mask]
    
    print(f'Disease dataset loaded:')
    print(f'  Nodes: {num_nodes}, Edges: {adj.nnz}')
    print(f'  Features shape: {features.shape}')
    print(f'  Labels shape: {labels.shape}')
    print(f'  Train/Val/Test: {train_mask.sum()}/{val_mask.sum()}/{test_mask.sum()}')
    print(f'  Adj type: {type(adj)}, format: {adj.format if hasattr(adj, "format") else "N/A"}')
    
    np.random.seed(None)
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask