"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import re
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None

from .exptable2graph_loader import is_exptable_dataset, load_exptable2graph_folder


def _torch_load_pt(path, map_location='cpu'):
    """torch.load with optional weights_only (PyTorch version differences)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _film_dataset_names():
    """WebKB + Film datasets using raw/out1_*.txt layout."""
    return frozenset({
        'cornell', 'wisconsin', 'texas', 'actor',
        'Cornell', 'Wisconsin', 'Texas', 'Actor',
    })


def load_film_style_graph(data_path, use_feats=True):
    """Parse raw/out1_node_feature_label.txt and out1_graph_edges.txt (bag-of-words or 0/1 feats)."""
    raw = os.path.join(data_path, 'raw')
    node_path = os.path.join(raw, 'out1_node_feature_label.txt')
    edge_path = os.path.join(raw, 'out1_graph_edges.txt')
    feat_dim = 931
    nodes_order = []
    feat_rows = []
    raw_labs = []
    bow_mode = False

    with open(node_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline()
        low = header.lower()
        if 'feature_amount' in low:
            bow_mode = True
            m = re.search(r'feature_amount:(\d+)', header)
            if m:
                feat_dim = int(m.group(1))
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            feat_str = parts[1]
            lab = int(parts[2])
            nodes_order.append(nid)
            feat_rows.append(feat_str)
            raw_labs.append(lab)

    n = len(nodes_order)
    if n == 0:
        raise ValueError('no nodes parsed from {}'.format(node_path))
    nid2idx = {nid: i for i, nid in enumerate(nodes_order)}
    labels = np.asarray(raw_labs, dtype=np.int64)
    labels = labels - labels.min()

    if not use_feats:
        features = sp.eye(n)
    elif bow_mode:
        X = sp.lil_matrix((n, feat_dim), dtype=np.float32)
        for i, feat_str in enumerate(feat_rows):
            for tok in feat_str.split(','):
                t = tok.strip()
                if not t:
                    continue
                j = int(t)
                if 0 <= j < feat_dim:
                    X[i, j] = 1.0
        features = X.tocsr()
    else:
        # dense 0/1 feature strings (WebKB)
        row_vecs = []
        for feat_str in feat_rows:
            vec = np.asarray([float(x) for x in feat_str.split(',')], dtype=np.float32)
            row_vecs.append(vec)
        feat_mat = np.vstack(row_vecs)
        features = sp.csr_matrix(feat_mat)

    adj = sp.lil_matrix((n, n), dtype=np.float32)
    with open(edge_path, 'r', encoding='utf-8', errors='ignore') as ef:
        ef.readline()
        for line in ef:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split('\t') if '\t' in line else line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u in nid2idx and v in nid2idx:
                iu, iv = nid2idx[u], nid2idx[v]
                if iu != iv:
                    adj[iu, iv] = 1.0
                    adj[iv, iu] = 1.0
    adj = adj.tocsr()
    return adj, features, labels


def load_film_style_nc(data_path, split_seed, use_feats=True):
    """Stratified 60/20/20 train/val/test split."""
    if train_test_split is None:
        raise RuntimeError('scikit-learn required: pip install scikit-learn')
    adj, features, labels = load_film_style_graph(data_path, use_feats)
    inds = np.arange(labels.shape[0])
    idx_tr, idx_tmp, _, y_tmp = train_test_split(
        inds, labels, test_size=0.4, stratify=labels, random_state=int(split_seed))
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=0.5, stratify=y_tmp, random_state=int(split_seed))
    return adj, features, labels, idx_tr.tolist(), idx_va.tolist(), idx_te.tolist()


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    else:
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        adj = data['adj_train']
        if args.task == 'lp':
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])
    return data


# ############### FEATURES PROCESSING ####################################


def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    """Edge split without dense (1-A) matrix (memory-safe on large graphs)."""
    np.random.seed(seed)
    adj = adj.tocsr()
    n = adj.shape[0]
    x, y = sp.triu(adj, k=1).nonzero()
    pos_edges = np.stack([x, y], axis=1)
    if len(pos_edges) == 0:
        raise ValueError('no edges in upper triangle; cannot split for link prediction')
    np.random.shuffle(pos_edges)
    m_pos = len(pos_edges)
    n_val = max(1, int(m_pos * val_prop))
    n_test = max(1, int(m_pos * test_prop))
    if n_val + n_test >= m_pos:
        n_val = max(1, m_pos // 5)
        n_test = max(1, m_pos // 5)
        if n_val + n_test >= m_pos:
            n_test = m_pos - n_val - 1
    val_edges = pos_edges[:n_val]
    test_edges = pos_edges[n_val:n_val + n_test]
    train_edges = pos_edges[n_val + n_test:]

    def _sample_neg(n_need, forbidden_csr):
        """Sample random non-edges u<v not in forbidden."""
        out = []
        tries = 0
        max_tries = n_need * 50 + 1000
        while len(out) < n_need and tries < max_tries:
            tries += 1
            u = np.random.randint(0, n)
            v = np.random.randint(0, n)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            if forbidden_csr[u, v] != 0:
                continue
            out.append([u, v])
        if len(out) < n_need:
            raise RuntimeError('negative edge sampling failed; reduce val/test ratio or check density')
        return np.asarray(out[:n_need], dtype=np.int64)

    adj_und = adj + adj.T
    adj_und.setdiag(1)
    forbidden = adj_und.tocsr()
    val_edges_false = _sample_neg(n_val, forbidden)
    test_edges_false = _sample_neg(n_test, forbidden)
    n_tr_neg = min(len(train_edges) * 2, n * (n - 1) // 2)
    n_tr_neg = max(len(train_edges), min(n_tr_neg, 500000))
    train_edges_false = _sample_neg(n_tr_neg, forbidden)

    adj_train = sp.csr_matrix(
        (np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])),
        shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def _torch_adj_to_scipy_csr(adj):
    """Torch dense or sparse COO adj -> scipy csr float32."""
    if torch.is_tensor(adj):
        is_sp = bool(getattr(adj, 'is_sparse', False))
        if not is_sp and hasattr(adj, 'layout'):
            is_sp = str(adj.layout).endswith('sparse_coo')
        if is_sp:
            adj = adj.coalesce()
            idx = adj.indices().detach().cpu().numpy()
            vals = adj.values().detach().cpu().numpy().astype(np.float32)
            n = int(adj.size(0))
            mat = sp.csr_matrix((vals, (idx[0], idx[1])), shape=(n, n), dtype=np.float32)
            return mat
        a = adj.detach().cpu().numpy()
        if a.ndim != 2:
            raise ValueError('adj must be 2D, got shape {}'.format(a.shape))
        mat = sp.csr_matrix(a.astype(np.float32))
        return mat
    raise TypeError('unsupported adj type: {}'.format(type(adj)))


def _torch_feats_to_sparse(features, use_feats, n_nodes):
    if not use_feats:
        return sp.eye(n_nodes, dtype=np.float32, format='csr')
    if not torch.is_tensor(features):
        raise TypeError('unsupported feats type: {}'.format(type(features)))
    x = features.detach().cpu().numpy()
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] != n_nodes:
        raise ValueError('feats rows {} != n_nodes {}'.format(x.shape[0], n_nodes))
    return sp.csr_matrix(x.astype(np.float32))


def _resolve_cs_phds_pt_file(data_path, logical_name, candidates):
    """Resolve a .pt path: try candidates, case-insensitive basename, then fuzzy single match."""
    tried = list(candidates)
    for name in candidates:
        p = os.path.join(data_path, name)
        if os.path.isfile(p):
            return p
    if os.path.isdir(data_path):
        lower_index = {}
        for fn in os.listdir(data_path):
            if fn.lower().endswith('.pt'):
                lower_index.setdefault(fn.lower(), fn)
        for name in candidates:
            key = name.lower()
            if key in lower_index:
                return os.path.join(data_path, lower_index[key])

        pt_files = sorted(
            fn for fn in os.listdir(data_path) if fn.lower().endswith('.pt'))
        if logical_name == 'adj':
            hits = [
                fn for fn in pt_files
                if 'dists' not in fn.lower() and 'dist' not in fn.lower()
                and ('adj' in fn.lower() or fn.lower() in ('a.pt',))
                and not fn.lower().startswith('dist')]
        elif logical_name == 'feats':
            hits = [
                fn for fn in pt_files
                if 'label' not in fn.lower() and 'dists' not in fn.lower()
                and ('feat' in fn.lower() or 'feature' in fn.lower()
                     or fn.lower() in ('x.pt',))]
        else:
            hits = [
                fn for fn in pt_files
                if 'label' in fn.lower() or fn.lower() in ('y.pt',)]
        if len(hits) == 1:
            return os.path.join(data_path, hits[0])
        if len(hits) > 1:
            raise FileNotFoundError(
                'cs_phds: multiple .pt files match "{}": rename to one of {} or keep a single match: {}'
                .format(logical_name, list(candidates)[:3], hits))

    listing = ''
    if os.path.isdir(data_path):
        listing = ', '.join(sorted(os.listdir(data_path))) or '(empty)'
    else:
        listing = '(missing dir)'
    raise FileNotFoundError(
        'cs_phds: no .pt for "{}". dir: {}\n  tried: {}\n  listing: {}'.format(
            logical_name, data_path, tried[:6], listing))


_CS_PHDS_ADJ_CANDIDATES = (
    'adj_train.pt', 'Adj_train.pt',
    'adj.pt', 'Adj.pt', 'ADJ.pt', 'adjacency.pt', 'graph_adj.pt', 'A.pt',
)
_CS_PHDS_FEATS_CANDIDATES = (
    'feats_train.pt', 'Feats_train.pt',
    'feats.pt', 'Feats.pt', 'features.pt', 'feature.pt', 'feat.pt', 'x.pt', 'X.pt',
)
_CS_PHDS_LABELS_CANDIDATES = (
    'labels_train.pt', 'Labels_train.pt',
    'labels.pt', 'Labels.pt', 'label.pt', 'y.pt', 'Y.pt',
)


def load_cs_phds_ready_folder(data_path, use_feats, need_labels, split_seed):
    """Load adj/feats[/labels] .pt from cs_phds_*_ready (adj_train.pt etc.; dists*.pt ignored)."""
    adj_path = _resolve_cs_phds_pt_file(data_path, 'adj', _CS_PHDS_ADJ_CANDIDATES)
    feats_path = _resolve_cs_phds_pt_file(data_path, 'feats', _CS_PHDS_FEATS_CANDIDATES)

    labels_path = None
    if need_labels:
        labels_path = _resolve_cs_phds_pt_file(
            data_path, 'labels', _CS_PHDS_LABELS_CANDIDATES)

    adj = _torch_load_pt(adj_path)
    feats = _torch_load_pt(feats_path)
    adj = _torch_adj_to_scipy_csr(adj)
    n = adj.shape[0]
    features = _torch_feats_to_sparse(feats, use_feats, n)

    labels = None
    idx_train = idx_val = idx_test = None
    if need_labels:
        lab_t = _torch_load_pt(labels_path)
        if torch.is_tensor(lab_t):
            labels = lab_t.detach().cpu().numpy().reshape(-1).astype(np.int64)
        else:
            labels = np.asarray(lab_t, dtype=np.int64).reshape(-1)
        if labels.shape[0] != n:
            raise ValueError('labels len {} != n_nodes {}'.format(len(labels), n))
        if train_test_split is None:
            raise RuntimeError('scikit-learn required: pip install scikit-learn')
        inds = np.arange(n)
        idx_tr, idx_tmp, _, y_tmp = train_test_split(
            inds, labels, test_size=0.4, stratify=labels, random_state=int(split_seed))
        idx_va, idx_te = train_test_split(
            idx_tmp, test_size=0.5, stratify=y_tmp, random_state=int(split_seed))
        idx_train, idx_val, idx_test = idx_tr.tolist(), idx_va.tolist(), idx_te.tolist()

    return adj, features, labels, idx_train, idx_val, idx_test


def load_telecom_layout(data_path, use_feats):
    """Read edges.csv, feats.npz, labels.npy (see telecom/convert_telecom_to_gcn.py)."""
    edges_path = os.path.join(data_path, 'edges.csv')
    feats_path = os.path.join(data_path, 'feats.npz')
    labels_path = os.path.join(data_path, 'labels.npy')
    if not os.path.isfile(edges_path):
        raise FileNotFoundError(
            'missing {}; run telecom/convert_telecom_to_gcn.py in that folder'.format(edges_path))
    rows, cols = [], []
    max_id = 0
    with open(edges_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            rows.append(u)
            cols.append(v)
            max_id = max(max_id, u, v)
    n_min = max_id + 1
    labels = None
    if os.path.isfile(labels_path):
        labels = np.load(labels_path)
    features = None
    if use_feats and os.path.isfile(feats_path):
        features = sp.load_npz(feats_path)
    n = n_min
    if features is not None:
        n = max(n, features.shape[0])
    if labels is not None:
        n = max(n, len(labels))
    if features is None:
        features = sp.eye(n)
    elif features.shape[0] != n:
        raise ValueError(
            'telecom: feats rows {} != n {}'.format(features.shape[0], n))
    if labels is None:
        labels = np.zeros(n, dtype=np.int64)
    elif len(labels) != n:
        raise ValueError(
            'telecom: labels len {} != n {}'.format(len(labels), n))
    adj = sp.lil_matrix((n, n), dtype=np.float32)
    for u, v in zip(rows, cols):
        if u != v:
            adj[u, v] = 1.0
            adj[v, u] = 1.0
    adj = adj.tocsr()
    return adj, features, labels


# ############### LINK PREDICTION DATA LOADERS ####################################


def load_data_lp(dataset, use_feats, data_path):
    if dataset in ['cora', 'pubmed']:
        adj, features = load_citation_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'disease_lp':
        adj, features = load_synthetic_data(dataset, use_feats, data_path)[:2]
    elif dataset == 'airport':
        adj, features = load_data_airport(dataset, data_path, return_label=False)
    elif dataset == 'citeseer':
        adj, features, _ = load_synthetic_data('citeseer', use_feats, data_path)
    elif dataset in _film_dataset_names():
        adj, features = load_film_style_graph(data_path, use_feats)[:2]
    elif is_exptable_dataset(dataset):
        adj, features, _, _, _, _ = load_exptable2graph_folder(
            data_path, use_feats=use_feats, split_seed=0)
    elif dataset == 'telecom_lp':
        adj, features, _ = load_telecom_layout(data_path, use_feats)
    elif dataset == 'cs_phds_lp':
        adj, features, _, _, _, _ = load_cs_phds_ready_folder(
            data_path, use_feats, need_labels=False, split_seed=0)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    elif dataset in _film_dataset_names():
        adj, features, labels, idx_train, idx_val, idx_test = load_film_style_nc(
            data_path, split_seed, use_feats
        )
    elif is_exptable_dataset(dataset):
        adj, features, labels, idx_train, idx_val, idx_test = load_exptable2graph_folder(
            data_path, use_feats=use_feats, split_seed=split_seed
        )
    elif dataset == 'cs_phds_nc':
        adj, features, labels, idx_train, idx_val, idx_test = load_cs_phds_ready_folder(
            data_path, use_feats, need_labels=True, split_seed=split_seed)
    else:
        if dataset == 'disease_nc':
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'citeseer':
            adj, features, labels = load_synthetic_data('citeseer', use_feats, data_path)
            val_prop, test_prop = 0.10, 0.10
        elif dataset == 'telecom_nc':
            adj, features, labels = load_telecom_layout(data_path, use_feats)
            val_prop, test_prop = 0.10, 0.10
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    return data


# ############### DATASETS ####################################


def load_citation_data(dataset_str, use_feats, data_path, split_seed=None):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.{}.test.index".format(dataset_str)))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + min(1000, len(labels) - len(y) - len(idx_test)))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    if not use_feats:
        features = sp.eye(adj.shape[0])
    return adj, features, labels, idx_train, idx_val, idx_test


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_synthetic_data(dataset_str, use_feats, data_path):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(os.path.join(data_path, "{}.edges.csv".format(dataset_str)), 'r') as f:
        all_edges = f.readlines()
    for line in all_edges:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
    labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels


def load_data_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    
    features = np.array([graph.nodes[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features

