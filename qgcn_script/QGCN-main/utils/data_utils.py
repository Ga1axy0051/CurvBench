"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from .preprocessing import *
import random

HETERO_DATASET_SPECS = {
    'carcinogenesis_data': {
        'dir_name': 'Carcinogenesis_data',
        'pt_file': 'Carcinogenesis_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'target_node_type': 'canc',
        'label_csv': os.path.join('csv', 'canc.csv'),
        'label_column': 'class',
    },
    'hepatitis_std_data': {
        'dir_name': 'Hepatitis_std_data',
        'pt_file': 'Hepatitis_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'target_node_type': 'dispat',
        'label_csv': os.path.join('csv', 'dispat.csv'),
        'label_column': 'Type',
    },
    'hockey_data': {
        'dir_name': 'Hockey_data',
        'pt_file': 'Hockey_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'target_node_type': 'Master',
        'label_csv': os.path.join('csv', 'Master.csv'),
        'label_column': 'pos',
    },
    'f1_ultimate_hetero_graph': {
        'dir_name': 'f1_ultimate_hetero_graph',
        'pt_file': 'f1_ultimate_hetero_graph.pt',
        'unified_dir_name': 'f1',
        'unified_file': 'unified_data.pt',
    },
    'pte': {
        'dir_name': 'PTE',
        'pt_file': 'PTE_Giant_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'target_node_type': 'pte_active',
        'label_csv': os.path.join('csv', 'pte_active.csv'),
        'label_column': 'is_active',
    },
    'toxicology_data': {
        'dir_name': 'Toxicology_data',
        'pt_file': 'Toxicology_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'target_node_type': 'molecule',
        'label_csv': os.path.join('csv', 'molecule.csv'),
        'label_column': 'label',
    },
}

HETERO_LP_DATASETS = set(HETERO_DATASET_SPECS)
HETERO_NC_DATASETS = {
    name for name, spec in HETERO_DATASET_SPECS.items()
    if 'target_node_type' in spec and 'label_csv' in spec and 'label_column' in spec
}
READY_DATASET_SPECS = {
    'cs_phds': {
        'base_dir': 'cs_phds',
        'nc_dir': 'cs_phds_nc_ready',
        'lp_dir': 'cs_phds_lp_ready',
    },
}


def sanitize_numpy_array(name, arr):
    arr = np.asarray(arr, dtype=np.float32)
    invalid_mask = ~np.isfinite(arr)
    invalid_count = int(invalid_mask.sum())
    if invalid_count:
        print(f"[DataSanitize] {name}: replaced {invalid_count} non-finite values with 0.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def sanitize_torch_tensor(name, tensor):
    tensor = tensor.detach().cpu().float()
    invalid_count = int((~torch.isfinite(tensor)).sum().item())
    if invalid_count:
        print(f"[DataSanitize] {name}: replaced {invalid_count} non-finite values with 0.")
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def get_baseline_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def get_ready_dataset_spec(dataset_str):
    return READY_DATASET_SPECS.get(dataset_str.lower())


def resolve_ready_dataset_root(dataset_str, data_path, task):
    spec = get_ready_dataset_spec(dataset_str)
    if spec is None:
        return None

    task_key = 'lp_dir' if str(task).lower() == 'lp' else 'nc_dir'
    ready_dir_name = spec[task_key]
    norm_path = os.path.normpath(data_path)
    candidates = []

    if os.path.basename(norm_path).lower() == ready_dir_name.lower():
        candidates.append(norm_path)
    if os.path.basename(norm_path).lower() == spec['base_dir'].lower():
        candidates.append(os.path.join(norm_path, ready_dir_name))
    candidates.append(os.path.join(norm_path, spec['base_dir'], ready_dir_name))
    candidates.append(os.path.join(get_baseline_root(), spec['base_dir'], ready_dir_name))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isdir(normalized):
            return normalized
    return None


def load_torch_payload(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def dense_tensor_to_sparse_adj(tensor):
    tensor = tensor.detach().cpu().float()
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f'Adjacency tensor must be square, got shape {tuple(tensor.shape)}')
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    idx = (tensor > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return sp.csr_matrix(tensor.shape, dtype=np.float32)
    values = np.ones(idx.shape[0], dtype=np.float32)
    rows = idx[:, 0].numpy()
    cols = idx[:, 1].numpy()
    adj = sp.coo_matrix((values, (rows, cols)), shape=tuple(tensor.shape))
    adj = adj.maximum(adj.T)
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


def ready_edge_tensor_to_pairs(tensor):
    tensor = tensor.detach().cpu().long()
    if tensor.dim() != 2:
        raise ValueError(f'Edge tensor must be rank-2, got shape {tuple(tensor.shape)}')
    if tensor.shape[0] == 2:
        return tensor.t().contiguous()
    if tensor.shape[1] == 2:
        return tensor.contiguous()
    raise ValueError(f'Unsupported edge tensor shape: {tuple(tensor.shape)}')


def load_ready_dataset_lp(dataset_str, use_feats, data_path):
    dataset_root = resolve_ready_dataset_root(dataset_str, data_path, 'lp')
    if dataset_root is None:
        raise FileNotFoundError(f'Ready LP dataset {dataset_str} not found from {data_path}')

    adj_train = dense_tensor_to_sparse_adj(load_torch_payload(os.path.join(dataset_root, 'adj_train.pt')))
    features_tensor = sanitize_torch_tensor(
        f'{dataset_str}.lp.features',
        load_torch_payload(os.path.join(dataset_root, 'feats.pt')),
    )
    splits = load_torch_payload(os.path.join(dataset_root, 'splits.pt'))
    if not isinstance(splits, dict):
        raise ValueError(f'{dataset_str} LP splits.pt must be a dict.')

    num_nodes = features_tensor.size(0)
    features = features_tensor if use_feats else sp.eye(num_nodes)
    return {
        'adj_train': adj_train,
        'features': features,
        'train_edges': ready_edge_tensor_to_pairs(splits['train_pos']),
        'train_edges_false': ready_edge_tensor_to_pairs(splits['train_neg']),
        'val_edges': ready_edge_tensor_to_pairs(splits['val_pos']),
        'val_edges_false': ready_edge_tensor_to_pairs(splits['val_neg']),
        'test_edges': ready_edge_tensor_to_pairs(splits['test_pos']),
        'test_edges_false': ready_edge_tensor_to_pairs(splits['test_neg']),
        'predefined_lp_splits': True,
    }


def load_ready_dataset_nc(dataset_str, use_feats, data_path, split_seed):
    dataset_root = resolve_ready_dataset_root(dataset_str, data_path, 'nc')
    if dataset_root is None:
        raise FileNotFoundError(f'Ready NC dataset {dataset_str} not found from {data_path}')

    adj = dense_tensor_to_sparse_adj(load_torch_payload(os.path.join(dataset_root, 'adj.pt')))
    features_tensor = sanitize_torch_tensor(
        f'{dataset_str}.nc.features',
        load_torch_payload(os.path.join(dataset_root, 'feats.pt')),
    )
    labels = load_torch_payload(os.path.join(dataset_root, 'labels.pt'))
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu().numpy()

    labeled_idx = np.where(labels >= 0)[0]
    if labeled_idx.size == 0:
        raise ValueError(f'{dataset_str} ready NC labels have no labeled nodes.')
    idx_train_rel, idx_val_rel, idx_test_rel = stratified_split_indices(
        labels[labeled_idx], val_prop=0.2, test_prop=0.2, seed=split_seed
    )
    idx_train = labeled_idx[np.asarray(idx_train_rel, dtype=np.int64)].tolist()
    idx_val = labeled_idx[np.asarray(idx_val_rel, dtype=np.int64)].tolist()
    idx_test = labeled_idx[np.asarray(idx_test_rel, dtype=np.int64)].tolist()

    num_nodes = features_tensor.size(0)
    features = features_tensor if use_feats else sp.eye(num_nodes)
    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(args, datapath):
    if args.task == 'nc':
        data = load_data_nc(args.dataset, args.use_feats, datapath, args.split_seed)
    elif args.task == 'lp':
        data = load_data_lp(args.dataset, args.use_feats, datapath)
        if not data.get('predefined_lp_splits'):
            adj = data['adj_train']
            adj_train, train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = mask_edges(
                    adj, args.val_prop, args.test_prop, args.split_seed
            )
            data['adj_train'] = adj_train
            data['train_edges'], data['train_edges_false'] = train_edges, train_edges_false
            data['val_edges'], data['val_edges_false'] = val_edges, val_edges_false
            data['test_edges'], data['test_edges_false'] = test_edges, test_edges_false
    else:
        raise ValueError(f'Unsupported task: {args.task}')
    
    data['adj_train_norm'], data['features'] = process(
            data['adj_train'], data['features'], args.normalize_adj, args.normalize_feats
    )
    # print(data['features'].max())
    # print(data['adj_train_norm'])
    if args.dataset == 'airport':
        data['features'] = augment(data['adj_train'], data['features'])

    # feature = data['features']
    # norm_feature = F.normalize(feature)
    # norm_feature[:,0] = 1 + norm_feature[:,0]
    # print(norm_feature.max().item(),norm_feature.min().item())
    # ones = torch.ones(norm_feature.shape[0],1)
    # new_feature = torch.cat((ones,norm_feature),1)
    # new_feature[] = torch.cat((ones,norm_feature),1)
    # data['features'] = norm_feature
    # k = 2/(feature.max()-feature.min()) 
    # new_feature = -1+k*(feature-feature.min())
    # data['features'] = norm_feature

    return data


def resolve_dataset_root(dataset_str, data_path):
    spec = HETERO_DATASET_SPECS.get(dataset_str.lower())
    if spec is None:
        return data_path

    norm_path = os.path.normpath(data_path)
    if os.path.basename(norm_path).lower() == spec['dir_name'].lower():
        return norm_path
    return os.path.join(norm_path, spec['dir_name'])


def iter_unified_dataset_roots(dataset_str, data_path):
    spec = get_hetero_dataset_spec(dataset_str)
    norm_path = os.path.normpath(data_path)
    dir_name = spec.get('unified_dir_name', spec['dir_name'])
    candidates = []

    if os.path.basename(norm_path).lower() == dir_name.lower():
        candidates.append(norm_path)
    candidates.append(os.path.join(norm_path, dir_name))
    candidates.append(os.path.join(norm_path, 'exptable2graph', 'exptable2graph', dir_name))

    baseline_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    candidates.append(os.path.join(baseline_root, 'exptable2graph', 'exptable2graph', dir_name))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isdir(normalized):
            yield normalized


def resolve_unified_data_file(dataset_str, data_path):
    spec = get_hetero_dataset_spec(dataset_str)
    unified_file = spec.get('unified_file')
    if not unified_file:
        return None
    for dataset_root in iter_unified_dataset_roots(dataset_str, data_path):
        candidate = os.path.join(dataset_root, unified_file)
        if os.path.exists(candidate):
            return candidate, dataset_root
    return None


def get_hetero_dataset_spec(dataset_str):
    dataset_lower = dataset_str.lower()
    if dataset_lower not in HETERO_DATASET_SPECS:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset_str))
    return HETERO_DATASET_SPECS[dataset_lower]


def encode_label_series(labels):
    series = pd.Series(labels).fillna('__nan__')
    encoded, _ = pd.factorize(series, sort=True)
    return encoded.astype(np.int64)


def build_hetero_dataset_payload(dataset_str, use_feats, data_path):
    spec = get_hetero_dataset_spec(dataset_str)
    unified_resolved = resolve_unified_data_file(dataset_str, data_path)
    if unified_resolved is not None:
        unified_file, dataset_root = unified_resolved
        data_obj = torch.load(unified_file, map_location='cpu')
        if isinstance(data_obj, tuple):
            data_obj = data_obj[0]
        if not hasattr(data_obj, 'x') or not hasattr(data_obj, 'edge_index'):
            raise ValueError('Unified data file {} is missing x/edge_index.'.format(unified_file))

        x = sanitize_torch_tensor(f"{dataset_str}.unified_features", data_obj.x)
        edge_index = data_obj.edge_index.detach().cpu().long()
        num_nodes = x.size(0)
        payload = {
            'adj': edge_index_to_adj(edge_index, num_nodes),
            'features': x if use_feats else sp.eye(num_nodes),
            'dataset_root': dataset_root,
            'spec': spec,
            'source': 'unified',
            'data_obj': data_obj,
        }
        return payload

    dataset_root = resolve_dataset_root(dataset_str, data_path)
    data_file = os.path.join(dataset_root, spec['pt_file'])
    data_obj = torch.load(data_file, map_location='cpu')

    node_offsets = {}
    feature_blocks = []
    total_nodes = 0
    for node_type in data_obj.node_types:
        x = getattr(data_obj[node_type], 'x', None)
        if x is None:
            raise ValueError('Node type {} in dataset {} has no features.'.format(node_type, dataset_str))
        num_nodes = x.size(0)
        node_offsets[node_type] = (total_nodes, total_nodes + num_nodes)
        total_nodes += num_nodes
        if use_feats:
            feature_blocks.append(sanitize_torch_tensor(f"{dataset_str}.{node_type}.features", x))

    if use_feats:
        features = torch.cat(feature_blocks, dim=0)
    else:
        features = sp.eye(total_nodes)

    edge_blocks = []
    for src_type, _, dst_type in data_obj.edge_types:
        edge_index = data_obj[(src_type, _, dst_type)].edge_index.detach().cpu().numpy()
        src_offset = node_offsets[src_type][0]
        dst_offset = node_offsets[dst_type][0]
        edge_blocks.append(
            np.vstack([
                edge_index[0] + src_offset,
                edge_index[1] + dst_offset,
            ])
        )

    if edge_blocks:
        merged_edge_index = np.hstack(edge_blocks)
        adj = edge_index_to_adj(torch.from_numpy(merged_edge_index), total_nodes)
    else:
        adj = sp.csr_matrix((total_nodes, total_nodes), dtype=np.float32)

    payload = {
        'adj': adj,
        'features': features,
        'node_offsets': node_offsets,
        'dataset_root': dataset_root,
        'spec': spec,
        'source': 'legacy',
    }
    return payload


def load_hetero_graph_lp(dataset_str, use_feats, data_path):
    payload = build_hetero_dataset_payload(dataset_str, use_feats, data_path)
    return payload['adj'], payload['features']


def load_hetero_graph_nc(dataset_str, use_feats, data_path, split_seed):
    payload = build_hetero_dataset_payload(dataset_str, use_feats, data_path)
    spec = payload['spec']

    if payload.get('source') == 'unified':
        data_obj = payload['data_obj']
        y = getattr(data_obj, 'y', None)
        if y is None:
            raise ValueError('Unified data for {} is missing labels.'.format(dataset_str))
        labels = y.detach().cpu().long().numpy()
        train_mask = getattr(data_obj, 'train_mask', None)
        val_mask = getattr(data_obj, 'val_mask', None)
        test_mask = getattr(data_obj, 'test_mask', None)
        if train_mask is not None and val_mask is not None and test_mask is not None:
            idx_train, idx_val, idx_test = get_split_indices_from_masks(
                train_mask, val_mask, test_mask, split_seed, labels=labels
            )
        else:
            labeled_idx = np.where(np.asarray(labels) >= 0)[0]
            if labeled_idx.size == 0:
                raise ValueError('Unified data for {} has no labeled nodes.'.format(dataset_str))
            idx_train, idx_val, idx_test = stratified_split_indices(
                np.asarray(labels)[labeled_idx], val_prop=0.2, test_prop=0.2, seed=split_seed
            )
            idx_train = labeled_idx[np.asarray(idx_train, dtype=np.int64)].tolist()
            idx_val = labeled_idx[np.asarray(idx_val, dtype=np.int64)].tolist()
            idx_test = labeled_idx[np.asarray(idx_test, dtype=np.int64)].tolist()
        return payload['adj'], payload['features'], labels, idx_train, idx_val, idx_test

    target_node_type = spec['target_node_type']
    start_idx, end_idx = payload['node_offsets'][target_node_type]

    label_df = pd.read_csv(os.path.join(payload['dataset_root'], spec['label_csv']))
    raw_labels = label_df[spec['label_column']]
    labels_target = encode_label_series(raw_labels)
    expected_size = end_idx - start_idx
    if len(labels_target) != expected_size:
        raise ValueError(
            'Label count mismatch for dataset {}: expected {}, got {}'.format(
                dataset_str, expected_size, len(labels_target)
            )
        )

    labels = np.full(payload['adj'].shape[0], -1, dtype=np.int64)
    labels[start_idx:end_idx] = labels_target

    idx_train, idx_val, idx_test = stratified_split_indices(
        np.array(labels_target), val_prop=0.2, test_prop=0.2, seed=split_seed
    )
    idx_train = [start_idx + idx for idx in idx_train]
    idx_val = [start_idx + idx for idx in idx_val]
    idx_test = [start_idx + idx for idx in idx_test]

    return payload['adj'], payload['features'], labels, idx_train, idx_val, idx_test


# ############### FEATURES PROCESSING ####################################

def process(adj, features, normalize_adj, normalize_feats):
    if torch.is_tensor(features):
        features = features.detach().cpu().numpy()
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    features = sanitize_numpy_array("features.pre_normalize", features)
    if normalize_feats:
        features = normalize(features)
    features = sanitize_numpy_array("features.post_normalize", features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    with np.errstate(divide='ignore', invalid='ignore'):
        r_inv = np.power(rowsum, -1).flatten()
    r_inv[~np.isfinite(r_inv)] = 0.
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


def sample_negative_edges(num_nodes, forbidden_edges, num_samples, rng):
    samples = set()
    while len(samples) < num_samples:
        remaining = num_samples - len(samples)
        batch_size = max(remaining * 2, 1024)
        row = rng.randint(0, num_nodes, size=batch_size)
        col = rng.randint(0, num_nodes, size=batch_size)
        mask = row < col
        row = row[mask]
        col = col[mask]
        for u, v in zip(row, col):
            edge = (int(u), int(v))
            if edge in forbidden_edges or edge in samples:
                continue
            samples.add(edge)
            if len(samples) >= num_samples:
                break
    return np.array(list(samples), dtype=np.int64)


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    rng = np.random.RandomState(seed)
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)), dtype=np.int64)
    rng.shuffle(pos_edges)
    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    all_pos_edge_set = {tuple(edge) for edge in pos_edges.tolist()}
    val_test_neg_edges = sample_negative_edges(adj.shape[0], all_pos_edge_set, n_val + n_test, rng)
    val_edges_false = val_test_neg_edges[:n_val]
    test_edges_false = val_test_neg_edges[n_val:n_val + n_test]
    train_edge_set = {tuple(edge) for edge in train_edges.tolist()}
    train_neg_pool = sample_negative_edges(adj.shape[0], train_edge_set, max(m_pos, 1), rng)
    train_edges_false = np.concatenate([train_neg_pool, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
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


# ############### LINK PREDICTION DATA LOADERS ####################################
def load_data_md(dataset, use_feats, data_path):
    if dataset in ['disease_md','grid','tree','tree_cycle','tree_grid','cycle_tree','sphere','cycle', 'cs_phd', 'power', 'facebook', 'random', 'club','nips','bio-diseasome','bio-wormnet','california','grqc','road-m','web-edu']:
        adj, features,labels, G = load_synthetic_md_data(dataset, False, data_path)[:4]
    elif dataset in ['toroidal','spherical','uniform_tree','random_geometric_graph','ring_of_tree','erdos_graph','tree_with_random_cycle']:
        adj, features,labels, G = load_synthetic_rec_data(dataset, False, data_path)[:4]
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    # if adj_dict == None:
    data = {'adj_train': adj, 'features': features,'labels': labels, 'G': G}
    # else:
    #     data = {'adj_train': adj, 'features': features,'labels': labels, "adj_dict":adj_dict}
    return data

def load_synthetic_rec_data(dataset_str, use_feats, data_path):
    if dataset_str == 'toroidal':
        g = ToroidalGraph(num_nodes=1000, R=0.01)
    elif dataset_str == 'spherical':
        g = SphericalGraph(num_nodes=1000, R=0.2)
    elif dataset_str == "uniform_tree":
        g = UniformTree(depth=5, branching_factor=4)
    elif dataset_str == 'random_geometric_graph':
        g = nx.random_geometric_graph(1000, 0.125)
    elif dataset_str == 'erdos_graph':
        g = ErdosGraph(num_nodes=1000, p=0.005)
    elif dataset_str == 'cycle':
        g = Cycle(1000)
    elif dataset_str == 'ring_of_tree':
        g = RingOfTrees(order=3, branching_factor=4)
    elif dataset_str == "tree_with_random_cycle":
        g = RandomTree(depth=5, branching_factor=4, num=500)
        # G = nx.Graph(g.edge_matrix)
        # g = random_edge(G, del_orig=True, num=100)
    elif dataset_str == 'club':
        g = nx.karate_club_graph()
    
    adj = g.edge_matrix
    # print(adj[0])
    labels = g.get_weight_matrix()
    features = np.eye(g.get_num_nodes())
    # features = adj
    # features[:,0] = 1
    G = nx.Graph(g.edge_matrix)
    # edges = G.edges()
    # positive_pairs = set(list(edges))
    # all_pairs = set([(i,j) for j in range(G.number_of_nodes()) for i in range(G.number_of_nodes())])
    # negative_pairs = all_pairs - positive_pairs
    # positive_pairs = torch.Tensor(list(positive_pairs)).long()
    # negative_pairs = torch.Tensor(list(negative_pairs)).long()
    return adj, features, labels, G


from multiprocessing import Pool
import scipy.sparse.csgraph as csg

def build_distance(G):
    length = dict(nx.all_pairs_shortest_path_length(G))
    R = np.array([[length.get(m, {}).get(n, 0) for m in G.nodes] for n in G.nodes], dtype=np.int32)
    return R

def load_synthetic_md_data(dataset_str, use_feats, data_path):
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
    adj_dict = {}
    for i,j in edges:
        adj_dict[i] = set()
    for i, j in edges:
        adj_dict[i].add(j)
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    if use_feats:
        features = sp.load_npz(os.path.join(data_path, "{}.feats.npz".format(dataset_str)))
    else:
        features = sp.eye(adj.shape[0])
        # print(features)
        # features = 
    G = nx.from_numpy_matrix(adj)
    labels = build_distance(G)
    labels = labels
    # G = nx.Graph(G)
    # print(features.shape, adj.shape)
    return sp.csr_matrix(adj), features, labels,G

def load_data_lp(dataset, use_feats, data_path):
    dataset_lower = dataset.lower()
    if dataset_lower in ['cora', 'pubmed', 'citeseer']:
        adj, features = load_citation_data(dataset_lower, use_feats, data_path)[:2]
    elif dataset_lower in ['disease_lp']:
        adj, features = load_synthetic_data(dataset_lower, use_feats, data_path)[:2]
    elif dataset_lower == 'airport':
        adj, features = load_data_airport(dataset_lower, data_path, return_label=False)
    elif dataset_lower in READY_DATASET_SPECS:
        return load_ready_dataset_lp(dataset_lower, use_feats, data_path)
    elif dataset_lower in ['telecom', 'actor', 'cornell']:
        adj, features = load_pyg_graph_data(dataset_lower, use_feats, data_path)[:2]
    elif dataset_lower in HETERO_LP_DATASETS:
        adj, features = load_hetero_graph_lp(dataset_lower, use_feats, data_path)
    else:
        raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
    data = {'adj_train': adj, 'features': features}
    return data


# ############### NODE CLASSIFICATION DATA LOADERS ####################################
def load_data_nc_md(dataset, use_feats, data_path, split_seed):
    if dataset in ['cora', 'pubmed','citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset, use_feats, data_path, split_seed
        )
    else:
        if dataset in ['disease_nc','tree_cycle','tree_grid','ba_shape']:
            adj, features, labels = load_synthetic_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.0, 0.0
        elif dataset == 'airport':
            adj, features, labels = load_data_airport(dataset, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'deezer':
            dj, features, labels = load_json_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        else:
            raise FileNotFoundError('Dataset {} is not supported.'.format(dataset))
        idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop, seed=split_seed)

    # labels = torch.LongTensor(labels)
    # data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}
    adj = nx.to_scipy_sparse_matrix(adj)
    G = nx.from_numpy_matrix(adj)
    labels = build_distance(G)
    labels = labels
    # G = nx.Graph(G)
    # print(features.shape, adj.shape)
    return sp.csr_matrix(adj), features, labels,G



def load_data_nc(dataset, use_feats, data_path, split_seed):
    import os, sys
    sys.path.append("/data/hxz/WXY/CurvBench")
    from parquet_loader import _get_parquet_path, load_parquet_as_hgcn
    if os.path.exists(_get_parquet_path(dataset.lower())):
        return load_parquet_as_hgcn(dataset.lower())

    dataset_lower = dataset.lower()
    if dataset_lower in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            dataset_lower, use_feats, data_path, split_seed
        )
    elif dataset_lower in READY_DATASET_SPECS:
        adj, features, labels, idx_train, idx_val, idx_test = load_ready_dataset_nc(
            dataset_lower, use_feats, data_path, split_seed
        )
    elif dataset_lower in ['telecom', 'actor', 'cornell']:
        adj, features, labels, idx_train, idx_val, idx_test = load_pyg_graph_data(
            dataset_lower, use_feats, data_path, split_seed
        )
    elif dataset_lower in HETERO_NC_DATASETS or (
        dataset_lower in HETERO_LP_DATASETS and resolve_unified_data_file(dataset_lower, data_path) is not None
    ):
        adj, features, labels, idx_train, idx_val, idx_test = load_hetero_graph_nc(
            dataset_lower, use_feats, data_path, split_seed
        )
    else:
        if dataset_lower in ['disease_nc', 'tree_cycle', 'tree_grid', 'ba_shape']:
            adj, features, labels = load_synthetic_data(dataset_lower, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
        elif dataset_lower == 'airport':
            adj, features, labels = load_data_airport(dataset_lower, data_path, return_label=True)
            val_prop, test_prop = 0.15, 0.15
        elif dataset == 'deezer':
            dj, features, labels = load_json_data(dataset, use_feats, data_path)
            val_prop, test_prop = 0.10, 0.60
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

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

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
    if dataset_str in ['tree_cycle','tree_grid','ba_shape']:
        import pandas as pd
        labels = pd.read_csv(os.path.join(data_path, "{}.label.csv".format(dataset_str))).values[:,0]
        print(labels.shape)
    else:
        labels = np.load(os.path.join(data_path, "{}.labels.npy".format(dataset_str)))
    return sp.csr_matrix(adj), features, labels

def load_json_data(dataset_str, use_feats, data_path):
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


def load_pyg_graph_data(dataset_str, use_feats, data_path, split_seed=None):
    dataset_lower = dataset_str.lower()
    if dataset_lower == 'telecom':
        data_file = os.path.join(data_path, 'telecom_graph.pt')
    else:
        data_file = os.path.join(data_path, 'processed', 'data.pt')
    raw_obj = torch.load(data_file, map_location='cpu')
    if isinstance(raw_obj, tuple):
        data_obj = raw_obj[0]
    else:
        data_obj = raw_obj

    if isinstance(data_obj, dict):
        x = data_obj.get('x')
        edge_index = data_obj.get('edge_index')
        y = data_obj.get('y')
        train_mask = data_obj.get('train_mask')
        val_mask = data_obj.get('val_mask')
        test_mask = data_obj.get('test_mask')
    else:
        x = data_obj.x
        edge_index = data_obj.edge_index
        y = data_obj.y
        train_mask = getattr(data_obj, 'train_mask', None)
        val_mask = getattr(data_obj, 'val_mask', None)
        test_mask = getattr(data_obj, 'test_mask', None)

    n_nodes = x.size(0)
    adj = edge_index_to_adj(edge_index, n_nodes)

    if not use_feats:
        features = sp.eye(n_nodes)
    else:
        features = x

    if y is None:
        labels = None
    elif y.dim() > 1:
        labels = torch.argmax(y, dim=1).cpu().numpy()
    else:
        labels = y.cpu().numpy()

    if train_mask is not None and val_mask is not None and test_mask is not None:
        idx_train, idx_val, idx_test = get_split_indices_from_masks(
            train_mask, val_mask, test_mask, split_seed
        )
    elif labels is not None:
        idx_train, idx_val, idx_test = stratified_split_indices(
            labels, val_prop=0.2, test_prop=0.2, seed=split_seed
        )
    else:
        idx_train = idx_val = idx_test = None

    return adj, features, labels, idx_train, idx_val, idx_test


def edge_index_to_adj(edge_index, n_nodes):
    edge_index = edge_index.cpu().numpy()
    rows = edge_index[0]
    cols = edge_index[1]
    data = np.ones(rows.shape[0], dtype=np.float32)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    return adj.tocsr()


def get_split_indices_from_masks(train_mask, val_mask, test_mask, split_seed, labels=None):
    train_mask = train_mask.cpu().numpy()
    val_mask = val_mask.cpu().numpy()
    test_mask = test_mask.cpu().numpy()

    if train_mask.ndim == 2:
        split_id = 0 if split_seed is None else int(split_seed) % train_mask.shape[1]
        train_mask = train_mask[:, split_id]
        val_mask = val_mask[:, split_id]
        test_mask = test_mask[:, split_id]

    idx_train = np.where(train_mask)[0].tolist()
    idx_val = np.where(val_mask)[0].tolist()
    idx_test = np.where(test_mask)[0].tolist()

    if labels is not None:
        labels = np.asarray(labels)
        idx_train = [i for i in idx_train if labels[i] >= 0]
        idx_val = [i for i in idx_val if labels[i] >= 0]
        idx_test = [i for i in idx_test if labels[i] >= 0]
    return idx_train, idx_val, idx_test


def stratified_split_indices(labels, val_prop, test_prop, seed=None):
    if labels is None:
        raise ValueError('Labels are required to create splits.')
    rng = np.random.RandomState(seed)
    labels = np.array(labels)
    idx_train, idx_val, idx_test = [], [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_total = cls_idx.shape[0]
        n_val = int(round(val_prop * n_total))
        n_test = int(round(test_prop * n_total))
        idx_val.extend(cls_idx[:n_val].tolist())
        idx_test.extend(cls_idx[n_val:n_val + n_test].tolist())
        idx_train.extend(cls_idx[n_val + n_test:].tolist())
    return idx_train, idx_val, idx_test

def load_airport(dataset_str, data_path, return_label=False):
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph.node[u]['feat'] for u in graph.nodes()])
    if return_label:
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0/7, 8.0/7, 9.0/7])
        return sp.csr_matrix(adj), features, labels
    else:
        return sp.csr_matrix(adj), features


import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


class ToroidalGraph():
    def __init__(self, R, num_nodes, num_classes=None):
        self.R = R
        self.num_nodes = num_nodes
        self.num_classes = num_classes

        self.samples = self.get_samples()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()
        if num_classes is not None:
            self.labels = self.random_labeling(self.num_classes)

    def torus_distance(self, x, y):
        dx = np.abs(x[0] - y[0])
        dy = np.abs(x[1] - y[1])

        if dx > 0.5:
            dx = 1 - dx

        if dy > 0.5:
            dy = 1 - dy

        return dx**2 + dy**2

    def get_num_nodes(self):
        return self.num_nodes

    def get_samples(self):
        samples = np.random.uniform(low=-1, high=1, size=(self.num_nodes,2))

        return samples

    def construct_adjacency(self):
        edge_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.torus_distance(x, y)
                if dist < self.R and i != j:
                    edge_matrix[i, j] = 1

        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(graph):
            print('Graph is not connected')
        else:
            print('Everything fine')
            return graph

    def random_labeling(self, num_classes):
        labels = np.ones(shape=self.num_nodes) * 1000
        A = np.array(self.edge_matrix.todense(), dtype=np.float64)
        # let's evaluate the degree matrix D
        D = np.diag(np.sum(A, axis=0))
        length = 3 * self.num_nodes // num_classes
        for k in range(num_classes):
            source = np.random.randint(low=0, high=self.num_nodes, size=1)
            labels[source] = k
            visited = list()
            for _ in range(length):
                # evaluate the next state vector
                count = 0
                nn = np.random.randint(0, D[source, source], 1)
                for i in range(self.num_nodes):
                    if count == nn:
                        source = i
                        visited.append(i)
                        if labels[i] == 1000:
                            labels[i] = k
                        else:
                            p = np.random.uniform(low=0, high=1, size=1)
                            if p > 0.5:
                                labels[i] = k
                        break
                    if A[source, i] == 1:
                        count += 1
        labels[labels == 1000] = 0
        return labels

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

    def get_features(self, dim):
        ni

import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


class SphericalGraph():
    def __init__(self, R, num_nodes):
        self.R = R
        self.num_nodes = num_nodes
        self.samples = self.get_samples()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()

    def stereo_distance(self, x, y):
        eps = 1e-10
        arg = np.clip(np.sum(x*y), a_max=1-eps, a_min=-1+eps)
        d = np.arccos(arg)

        return d

    def get_num_nodes(self):
        return self.num_nodes

    def get_samples(self):
        phi = np.random.uniform(low=0, high=2*np.pi, size=self.num_nodes)
        psi = np.random.uniform(low=0, high=2 * np.pi, size=self.num_nodes)
        samples = np.concatenate([np.reshape(np.sin(phi)*np.cos(psi), (-1, 1)), np.reshape(np.sin(phi)*np.sin(psi), (-1, 1)), np.reshape(np.cos(phi), (-1, 1))], axis=1)

        return samples

    def construct_adjacency(self):
        edge_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.stereo_distance(x, y)
                if dist < self.R and i != j:
                    edge_matrix[i, j] = 1

        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(graph):
            print('Graph is not connected')
        else:
            print('Everything fine')
            return graph

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)
        return weight_matrix


import networkx as nx
import scipy.sparse.csgraph as graph
from scipy import sparse
import numpy as np


class HyperbolicGraph():
    def __init__(self, R, num_nodes):
        self.R = R
        self.num_nodes = num_nodes

        self.samples = self.get_samples()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()

    def minkowski(self, x, y):
        a = x[-1] * y[-1] - np.sum(x[0:-1] * y[0:-1])

        return a

    def lorentz_distance(self, x, y):
        d = np.arccosh(self.minkowski(x, y))

        return d

    def get_num_nodes(self):
        return self.num_nodes

    def get_samples(self):
        phi = np.random.uniform(low=0, high=1, size=self.num_nodes)
        psi = np.random.uniform(low=0, high=2 * np.pi, size=self.num_nodes)
        samples = np.concatenate([np.reshape(np.cosh(phi)*np.cos(psi), (-1, 1)), np.reshape(np.cosh(phi)*np.sin(psi), (-1, 1)), np.reshape(np.sinh(phi), (-1, 1))], axis=1)

        return samples

    def construct_adjacency(self):
        edge_matrix = np.zeros(shape=(self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.lorentz_distance(x, y)
                if dist < self.R and i != j:
                    edge_matrix[i, j] = 1
         
        edge_matrix = edge_matrix + np.transpose(edge_matrix)

        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(graph):
            print('Graph is not connected')
        else:
            print('Everything fine')
            return graph

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix


import numpy as np
import scipy.sparse.csgraph as graph
from scipy import sparse
import networkx as nx


class UniformTree:

    def __init__(self, depth, branching_factor):
        self.depth = depth
        self.b_factor = branching_factor

        self.num_nodes = self.get_num_nodes()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)

    def get_num_nodes(self):
        num_nodes = int((self.b_factor**(self.depth+1)-1)/(self.b_factor-1))

        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes-self.b_factor**self.depth):
            edge_matrix[i, self.b_factor*i+1:self.b_factor*(i+1)+1] = 1

        edge_matrix = edge_matrix + np.transpose(edge_matrix)

        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

    def get_batch_distances(self, indices):
        batch = []
        for index in indices:
            batch.append(self.weight_matrix[index[0], index[1]])
        return batch

class RandomTree:

    def __init__(self, depth, branching_factor, num=100):
        self.depth = depth
        self.b_factor = branching_factor
        self.num = num
        self.num_nodes = self.get_num_nodes()
        self.samples = self.get_samples()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)
        

    def get_samples(self):
        phi = np.random.uniform(low=0, high=2*np.pi, size=self.num_nodes)
        psi = np.random.uniform(low=0, high=2 * np.pi, size=self.num_nodes)
        samples = np.concatenate([np.reshape(np.sin(phi)*np.cos(psi), (-1, 1)), np.reshape(np.sin(phi)*np.sin(psi), (-1, 1)), np.reshape(np.cos(phi), (-1, 1))], axis=1)

        return samples

    def stereo_distance(self, x, y):
        eps = 1e-10
        arg = np.clip(np.sum(x*y), a_max=1-eps, a_min=-1+eps)
        d = np.arccos(arg)
        return d

    def get_num_nodes(self):
        num_nodes = int((self.b_factor**(self.depth+1)-1)/(self.b_factor-1))
        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes-self.b_factor**self.depth):
            # tree_edge.append( (i, self.b_factor*i+1:self.b_factor*(i+1)+1)) 
            edge_matrix[i, self.b_factor*i+1:self.b_factor*(i+1)+1] = 1

        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                x = self.samples[i, :]
                y = self.samples[j, :]
                dist = self.stereo_distance(x, y)
                if dist < 0.2 and i != j:
                    edge_matrix[i, j] = 1

        # count = 0 
        # tree_edge = edge_matrix.copy()

        # while count< self.num:
        #     a = random.choice(range(self.num_nodes-self.b_factor**self.depth))
        #     b = random.choice(range(0, self.num_nodes))
        #     if a!=b and edge_matrix[a,b]==0:
        #         edge_matrix[a,b] = 1
        #         count=count+1
                # drop = random.choice(range(self.b_factor*a+1, self.b_factor*(a+1)+2))
                # print(a,b,drop)
                # edge_matrix[a, drop] = 0
                # edge_matrix[drop_tree_edge[0], drop_tree_edge[1]] = 0
            
        edge_matrix = edge_matrix + np.transpose(edge_matrix)
        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

    def get_batch_distances(self, indices):
        batch = []
        for index in indices:
            batch.append(self.weight_matrix[index[0], index[1]])
        return batch

class Cycle:

    def __init__(self, order):
        self.order = order

        self.num_nodes = self.get_num_nodes()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)

    def get_num_nodes(self):
        num_nodes = self.order

        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))

        for i in range(self.num_nodes):
            left = i - 1
            right = i + 1
            if left < 0:
                left = self.num_nodes - 1
            if right == self.num_nodes:
                right = 0
            edge_matrix[i, left] = 1
            edge_matrix[i, right] = 1

        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

class RingOfTrees:

    def __init__(self, order, branching_factor):
        self.order = order
        self.b = branching_factor

        self.num_nodes = self.get_num_nodes()
        self.edge_matrix = self.get_edge_matrix()
        self.weight_matrix = self.get_weight_matrix()
        self.graph = nx.from_scipy_sparse_matrix(self.edge_matrix)

    def get_num_nodes(self):
        num_nodes = 2 * self.order + self.b * self.order

        return num_nodes

    def get_edge_matrix(self):
        edge_matrix = np.zeros((self.num_nodes, self.num_nodes))
        cycle_matrix = np.zeros((self.order, self.order))

        for i in range(self.order):
            left = i - 1
            right = i + 1
            if left < 0:
                left = self.order - 1
            if right == self.order:
                right = 0
            cycle_matrix[i, left] = 1
            cycle_matrix[i, right] = 1

        edge_matrix[0:self.order, 0:self.order] = cycle_matrix

        for i in range(self.order):
            edge_matrix[i, self.order+i] = 1
            edge_matrix[self.order + i, i] = 1

        for i in range(self.order):
            for j in range(2 * self.order + i * self.b, 2 * self.order + (i+1) * self.b):
                edge_matrix[i + self.order, j] = 1
                edge_matrix[j, i + self.order] = 1

        return sparse.csr_matrix(edge_matrix)

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)

        return weight_matrix

class ErdosGraph():
    def __init__(self, p, num_nodes):
        self.p = p
        self.num_nodes = num_nodes
        self.setting()
        self.edge_matrix = self.construct_adjacency()
        self.graph = self.construct_graph()
        self.weight_matrix = self.get_weight_matrix()

    def get_num_nodes(self):
        return self.num_nodes

    def setting(self):
        if self.p >= (np.log(self.num_nodes)/self.num_nodes) ** (1/3):
            print('Spherical setting')
        if self.p > (np.log(self.num_nodes)/self.num_nodes ** 2) ** (1/3) and self.p < 1 /np.sqrt(self.num_nodes):
            print('Hyperbolic setting')

    def construct_adjacency(self):
        edge_matrix = np.random.binomial(2, self.p, size=(self.num_nodes, self.num_nodes))
        return sparse.csr_matrix(edge_matrix)

    def construct_graph(self):
        G = nx.from_scipy_sparse_matrix(self.edge_matrix)
        if not nx.is_connected(G):
            print('Graph is not connected')
            Gs = nx.connected_component_subgraphs(G, copy=True)
        else:
            print('Everything fine')
            return G

    def get_weight_matrix(self):
        weight_matrix = graph.shortest_path(self.edge_matrix)
        return weight_matrix
        
#H = HyperbolicGraph(num_nodes=1000, R=1)
def random_edge(graph, del_orig=True, num=100):
    '''
    Create a new random edge and delete one of its current edge if del_orig is True.
    :param graph: networkx graph
    :param del_orig: bool
    :return: networkx graph
    '''
    for i in range(num):
        edges = list(graph.edges)
        nonedges = list(nx.non_edges(graph))
        # random edge choice
        chosen_edge = random.choice(edges)
        chosen_nonedge = random.choice([x for x in nonedges if chosen_edge[0] == x[0]])
        if del_orig:
            # delete chosen edge
            graph.remove_edge(chosen_edge[0], chosen_edge[1])
        # add new edge
        graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
    return graph
