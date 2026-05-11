import csv
import os
import pickle as pkl
import random
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.data
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import negative_sampling

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_SEED = 3047
random.seed(DEFAULT_SEED)
torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)

HETERO_GRAPH_DATASETS = {
    'carcinogenesis_data': {
        'graph_file': 'Carcinogenesis_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'nc_target_node_type': 'canc',
        'nc_label_file': os.path.join('csv', 'canc.csv'),
        'nc_label_column': 'class',
    },
    'hepatitis_std_data': {
        'graph_file': 'Hepatitis_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'nc_target_node_type': 'dispat',
        'nc_label_file': os.path.join('csv', 'dispat.csv'),
        'nc_label_column': 'Type',
    },
    'hockey_data': {
        'graph_file': 'Hockey_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'nc_target_node_type': 'Master',
        'nc_label_file': os.path.join('csv', 'Master.csv'),
        'nc_label_column': 'pos',
    },
    'f1_ultimate_hetero_graph': {
        'graph_file': 'f1_ultimate_hetero_graph.pt',
        'unified_file': 'unified_data.pt',
    },
    'pte': {
        'graph_file': 'PTE_Giant_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'nc_target_node_type': 'pte_active',
        'nc_label_file': os.path.join('csv', 'pte_active.csv'),
        'nc_label_column': 'is_active',
    },
    'toxicology_data': {
        'graph_file': 'Toxicology_HeteroGraph.pt',
        'unified_file': 'unified_data.pt',
        'nc_target_node_type': 'molecule',
        'nc_label_file': os.path.join('csv', 'molecule.csv'),
        'nc_label_column': 'label',
    },
}

BOOLEAN_LABELS = {
    '0': 0,
    '1': 1,
    'false': 0,
    'true': 1,
    'f': 0,
    't': 1,
    'no': 0,
    'yes': 1,
    'n': 0,
    'y': 1,
    '-': 0,
    '+': 1,
    'negative': 0,
    'positive': 1,
}

HETERO_VAL_PROP = 0.2
HETERO_TEST_PROP = 0.2
READY_DATASET_SPECS = {
    'cs_phds': {
        'base_dir': 'cs_phds',
        'nc_dir': 'cs_phds_nc_ready',
        'lp_dir': 'cs_phds_lp_ready',
    },
}


def get_mask(idx, length):
    mask = torch.zeros(length, dtype=torch.bool)
    mask[idx] = 1
    return mask


def canonical_dataset_name(data_name: str) -> str:
    mapping = {
        'cora': 'Cora',
        'citeseer': 'Citeseer',
        'pubmed': 'Pubmed',
        'photo': 'photo',
        'airport': 'airport',
        'actor': 'Actor',
        'cornell': 'cornell',
        'telecom': 'telecom',
        'disease_lp': 'disease_lp',
        'disease_nc': 'disease_nc',
        'cs_phds': 'cs_phds',
        'carcinogenesis_data': 'Carcinogenesis_data',
        'hepatitis_std_data': 'Hepatitis_std_data',
        'hockey_data': 'Hockey_data',
        'pte': 'PTE',
        'toxicology_data': 'Toxicology_data',
    }
    return mapping.get(data_name.lower(), data_name)


def load_data(root: str, data_name: str, split='public', downstream_task='NC', split_seed=DEFAULT_SEED, **kwargs):
    data_name = canonical_dataset_name(data_name)
    dataset_lower = data_name.lower()
    task = downstream_task.upper()

    if dataset_lower in HETERO_GRAPH_DATASETS:
        return load_hetero_graph_data(root, data_name, task, split_seed)

    if dataset_lower in {'actor', 'cornell', 'telecom', 'cs_phds'}:
        return load_pyg_graph_data(root, data_name, task, split_seed)

    if dataset_lower in {'disease_lp', 'disease_nc'}:
        return load_disease_data(root, data_name, task, split_seed)

    if dataset_lower in {'cora', 'citeseer', 'pubmed'}:
        dataset = Planetoid(root=root, name=data_name, split=split)
        train_mask = dataset.data.train_mask
        val_mask = dataset.data.val_mask
        test_mask = dataset.data.test_mask
    elif dataset_lower == 'airport':
        dataset = Airport(root)
        train_mask, val_mask, test_mask = dataset.data.mask
    elif dataset_lower == 'photo':
        dataset = Amazon(root=root, name='Photo')
        labels = dataset.data.y.tolist()
        val_prop, test_prop = 0.15, 0.15
        val_mask, test_mask, train_mask = split_data(labels, val_prop, test_prop, seed=split_seed)
        mask = (train_mask, val_mask, test_mask)
        features = dataset.data.x.float()
        num_features = dataset.num_features
        edge_index = dataset.data.edge_index.long()
        neg_edges = negative_sampling(edge_index).long()
        num_classes = dataset.num_classes
        labels = torch.tensor(labels, dtype=torch.long)
        return features, num_features, labels, edge_index, neg_edges, mask, num_classes
    elif os.path.exists(os.path.join(root, f'{data_name}.pkl')):
        return load_synthetic_data(root, data_name)
    else:
        raise NotImplementedError

    mask = (train_mask, val_mask, test_mask)
    features = dataset.data.x.float()
    num_features = dataset.num_features
    labels = dataset.data.y.long()
    edge_index = dataset.data.edge_index.long()
    neg_edges = negative_sampling(edge_index).long()
    num_classes = dataset.num_classes
    return features, num_features, labels, edge_index, neg_edges, mask, num_classes


def load_synthetic_data(root: str, data_name: str):
    with open(f'{root}/{data_name}.pkl', 'rb') as f:
        graph = pkl.load(f)
    with open(f'{root}/{data_name}_feature.pkl', 'rb') as f:
        features = pkl.load(f)
    features = torch.tensor(features).float()
    num_features = features.shape[-1]
    edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
    neg_edges = negative_sampling(edge_index).long()
    perm = torch.randperm(edge_index.shape[-1])
    edge_index = edge_index[:, perm]
    perm = torch.randperm(neg_edges.shape[-1])
    neg_edges = neg_edges[:, perm]
    labels = torch.tensor([], dtype=torch.long)
    mask = torch.tensor([])
    num_classes = None
    return features, num_features, labels, edge_index, neg_edges, mask, num_classes


def resolve_dataset_dir(root: str, data_name: str) -> str:
    root = os.path.abspath(root)
    candidates = []
    if os.path.basename(os.path.normpath(root)).lower() == data_name.lower():
        candidates.append(root)
    candidates.append(os.path.join(root, data_name))
    baseline_root = os.path.abspath(os.path.join(PROJECT_ROOT.parent))
    candidates.append(os.path.join(baseline_root, data_name))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f'Unable to locate dataset directory for {data_name} under {root}')


def get_ready_dataset_spec(data_name: str):
    return READY_DATASET_SPECS.get(data_name.lower())


def resolve_ready_dataset_root(root: str, data_name: str, task: str) -> str | None:
    spec = get_ready_dataset_spec(data_name)
    if spec is None:
        return None

    task_key = 'lp_dir' if str(task).upper() == 'LP' else 'nc_dir'
    ready_dir_name = spec[task_key]
    root = os.path.abspath(root)
    candidates = []

    if os.path.basename(os.path.normpath(root)).lower() == ready_dir_name.lower():
        candidates.append(root)
    if os.path.basename(os.path.normpath(root)).lower() == spec['base_dir'].lower():
        candidates.append(os.path.join(root, ready_dir_name))
    candidates.append(os.path.join(root, spec['base_dir'], ready_dir_name))
    candidates.append(os.path.join(resolve_dataset_dir(root, data_name), ready_dir_name))
    candidates.append(os.path.join(os.path.abspath(os.path.join(PROJECT_ROOT.parent)), spec['base_dir'], ready_dir_name))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isdir(normalized):
            return normalized
    return None


def iter_unified_dataset_dirs(root: str, data_name: str):
    root = os.path.abspath(root)
    dir_name = 'f1' if data_name.lower() == 'f1_ultimate_hetero_graph' else data_name
    candidates = []
    if os.path.basename(os.path.normpath(root)).lower() == dir_name.lower():
        candidates.append(root)
    candidates.append(os.path.join(root, dir_name))
    candidates.append(os.path.join(root, 'exptable2graph', 'exptable2graph', dir_name))
    baseline_root = os.path.abspath(os.path.join(PROJECT_ROOT.parent))
    candidates.append(os.path.join(baseline_root, 'exptable2graph', 'exptable2graph', dir_name))

    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.isdir(normalized):
            yield normalized


def resolve_unified_data_file(root: str, data_name: str):
    dataset_meta = HETERO_GRAPH_DATASETS.get(data_name.lower())
    if dataset_meta is None or 'unified_file' not in dataset_meta:
        return None, None
    for dataset_dir in iter_unified_dataset_dirs(root, data_name):
        unified_path = os.path.join(dataset_dir, dataset_meta['unified_file'])
        if os.path.exists(unified_path):
            return unified_path, dataset_dir
    return None, None


def load_disease_data(root: str, data_name: str, downstream_task: str, split_seed: int):
    dataset_dir = resolve_dataset_dir(root, data_name)
    edge_path = os.path.join(dataset_dir, f'{data_name}.edges.csv')
    feat_path = os.path.join(dataset_dir, f'{data_name}.feats.npz')
    label_path = os.path.join(dataset_dir, f'{data_name}.labels.npy')

    object_to_idx = {}
    idx_counter = 0
    edges = []
    with open(edge_path, 'r', encoding='utf-8') as f:
        for line in f:
            n1, n2 = line.rstrip().split(',')
            if n1 not in object_to_idx:
                object_to_idx[n1] = idx_counter
                idx_counter += 1
            if n2 not in object_to_idx:
                object_to_idx[n2] = idx_counter
                idx_counter += 1
            edges.append((object_to_idx[n1], object_to_idx[n2]))

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    features = sp.load_npz(feat_path)
    features = torch.tensor(features.toarray(), dtype=torch.float)
    num_features = features.shape[-1]
    neg_edges = negative_sampling(edge_index).long()

    if downstream_task == 'LP':
        edge_index, neg_edges = shuffle_edges(edge_index, neg_edges, split_seed)
        labels = torch.tensor([], dtype=torch.long)
        mask = torch.tensor([])
        num_classes = None
        return features, num_features, labels, edge_index, neg_edges, mask, num_classes

    labels = np.load(label_path).astype(np.int64)
    val_prop, test_prop = 0.10, 0.60
    idx_val, idx_test, idx_train = split_data(labels.tolist(), val_prop, test_prop, seed=split_seed)
    mask = (idx_train, idx_val, idx_test)
    num_classes = int(labels.max()) + 1
    return features, num_features, torch.tensor(labels, dtype=torch.long), edge_index.long(), neg_edges, mask, num_classes


def load_torch_graph_file(path: str):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def dense_tensor_to_edge_index(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(tensor).detach().cpu().float()
    if tensor.dim() != 2 or tensor.shape[0] != tensor.shape[1]:
        raise ValueError(f'Adjacency tensor must be square, got shape {tuple(tensor.shape)}')
    idx = (tensor > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = idx.t().contiguous().long()
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index


def ready_edge_tensor_to_index(tensor: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(tensor).detach().cpu().long()
    if tensor.dim() != 2:
        raise ValueError(f'Edge tensor must be rank-2, got shape {tuple(tensor.shape)}')
    if tensor.shape[0] == 2:
        return tensor.contiguous()
    if tensor.shape[1] == 2:
        return tensor.t().contiguous()
    raise ValueError(f'Unsupported edge tensor shape: {tuple(tensor.shape)}')


def load_ready_dataset(root: str, data_name: str, downstream_task: str, split_seed: int):
    dataset_root = resolve_ready_dataset_root(root, data_name, downstream_task)
    if dataset_root is None:
        raise FileNotFoundError(f'Unable to locate ready dataset directory for {data_name} ({downstream_task}).')

    task = downstream_task.upper()
    if task == 'NC':
        features = load_torch_graph_file(os.path.join(dataset_root, 'feats.pt')).detach().cpu().float()
        labels = torch.as_tensor(load_torch_graph_file(os.path.join(dataset_root, 'labels.pt')), dtype=torch.long).detach().cpu()
        edge_index = dense_tensor_to_edge_index(load_torch_graph_file(os.path.join(dataset_root, 'adj.pt')))
        neg_edges = negative_sampling(edge_index, num_nodes=features.shape[0]).long() if edge_index.numel() else torch.empty((2, 0), dtype=torch.long)

        labeled_idx = torch.where(labels >= 0)[0]
        if labeled_idx.numel() == 0:
            raise ValueError(f'{data_name} ready NC labels have no labeled nodes.')
        idx_val_rel, idx_test_rel, idx_train_rel = split_data(labels[labeled_idx].tolist(), 0.2, 0.2, seed=split_seed)
        idx_train = labeled_idx[torch.tensor(idx_train_rel, dtype=torch.long)].tolist()
        idx_val = labeled_idx[torch.tensor(idx_val_rel, dtype=torch.long)].tolist()
        idx_test = labeled_idx[torch.tensor(idx_test_rel, dtype=torch.long)].tolist()
        mask = (idx_train, idx_val, idx_test)
        num_classes = int(labels.max().item()) + 1
        return features, int(features.shape[-1]), labels.long(), edge_index, neg_edges, mask, num_classes

    features = load_torch_graph_file(os.path.join(dataset_root, 'feats.pt')).detach().cpu().float()
    splits = load_torch_graph_file(os.path.join(dataset_root, 'splits.pt'))
    if not isinstance(splits, dict):
        raise ValueError(f'{data_name} LP splits.pt must be a dict.')

    edge_index = torch.cat([
        ready_edge_tensor_to_index(splits['val_pos']),
        ready_edge_tensor_to_index(splits['test_pos']),
        ready_edge_tensor_to_index(splits['train_pos']),
    ], dim=1).long()
    neg_edges = torch.cat([
        ready_edge_tensor_to_index(splits['val_neg']),
        ready_edge_tensor_to_index(splits['test_neg']),
        ready_edge_tensor_to_index(splits['train_neg']),
    ], dim=1).long()
    labels = torch.tensor([], dtype=torch.long)
    mask = torch.tensor([])
    num_classes = None
    return features, int(features.shape[-1]), labels, edge_index, neg_edges, mask, num_classes


def load_hetero_graph_data(root: str, data_name: str, downstream_task: str, split_seed: int):
    dataset_meta = HETERO_GRAPH_DATASETS[data_name.lower()]
    unified_path, unified_dir = resolve_unified_data_file(root, data_name)
    if unified_path is not None:
        raw_obj = load_torch_graph_file(unified_path)
        data_obj = raw_obj[0] if isinstance(raw_obj, tuple) else raw_obj
        if not hasattr(data_obj, 'x') or not hasattr(data_obj, 'edge_index'):
            raise ValueError(f'{data_name} unified_data.pt is missing x/edge_index.')

        features = data_obj.x.detach().cpu().float()
        num_features = int(features.shape[-1])
        edge_index = data_obj.edge_index.detach().cpu().long()
        neg_edges = negative_sampling(edge_index, num_nodes=features.shape[0]).long() if edge_index.numel() else torch.empty((2, 0), dtype=torch.long)

        if downstream_task == 'LP':
            edge_index, neg_edges = shuffle_edges(edge_index, neg_edges, split_seed)
            labels = torch.tensor([], dtype=torch.long)
            mask = torch.tensor([])
            num_classes = None
            return features, num_features, labels, edge_index, neg_edges, mask, num_classes

        y = getattr(data_obj, 'y', None)
        if y is None:
            raise ValueError(f'{data_name} unified_data.pt is missing y for node classification.')
        train_mask = getattr(data_obj, 'train_mask', None)
        val_mask = getattr(data_obj, 'val_mask', None)
        test_mask = getattr(data_obj, 'test_mask', None)
        labels = extract_labels(y)
        if train_mask is not None and val_mask is not None and test_mask is not None:
            idx_train, idx_val, idx_test = get_split_indices_from_masks(
                train_mask, val_mask, test_mask, split_seed, labels=labels
            )
        else:
            labeled_idx = torch.where(labels >= 0)[0]
            if labeled_idx.numel() == 0:
                raise ValueError(f'{data_name} unified_data.pt has no labeled nodes (y>=0).')
            labels_labeled = labels[labeled_idx]
            idx_val_rel, idx_test_rel, idx_train_rel = split_data(
                labels_labeled.tolist(), 0.2, 0.2, seed=split_seed
            )
            idx_train = labeled_idx[torch.tensor(idx_train_rel, dtype=torch.long)].tolist()
            idx_val = labeled_idx[torch.tensor(idx_val_rel, dtype=torch.long)].tolist()
            idx_test = labeled_idx[torch.tensor(idx_test_rel, dtype=torch.long)].tolist()
        mask = (idx_train, idx_val, idx_test)
        num_classes = int(labels.max().item()) + 1
        return features, num_features, labels.long(), edge_index, neg_edges, mask, num_classes

    dataset_dir = resolve_dataset_dir(root, data_name)
    data_file = os.path.join(dataset_dir, dataset_meta['graph_file'])
    raw_obj = load_torch_graph_file(data_file)
    data_obj = raw_obj[0] if isinstance(raw_obj, tuple) else raw_obj

    if not hasattr(data_obj, 'node_types') or not hasattr(data_obj, 'edge_types'):
        raise ValueError(f'{data_name} is not a HeteroData graph file.')

    features, edge_index, node_offsets = flatten_hetero_graph(data_obj)
    num_features = int(features.shape[-1])
    neg_edges = negative_sampling(edge_index, num_nodes=features.shape[0]).long() if edge_index.numel() else torch.empty((2, 0), dtype=torch.long)

    if downstream_task == 'LP':
        edge_index, neg_edges = shuffle_edges(edge_index, neg_edges, split_seed)
        labels = torch.tensor([], dtype=torch.long)
        mask = torch.tensor([])
        num_classes = None
        return features, num_features, labels, edge_index, neg_edges, mask, num_classes

    if 'nc_target_node_type' not in dataset_meta:
        raise ValueError(f'Dataset {data_name} does not currently support node classification.')

    labels, mask, num_classes = load_hetero_nc_labels(dataset_dir, dataset_meta, node_offsets, features.shape[0], split_seed)
    return features, num_features, labels, edge_index.long(), neg_edges, mask, num_classes


def flatten_hetero_graph(data_obj):
    node_offsets = {}
    features = []
    total_nodes = 0
    feature_dim = None

    for node_type in data_obj.node_types:
        x = getattr(data_obj[node_type], 'x', None)
        if x is None:
            raise ValueError(f'Node type {node_type} does not contain features.')
        x = x.float()
        if x.dim() != 2:
            raise ValueError(f'Node type {node_type} features must be rank-2, got shape {tuple(x.shape)}.')
        if feature_dim is None:
            feature_dim = x.shape[1]
        elif x.shape[1] != feature_dim:
            raise ValueError(f'Inconsistent feature dims in hetero graph: expected {feature_dim}, got {x.shape[1]} for {node_type}.')
        start = total_nodes
        total_nodes += x.shape[0]
        node_offsets[node_type] = (start, total_nodes)
        features.append(x)

    merged_features = torch.cat(features, dim=0)
    edge_parts = []
    for edge_type in data_obj.edge_types:
        src_type, _, dst_type = edge_type
        edge_index = getattr(data_obj[edge_type], 'edge_index', None)
        if edge_index is None or edge_index.numel() == 0:
            continue
        src_start, _ = node_offsets[src_type]
        dst_start, _ = node_offsets[dst_type]
        offset = torch.tensor([[src_start], [dst_start]], dtype=torch.long)
        edge_parts.append(edge_index.long() + offset)

    if edge_parts:
        merged_edge_index = torch.cat(edge_parts, dim=1)
        merged_edge_index = torch.cat([merged_edge_index, merged_edge_index.flip(0)], dim=1)
        merged_edge_index = torch.unique(merged_edge_index, dim=1)
    else:
        merged_edge_index = torch.empty((2, 0), dtype=torch.long)

    return merged_features, merged_edge_index, node_offsets


def load_hetero_nc_labels(dataset_dir: str, dataset_meta: dict, node_offsets: dict, total_nodes: int, split_seed: int):
    target_node_type = dataset_meta['nc_target_node_type']
    label_path = os.path.join(dataset_dir, dataset_meta['nc_label_file'])
    label_column = dataset_meta['nc_label_column']
    start, end = node_offsets[target_node_type]
    target_node_count = end - start

    raw_labels = []
    with open(label_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_labels.append(row.get(label_column))

    if len(raw_labels) != target_node_count:
        raise ValueError(
            f'Label row count mismatch for {target_node_type}: expected {target_node_count}, found {len(raw_labels)} in {label_path}.'
        )

    encoded_labels = encode_label_values(raw_labels)
    labeled_positions = [idx for idx, value in enumerate(encoded_labels) if value is not None]
    labeled_values = [encoded_labels[idx] for idx in labeled_positions]
    if not labeled_values:
        raise ValueError(f'No usable node-classification labels found in {label_path}.')

    labels = torch.zeros(total_nodes, dtype=torch.long)
    for idx, value in enumerate(encoded_labels):
        if value is not None:
            labels[start + idx] = int(value)

    idx_val_rel, idx_test_rel, idx_train_rel = split_data(
        labeled_values,
        HETERO_VAL_PROP,
        HETERO_TEST_PROP,
        seed=split_seed,
    )
    idx_train = [start + labeled_positions[idx] for idx in idx_train_rel]
    idx_val = [start + labeled_positions[idx] for idx in idx_val_rel]
    idx_test = [start + labeled_positions[idx] for idx in idx_test_rel]
    num_classes = int(max(labeled_values)) + 1
    return labels, (idx_train, idx_val, idx_test), num_classes


def encode_label_values(raw_labels):
    normalized = []
    categorical_values = []

    for raw_value in raw_labels:
        if raw_value is None:
            normalized.append(None)
            continue
        value = str(raw_value).strip()
        if not value:
            normalized.append(None)
            continue

        lowered = value.lower()
        if lowered in BOOLEAN_LABELS:
            normalized.append(BOOLEAN_LABELS[lowered])
            continue

        try:
            numeric_value = float(value)
        except ValueError:
            normalized.append(value)
            categorical_values.append(value)
            continue

        if numeric_value.is_integer():
            normalized.append(int(numeric_value))
        else:
            normalized.append(value)
            categorical_values.append(value)

    categorical_map = {value: idx for idx, value in enumerate(sorted(set(categorical_values)))}
    encoded_labels = []
    for value in normalized:
        if value is None:
            encoded_labels.append(None)
        elif isinstance(value, str):
            encoded_labels.append(categorical_map[value])
        else:
            encoded_labels.append(int(value))
    return encoded_labels


def load_pyg_graph_data(root: str, data_name: str, downstream_task: str, split_seed: int):
    if data_name.lower() in READY_DATASET_SPECS:
        return load_ready_dataset(root, data_name, downstream_task, split_seed)

    dataset_dir = resolve_dataset_dir(root, data_name)
    if data_name.lower() == 'telecom':
        data_file = os.path.join(dataset_dir, 'telecom_graph.pt')
    else:
        data_file = os.path.join(dataset_dir, 'processed', 'data.pt')

    raw_obj = load_torch_graph_file(data_file)
    data_obj = raw_obj[0] if isinstance(raw_obj, tuple) else raw_obj

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

    features = x.float()
    num_features = features.shape[-1]
    edge_index = edge_index.long()
    neg_edges = negative_sampling(edge_index).long()

    if downstream_task == 'LP':
        edge_index, neg_edges = shuffle_edges(edge_index, neg_edges, split_seed)
        labels = torch.tensor([], dtype=torch.long)
        mask = torch.tensor([])
        num_classes = None
        return features, num_features, labels, edge_index, neg_edges, mask, num_classes

    labels = extract_labels(y)
    if train_mask is not None and val_mask is not None and test_mask is not None:
        idx_train, idx_val, idx_test = get_split_indices_from_masks(train_mask, val_mask, test_mask, split_seed)
    else:
        idx_val, idx_test, idx_train = split_data(labels.tolist(), 0.2, 0.2, seed=split_seed)
    mask = (idx_train, idx_val, idx_test)
    num_classes = int(labels.max().item()) + 1
    return features, num_features, labels.long(), edge_index, neg_edges, mask, num_classes


def extract_labels(y):
    if y is None:
        raise ValueError('Node classification requires labels.')
    if y.dim() > 1:
        return torch.argmax(y, dim=1).long()
    return y.long()


def get_split_indices_from_masks(train_mask, val_mask, test_mask, split_seed, labels=None):
    train_mask = train_mask.cpu().numpy()
    val_mask = val_mask.cpu().numpy()
    test_mask = test_mask.cpu().numpy()

    if train_mask.ndim == 2:
        split_id = int(split_seed) % train_mask.shape[1]
        train_mask = train_mask[:, split_id]
        val_mask = val_mask[:, split_id]
        test_mask = test_mask[:, split_id]

    idx_train = np.where(train_mask)[0].tolist()
    idx_val = np.where(val_mask)[0].tolist()
    idx_test = np.where(test_mask)[0].tolist()
    if labels is not None:
        if torch.is_tensor(labels):
            labels_arr = labels.detach().cpu().numpy()
        else:
            labels_arr = np.asarray(labels)
        idx_train = [i for i in idx_train if labels_arr[i] >= 0]
        idx_val = [i for i in idx_val if labels_arr[i] >= 0]
        idx_test = [i for i in idx_test if labels_arr[i] >= 0]
    return idx_train, idx_val, idx_test


def shuffle_edges(edge_index, neg_edges, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    pos_perm = torch.randperm(edge_index.shape[1], generator=generator)
    neg_perm = torch.randperm(neg_edges.shape[1], generator=generator)
    return edge_index[:, pos_perm], neg_edges[:, neg_perm]


def mask_edges(edge_index, neg_edges, val_prop, test_prop):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val = edge_index[:, :n_val]
    edge_test = edge_index[:, n_val:n_val + n_test]
    edge_train = edge_index[:, n_val + n_test:]
    val_edges_neg = neg_edges[:, :n_val]
    test_edges_neg = neg_edges[:, n_val:n_test + n_val]
    train_edges_neg = torch.concat([neg_edges, edge_val, edge_test], dim=-1)
    return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.shape[0], 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


def split_data(labels, val_prop, test_prop, seed):
    random.seed(seed)
    num_class = np.max(labels) + 1
    label_dict = {i: [] for i in range(num_class)}
    for i, label in enumerate(labels):
        label_dict[int(label)].append(i)

    idx_train, idx_val, idx_test = [], [], []
    for i in range(num_class):
        random.shuffle(label_dict[i])
        num_val = round(val_prop * len(label_dict[i]))
        num_test = round(test_prop * len(label_dict[i]))
        idx_val += label_dict[i][:num_val]
        idx_test += label_dict[i][num_val:num_val + num_test]
        idx_train += label_dict[i][num_val + num_test:]
    return idx_val, idx_test, idx_train


class Airport(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        val_prop, test_prop = 0.15, 0.15
        graph = pkl.load(open(os.path.join(root, 'airport', 'airport.p'), 'rb'))
        adj = nx.adjacency_matrix(graph).toarray()
        row, col = np.nonzero(adj)
        edge_index = np.concatenate([row[None], col[None]], axis=0)
        features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
        features = augment(adj, torch.tensor(features).float())
        label_idx = 4
        labels = features[:, label_idx]
        features = features[:, :label_idx]
        labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])

        idx_val, idx_test, idx_train = split_data(labels.tolist(), val_prop, test_prop, DEFAULT_SEED)
        mask = (idx_train, idx_val, idx_test)

        self.data = torch_geometric.data.Data(
            x=features,
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
            mask=mask,
        )

    @property
    def num_features(self) -> int:
        return self.data.x.shape[-1]

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        return

    def process(self):
        return
