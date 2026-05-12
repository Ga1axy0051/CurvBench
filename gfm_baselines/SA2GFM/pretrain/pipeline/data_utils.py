"""Load graph + build sparse adj for pretrain (single dataset)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import to_undirected

_PIPELINE = Path(__file__).resolve().parent
_SA2GFM = _PIPELINE.parents[2]
_AG = _SA2GFM / "attack_data_gen"
sys.path.insert(0, str(_AG))
from lib.data_utils import load_graph, normalize_dataset_name  # noqa: E402


def load_dataset_pt(dataset_name: str):
    """
    Returns:
        features_np: np.ndarray (N, F)
        adj_sp: scipy.sparse.csr_matrix
        num_nodes: int
    """
    ori_data = load_graph(normalize_dataset_name(dataset_name), require_enhanced=True)
    features = getattr(ori_data, "enhanced_x_64", None)
    if features is None:
        features = getattr(ori_data, "enhanced_x", None)
    if features is None:
        raise AttributeError(f"{dataset_name} has no enhanced_x or enhanced_x_64")
    edge_index = ori_data.edge_index
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if isinstance(features, torch.Tensor):
        features = features.detach().float()
    else:
        features = torch.as_tensor(features, dtype=torch.float32)
    num_nodes = int(features.shape[0])
    edge_index = to_undirected(edge_index)
    adj = edge_index_to_sparse_adj(edge_index, num_nodes)
    return features.cpu().numpy().astype(np.float32), adj, num_nodes


def edge_index_to_sparse_adj(edge_index, num_nodes):
    edge_index = edge_index.cpu().numpy()
    row, col = edge_index[0], edge_index[1]
    data = np.ones(len(row))
    adj = sp.csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()


def get_negative_samples(adj, num_nodes, neg_samples=50):
    adj_dense = adj.todense()
    neg_samples_list = []
    for i in range(num_nodes):
        non_neighbors = np.where(adj_dense[i].A1 == 0)[0]
        if len(non_neighbors) > 0:
            neg = np.random.choice(non_neighbors, size=min(neg_samples, len(non_neighbors)), replace=False)
            neg_samples_list.append(neg)
        else:
            neg = np.random.choice(np.arange(num_nodes), size=neg_samples, replace=False)
            neg_samples_list.append(neg)
    return np.array(neg_samples_list)
