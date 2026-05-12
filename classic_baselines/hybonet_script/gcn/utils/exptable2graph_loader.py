"""Load ``data/exptable2graph/<subdir>/`` (unified_data.pt or *_HeteroGraph.pt) to scipy adj/feats/labels."""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    train_test_split = None


# Keys match --dataset exptable_*; values are subdirs under data/exptable2graph/
EXPTABLE2GRAPH_SUBDIRS: Dict[str, str] = {
    'exptable_carcinogenesis': 'Carcinogenesis_data',
    'exptable_hepatitis_std': 'Hepatitis_std_data',
    'exptable_hockey': 'Hockey_data',
    'exptable_pte': 'PTE',
    'exptable_toxicology': 'Toxicology_data',
    'exptable_f1': 'f1',
}


def is_exptable_dataset(name: str) -> bool:
    return name in EXPTABLE2GRAPH_SUBDIRS


def exptable_data_dir(repo_data_root: str, dataset_key: str) -> str:
    sub = EXPTABLE2GRAPH_SUBDIRS[dataset_key]
    return os.path.join(repo_data_root, 'exptable2graph', sub)


def _as_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def align_y_to_nodes(y: Any, n_nodes: int) -> np.ndarray:
    """Align labels to length n_nodes int64 vector."""
    if y is None:
        raise ValueError('exptable2graph: missing label tensor y')
    y = _as_numpy(y)
    if y.ndim == 2:
        if y.shape[0] != n_nodes:
            raise ValueError('y rows {} != n_nodes {}'.format(y.shape[0], n_nodes))
        if y.shape[1] == 1:
            y = y[:, 0]
        else:
            y = np.argmax(y, axis=1)
    y = y.reshape(-1)
    if y.shape[0] == n_nodes:
        out = y.astype(np.int64)
    elif y.shape[0] % n_nodes == 0:
        k = y.shape[0] // n_nodes
        y2 = y.reshape(n_nodes, k)
        out = np.argmax(y2, axis=1).astype(np.int64) if k > 1 else y2.ravel().astype(np.int64)
    else:
        raise ValueError('label length {} cannot align to n_nodes {}'.format(y.shape[0], n_nodes))
    return out - out.min()


def _edge_index_to_adj(n: int, edge_index: np.ndarray, symmetric: bool = True) -> sp.csr_matrix:
    ei = np.asarray(edge_index)
    if ei.ndim != 2:
        raise ValueError('invalid edge_index shape')
    if ei.shape[0] == 2:
        row, col = ei[0], ei[1]
    else:
        row, col = ei[:, 0], ei[:, 1]
    row = row.astype(np.int64)
    col = col.astype(np.int64)
    data = np.ones(len(row), dtype=np.float32)
    adj = sp.csr_matrix((data, (row, col)), shape=(n, n))
    if symmetric:
        adj = adj + adj.T
        adj.data = np.minimum(adj.data, 1.0)
        adj.eliminate_zeros()
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj.tocsr()


def _extract_xy_ei_masks(obj: Any) -> Tuple[Any, Any, Any, Optional[Any], Optional[Any], Optional[Any]]:
    if isinstance(obj, dict):
        if 'data' in obj and hasattr(obj['data'], 'x'):
            obj = obj['data']
        elif 'x' in obj:
            return (
                obj['x'], obj['edge_index'], obj.get('y'),
                obj.get('train_mask'), obj.get('val_mask'), obj.get('test_mask'),
            )
        else:
            raise ValueError('dict missing x or edge_index')
    if hasattr(obj, 'x') and hasattr(obj, 'edge_index'):
        return (
            obj.x, obj.edge_index, getattr(obj, 'y', None),
            getattr(obj, 'train_mask', None),
            getattr(obj, 'val_mask', None),
            getattr(obj, 'test_mask', None),
        )
    raise ValueError('unrecognized object type: {}'.format(type(obj)))


def _hetero_to_homogeneous(path: str) -> Any:
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if hasattr(obj, 'to_homogeneous'):
        return obj.to_homogeneous()
    raise ValueError('{}: to_homogeneous failed; need torch_geometric and valid hetero data'.format(path))


def load_exptable2graph_folder(
    folder: str,
    use_feats: bool = True,
    split_seed: int = 0,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray, List[int], List[int], List[int]]:
    """Prefer unified_data.pt; else first *_HeteroGraph.pt homogenized."""
    unified = os.path.join(folder, 'unified_data.pt')
    hetero_glob = glob.glob(os.path.join(folder, '*_HeteroGraph.pt'))
    hetero_path = hetero_glob[0] if hetero_glob else None

    raw_obj = None
    if os.path.isfile(unified):
        raw_obj = torch.load(unified, map_location='cpu', weights_only=False)
    elif hetero_path:
        raw_obj = _hetero_to_homogeneous(hetero_path)
    else:
        raise FileNotFoundError(
            'folder {}: no unified_data.pt or *_HeteroGraph.pt'.format(folder))

    try:
        x, ei, y, tr_m, va_m, te_m = _extract_xy_ei_masks(raw_obj)
    except ValueError:
        if hetero_path and os.path.isfile(unified):
            raw_obj = _hetero_to_homogeneous(hetero_path)
            x, ei, y, tr_m, va_m, te_m = _extract_xy_ei_masks(raw_obj)
        else:
            raise

    x_np = _as_numpy(x)
    ei_np = _as_numpy(ei)
    n = x_np.shape[0]
    labels = align_y_to_nodes(y, n)

    if not use_feats:
        feats = sp.eye(n, format='csr', dtype=np.float32)
    else:
        if x_np.ndim != 2:
            raise ValueError('features x must be 2D, got shape {}'.format(x_np.shape))
        feats = sp.csr_matrix(x_np.astype(np.float32))

    adj = _edge_index_to_adj(n, ei_np, symmetric=True)

    def _mask_to_idx(m: Optional[Any]) -> Optional[List[int]]:
        if m is None:
            return None
        arr = _as_numpy(m).astype(bool).ravel()
        if arr.shape[0] != n:
            return None
        return np.where(arr)[0].tolist()

    idx_tr = _mask_to_idx(tr_m)
    idx_va = _mask_to_idx(va_m)
    idx_te = _mask_to_idx(te_m)

    if idx_tr is None or idx_va is None or idx_te is None:
        if train_test_split is None:
            raise RuntimeError('sklearn required to build random splits')
        inds = np.arange(n)
        idx_tr_a, idx_tmp, _, y_tmp = train_test_split(
            inds, labels, test_size=0.4, stratify=labels, random_state=int(split_seed))
        idx_va_a, idx_te_a = train_test_split(
            idx_tmp, test_size=0.5, stratify=y_tmp, random_state=int(split_seed))
        idx_tr = idx_tr_a.tolist()
        idx_va = idx_va_a.tolist()
        idx_te = idx_te_a.tolist()

    return adj, feats, labels, idx_tr, idx_va, idx_te


def load_exptable2graph_nc(dataset_key: str, repo_data_root: str, use_feats: bool, split_seed: int):
    folder = exptable_data_dir(repo_data_root, dataset_key)
    return load_exptable2graph_folder(folder, use_feats=use_feats, split_seed=split_seed)


def load_exptable2graph_lp(dataset_key: str, repo_data_root: str, use_feats: bool):
    adj, feats, _, _, _, _ = load_exptable2graph_nc(dataset_key, repo_data_root, use_feats, split_seed=0)
    return adj, feats
