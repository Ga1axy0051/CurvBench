"""Unified graph loading utilities shared across SA2GFM stages."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, AttributedGraphDataset, FacebookPagePage, Planetoid, Reddit, WebKB
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import coalesce, to_undirected

from .paths import paths

NAME_ALIASES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "citeseers": "CiteSeer",
    "pubmed": "PubMed",
    "wikics": "WikiCS",
    "arxiv": "ogbn-arxiv",
    "p-home": "P-home",
    "phome": "P-home",
    "p-tech": "P-tech",
    "ptech": "P-tech",
    "airport": "Airport",
    "actor": "Actor",
    "cs-phds": "cs_phds",
    "cs_phd": "cs_phds",
    "cs_phds": "cs_phds",
    "disease": "Disease",
    "disease_nc": "Disease",
    "cornell": "Cornell",
}

NAME_VARIANTS = {
    "cora": ["Cora"],
    "citeseer": ["CiteSeer", "Citeseer"],
    "pubmed": ["PubMed", "Pubmed"],
    "wikics": ["WikiCS"],
    "arxiv": ["Arxiv", "arXiv", "ogbn-arxiv"],
    "P-home": ["p-home", "PHome"],
    "P-tech": ["p-tech", "PTech"],
    "Airport": ["airport"],
    "Actor": ["actor"],
    "cs_phds": ["cs-phds", "cs_phd", "CS_PHDS"],
    "Disease": ["disease", "disease_nc"],
    "Cornell": ["cornell"],
}

_TENSOR_LIKE_KEYS = {
    "x",
    "y",
    "edge_index",
    "edge_weight",
    "enhanced_x",
    "enhanced_x_64",
    "svd_x",
    "text_svd_embedding",
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip()
    return NAME_ALIASES.get(key.lower(), key)


def _candidate_names(name: str) -> list[str]:
    normalized = normalize_dataset_name(name)
    candidates: list[str] = []
    for value in [
        normalized,
        name,
        normalized.lower(),
        name.lower(),
        normalized.upper(),
        name.upper(),
    ]:
        if value and value not in candidates:
            candidates.append(value)

    for alias, target in NAME_ALIASES.items():
        if target == normalized:
            for value in [alias, alias.lower(), alias.upper()]:
                if value and value not in candidates:
                    candidates.append(value)
    for value in NAME_VARIANTS.get(normalized, []):
        if value and value not in candidates:
            candidates.append(value)
    return candidates


def canonical_graph_path(dataset_name: str) -> Path:
    return paths.graph_ori_dir / f"{normalize_dataset_name(dataset_name)}.pt"


def _first_existing_path(paths_to_check: Iterable[Path]) -> Optional[Path]:
    return next((path for path in paths_to_check if path.exists()), None)


def _torch_load_local(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_pt_or_dict(path: Path) -> Data:
    obj = _torch_load_local(path)

    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif isinstance(value, list) and key in _TENSOR_LIKE_KEYS:
                value = torch.tensor(value)
            normalized[key] = value
        return Data(**normalized)
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        first = obj[0]
        if isinstance(first, Data):
            return first
        if isinstance(first, dict):
            normalized = {}
            for key, value in first.items():
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                elif isinstance(value, list) and key in _TENSOR_LIKE_KEYS:
                    value = torch.tensor(value)
                normalized[key] = value
            return Data(**normalized)

    raise ValueError(f"Unable to parse a PyG Data object from {path}.")


def flatten_node_labels(y: torch.Tensor) -> torch.Tensor:
    if y.dim() == 0:
        y = y.view(1)
    elif y.dim() == 2 and y.size(-1) == 1:
        y = y.view(-1)
    elif y.dim() != 1:
        raise ValueError("Node labels must be a 1D tensor or shape [N, 1].")
    return y.long()


def remap_labels(y: torch.Tensor) -> torch.Tensor:
    y = flatten_node_labels(y)
    if (y < 0).any():
        raise ValueError("Node labels contain negative values, which are unsupported for few-shot node classification.")

    unique_labels = torch.unique(y).cpu().tolist()
    unique_labels = sorted(int(v) for v in unique_labels)
    mapping = {old: new for new, old in enumerate(unique_labels)}
    return torch.tensor([mapping[int(v)] for v in y.cpu().tolist()], dtype=torch.long)


def has_enhanced_features(data: Data) -> bool:
    return (
        getattr(data, "enhanced_x_64", None) is not None or
        getattr(data, "enhanced_x", None) is not None
    )


def get_num_nodes(data: Data) -> int:
    for attr in ["x", "enhanced_x_64", "enhanced_x", "svd_x", "text_svd_embedding"]:
        tensor = getattr(data, attr, None)
        if isinstance(tensor, torch.Tensor):
            return int(tensor.size(0))

    if getattr(data, "num_nodes", None) is not None:
        return int(data.num_nodes)

    if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
        return int(data.edge_index.max().item()) + 1

    raise ValueError("Unable to infer num_nodes from graph data.")


def get_num_classes(data: Data, ignore_negative: bool = True) -> int:
    if getattr(data, "y", None) is None:
        raise ValueError("Graph data is missing labels y.")
    y = flatten_node_labels(data.y)
    if ignore_negative:
        y = y[y >= 0]
    if y.numel() == 0:
        raise ValueError("Graph data does not contain any valid labels.")
    return int(torch.unique(y).numel())


def get_feature_tensor(data: Data, prefer_enhanced: bool = False, require_features: bool = True) -> Optional[torch.Tensor]:
    attr_order = ["enhanced_x_64", "enhanced_x", "x"] if prefer_enhanced else ["x", "enhanced_x_64", "enhanced_x"]
    for attr in attr_order:
        value = getattr(data, attr, None)
        if value is not None:
            return value.float()
    if require_features:
        raise ValueError("Graph data is missing usable node features.")
    return None


def prepare_few_shot_labels(data: Data, data_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if getattr(data, "y", None) is None:
        raise ValueError(f"{data_name} is missing node labels y, so it cannot be used for few-shot node classification.")

    labels = flatten_node_labels(data.y).clone()
    valid_mask = labels >= 0
    if not valid_mask.any():
        raise ValueError(f"{data_name} does not contain any non-negative node labels for few-shot node classification.")

    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    remapped = remap_labels(labels[valid_idx])
    return valid_idx, remapped


def _candidate_graph_files(dataset_name: str) -> list[Path]:
    result: list[Path] = []
    for base_dir in [paths.graph_ori_dir, paths.data_root]:
        for name in _candidate_names(dataset_name):
            for filename in [f"{name}.pt", f"{name}_enhanced_x64.pt"]:
                path = base_dir / filename
                if path.is_file() and path not in result:
                    result.append(path)
    return result


def _candidate_dataset_dirs(dataset_name: str) -> list[Path]:
    result: list[Path] = []
    for base_dir in [paths.graph_ori_dir, paths.data_root]:
        for name in _candidate_names(dataset_name):
            path = base_dir / name
            if path.is_dir() and path not in result:
                result.append(path)
    return result


def _has_local_node_data(data_dir: Path) -> bool:
    return (data_dir / "data.pt").exists() or (data_dir / "edge_index.npy").exists()


def _has_cornell_raw_data(data_dir: Path) -> bool:
    candidate_feature_paths = [
        data_dir / "out1_node_feature_label.txt",
        data_dir / "raw" / "out1_node_feature_label.txt",
    ]
    candidate_edge_paths = [
        data_dir / "out1_graph_edges.txt",
        data_dir / "raw" / "out1_graph_edges.txt",
    ]
    return _first_existing_path(candidate_feature_paths) is not None and _first_existing_path(candidate_edge_paths) is not None


def _has_actor_raw_data(data_dir: Path) -> bool:
    return (
        ((data_dir / "actor.types").exists() and (data_dir / "actor.edges").exists()) or
        (((data_dir / "raw") / "actor.types").exists() and ((data_dir / "raw") / "actor.edges").exists()) or
        (((data_dir / "out1_node_feature_label.txt").exists() and (data_dir / "out1_graph_edges.txt").exists())) or
        ((((data_dir / "raw") / "out1_node_feature_label.txt").exists()) and (((data_dir / "raw") / "out1_graph_edges.txt").exists()))
    )


def _has_disease_raw_data(data_dir: Path) -> bool:
    edge_paths = [
        data_dir / "disease_nc.edges.csv",
        data_dir / "edges.csv",
        data_dir / "raw" / "disease_nc.edges.csv",
        data_dir / "raw" / "edges.csv",
    ]
    feat_paths = [
        data_dir / "disease_nc.feats.npz",
        data_dir / "feats.npz",
        data_dir / "raw" / "disease_nc.feats.npz",
        data_dir / "raw" / "feats.npz",
    ]
    label_paths = [
        data_dir / "disease_nc.labels.npy",
        data_dir / "labels.npy",
        data_dir / "raw" / "disease_nc.labels.npy",
        data_dir / "raw" / "labels.npy",
    ]
    return (
        _first_existing_path(edge_paths) is not None and
        _first_existing_path(feat_paths) is not None and
        _first_existing_path(label_paths) is not None
    )


def _has_cs_phds_raw_data(data_dir: Path) -> bool:
    return all(
        (data_dir / filename).exists()
        for filename in ["adj.pt", "feats.pt", "labels.pt"]
    )


def _load_dense_feature_npz(npz_path: Path) -> np.ndarray:
    npz_obj = np.load(npz_path, allow_pickle=True)

    if isinstance(npz_obj, np.ndarray):
        return np.asarray(npz_obj)

    keys = list(npz_obj.files)
    if len(keys) == 1:
        return np.asarray(npz_obj[keys[0]])

    if {"data", "indices", "indptr", "shape"}.issubset(keys):
        shape = tuple(int(v) for v in np.asarray(npz_obj["shape"]).tolist())
        data = np.asarray(npz_obj["data"])
        indices = np.asarray(npz_obj["indices"], dtype=np.int64)
        indptr = np.asarray(npz_obj["indptr"], dtype=np.int64)
        dense = np.zeros(shape, dtype=data.dtype)
        for row in range(shape[0]):
            start = indptr[row]
            end = indptr[row + 1]
            dense[row, indices[start:end]] = data[start:end]
        return dense

    if {"row", "col", "data", "shape"}.issubset(keys):
        shape = tuple(int(v) for v in np.asarray(npz_obj["shape"]).tolist())
        row = np.asarray(npz_obj["row"], dtype=np.int64)
        col = np.asarray(npz_obj["col"], dtype=np.int64)
        data = np.asarray(npz_obj["data"])
        dense = np.zeros(shape, dtype=data.dtype)
        np.add.at(dense, (row, col), data)
        return dense

    raise ValueError(
        f"Unsupported feature npz format in {npz_path}. "
        "Expected a single dense array or a sparse matrix export."
    )


def _remap_edge_ids(edge_pairs: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    flat_edges = edge_pairs.reshape(-1)
    remapped = np.searchsorted(node_ids, flat_edges)
    if remapped.size == 0:
        return remapped.reshape(0, 2)

    if remapped.max(initial=0) >= len(node_ids):
        raise ValueError("Edge list contains node ids that are missing from the node metadata file.")

    if not np.array_equal(node_ids[remapped], flat_edges):
        raise ValueError("Edge list contains node ids that are missing from the node metadata file.")

    return remapped.reshape(-1, 2)


def _normalize_data_object(data: Data, data_name: str) -> Data:
    if getattr(data, "edge_index", None) is None:
        raise ValueError(f"{data_name} is missing edge_index.")

    data.edge_index = data.edge_index.long().contiguous()

    for attr in ["x", "enhanced_x_64", "enhanced_x", "svd_x", "text_svd_embedding"]:
        tensor = getattr(data, attr, None)
        if tensor is not None:
            setattr(data, attr, tensor.float())

    if getattr(data, "y", None) is not None:
        data.y = flatten_node_labels(data.y)

    data.num_nodes = get_num_nodes(data)

    if getattr(data, "edge_weight", None) is None:
        data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)
    else:
        data.edge_weight = data.edge_weight.float()

    return data


def _load_local_node_data(data_dir: Path, data_name: str) -> Data:
    pt_path = data_dir / "data.pt"
    x_path = data_dir / "x.npy"
    y_path = data_dir / "y.npy"
    edge_index_path = data_dir / "edge_index.npy"
    edge_weight_path = data_dir / "edge_weight.npy"

    if pt_path.exists():
        data = _load_pt_or_dict(pt_path)
    else:
        if not edge_index_path.exists():
            raise FileNotFoundError(
                f"{data_name} is missing data.pt or edge_index.npy, so the graph cannot be constructed."
            )

        edge_index = torch.from_numpy(np.load(edge_index_path)).long()
        if edge_index.ndim != 2:
            raise ValueError(f"{data_name} edge_index must be a 2D array.")
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        if edge_index.shape[0] != 2:
            raise ValueError(f"{data_name} edge_index must have shape [2, E] or [E, 2].")

        x = torch.from_numpy(np.load(x_path)).float() if x_path.exists() else None
        y = torch.from_numpy(np.load(y_path)) if y_path.exists() else None
        edge_weight = torch.from_numpy(np.load(edge_weight_path)).float() if edge_weight_path.exists() else None
        data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)

    return _normalize_data_object(data, data_name)


def _load_cornell_raw_data(data_dir: Path, data_name: str) -> Data:
    feature_path = _first_existing_path([
        data_dir / "out1_node_feature_label.txt",
        data_dir / "raw" / "out1_node_feature_label.txt",
    ])
    edge_path = _first_existing_path([
        data_dir / "out1_graph_edges.txt",
        data_dir / "raw" / "out1_graph_edges.txt",
    ])
    if feature_path is None or edge_path is None:
        raise FileNotFoundError(
            "Cornell raw data is incomplete. Expected out1_node_feature_label.txt and out1_graph_edges.txt."
        )

    node_ids = []
    xs = []
    ys = []
    with open(feature_path, "r", encoding="utf-8") as file_obj:
        next(file_obj, None)
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            node_id_str, feature_str, label_str = line.split("\t")
            node_ids.append(int(node_id_str))
            xs.append([float(value) for value in feature_str.split(",") if value])
            ys.append(int(label_str))

    node_ids = np.asarray(node_ids, dtype=np.int64)
    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    x = torch.tensor(np.asarray(xs, dtype=np.float32)[order], dtype=torch.float)
    y = torch.tensor(np.asarray(ys, dtype=np.int64)[order], dtype=torch.long)

    edge_pairs = np.loadtxt(edge_path, delimiter="\t", skiprows=1, dtype=np.int64)
    if edge_pairs.ndim == 1:
        edge_pairs = edge_pairs.reshape(1, -1)
    if edge_pairs.shape[1] != 2:
        raise ValueError("Cornell edge file must have exactly two columns.")

    remapped_edges = _remap_edge_ids(edge_pairs, node_ids)
    edge_index = torch.from_numpy(remapped_edges.T.copy()).long()
    edge_index = coalesce(edge_index, num_nodes=x.size(0))
    return _normalize_data_object(Data(x=x, y=y, edge_index=edge_index), data_name)


def _load_actor_raw_data(data_dir: Path, data_name: str) -> Data:
    geom_feature_path = _first_existing_path([
        data_dir / "out1_node_feature_label.txt",
        data_dir / "raw" / "out1_node_feature_label.txt",
    ])
    geom_edge_path = _first_existing_path([
        data_dir / "out1_graph_edges.txt",
        data_dir / "raw" / "out1_graph_edges.txt",
    ])
    if geom_feature_path is not None and geom_edge_path is not None:
        node_ids = []
        xs = []
        ys = []
        with open(geom_feature_path, "r", encoding="utf-8") as file_obj:
            next(file_obj, None)
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                node_id_str, feature_str, label_str = line.split("\t")
                node_ids.append(int(node_id_str))
                feature_idx = [int(value) for value in feature_str.split(",") if value]
                xs.append(feature_idx)
                ys.append(int(label_str))

        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.size == 0:
            raise ValueError("Actor feature file is empty.")
        order = np.argsort(node_ids)
        node_ids = node_ids[order]
        ys = np.asarray(ys, dtype=np.int64)[order]
        xs = [xs[idx] for idx in order.tolist()]

        if np.unique(node_ids).size != node_ids.size:
            raise ValueError("Actor node ids must be unique.")

        feat_dim = 0
        if xs:
            feat_dim = max((max(feat) for feat in xs if feat), default=-1) + 1
        x_np = np.zeros((node_ids.size, feat_dim), dtype=np.float32)
        for row_idx, feat in enumerate(xs):
            if feat:
                x_np[row_idx, feat] = 1.0

        edge_pairs = np.loadtxt(geom_edge_path, delimiter="\t", skiprows=1, dtype=np.int64)
        if edge_pairs.ndim == 1:
            edge_pairs = edge_pairs.reshape(1, -1)
        if edge_pairs.shape[1] != 2:
            raise ValueError("Actor edge file must have exactly two columns.")

        remapped_edges = _remap_edge_ids(edge_pairs, node_ids)
        edge_index = torch.from_numpy(remapped_edges.T.copy()).long()
        edge_index = coalesce(edge_index, num_nodes=node_ids.size)
        x = torch.tensor(x_np, dtype=torch.float)
        y = torch.tensor(ys, dtype=torch.long)
        return _normalize_data_object(Data(x=x, y=y, edge_index=edge_index), data_name)

    type_path = _first_existing_path([
        data_dir / "actor.types",
        data_dir / "raw" / "actor.types",
    ])
    edge_path = _first_existing_path([
        data_dir / "actor.edges",
        data_dir / "raw" / "actor.edges",
    ])
    if type_path is None or edge_path is None:
        raise FileNotFoundError(
            "Actor raw data is incomplete. Expected either actor.types + actor.edges, or out1_node_feature_label.txt + out1_graph_edges.txt."
        )

    node_label_array = np.loadtxt(type_path, delimiter=",", dtype=np.int64)
    if node_label_array.ndim == 1:
        node_label_array = node_label_array.reshape(-1, 1)

    if node_label_array.shape[1] == 1:
        labels = node_label_array[:, 0]
        node_ids = np.arange(1, labels.shape[0] + 1, dtype=np.int64)
    else:
        node_ids = node_label_array[:, 0]
        labels = node_label_array[:, -1]

    order = np.argsort(node_ids)
    node_ids = node_ids[order]
    labels = labels[order]

    if np.unique(node_ids).size != node_ids.size:
        raise ValueError("Actor node ids must be unique.")

    edge_pairs = np.loadtxt(edge_path, delimiter=",", dtype=np.int64)
    if edge_pairs.ndim == 1:
        edge_pairs = edge_pairs.reshape(1, -1)
    if edge_pairs.shape[1] != 2:
        raise ValueError("Actor edge file must have exactly two columns.")

    remapped_edges = _remap_edge_ids(edge_pairs, node_ids)
    edge_index = torch.from_numpy(remapped_edges.T.copy()).long()
    y = torch.tensor(labels, dtype=torch.long)
    return _normalize_data_object(Data(edge_index=edge_index, y=y, num_nodes=y.size(0)), data_name)


def _load_disease_raw_data(data_dir: Path, data_name: str) -> Data:
    edge_path = _first_existing_path([
        data_dir / "disease_nc.edges.csv",
        data_dir / "edges.csv",
        data_dir / "raw" / "disease_nc.edges.csv",
        data_dir / "raw" / "edges.csv",
    ])
    feat_path = _first_existing_path([
        data_dir / "disease_nc.feats.npz",
        data_dir / "feats.npz",
        data_dir / "raw" / "disease_nc.feats.npz",
        data_dir / "raw" / "feats.npz",
    ])
    label_path = _first_existing_path([
        data_dir / "disease_nc.labels.npy",
        data_dir / "labels.npy",
        data_dir / "raw" / "disease_nc.labels.npy",
        data_dir / "raw" / "labels.npy",
    ])
    if edge_path is None or feat_path is None or label_path is None:
        raise FileNotFoundError(
            "Disease raw data is incomplete. Expected disease_nc.edges.csv, disease_nc.feats.npz, and disease_nc.labels.npy."
        )

    edge_pairs = np.loadtxt(edge_path, delimiter=",", dtype=np.int64)
    if edge_pairs.ndim == 1:
        edge_pairs = edge_pairs.reshape(1, -1)
    if edge_pairs.shape[1] != 2:
        raise ValueError("Disease edge file must have exactly two columns.")

    x_np = np.asarray(_load_dense_feature_npz(feat_path), dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError("Disease features must be a 2D matrix.")

    y_np = np.asarray(np.load(label_path))
    if y_np.ndim == 2:
        if y_np.shape[1] == 1:
            y_np = y_np.reshape(-1)
        else:
            y_np = y_np.argmax(axis=-1)
    else:
        y_np = y_np.reshape(-1)
    y_np = y_np.astype(np.int64)

    num_nodes = max(int(edge_pairs.max()) + 1, x_np.shape[0], y_np.shape[0])
    if x_np.shape[0] != num_nodes:
        raise ValueError(
            f"Disease feature matrix row count ({x_np.shape[0]}) does not match inferred node count ({num_nodes})."
        )
    if y_np.shape[0] != num_nodes:
        raise ValueError(
            f"Disease label count ({y_np.shape[0]}) does not match inferred node count ({num_nodes})."
        )

    edge_index = torch.from_numpy(edge_pairs.T.copy()).long()
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    x = torch.tensor(x_np, dtype=torch.float)
    y = torch.tensor(y_np, dtype=torch.long)
    return _normalize_data_object(Data(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes), data_name)


def _load_cs_phds_raw_data(data_dir: Path, data_name: str) -> Data:
    adj_path = data_dir / "adj.pt"
    feat_path = data_dir / "feats.pt"
    label_path = data_dir / "labels.pt"

    missing = [path.name for path in [adj_path, feat_path, label_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"cs_phds raw data is incomplete. Missing files: {', '.join(missing)}"
        )

    adj = _torch_load_local(adj_path)
    x = _torch_load_local(feat_path).float()
    y = flatten_node_labels(_torch_load_local(label_path))

    if x.dim() != 2:
        raise ValueError("cs_phds feats.pt must be a 2D tensor [num_nodes, num_features].")

    num_nodes = x.size(0)
    if y.size(0) != num_nodes:
        raise ValueError(
            f"cs_phds label count ({y.size(0)}) does not match feature rows ({num_nodes})."
        )

    if adj.is_sparse:
        adj = adj.coalesce()
        edge_index = adj.indices().long()
        edge_weight = adj.values().float()
    elif adj.dim() == 2 and adj.size(0) == 2 and adj.size(1) != adj.size(0):
        edge_index = adj.long().contiguous()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    elif adj.dim() == 2 and adj.size(1) == 2 and adj.size(0) != adj.size(1):
        edge_index = adj.t().long().contiguous()
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    elif adj.dim() == 2 and adj.size(0) == num_nodes and adj.size(1) == num_nodes:
        edge_index = (adj != 0).nonzero(as_tuple=False).t().contiguous().long()
        edge_weight = adj[edge_index[0], edge_index[1]].float()
    else:
        raise ValueError("cs_phds adj.pt must be a sparse adjacency, dense [N, N] adjacency, or edge_index.")

    edge_index, edge_weight = coalesce(edge_index, edge_weight, num_nodes=num_nodes)
    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)
    return _normalize_data_object(data, data_name)


def _build_node2vec_features(
    data: Data,
    embed_dim: int = 64,
    batch_size: int = 128,
    walk_length: int = 20,
    context_size: int = 10,
    lr: float = 0.01,
    walks_per_node: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    num_epochs: int = 100,
) -> Data:
    from torch_geometric.nn import Node2Vec

    edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Node2Vec(
        edge_index,
        num_nodes=data.num_nodes,
        embedding_dim=embed_dim,
        walk_length=walk_length,
        context_size=context_size,
        walks_per_node=walks_per_node,
        p=p,
        q=q,
        sparse=True,
    ).to(device)

    optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        loader = model.loader(batch_size=batch_size, shuffle=True)
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Node2Vec Epoch {epoch}, Loss: {total_loss / max(1, len(loader)):.4f}")

    out = data.clone()
    out.x = model.embedding.weight.detach().cpu().float()
    return _normalize_data_object(out, "Node2VecGraph")


def _load_builtin_single_graph_data(dataset_name: str) -> Optional[Data]:
    data_root = str(paths.data_root)
    if dataset_name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset

        data = PygNodePropPredDataset(root=data_root, name=dataset_name)[0]
    elif dataset_name in ["Computers", "Photo"]:
        data = Amazon(data_root, dataset_name)[0]
    elif dataset_name == "Reddit":
        data = Reddit(f"{data_root}/{dataset_name}")[0]
    elif dataset_name == "FacebookPagePage":
        data = FacebookPagePage(f"{data_root}/{dataset_name}")[0]
    elif dataset_name == "PPI":
        data = AttributedGraphDataset(data_root, name=dataset_name.lower())[0]
    elif dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        data = Planetoid(data_root, dataset_name)[0]
    elif dataset_name == "Cornell":
        data = WebKB(data_root, dataset_name)[0]
    else:
        return None

    data = _normalize_data_object(data, dataset_name)
    if getattr(data, "edge_weight", None) is None:
        data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)
    return data


def load_graph(
    dataset_name: str,
    require_features: bool = False,
    require_enhanced: bool = False,
    build_node2vec_if_missing: bool = False,
) -> Data:
    normalized_name = normalize_dataset_name(dataset_name)
    fallback_reason: Optional[str] = None

    for path in _candidate_graph_files(dataset_name):
        data = _normalize_data_object(_load_pt_or_dict(path), normalized_name)
        if require_enhanced and not has_enhanced_features(data):
            fallback_reason = f"{path} exists but has no enhanced_x_64 / enhanced_x"
            continue
        if require_features and getattr(data, "x", None) is None:
            if build_node2vec_if_missing:
                data = _build_node2vec_features(data)
            else:
                raise ValueError(f"{path} is missing raw node features x.")
        return data

    for data_dir in _candidate_dataset_dirs(dataset_name):
        data: Optional[Data] = None
        if _has_local_node_data(data_dir):
            data = _load_local_node_data(data_dir, normalized_name)
        elif normalized_name == "Cornell" and _has_cornell_raw_data(data_dir):
            data = _load_cornell_raw_data(data_dir, normalized_name)
        elif normalized_name == "Actor" and _has_actor_raw_data(data_dir):
            data = _load_actor_raw_data(data_dir, normalized_name)
        elif normalized_name == "Disease" and _has_disease_raw_data(data_dir):
            data = _load_disease_raw_data(data_dir, normalized_name)
        elif normalized_name == "cs_phds" and _has_cs_phds_raw_data(data_dir):
            data = _load_cs_phds_raw_data(data_dir, normalized_name)

        if data is None:
            continue

        if require_enhanced and not has_enhanced_features(data):
            fallback_reason = f"{data_dir} can be parsed, but enhanced_x_64 / enhanced_x is still missing"
            continue
        if require_features and getattr(data, "x", None) is None:
            if build_node2vec_if_missing:
                data = _build_node2vec_features(data)
            else:
                raise ValueError(f"{data_dir} is missing raw node features x.")
        return data

    builtin_data = _load_builtin_single_graph_data(normalized_name)
    if builtin_data is not None:
        if require_enhanced and not has_enhanced_features(builtin_data):
            fallback_reason = f"{normalized_name} can be loaded from a built-in dataset, but enhanced_x_64 / enhanced_x is still missing"
        else:
            data = builtin_data
            if require_features and getattr(data, "x", None) is None:
                if build_node2vec_if_missing:
                    data = ToUndirected()(data)
                    data = _build_node2vec_features(data)
                else:
                    raise ValueError(f"{normalized_name} is missing raw node features x.")
            return data

    if require_enhanced and fallback_reason is not None:
        raise ValueError(
            f"{dataset_name} was found, but enhanced features are missing. {fallback_reason}. "
            "Run node_feature_enhance and save the output back to the canonical ori/{dataset}.pt file."
        )

    raise FileNotFoundError(
        f"Missing graph data for dataset {dataset_name!r}.\n"
        f"Looked for GraphGlue-style local imports under {paths.graph_ori_dir} and {paths.data_root}, "
        "including *.pt, data.pt, edge_index.npy, and supported raw files for Actor/Cornell/Disease.\n"
        f"Set SA2GFM_DATA_ROOT to the directory that contains `ori/` (current data_root={paths.data_root})."
    )
