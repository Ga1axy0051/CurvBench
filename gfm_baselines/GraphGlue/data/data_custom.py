import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce


NAME_ALIASES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "citeseers": "CiteSeer",
    "pubmed": "PubMed",
    "cs-phds": "cs_phds",
    "cs_phd": "cs_phds",
    "cs_phds": "cs_phds",
    "airport": "Airport",
    "actor": "Actor",
    "disease": "Disease",
    "disease_nc": "Disease",
    "telecom": "Telecom",
    "cornell": "Cornell",
}


def normalize_dataset_name(name: str) -> str:
    key = name.strip()
    return NAME_ALIASES.get(key.lower(), key)


def resolve_dataset_dir(root: str, data_name: str) -> str:
    normalized = normalize_dataset_name(data_name)
    candidates = []
    for name in [
        normalized,
        data_name,
        normalized.lower(),
        data_name.lower(),
        normalized.upper(),
        data_name.upper(),
    ]:
        if name and name not in candidates:
            candidates.append(name)

    for alias, target in NAME_ALIASES.items():
        if target == normalized:
            for name in [alias, alias.lower(), alias.upper()]:
                if name and name not in candidates:
                    candidates.append(name)

    for name in candidates:
        path = os.path.join(root, name)
        if os.path.isdir(path):
            return path

    return os.path.join(root, normalized)


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


def _torch_load_local(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _load_pt_or_dict(pt_path: str) -> Data:
    obj = _torch_load_local(pt_path)

    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            elif isinstance(value, list) and key in {"x", "y", "edge_index", "edge_weight"}:
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
                elif isinstance(value, list) and key in {"x", "y", "edge_index", "edge_weight"}:
                    value = torch.tensor(value)
                normalized[key] = value
            return Data(**normalized)

    raise ValueError(f"Unable to parse a PyG Data object from {pt_path}.")


def load_local_node_data(root: str, data_name: str) -> Data:
    name = normalize_dataset_name(data_name)
    data_dir = resolve_dataset_dir(root, name)

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    pt_path = os.path.join(data_dir, "data.pt")
    x_path = os.path.join(data_dir, "x.npy")
    y_path = os.path.join(data_dir, "y.npy")
    edge_index_path = os.path.join(data_dir, "edge_index.npy")
    edge_weight_path = os.path.join(data_dir, "edge_weight.npy")

    if os.path.exists(pt_path):
        data = _load_pt_or_dict(pt_path)
    else:
        if not os.path.exists(edge_index_path):
            raise FileNotFoundError(
                f"{name} is missing data.pt or edge_index.npy, so the graph cannot be constructed."
            )

        edge_index = torch.from_numpy(np.load(edge_index_path)).long()
        if edge_index.ndim != 2:
            raise ValueError(f"{name} edge_index must be a 2D array.")
        if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
            edge_index = edge_index.t().contiguous()
        if edge_index.shape[0] != 2:
            raise ValueError(f"{name} edge_index must have shape [2, E] or [E, 2].")

        x = torch.from_numpy(np.load(x_path)).float() if os.path.exists(x_path) else None
        y = torch.from_numpy(np.load(y_path)) if os.path.exists(y_path) else None
        edge_weight = (
            torch.from_numpy(np.load(edge_weight_path)).float()
            if os.path.exists(edge_weight_path)
            else None
        )

        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_weight=edge_weight,
        )

    if getattr(data, "edge_index", None) is None:
        raise ValueError(f"{name} is missing edge_index.")

    data.edge_index = data.edge_index.long().contiguous()

    if getattr(data, "x", None) is not None:
        data.x = data.x.float()

    if getattr(data, "y", None) is not None:
        data.y = flatten_node_labels(data.y)

    if getattr(data, "num_nodes", None) is None:
        if getattr(data, "x", None) is not None:
            data.num_nodes = data.x.size(0)
        else:
            data.num_nodes = int(data.edge_index.max().item()) + 1

    if getattr(data, "edge_weight", None) is None:
        data.edge_weight = torch.ones(data.edge_index.size(1), dtype=torch.float)

    return data


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


def _first_existing_path(paths: List[str]) -> Optional[str]:
    return next((path for path in paths if os.path.exists(path)), None)


def _load_dense_feature_npz(npz_path: str) -> np.ndarray:
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
        f"Expected a single dense array or a sparse matrix export."
    )


def load_cornell_raw_data(root: str, data_name: str = "Cornell") -> Data:
    data_dir = resolve_dataset_dir(root, data_name)
    raw_dir = os.path.join(data_dir, "raw")
    feature_path = os.path.join(raw_dir, "out1_node_feature_label.txt")
    edge_path = os.path.join(raw_dir, "out1_graph_edges.txt")

    if not os.path.exists(feature_path) or not os.path.exists(edge_path):
        raise FileNotFoundError(
            "Cornell raw data is incomplete. Expected raw/out1_node_feature_label.txt and raw/out1_graph_edges.txt."
        )

    node_ids = []
    xs = []
    ys = []
    with open(feature_path, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            node_id_str, feature_str, label_str = line.split("\t")
            node_ids.append(int(node_id_str))
            xs.append([float(value) for value in feature_str.split(",")])
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

    return Data(x=x, y=y, edge_index=edge_index)


def load_actor_raw_data(root: str, data_name: str = "Actor") -> Data:
    data_dir = resolve_dataset_dir(root, data_name)
    candidate_geom_feature_paths = [
        os.path.join(data_dir, "out1_node_feature_label.txt"),
        os.path.join(data_dir, "raw", "out1_node_feature_label.txt"),
    ]
    candidate_geom_edge_paths = [
        os.path.join(data_dir, "out1_graph_edges.txt"),
        os.path.join(data_dir, "raw", "out1_graph_edges.txt"),
    ]
    candidate_type_paths = [
        os.path.join(data_dir, "actor.types"),
        os.path.join(data_dir, "raw", "actor.types"),
    ]
    candidate_edge_paths = [
        os.path.join(data_dir, "actor.edges"),
        os.path.join(data_dir, "raw", "actor.edges"),
    ]

    geom_feature_path = next((path for path in candidate_geom_feature_paths if os.path.exists(path)), None)
    geom_edge_path = next((path for path in candidate_geom_edge_paths if os.path.exists(path)), None)
    if geom_feature_path is not None and geom_edge_path is not None:
        node_ids = []
        xs = []
        ys = []
        with open(geom_feature_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
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
        return Data(x=x, y=y, edge_index=edge_index)

    type_path = next((path for path in candidate_type_paths if os.path.exists(path)), None)
    edge_path = next((path for path in candidate_edge_paths if os.path.exists(path)), None)

    if type_path is None or edge_path is None:
        raise FileNotFoundError(
            "Actor raw data is incomplete. Expected either actor.types + actor.edges, or out1_node_feature_label.txt + out1_graph_edges.txt in the dataset directory."
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

    return Data(edge_index=edge_index, y=y, num_nodes=y.size(0))


def load_disease_raw_data(root: str, data_name: str = "Disease") -> Data:
    data_dir = resolve_dataset_dir(root, data_name)
    candidate_edge_paths = [
        os.path.join(data_dir, "disease_nc.edges.csv"),
        os.path.join(data_dir, "edges.csv"),
        os.path.join(data_dir, "raw", "disease_nc.edges.csv"),
        os.path.join(data_dir, "raw", "edges.csv"),
    ]
    candidate_feat_paths = [
        os.path.join(data_dir, "disease_nc.feats.npz"),
        os.path.join(data_dir, "feats.npz"),
        os.path.join(data_dir, "raw", "disease_nc.feats.npz"),
        os.path.join(data_dir, "raw", "feats.npz"),
    ]
    candidate_label_paths = [
        os.path.join(data_dir, "disease_nc.labels.npy"),
        os.path.join(data_dir, "labels.npy"),
        os.path.join(data_dir, "raw", "disease_nc.labels.npy"),
        os.path.join(data_dir, "raw", "labels.npy"),
    ]

    edge_path = _first_existing_path(candidate_edge_paths)
    feat_path = _first_existing_path(candidate_feat_paths)
    label_path = _first_existing_path(candidate_label_paths)

    if edge_path is None or feat_path is None or label_path is None:
        raise FileNotFoundError(
            "Disease raw data is incomplete. Expected disease_nc.edges.csv, disease_nc.feats.npz, and disease_nc.labels.npy."
        )

    edge_pairs = np.loadtxt(edge_path, delimiter=",", dtype=np.int64)
    if edge_pairs.ndim == 1:
        edge_pairs = edge_pairs.reshape(1, -1)
    if edge_pairs.shape[1] != 2:
        raise ValueError("Disease edge file must have exactly two columns.")

    x_np = _load_dense_feature_npz(feat_path)
    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim != 2:
        raise ValueError("Disease features must be a 2D matrix.")

    y_np = np.load(label_path)
    y_np = np.asarray(y_np)
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
    return Data(x=x, y=y, edge_index=edge_index, num_nodes=num_nodes)


def load_cs_phds_raw_data(root: str, data_name: str = "cs_phds") -> Data:
    data_dir = resolve_dataset_dir(root, data_name)
    adj_path = os.path.join(data_dir, "adj.pt")
    feat_path = os.path.join(data_dir, "feats.pt")
    label_path = os.path.join(data_dir, "labels.pt")

    missing = [path for path in [adj_path, feat_path, label_path] if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(
            f"cs_phds raw data is incomplete. Missing files: {', '.join(os.path.basename(p) for p in missing)}"
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
    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, num_nodes=num_nodes)


class FB15k_237(InMemoryDataset):
    """
    The same with torch_geometric.datasets.FB15K_237.
    But applying pre_transform.
    """
    url = ('https://raw.githubusercontent.com/villmow/'
           'datasets_knowledge_embedding/master/FB15k-237')

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)

        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid 'split' argument (got {split})")

        path = self.processed_paths[['train', 'val', 'test'].index(split)]
        self.load(path)

    @property
    def raw_file_names(self) -> List[str]:
        return ['train.txt', 'valid.txt', 'test.txt']

    @property
    def processed_file_names(self) -> List[str]:
        return ['train_data.pt', 'val_data.pt', 'test_data.pt']

    def download(self) -> None:
        for filename in self.raw_file_names:
            download_url(f'{self.url}/{filename}', self.raw_dir)

    def process(self) -> None:
        data_list: List[Data] = []
        node_dict: Dict[str, int] = {}
        rel_dict: Dict[str, int] = {}

        for path in self.raw_paths:
            with open(path) as f:
                lines = [x.split('\t') for x in f.read().split('\n')[:-1]]

            edge_index = torch.empty((2, len(lines)), dtype=torch.long)
            edge_type = torch.empty(len(lines), dtype=torch.long)
            for i, (src, rel, dst) in enumerate(lines):
                if src not in node_dict:
                    node_dict[src] = len(node_dict)
                if dst not in node_dict:
                    node_dict[dst] = len(node_dict)
                if rel not in rel_dict:
                    rel_dict[rel] = len(rel_dict)

                edge_index[0, i] = node_dict[src]
                edge_index[1, i] = node_dict[dst]
                edge_type[i] = rel_dict[rel]

            data = Data(edge_index=edge_index, edge_type=edge_type)
            data_list.append(data)

        for data, path in zip(data_list, self.processed_paths):
            data.num_nodes = len(node_dict)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            self.save([data], path)
