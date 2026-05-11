from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.io import read_planetoid_data
from torch_geometric.utils import degree

PLANETOID_SPECS = {
    "cora": {
        "prefix": "ind.cora",
        "raw_names": (
            "ind.cora.x",
            "ind.cora.tx",
            "ind.cora.ty",
            "ind.cora.allx",
            "ind.cora.ally",
            "ind.cora.graph",
            "ind.cora.test.index",
        ),
        "text_dir": "cora",
        "train_n": 140,
        "val_n": 500,
    },
    "citeseer": {
        "prefix": "ind.citeseer",
        "raw_names": (
            "ind.citeseer.x",
            "ind.citeseer.tx",
            "ind.citeseer.ty",
            "ind.citeseer.allx",
            "ind.citeseer.ally",
            "ind.citeseer.graph",
            "ind.citeseer.test.index",
        ),
        "text_dir": "citeseer",
        "train_n": 120,
        "val_n": 500,
    },
    "pubmed": {
        "prefix": "ind.pubmed",
        "raw_names": (
            "ind.pubmed.x",
            "ind.pubmed.tx",
            "ind.pubmed.ty",
            "ind.pubmed.allx",
            "ind.pubmed.ally",
            "ind.pubmed.graph",
            "ind.pubmed.test.index",
        ),
        "text_dir": "pubmed",
        "train_n": 60,
        "val_n": 500,
    },
}


def _planetoid_capital_dir(name: str, data_root: Path) -> Path:
    return data_root / name.capitalize() / "raw"


def _has_all_raw(raw_dir: Path, names: tuple[str, ...]) -> bool:
    return all((raw_dir / n).is_file() for n in names)


def _iter_planetoid_raw_dirs(data_root: Path, key: str) -> list[Path]:
    k = key.lower()
    k_cap = key.capitalize()
    return [
        data_root / k_cap / "raw",
        data_root / k / "raw",
        data_root / k,
    ]


def load_planetoid_from_ind_raw(raw_dir: Path, key: str) -> Tuple[Data, SimpleNamespace]:
    data = read_planetoid_data(str(raw_dir), key.capitalize())
    meta = SimpleNamespace(
        num_features=int(data.x.size(1)),
        num_classes=int(data.y.max().item()) + 1,
        source=f"planetoid_{key}_raw",
    )
    return data, meta


def load_planetoid_official(data_root: Path, key: str, download: bool = False) -> Tuple[Data, SimpleNamespace]:
    name = key.capitalize()
    ds = Planetoid(root=str(data_root), name=name, download=download)
    meta = SimpleNamespace(
        num_features=ds.num_features,
        num_classes=ds.num_classes,
        source=f"planetoid_{key}_raw",
    )
    return ds[0], meta


def _load_planetoid_text(
    data_root: Path,
    key: str,
    spec: dict[str, Any],
    split_seed: int,
) -> Data:
    sub = spec["text_dir"]
    content_name = f"{sub}.content"
    cites_name = f"{sub}.cites"
    content_p = data_root / sub / content_name
    cites_p = data_root / sub / cites_name
    test_idx_p = data_root / sub / "raw" / f"{spec['prefix']}.test.index"
    if not test_idx_p.is_file():
        test_idx_p = data_root / f"{spec['prefix']}.test.index"

    if not content_p.is_file() or not cites_p.is_file():
        raise FileNotFoundError(f"Missing {content_p} or {cites_p}")
    if not test_idx_p.is_file():
        raise FileNotFoundError(f"Missing test index: {test_idx_p}")

    paper_ids: list[int] = []
    features: list[list[float]] = []
    labels_str: list[str] = []
    classes: dict[str, int] = {}

    with open(content_p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            pid = int(parts[0])
            feats = [float(x) for x in parts[1:-1]]
            cls_name = parts[-1]
            if cls_name not in classes:
                classes[cls_name] = len(classes)
            paper_ids.append(pid)
            features.append(feats)
            labels_str.append(cls_name)

    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}
    num_nodes = len(paper_ids)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor([classes[s] for s in labels_str], dtype=torch.long)

    edges: list[list[int]] = []
    with open(cites_p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            a, b = line.strip().split("\t")
            ia, ib = id_to_idx[int(a)], id_to_idx[int(b)]
            edges.append([ia, ib])
            edges.append([ib, ia])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    test_ids: set[int] = set()
    with open(test_idx_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_ids.add(int(line))

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for pid in test_ids:
        test_mask[id_to_idx[pid]] = True

    non_test = np.where(~test_mask.numpy())[0]
    y_np = y.numpy()
    tn, vn = spec["train_n"], spec["val_n"]
    idx_train, idx_rest = train_test_split(
        non_test,
        train_size=tn,
        stratify=y_np[non_test],
        random_state=split_seed,
    )
    idx_val, _ = train_test_split(
        idx_rest,
        train_size=vn,
        stratify=y_np[idx_rest],
        random_state=split_seed,
    )
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True

    return Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)


def load_planetoid_dataset(data_root: Path, key: str, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    spec = PLANETOID_SPECS[key]
    standard_raw = _planetoid_capital_dir(key, data_root)
    if _has_all_raw(standard_raw, spec["raw_names"]):
        return load_planetoid_official(data_root, key, download=False)

    for cand in _iter_planetoid_raw_dirs(data_root, key):
        if cand == standard_raw:
            continue
        if _has_all_raw(cand, spec["raw_names"]):
            return load_planetoid_from_ind_raw(cand, key)

    data = _load_planetoid_text(data_root, key, spec, split_seed)
    meta = SimpleNamespace(
        num_features=int(data.x.size(1)),
        num_classes=int(data.y.max().item() + 1),
        source=f"{key}_text",
    )
    return data, meta


def _webkb_bow_dim_from_header(header: str) -> Optional[int]:
    m = re.search(r"feature_amount\D*(\d+)", header, re.I)
    return int(m.group(1)) if m else None


def load_webkb(data_root: Path, subdir: str, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    base = data_root / subdir / "raw"
    node_path = base / "out1_node_feature_label.txt"
    edge_path = base / "out1_graph_edges.txt"
    if not node_path.is_file() or not edge_path.is_file():
        raise FileNotFoundError(f"Missing WebKB files: {node_path}, {edge_path}")

    nodes_order: list[int] = []
    feats_list: list[list[float]] = []
    labels: list[int] = []
    bow_dim: Optional[int] = None

    with open(node_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        bow_dim = _webkb_bow_dim_from_header(header)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            nid = int(parts[0])
            feat_str = parts[1]
            lab = int(parts[2])
            if bow_dim is not None:
                vec = [0.0] * bow_dim
                for tok in feat_str.split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    wi = int(tok)
                    if 0 <= wi < bow_dim:
                        vec[wi] = 1.0
                feats = vec
            else:
                feats = [float(x) for x in feat_str.split(",")]
            nodes_order.append(nid)
            feats_list.append(feats)
            labels.append(lab)

    id_to_idx = {nid: i for i, nid in enumerate(nodes_order)}
    num_nodes = len(nodes_order)
    x = torch.tensor(feats_list, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    num_classes = int(y.max().item() + 1)

    edges: list[list[int]] = []
    with open(edge_path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b = re.split(r"\s+", line, maxsplit=1)
            ia, ib = id_to_idx[int(a)], id_to_idx[int(b)]
            edges.append([ia, ib])
            edges.append([ib, ia])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    idx = np.arange(num_nodes)
    y_np = y.numpy()
    strat = _stratify_or_none(y_np)
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, stratify=strat, random_state=split_seed
    )
    strat2 = _stratify_or_none(y_np[idx_temp])
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=strat2, random_state=split_seed
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    feat_note = f"_bow{bow_dim}" if bow_dim is not None else "_dense"
    meta = SimpleNamespace(
        num_features=int(data.x.size(1)),
        num_classes=num_classes,
        source=f"webkb_{subdir.lower()}{feat_note}",
    )
    return data, meta


def _edges_csv_to_graph(
    path: Path, delimiter: str = ",", comment: Optional[str] = None
) -> Tuple[torch.Tensor, int]:
    rows: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in raw_lines:
        if comment and line.startswith(comment):
            continue
        try:
            if "," in line:
                a, b = line.split(",")[:2]
                rows.append((int(a.strip()), int(b.strip())))
            else:
                parts = re.split(r"\s+", line)
                if len(parts) >= 2:
                    rows.append((int(parts[0]), int(parts[1])))
        except ValueError:
            continue

    if not rows:
        flat: list[int] = []
        for tok in text.replace("\n", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                flat.append(int(tok))
            except ValueError:
                continue
        for i in range(0, len(flat) - 1, 2):
            rows.append((flat[i], flat[i + 1]))

    if not rows:
        raise ValueError(f"No edges parsed from {path}")
    nodes_set = set()
    for a, b in rows:
        nodes_set.add(a)
        nodes_set.add(b)
    sorted_ids = sorted(nodes_set)
    id_map = {oid: i for i, oid in enumerate(sorted_ids)}
    edges: list[list[int]] = []
    for a, b in rows:
        ia, ib = id_map[a], id_map[b]
        edges.append([ia, ib])
        edges.append([ib, ia])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    n = len(sorted_ids)
    return edge_index, n


def _torch_load_any(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_long_1d(t: torch.Tensor) -> torch.Tensor:
    if t.dim() > 1:
        t = t.squeeze(-1)
    return t.long().view(-1)


def _load_disease_nc_feats_npz(path: Path) -> torch.Tensor:
    d = np.load(path)
    if "indices" in d.files and "indptr" in d.files and "data" in d.files and "shape" in d.files:
        from scipy.sparse import csr_matrix

        sp = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
        return torch.tensor(sp.toarray(), dtype=torch.float32)
    key = d.files[0]
    return torch.tensor(d[key], dtype=torch.float32)


def _try_load_xy_optional(folder: Path, num_nodes: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    feats_npz = folder / "disease_nc.feats.npz"
    labels_npy = folder / "disease_nc.labels.npy"
    if feats_npz.is_file() and labels_npy.is_file():
        x = _load_disease_nc_feats_npz(feats_npz)
        yraw = np.load(labels_npy)
        y = torch.tensor(yraw, dtype=torch.long).view(-1)
        if y.size(0) != num_nodes or x.size(0) != num_nodes:
            raise ValueError(
                f"{feats_npz} / {labels_npy} row count mismatch edges: "
                f"N={num_nodes}, x={x.shape}, y={y.shape}"
            )
        return x, y

    pt_pairs = [
        ("x.pt", "y.pt"),
        ("features.pt", "labels.pt"),
    ]
    for xf, yf in pt_pairs:
        xp, yp = folder / xf, folder / yf
        if xp.is_file() and yp.is_file():
            xv = _torch_load_any(xp)
            yv = _torch_load_any(yp)
            if not isinstance(xv, torch.Tensor):
                xv = torch.as_tensor(xv, dtype=torch.float32)
            if not isinstance(yv, torch.Tensor):
                yv = torch.as_tensor(yv)
            yv = _as_long_1d(yv)
            if xv.size(0) != num_nodes or yv.size(0) != num_nodes:
                raise ValueError(
                    f"{xp} / {yp} must have {num_nodes} rows; got x={xv.shape}, y={yv.shape}"
                )
            return xv.float(), yv

    xn, yn = folder / "x.npy", folder / "y.npy"
    if xn.is_file() and yn.is_file():
        xv = torch.from_numpy(np.load(xn)).float()
        yv = _as_long_1d(torch.from_numpy(np.load(yn)))
        if xv.size(0) != num_nodes or yv.size(0) != num_nodes:
            raise ValueError(f"{xn} / {yn} length must be {num_nodes}")
        return xv, yv

    fc, lc = folder / "node_features.csv", folder / "node_labels.csv"
    if fc.is_file() and lc.is_file():
        xf = np.loadtxt(fc, delimiter=",", dtype=np.float64)
        yraw = np.loadtxt(lc, delimiter=",", dtype=np.int64)
        if yraw.ndim > 1:
            yraw = yraw[:, 0]
        xv = torch.tensor(xf, dtype=torch.float32)
        yv = torch.tensor(yraw, dtype=torch.long).view(-1)
        if xv.size(0) != num_nodes or yv.size(0) != num_nodes:
            raise ValueError(f"{fc} / {lc} must have {num_nodes} rows")
        return xv, yv

    return None


def _synthetic_deg_xy(edge_index: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    d1 = torch.log(deg + 1.0).unsqueeze(1)
    d2 = torch.sqrt(deg).unsqueeze(1)
    parts = [d1, d2]
    while sum(p.size(1) for p in parts) < 16:
        parts.append(d1)
    x = torch.cat(parts, dim=1)[:, :16].contiguous()
    med = deg.median()
    y = (deg > med).long()
    return x, y


def load_disease_nc_folder(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    rel = "disease_nc"
    folder = data_root / rel
    path = folder / "disease_nc.edges.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    edge_index, num_nodes = _edges_csv_to_graph(path)
    xy = _try_load_xy_optional(folder, num_nodes)

    if xy is not None:
        x, y = xy
        if (folder / "disease_nc.feats.npz").is_file():
            source = "disease_nc_npz"
        else:
            source = "disease_nc_tabular"
    else:
        print("[disease_nc] synthetic x/y")
        x, y = _synthetic_deg_xy(edge_index, num_nodes)
        source = "disease_nc_synthetic_deg"

    idx = np.arange(num_nodes)
    y_np = y.numpy()
    uniq = int(y.max().item()) + 1
    strat = y_np if uniq > 1 and len(np.unique(y_np)) > 1 else None
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, random_state=split_seed, stratify=strat
    )
    strat2 = y_np[idx_temp] if strat is not None else None
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=split_seed, stratify=strat2
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    meta = SimpleNamespace(
        num_features=int(x.size(1)),
        num_classes=int(y.max().item()) + 1,
        source=source,
    )
    return data, meta


def load_edge_only_synthetic(
    data_root: Path,
    rel_path: str,
    split_seed: int,
    source_tag: str,
) -> Tuple[Data, SimpleNamespace]:
    path = data_root / rel_path
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"[{source_tag}] synthetic")
    edge_index, num_nodes = _edges_csv_to_graph(path)
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    d1 = torch.log(deg + 1.0).unsqueeze(1)
    d2 = torch.sqrt(deg).unsqueeze(1)
    parts = [d1, d2]
    while sum(p.size(1) for p in parts) < 16:
        parts.append(d1)
    x = torch.cat(parts, dim=1)[:, :16].contiguous()
    med = deg.median()
    y = (deg > med).long()
    idx = np.arange(num_nodes)
    y_np = y.numpy()
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, stratify=y_np, random_state=split_seed
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_np[idx_temp], random_state=split_seed
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    meta = SimpleNamespace(num_features=16, num_classes=2, source=source_tag + "_synthetic_deg")
    return data, meta


def load_telecom_folder(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    folder = data_root / "telecom"
    path = folder / "edges.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    edge_index, num_nodes = _edges_csv_to_graph(path)

    feats_npz = folder / "feats.npz"
    labels_npy = folder / "labels.npy"
    pt_path = folder / "telecom_graph.pt"

    if feats_npz.is_file() and labels_npy.is_file():
        x = _load_disease_nc_feats_npz(feats_npz)
        yraw = np.load(labels_npy)
        y = torch.tensor(yraw, dtype=torch.long).view(-1)
        if y.size(0) != num_nodes or x.size(0) != num_nodes:
            raise ValueError(
                f"telecom feats.npz/labels.npy size mismatch: N={num_nodes}, x={x.shape}, y={y.shape}"
            )
        source = "telecom_npz"
    elif pt_path.is_file():
        raw = _torch_load_any(pt_path)
        if not isinstance(raw, Data):
            raise TypeError(f"telecom_graph.pt must be torch_geometric.data.Data, got {type(raw)}")
        if raw.x is None or raw.y is None:
            raise ValueError("telecom_graph.pt missing x or y")
        x = raw.x.float()
        yn = raw.y
        if yn.dim() == 2:
            y = yn.argmax(dim=1).long()
        else:
            y = yn.long().view(-1)
        if x.size(0) != num_nodes or y.size(0) != num_nodes:
            raise ValueError(
                f"telecom_graph.pt node count mismatch: N={num_nodes}, x={x.shape}, y={y.shape}"
            )
        source = "telecom_pt"
    else:
        print("[telecom] synthetic")
        x, y = _synthetic_deg_xy(edge_index, num_nodes)
        source = "telecom_synthetic_deg"

    idx = np.arange(num_nodes)
    y_np = y.numpy()
    strat = _stratify_or_none(y_np)
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, random_state=split_seed, stratify=strat
    )
    strat2 = _stratify_or_none(y_np[idx_temp])
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=split_seed, stratify=strat2
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    meta = SimpleNamespace(
        num_features=int(x.size(1)),
        num_classes=int(y.max().item()) + 1,
        source=source,
    )
    return data, meta


def _stratify_or_none(y_np: np.ndarray) -> Optional[np.ndarray]:
    if y_np.size == 0:
        return None
    _, counts = np.unique(y_np, return_counts=True)
    if len(counts) < 2 or bool((counts < 2).any()):
        return None
    return y_np


def load_airport_from_nx_and_table(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    import pickle

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("airport_alldata.p needs pandas: pip install pandas") from e

    p_nx = data_root / "airport" / "airport.p"
    p_df = data_root / "airport" / "airport_alldata.p"
    if not p_nx.is_file() or not p_df.is_file():
        raise FileNotFoundError(f"Need both {p_nx} and {p_df}")

    with open(p_nx, "rb") as f:
        G = pickle.load(f)

    with open(p_df, "rb") as f:
        df = pickle.load(f)

    label_col = 10
    id_col = 0
    meta_df = df[[id_col, label_col]].drop_duplicates(subset=[id_col])
    id_to_lab = dict(zip(meta_df[id_col].tolist(), meta_df[label_col].tolist()))

    nodes_sorted = sorted(G.nodes())
    feats: list[list[float]] = []
    labs: list[str] = []
    for nid in nodes_sorted:
        attrs = G.nodes[nid]
        if "feat" not in attrs:
            raise ValueError(f"node {nid} has no feat")
        feats.append([float(x) for x in attrs["feat"]])
        raw = id_to_lab.get(nid, "\\N")
        if pd.isna(raw):
            raw = "\\N"
        labs.append(str(raw))

    y = torch.tensor(LabelEncoder().fit_transform(labs), dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float32)

    id_to_idx = {nid: i for i, nid in enumerate(nodes_sorted)}
    edges: list[list[int]] = []
    for u, v in G.edges():
        iu, iv = id_to_idx[u], id_to_idx[v]
        edges.append([iu, iv])
        edges.append([iv, iu])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    n = len(nodes_sorted)
    idx = np.arange(n)
    y_np = y.numpy()
    strat = _stratify_or_none(y_np)
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, random_state=split_seed, stratify=strat
    )
    y_temp = y_np[idx_temp]
    strat2 = _stratify_or_none(y_temp)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=split_seed, stratify=strat2
    )
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    meta = SimpleNamespace(
        num_features=int(x.size(1)),
        num_classes=int(y.max().item()) + 1,
        source="airport_nx_alldata",
    )
    return data, meta


def load_airport_routes_synthetic(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    path = data_root / "airport" / "routes.dat"
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: list[tuple[int, int]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                src_id = int(parts[3])
                dst_id = int(parts[5])
            except ValueError:
                continue
            rows.append((src_id, dst_id))

    if not rows:
        raise ValueError("routes.dat: no valid edges")

    nodes_set = set()
    for a, b in rows:
        nodes_set.add(a)
        nodes_set.add(b)
    sorted_ids = sorted(nodes_set)
    id_map = {oid: i for i, oid in enumerate(sorted_ids)}
    edges: list[list[int]] = []
    for a, b in rows:
        ia, ib = id_map[a], id_map[b]
        edges.append([ia, ib])
        edges.append([ib, ia])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    num_nodes = len(sorted_ids)

    print("[airport] synthetic")
    deg = degree(edge_index[0], num_nodes=num_nodes).float()
    d1 = torch.log(deg + 1.0).unsqueeze(1)
    d2 = torch.sqrt(deg).unsqueeze(1)
    parts = [d1, d2]
    while sum(p.size(1) for p in parts) < 16:
        parts.append(d1)
    x = torch.cat(parts, dim=1)[:, :16].contiguous()
    med = deg.median()
    y = (deg > med).long()
    idx = np.arange(num_nodes)
    y_np = y.numpy()
    idx_train, idx_temp = train_test_split(
        idx, train_size=0.6, stratify=y_np, random_state=split_seed
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y_np[idx_temp], random_state=split_seed
    )
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    meta = SimpleNamespace(num_features=16, num_classes=2, source="airport_routes_synthetic")
    return data, meta


def load_airport_openflights(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    p_nx = data_root / "airport" / "airport.p"
    p_df = data_root / "airport" / "airport_alldata.p"
    if p_nx.is_file() and p_df.is_file():
        return load_airport_from_nx_and_table(data_root, split_seed)
    return load_airport_routes_synthetic(data_root, split_seed)


EXP_TABLE2GRAPH_DIRS = {
    "carcinogenesis": "Carcinogenesis_data",
    "hockey": "Hockey_data",
    "hepatitis": "Hepatitis_std_data",
    "pte": "PTE",
    "toxicology": "Toxicology_data",
    "f1": "f1",
}


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _hetero_to_homogeneous(obj: Any) -> Data:
    from torch_geometric.data import HeteroData

    if isinstance(obj, HeteroData):
        homo = obj.to_homogeneous()
        if homo.y is None:
            n = homo.num_nodes
            homo.y = torch.zeros(n, dtype=torch.long)
        return Data(
            x=homo.x,
            edge_index=homo.edge_index,
            y=homo.y,
        )
    raise TypeError(f"cannot convert {type(obj)} to homogeneous Data")


def _ensure_masks(
    data: Data, split_seed: int, train_ratio: float = 0.6
) -> Data:
    if (
        hasattr(data, "train_mask")
        and data.train_mask is not None
        and bool(data.train_mask.any())
    ):
        return data
    n = data.num_nodes
    y = data.y
    if y is None:
        y = torch.zeros(n, dtype=torch.long)
        data.y = y
    idx = np.arange(n)
    y_np = y.numpy()
    uniq = int(y.max().item()) + 1
    strat = y_np if uniq > 1 and len(np.unique(y_np)) > 1 else None
    idx_train, idx_temp = train_test_split(
        idx, train_size=train_ratio, random_state=split_seed, stratify=strat
    )
    strat2 = y_np[idx_temp] if strat is not None else None
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=0.5,
        random_state=split_seed,
        stratify=strat2,
    )
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[torch.from_numpy(idx_train)] = True
    val_mask[torch.from_numpy(idx_val)] = True
    test_mask[torch.from_numpy(idx_test)] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def _remap_y_contiguous_for_exptable2graph(data: Data) -> None:
    if data.y is None:
        return
    y = data.y.view(-1).long()
    m = y >= 0
    if not bool(m.any()):
        return
    raw = y[m].detach().cpu().numpy()
    enc = LabelEncoder()
    mapped = torch.as_tensor(enc.fit_transform(raw), dtype=torch.long)
    new_y = torch.full_like(y, -1)
    new_y[m] = mapped.to(device=y.device)
    data.y = new_y


def _hockey_master_y_from_store(st: Any) -> Optional[torch.Tensor]:
    if getattr(st, "y", None) is not None:
        return st.y.long().view(-1)
    if getattr(st, "pos", None) is None:
        return None
    pos = st.pos
    if pos.dtype in (torch.int64, torch.int32, torch.long):
        return pos.long().view(-1)
    if pos.dim() == 2 and pos.size(1) == 1:
        return pos.long().view(-1)
    if pos.dim() == 1:
        if pos.dtype.is_floating_point:
            arr = pos.detach().cpu().numpy()
            return torch.tensor(LabelEncoder().fit_transform(arr), dtype=torch.long)
        return pos.long().view(-1)
    return None


def _try_load_hockey_master_y_sidecar(folder: Path, n_expected: int) -> Optional[torch.Tensor]:
    for fn in ("master_y.npy", "Master_y.npy", "master_pos.npy", "Master_pos.npy"):
        p = folder / fn
        if not p.is_file():
            continue
        arr = np.load(p)
        t = torch.tensor(np.asarray(arr).reshape(-1), dtype=torch.long)
        if t.numel() != n_expected:
            raise ValueError(f"{p} length {t.numel()} != num Master nodes {n_expected}")
        return t
    return None


def _build_hockey_master_full_graph(
    hetero: Any,
    folder: Path,
    split_seed: int,
) -> Optional[Tuple[Data, SimpleNamespace]]:
    from torch_geometric.data import HeteroData

    if not isinstance(hetero, HeteroData) or "Master" not in hetero.node_types:
        return None

    st = hetero["Master"]
    n_m = st.num_nodes
    y_m = _hockey_master_y_from_store(st)
    if y_m is None:
        y_m = _try_load_hockey_master_y_sidecar(folder, n_m)
    if y_m is None:
        return None

    homo = hetero.to_homogeneous()
    mid = list(hetero.node_types).index("Master")
    mask_m = homo.node_type == mid
    if int(mask_m.sum().item()) != n_m:
        raise RuntimeError("Hockey Master node count mismatch vs homogeneous mask")

    y_full = torch.full((homo.num_nodes,), -1, dtype=torch.long)
    g_idx = torch.where(mask_m)[0]
    y_full[g_idx] = y_m.to(y_full.device)

    idx_all = np.arange(n_m, dtype=np.int64)
    y_np = y_m.numpy()
    strat = _stratify_or_none(y_np)
    idx_train, idx_temp = train_test_split(
        idx_all, train_size=0.6, random_state=split_seed, stratify=strat
    )
    strat2 = _stratify_or_none(y_np[idx_temp])
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=split_seed, stratify=strat2
    )

    n = homo.num_nodes
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[g_idx[idx_train]] = True
    val_mask[g_idx[idx_val]] = True
    test_mask[g_idx[idx_test]] = True

    data_obj = Data(
        x=homo.x,
        edge_index=homo.edge_index,
        y=y_full,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    nc = int(y_m.max().item()) + 1
    meta = SimpleNamespace(
        num_features=int(data_obj.x.size(1)),
        num_classes=nc,
        source="exptable2graph_hockey_master",
    )
    return data_obj, meta


def load_exptable2graph(
    data_root: Path,
    key: str,
    split_seed: int,
    exptable2graph_root: Optional[Path] = None,
) -> Tuple[Data, SimpleNamespace]:
    if key not in EXP_TABLE2GRAPH_DIRS:
        raise ValueError(f"unknown exptable2graph key {key}; expected one of {list(EXP_TABLE2GRAPH_DIRS)}")

    sub = EXP_TABLE2GRAPH_DIRS[key]
    base = exptable2graph_root if exptable2graph_root is not None else data_root / "exptable2graph"
    folder = base / sub
    if not folder.is_dir():
        raise FileNotFoundError(f"missing directory: {folder}")

    unified = folder / "unified_data.pt"
    hetero_files = sorted(folder.glob("*_HeteroGraph.pt"))

    data_obj: Optional[Data] = None

    if key == "hockey" and hetero_files:
        raw_h = _torch_load(hetero_files[0])
        built = _build_hockey_master_full_graph(raw_h, folder, split_seed)
        if built is not None:
            return built

    if unified.is_file():
        raw = _torch_load(unified)
        data_obj = _coerce_to_data(raw)
    elif hetero_files:
        raw = _torch_load(hetero_files[0])
        if isinstance(raw, Data):
            data_obj = raw
        else:
            data_obj = _hetero_to_homogeneous(raw)
    else:
        raise FileNotFoundError(f"{folder}: need unified_data.pt or *_HeteroGraph.pt")

    if key == "hockey":
        print("[hockey] fallback unified_data.pt")

    data_obj = _ensure_masks(data_obj, split_seed)
    if data_obj.x is None:
        n = data_obj.num_nodes
        data_obj.x = torch.randn(n, 16, generator=torch.Generator().manual_seed(split_seed))

    _remap_y_contiguous_for_exptable2graph(data_obj)
    if data_obj.x is not None and not torch.isfinite(data_obj.x).all():
        data_obj.x = torch.nan_to_num(data_obj.x, nan=0.0, posinf=0.0, neginf=0.0)

    if data_obj.y is not None:
        y_pos = data_obj.y[data_obj.y >= 0]
        nc = int(y_pos.max().item()) + 1 if y_pos.numel() > 0 else 2
    else:
        nc = 2
    meta = SimpleNamespace(
        num_features=int(data_obj.x.size(1)),
        num_classes=nc,
        source=f"exptable2graph_{key}",
    )
    return data_obj, meta


def _coerce_to_data(raw: Any) -> Data:
    from torch_geometric.data import HeteroData

    if isinstance(raw, Data):
        return raw
    if isinstance(raw, dict):
        for kx, ke, ky in [
            ("x", "edge_index", "y"),
            ("feat", "edge_index", "label"),
            ("features", "edges", "labels"),
        ]:
            if kx in raw and ke in raw:
                d = Data(
                    x=raw[kx],
                    edge_index=raw[ke],
                    y=raw.get(ky),
                )
                for mk in ("train_mask", "val_mask", "test_mask"):
                    if mk in raw:
                        setattr(d, mk, raw[mk])
                return d
        raise ValueError(f"unified_data.pt dict missing x/edge_index keys: {list(raw.keys())[:20]}")

    if isinstance(raw, HeteroData):
        return _hetero_to_homogeneous(raw)
    raise TypeError(f"unsupported type: {type(raw)}")


def _coerce_adjacency_to_edge_index(adj: Any) -> torch.Tensor:
    if hasattr(adj, "tocoo"):
        coo = adj.tocoo()
        ei = torch.stack(
            [torch.from_numpy(coo.row).long(), torch.from_numpy(coo.col).long()], dim=0
        )
        return ei.contiguous()
    if not isinstance(adj, torch.Tensor):
        adj = torch.as_tensor(adj)
    t = adj
    if t.dim() != 2:
        raise ValueError(f"adj must be 2-D, got shape={tuple(t.shape)}")
    r, c = t.size(0), t.size(1)
    if r == 2 or c == 2:
        if r == 2:
            return t.long().contiguous()
        return t.t().long().contiguous()
    if r != c:
        raise ValueError(f"adj must be square or [2,E], got shape={tuple(t.shape)}")
    if not t.dtype.is_floating_point and not t.dtype == torch.bool:
        t = t.float()
    nz = torch.nonzero(t != 0, as_tuple=False)
    if nz.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long)
    return nz.t().contiguous().long()


def _cs_phds_lp_pack_split(
    edge_index_gnn: torch.Tensor,
    pos: Optional[torch.Tensor],
    neg: Optional[torch.Tensor],
) -> Data:
    if pos is None:
        pos = torch.empty(2, 0, dtype=torch.long)
    else:
        pos = torch.as_tensor(pos, dtype=torch.long)
        if pos.dim() != 2:
            raise ValueError(f"pos edges must be [2,E], got shape={tuple(pos.shape)}")
        if pos.size(0) != 2 and pos.size(1) == 2:
            pos = pos.t().contiguous()
        elif pos.size(0) != 2:
            raise ValueError(f"invalid pos shape: {tuple(pos.shape)}")
    if neg is None:
        neg = torch.empty(2, 0, dtype=torch.long)
    else:
        neg = torch.as_tensor(neg, dtype=torch.long)
        if neg.dim() != 2:
            raise ValueError(f"neg edges must be [2,E], got shape={tuple(neg.shape)}")
        if neg.size(0) != 2 and neg.size(1) == 2:
            neg = neg.t().contiguous()
        elif neg.size(0) != 2:
            raise ValueError(f"invalid neg shape: {tuple(neg.shape)}")
    ei = torch.cat([pos, neg], dim=1)
    el = torch.cat(
        [torch.ones(pos.size(1), dtype=torch.float32), torch.zeros(neg.size(1), dtype=torch.float32)],
        dim=0,
    )
    return Data(edge_index=edge_index_gnn.clone(), edge_label_index=ei, edge_label=el)


def _assert_edges_within_nodes(ei: torch.Tensor, n: int, name: str) -> None:
    if ei.numel() == 0:
        return
    mx = int(ei.max().item())
    if mx >= n:
        raise ValueError(f"{name} has node index {mx} but N={n}")


def _parse_cs_phds_lp_splits(
    splits_obj: Any, edge_index_train: torch.Tensor, num_nodes: int
) -> Tuple[Data, Data, Data]:
    if splits_obj is None:
        raise ValueError("splits.pt is empty")

    def _subdict_pack(d: Dict[str, Any], prefix: str) -> Data:
        if "edge_label_index" in d:
            ei = torch.as_tensor(d["edge_label_index"], dtype=torch.long)
            if ei.dim() != 2 or (ei.size(0) != 2 and ei.size(1) != 2):
                raise ValueError(f"{prefix}.edge_label_index must be [2,E] or [E,2]")
            if ei.size(0) != 2:
                ei = ei.t().contiguous()
            el = d.get("edge_label")
            if el is None:
                raise ValueError(f"{prefix} missing edge_label")
            el = torch.as_tensor(el).float().view(-1)
            if el.numel() != ei.size(1):
                raise ValueError(f"{prefix} edge_label length != edge_label_index")
            out_d = Data(
                edge_index=edge_index_train.clone(),
                edge_label_index=ei.long(),
                edge_label=el,
            )
            _assert_edges_within_nodes(out_d.edge_label_index, num_nodes, prefix)
            return out_d
        pos = d.get(f"{prefix}_pos") or d.get("pos") or d.get("positive")
        neg = d.get(f"{prefix}_neg") or d.get("neg") or d.get("negative")
        out_d = _cs_phds_lp_pack_split(edge_index_train, pos, neg)
        _assert_edges_within_nodes(out_d.edge_label_index, num_nodes, prefix)
        return out_d

    if isinstance(splits_obj, dict):
        keys_lower = {str(k).lower(): k for k in splits_obj.keys()}
        if all(x in keys_lower for x in ("train", "val", "test")):
            out: List[Data] = []
            for split in ("train", "val", "test"):
                k = keys_lower[split]
                v = splits_obj[k]
                if isinstance(v, dict):
                    out.append(_subdict_pack(v, split))
                elif isinstance(v, (list, tuple)) and len(v) == 2:
                    ei, el = v
                    out.append(
                        _subdict_pack(
                            {"edge_label_index": ei, "edge_label": el},
                            split,
                        )
                    )
                else:
                    raise ValueError(f"splits['{split}'] must be dict or pair, got {type(v)}")
                _assert_edges_within_nodes(out[-1].edge_label_index, num_nodes, split)
            return out[0], out[1], out[2]

        def _get_ci(d: Dict[str, Any], *names: str) -> Any:
            lm = {str(k).lower(): k for k in d}
            for nm in names:
                k0 = lm.get(nm.lower())
                if k0 is not None:
                    return d[k0]
            return None

        train_d = _cs_phds_lp_pack_split(
            edge_index_train,
            _get_ci(splits_obj, "train_pos", "train_positive"),
            _get_ci(splits_obj, "train_neg", "train_negative"),
        )
        val_d = _cs_phds_lp_pack_split(
            edge_index_train,
            _get_ci(splits_obj, "val_pos", "valid_pos"),
            _get_ci(splits_obj, "val_neg", "valid_neg"),
        )
        test_d = _cs_phds_lp_pack_split(
            edge_index_train,
            _get_ci(splits_obj, "test_pos"),
            _get_ci(splits_obj, "test_neg"),
        )
        if (
            train_d.edge_label_index.size(1) == 0
            and val_d.edge_label_index.size(1) == 0
            and test_d.edge_label_index.size(1) == 0
        ):
            raise ValueError(
                "splits.pt format not recognized: use train/val/test dicts with edge_label_index+edge_label "
                "or train_pos/train_neg style keys"
            )
        for tag, td in ("train", train_d), ("val", val_d), ("test", test_d):
            _assert_edges_within_nodes(td.edge_label_index, num_nodes, tag)
        return train_d, val_d, test_d

    raise TypeError(f"unsupported splits.pt type: {type(splits_obj)}")


def load_cs_phds_nc(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    folder = data_root / "cs_phds" / "cs_phds_nc_ready"
    p_adj = folder / "adj.pt"
    p_feats = folder / "feats.pt"
    p_labels = folder / "labels.pt"
    for p in (p_adj, p_feats, p_labels):
        if not p.is_file():
            raise FileNotFoundError(f"cs_phds NC missing file: {p}")

    adj = _torch_load_any(p_adj)
    x = _torch_load_any(p_feats)
    y_raw = _torch_load_any(p_labels)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.float()
    y = _as_long_1d(torch.as_tensor(y_raw))
    edge_index = _coerce_adjacency_to_edge_index(adj)
    n = x.size(0)
    if y.numel() != n:
        raise ValueError(f"labels len {y.numel()} != feats rows {n}")
    if edge_index.numel() > 0:
        em = int(edge_index.max().item()) + 1
        if em > n:
            raise ValueError(f"adj max node index {em-1} but feats has N={n}")

    data = Data(x=x, edge_index=edge_index, y=y)
    data = _ensure_masks(data, split_seed)
    y_pos = data.y[data.y >= 0]
    if y_pos.numel() > 0:
        nc = int(y_pos.max().item()) + 1
    else:
        m = int(data.y.max().item())
        nc = max(m + 1, 2) if m >= 0 else 2
    meta = SimpleNamespace(
        num_features=int(x.size(1)),
        num_classes=nc,
        source="cs_phds_nc_ready",
    )
    return data, meta


def load_cs_phds_lp(data_root: Path, split_seed: int) -> Tuple[Data, SimpleNamespace]:
    _ = split_seed
    folder = data_root / "cs_phds" / "cs_phds_lp_ready"
    p_adj = folder / "adj_train.pt"
    p_feats = folder / "feats.pt"
    p_splits = folder / "splits.pt"
    for p in (p_adj, p_feats, p_splits):
        if not p.is_file():
            raise FileNotFoundError(f"cs_phds LP missing file: {p}")

    adj_train = _torch_load_any(p_adj)
    x = _torch_load_any(p_feats)
    splits_obj = _torch_load_any(p_splits)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    else:
        x = x.float()
    n = x.size(0)
    edge_index_train = _coerce_adjacency_to_edge_index(adj_train)
    if edge_index_train.numel() > 0:
        em = int(edge_index_train.max().item()) + 1
        if em > n:
            raise ValueError(f"adj_train node indices inconsistent with feats N={n}")

    train_d, val_d, test_d = _parse_cs_phds_lp_splits(splits_obj, edge_index_train, n)
    y_dummy = torch.zeros(n, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index_train, y=y_dummy)
    meta = SimpleNamespace(
        num_features=int(x.size(1)),
        num_classes=2,
        source="cs_phds_lp_ready",
        lp_precomputed_splits=(train_d, val_d, test_d),
    )
    return data, meta


def load_dataset(
    dataset_name: str,
    data_root: Path,
    split_seed: int = 42,
    exptable2graph_root: Optional[Path] = None,
    task: str = "nc",
) -> Tuple[Data, SimpleNamespace]:
    name = dataset_name.lower().strip()
    if name == "disease":
        name = "disease_nc"

    if name == "cs_phds":
        if (task or "nc").lower() == "lp":
            return load_cs_phds_lp(data_root, split_seed)
        return load_cs_phds_nc(data_root, split_seed)

    if name in PLANETOID_SPECS:
        return load_planetoid_dataset(data_root, name, split_seed)

    if name == "cornell":
        return load_webkb(data_root, "cornell", split_seed)

    if name == "actor":
        return load_webkb(data_root, "Actor", split_seed)

    if name == "airport":
        return load_airport_openflights(data_root, split_seed)

    if name == "disease_nc":
        return load_disease_nc_folder(data_root, split_seed)

    if name == "disease_lp":
        return load_edge_only_synthetic(
            data_root, "disease_lp/disease_lp.edges.csv", split_seed, "disease_lp"
        )

    if name == "telecom":
        return load_telecom_folder(data_root, split_seed)

    if name == "human":
        return load_edge_only_synthetic(
            data_root, "human/PP-Pathways_ppi.csv", split_seed, "human_ppi"
        )

    if name in EXP_TABLE2GRAPH_DIRS:
        return load_exptable2graph(data_root, name, split_seed, exptable2graph_root)

    raise ValueError(
        f"unknown dataset {dataset_name}. Supported: "
        f"{list(PLANETOID_SPECS)} cornell actor airport disease_nc disease_lp telecom human "
        f"cs_phds (NC/LP via --task) "
        f"{list(EXP_TABLE2GRAPH_DIRS)}"
    )
