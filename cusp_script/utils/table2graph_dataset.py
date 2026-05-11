# -*- coding: utf-8 -*-
"""
Load homogeneous graph from data/exptable2graph/<folder>/unified_data.pt for NC/LP.
Hetero backups (*_HeteroGraph.pt) are ignored; only unified_data.pt is used.

If train/val/test masks are missing, train.py applies stratified_nc_split(seed).
"""
import os
from typing import Optional

import torch
from torch_geometric.data import Data

DEFAULT_ROOT = os.path.join("data", "exptable2graph")

# CLI dataset name -> subfolder under exptable2graph
TABLE2GRAPH_FOLDER = {
    "Carcinogenesis": "Carcinogenesis_data",
    "Hockey": "Hockey_data",
    "Hepatitis_std": "Hepatitis_std_data",
    "Toxicology": "Toxicology_data",
    "PTE": "PTE",
    "F1": "f1",
}


def _coerce_data(obj) -> Data:
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        for key in ("data", "graph", "Data"):
            if key in obj and isinstance(obj[key], Data):
                return obj[key]
        raise ValueError(
            "unified_data.pt dict has no Data (try keys data/graph)."
        )
    raise TypeError("unified_data.pt must be Data or dict containing Data.")


def _ensure_xy(data: Data) -> None:
    if data.x is None:
        raise ValueError("Table2Graph Data missing x.")
    if data.y is None:
        raise ValueError("Table2Graph Data missing y.")
    if data.edge_index is None:
        raise ValueError("Table2Graph Data missing edge_index.")
    data.y = data.y.view(-1).long()


def sanitize_table2graph_features(data: Data, name: str) -> int:
    """Replace inf/nan in x with 0; returns count of bad values."""
    x = data.x
    if torch.isfinite(x).all():
        return 0
    bad = int((~torch.isfinite(x)).sum().item())
    data.x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    print(
        "[Table2Graph %s] WARN: %d inf/nan in x replaced with 0 (fix upstream if possible)."
        % (name, bad)
    )
    return bad


class Table2GraphDataset:
    """Planetoid-like API: dataset[0] -> Data, num_classes set."""

    def __init__(self, name: str, root: Optional[str] = None):
        if name not in TABLE2GRAPH_FOLDER:
            raise ValueError(
                "unknown Table2Graph dataset %r; choose one of: %s"
                % (name, ", ".join(sorted(TABLE2GRAPH_FOLDER)))
            )
        root = root or DEFAULT_ROOT
        sub = TABLE2GRAPH_FOLDER[name]
        folder = os.path.join(root, sub)
        unified_path = os.path.join(folder, "unified_data.pt")
        if not os.path.isfile(unified_path):
            raise FileNotFoundError(
                "missing %s (place unified_data.pt there)."
                % unified_path
            )
        raw = torch.load(unified_path, map_location="cpu", weights_only=False)
        data = _coerce_data(raw)
        _ensure_xy(data)
        sanitize_table2graph_features(data, name)
        self.data = data
        self.name = name
        y = data.y.view(-1)
        labeled = y >= 0
        if not bool(labeled.any()):
            raise ValueError(
                "%s: no labeled nodes (need y>=0; y=-1 means unlabeled)" % name
            )
        self.num_classes = int(y[labeled].max().item()) + 1
        if bool((y < 0).any()):
            nu = int((y < 0).sum().item())
            print(
                "[Table2Graph %s] %d nodes have y=-1 (unlabeled); split/loss use y>=0 only."
                % (name, nu)
            )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("Table2GraphDataset has a single graph only.")
        return self.data


def list_table2graph_names():
    return list(TABLE2GRAPH_FOLDER.keys())


def validate_table2graph_data(data: Data, num_classes: int, name: str) -> None:
    """Sanity check unified_data for NC API."""
    if data.x.dim() != 2:
        raise ValueError("%s: expect x (N, F), got %s" % (name, tuple(data.x.shape)))
    n = data.x.shape[0]
    if data.y.shape[0] != n:
        raise ValueError("%s: len(y)=%d != N=%d" % (name, data.y.shape[0], n))
    if not torch.isfinite(data.x).all():
        bad = int((~torch.isfinite(data.x)).sum().item())
        raise ValueError(
            "%s: x still has %d inf/nan (should be sanitized on load)" % (name, bad)
        )
    ymin = int(data.y.min().item())
    if ymin < -1:
        raise ValueError(
            "%s: ymin=%d; only -1 allowed as unlabeled, else 0..C-1" % (name, ymin)
        )
    yl = data.y[data.y >= 0]
    if yl.numel() == 0:
        raise ValueError("%s: no labeled nodes (y>=0)" % name)
    ymin_lab = int(yl.min().item())
    ymax_lab = int(yl.max().item())
    if ymin_lab != 0:
        raise ValueError(
            "%s: labeled classes must start at 0, got min(y|y>=0)=%d" % (name, ymin_lab)
        )
    if ymax_lab >= num_classes:
        raise ValueError(
            "%s: max labeled y %d >= num_classes=%d" % (name, ymax_lab, num_classes)
        )
    ei = data.edge_index
    if ei.numel() == 0:
        raise ValueError("%s: empty edge_index" % name)
    emin, emax = int(ei.min().item()), int(ei.max().item())
    if emin < 0 or emax >= n:
        raise ValueError("%s: edge_index out of range [0,%d): min=%d max=%d" % (name, n, emin, emax))


def table2graph_feature_summary(data: Data) -> str:
    """Short x stats for logging."""
    x = data.x
    return "x in [%.4g, %.4g], mean=%.4g" % (
        float(x.min().item()),
        float(x.max().item()),
        float(x.mean().item()),
    )
