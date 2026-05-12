"""
CS-PhDs layout (under ``<data_root>/cs_phds`` by default):

  cs_phds_nc_ready/  — node classification: adj.pt, dists.pt (opt), feats.pt, labels.pt
  cs_phds_lp_ready/  — link prediction: adj_train.pt, dists_train.pt (opt), feats.pt

  LP needs either labels.pt (then train.py runs train_test_split_edges) or splits.pt
  (fixed val/test pos/neg; optional labels.pt for num_classes only). Dummy y if no labels.

  splits.pt: dict or PyG Data; see train_* / val_* / test_* key aliases in _apply_lp_splits_from_file.
  If train positives are omitted, edges from adj_train.pt become train_pos_edge_index.

  adj: (2,E) long or dense square (nonzero = edge). feats -> x; dists -> data.dists (unused by CUSP forward).
"""
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _coerce_y(data: Data) -> None:
    """Normalize y to per-node class indices (same rules as telecom_dataset)."""
    if data.y is None:
        raise ValueError("missing labels / y")
    n = data.num_nodes
    y = data.y
    if y.dtype in (torch.float32, torch.float64):
        y = y.float()
    else:
        y = y.long()

    if y.dim() == 2:
        if y.shape[0] != n:
            raise ValueError("y 2D first dim must be num_nodes=%d, got %s" % (n, tuple(y.shape)))
        if y.shape[1] == 1:
            data.y = y.squeeze(1).long()
        else:
            data.y = y.argmax(dim=1).long()
        return

    if y.dim() != 1:
        raise ValueError("y must be 1D or 2D, got dim=%d" % y.dim())

    if y.shape[0] == n:
        data.y = y.long()
        return

    if y.numel() % n == 0 and y.numel() != n:
        k = y.numel() // n
        y2 = y.view(n, k)
        if k == 1:
            data.y = y2.squeeze(1).long()
        else:
            data.y = y2.argmax(dim=1).long()
        return

    raise ValueError("y length %d != num_nodes %d" % (y.shape[0], n))


def _adj_to_edge_index(adj: torch.Tensor, num_nodes_hint: Optional[int]) -> torch.Tensor:
    """Map adjacency to coalesced long edge_index (2, E)."""
    if not isinstance(adj, torch.Tensor):
        raise TypeError("adj must be torch.Tensor, got %s" % type(adj))

    if adj.dim() == 2 and adj.size(0) == 2:
        ei = adj.long().contiguous()
        if ei.numel() == 0:
            raise ValueError("empty edge_index")
        n = int(ei.max().item()) + 1 if num_nodes_hint is None else num_nodes_hint
        out = coalesce(ei, num_nodes=n)
        return out[0] if isinstance(out, tuple) else out

    if adj.dim() == 2 and adj.size(0) == adj.size(1):
        n = adj.size(0)
        if adj.is_sparse:
            adj = adj.to_dense()
        mask = adj if adj.dtype == torch.bool else (adj != 0)
        if mask.dtype != torch.bool:
            mask = mask.bool()
        ei = torch.nonzero(mask, as_tuple=False).t().long()
        if ei.numel() == 0:
            raise ValueError("dense adj has no edges")
        ei = coalesce(ei, num_nodes=n)
        return ei[0] if isinstance(ei, tuple) else ei

    raise ValueError("unsupported adj shape: %s" % (tuple(adj.shape),))


def _as_long_edge_index(x, name: str) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.dim() != 2 or t.size(0) != 2:
        raise ValueError("%s must be (2, E), got %s" % (name, tuple(t.shape)))
    return t.long().contiguous()


def _splits_to_dict(sp: Any) -> Dict[str, Any]:
    if isinstance(sp, dict):
        return sp
    if isinstance(sp, Data):
        if hasattr(sp, "to_dict"):
            return dict(sp.to_dict())
        ks = sp.keys
        if callable(ks):
            ks = ks()
        return {k: sp[k] for k in ks}
    raise TypeError("splits.pt must be dict or torch_geometric.data.Data, got %s" % type(sp))


def _pick_tensor(d: Dict[str, Any], keys: Tuple[str, ...]) -> Tuple[Optional[Any], Optional[str]]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k], k
    return None, None


def _coalesce_undirected(ei: torch.Tensor, num_nodes: int) -> torch.Tensor:
    ei = to_undirected(ei.long().contiguous())
    out = coalesce(ei, num_nodes=num_nodes)
    return out[0] if isinstance(out, tuple) else out


def _coalesce_directed(ei: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Negatives: coalesce only (keep directed semantics)."""
    ei = ei.long().contiguous()
    out = coalesce(ei, num_nodes=num_nodes)
    return out[0] if isinstance(out, tuple) else out


def _apply_lp_splits_from_file(
    data: Data,
    sp: Any,
    train_ei_default: torch.Tensor,
    num_nodes: int,
    verbose: bool,
) -> None:
    d = _splits_to_dict(sp)

    train_keys = ("train_pos_edge_index", "train_pos", "pos_train", "train_edges")
    val_keys = ("val_pos_edge_index", "val_pos", "val_edges")
    test_keys = ("test_pos_edge_index", "test_pos", "test_edges")
    val_neg_keys = ("val_neg_edge_index", "val_neg", "val_neg_edges")
    test_neg_keys = ("test_neg_edge_index", "test_neg", "test_neg_edges")

    raw_train, tk = _pick_tensor(d, train_keys)
    if raw_train is None:
        train_pos = train_ei_default
        tk = "<adj_train fallback>"
    else:
        train_pos = _as_long_edge_index(raw_train, tk or "train")

    raw_vp, vk = _pick_tensor(d, val_keys)
    raw_te_pos, tek = _pick_tensor(d, test_keys)
    raw_vn, vnk = _pick_tensor(d, val_neg_keys)
    raw_tn, tnk = _pick_tensor(d, test_neg_keys)

    if raw_vp is None or raw_te_pos is None or raw_vn is None or raw_tn is None:
        raise ValueError(
            "splits.pt must define val/test pos and neg edges (see module doc). "
            "Matched keys: train=%s val_pos=%s test_pos=%s val_neg=%s test_neg=%s"
            % (tk, vk, tek, vnk, tnk)
        )

    val_pos = _as_long_edge_index(raw_vp, vk or "val_pos")
    test_pos = _as_long_edge_index(raw_te_pos, tek or "test_pos")
    val_neg = _as_long_edge_index(raw_vn, vnk or "val_neg")
    test_neg = _as_long_edge_index(raw_tn, tnk or "test_neg")

    data.train_pos_edge_index = _coalesce_undirected(train_pos, num_nodes)
    data.val_pos_edge_index = _coalesce_undirected(val_pos, num_nodes)
    data.test_pos_edge_index = _coalesce_undirected(test_pos, num_nodes)
    data.val_neg_edge_index = _coalesce_directed(val_neg, num_nodes)
    data.test_neg_edge_index = _coalesce_directed(test_neg, num_nodes)

    data.lp_presplit = True
    data.lp_no_node_labels = True
    data.y = torch.zeros(num_nodes, dtype=torch.long)

    if verbose:
        print(
            "[cs_phds] LP splits from splits.pt | train_pos=%s | val_pos=%s | test_pos=%s | val_neg=%s | test_neg=%s"
            % (tk, vk, tek, vnk, tnk)
        )


def _load_cs_phds_nc_folder(folder: str, adj_name: str, dists_name: str, verbose: bool) -> Data:
    p_adj = os.path.join(folder, adj_name)
    p_dists = os.path.join(folder, dists_name)
    p_feats = os.path.join(folder, "feats.pt")
    p_labels = os.path.join(folder, "labels.pt")

    for p in (p_adj, p_feats, p_labels):
        if not os.path.isfile(p):
            raise FileNotFoundError("missing file: %s" % p)

    feats = _torch_load(p_feats)
    labels = _torch_load(p_labels)
    adj = _torch_load(p_adj)

    if not isinstance(feats, torch.Tensor):
        feats = torch.as_tensor(feats)
    x = torch.nan_to_num(feats.float())

    if not isinstance(labels, torch.Tensor):
        labels = torch.as_tensor(labels)

    n = x.size(0)
    edge_index = _adj_to_edge_index(adj, num_nodes_hint=n)

    data = Data(x=x, edge_index=edge_index, y=None)
    data.y = labels
    _coerce_y(data)

    if os.path.isfile(p_dists):
        dists = _torch_load(p_dists)
        if isinstance(dists, torch.Tensor):
            data.dists = torch.nan_to_num(dists.float())
        else:
            data.dists = torch.as_tensor(dists).float()

    if verbose:
        undir_e = edge_index.size(1) // 2
        print(
            "[cs_phds] dir=%s | N=%d | edges(undir)≈%d | C=%d | x=%s | dists=%s"
            % (
                folder,
                data.num_nodes,
                undir_e,
                int(data.y.max().item()) + 1,
                tuple(x.shape),
                "yes" if hasattr(data, "dists") else "no",
            )
        )
    return data


def _load_cs_phds_lp_folder(folder: str, verbose: bool) -> Data:
    p_adj = os.path.join(folder, "adj_train.pt")
    p_feats = os.path.join(folder, "feats.pt")
    p_labels = os.path.join(folder, "labels.pt")
    p_splits = os.path.join(folder, "splits.pt")
    p_dists = os.path.join(folder, "dists_train.pt")

    if not os.path.isfile(p_adj) or not os.path.isfile(p_feats):
        raise FileNotFoundError("LP needs at least: %s and %s" % (p_adj, p_feats))

    feats = _torch_load(p_feats)
    adj = _torch_load(p_adj)
    if not isinstance(feats, torch.Tensor):
        feats = torch.as_tensor(feats)
    x = torch.nan_to_num(feats.float())
    n = x.size(0)
    edge_index = _adj_to_edge_index(adj, num_nodes_hint=n)

    data = Data(x=x, edge_index=edge_index, y=None)

    if os.path.isfile(p_dists):
        dists = _torch_load(p_dists)
        if isinstance(dists, torch.Tensor):
            data.dists = torch.nan_to_num(dists.float())
        else:
            data.dists = torch.as_tensor(dists).float()

    has_labels = os.path.isfile(p_labels)
    has_splits = os.path.isfile(p_splits)

    if has_splits:
        sp = _torch_load(p_splits)
        _apply_lp_splits_from_file(data, sp, edge_index, n, verbose)
        if has_labels:
            ly = _torch_load(p_labels)
            if not isinstance(ly, torch.Tensor):
                ly = torch.as_tensor(ly)
            data.y = ly
            _coerce_y(data)
            data.lp_no_node_labels = False
            if verbose:
                print("[cs_phds] LP: labels.pt + splits.pt — edges from splits, y for num_classes")
    elif has_labels:
        ly = _torch_load(p_labels)
        if not isinstance(ly, torch.Tensor):
            ly = torch.as_tensor(ly)
        data.y = ly
        _coerce_y(data)
        data.lp_presplit = False
        data.lp_no_node_labels = False
    else:
        raise FileNotFoundError(
            "LP folder needs labels.pt (random split) or splits.pt (fixed val/test), got: %s" % folder
        )

    if verbose and not getattr(data, "lp_presplit", False):
        undir_e = edge_index.size(1) // 2
        print(
            "[cs_phds] dir=%s | N=%d | edges(undir)≈%d | C=%d | x=%s | LP=train_test_split_edges | dists=%s"
            % (
                folder,
                data.num_nodes,
                undir_e,
                int(data.y.max().item()) + 1,
                tuple(x.shape),
                "yes" if hasattr(data, "dists") else "no",
            )
        )
    elif verbose and getattr(data, "lp_presplit", False) and not has_labels:
        undir_e = edge_index.size(1) // 2
        print(
            "[cs_phds] dir=%s | N=%d | train_graph edges(undir)≈%d | x=%s | y=dummy | dists=%s"
            % (
                folder,
                data.num_nodes,
                undir_e,
                tuple(x.shape),
                "yes" if hasattr(data, "dists") else "no",
            )
        )
    return data


class CsPhdsDataset:
    """root: parent of cs_phds_nc_ready / cs_phds_lp_ready (default <data_root>/cs_phds). task: NC or LP."""

    def __init__(self, root, task, nc_subdir="cs_phds_nc_ready", lp_subdir="cs_phds_lp_ready", verbose=True):
        self.root = root
        self.task = task
        if task == "link_prediction":
            sub = os.path.join(root, lp_subdir)
            self._data = _load_cs_phds_lp_folder(sub, verbose=verbose)
        elif task == "node_classification":
            sub = os.path.join(root, nc_subdir)
            self._data = _load_cs_phds_nc_folder(sub, "adj.pt", "dists.pt", verbose=verbose)
        else:
            raise ValueError("CsPhdsDataset: unsupported task %s" % task)

        y = self._data.y
        if getattr(self._data, "lp_no_node_labels", False) and getattr(self._data, "lp_presplit", False):
            self.num_classes = 2
        else:
            valid = y >= 0
            if valid.any():
                self.num_classes = int(y[valid].max().item()) + 1
            else:
                self.num_classes = max(int(y.max().item()) + 1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._data
