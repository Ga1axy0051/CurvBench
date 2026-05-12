import argparse
import copy
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.transforms import RandomLinkSplit

from datasets_loader import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"

data: Optional[Data] = None
dataset_meta: Optional[SimpleNamespace] = None
device: torch.device = torch.device("cpu")


def resolve_device(preference: str) -> torch.device:
    pref = (preference or "auto").strip().lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optimizer_hp_from_config(section: Dict[str, Any]) -> Dict[str, float]:
    return {
        "lr": float(section.get("lr", 0.01)),
        "weight_decay_graph": float(section.get("weight_decay", 5e-4)),
        "weight_decay_mlp": float(section.get("weight_decay_mlp", 0.0)),
    }


def _adam_for_baseline(model_name: str, params, hp: Dict[str, float]) -> torch.optim.Adam:
    wd = hp["weight_decay_mlp"] if model_name == "mlp" else hp["weight_decay_graph"]
    return torch.optim.Adam(params, lr=hp["lr"], weight_decay=wd)


def _nc_balanced_ce_class_weight(data: Data, num_classes: int, dev: torch.device) -> torch.Tensor:
    y = data.y.view(-1).long()
    m = data.train_mask & (y >= 0)
    y_np = y[m].detach().cpu().numpy()
    if y_np.size == 0:
        return torch.ones(num_classes, dtype=torch.float32, device=dev)
    classes_present = np.unique(y_np)
    cw = compute_class_weight(class_weight="balanced", classes=classes_present, y=y_np)
    w = torch.zeros(num_classes, dtype=torch.float32)
    for c, wi in zip(classes_present, cw):
        w[int(c)] = float(wi)
    return w.to(dev)


def _gat_nc_kwargs_from_config(gat_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    g = gat_cfg or {}
    return {
        "hidden_dim_per_head": int(g.get("hidden_dim_per_head", 8)),
        "heads1": int(g.get("num_heads_layer1", 8)),
        "heads2": int(g.get("num_heads_layer2", 1)),
        "dropout": float(g.get("dropout", 0.6)),
    }


def load_json_config(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_exptable2graph_root(ex_root_str: str) -> Optional[Path]:
    s = (ex_root_str or "").strip()
    if not s:
        return None
    if os.name != "nt" and len(s) >= 3 and s[1] == ":" and s[0].isalpha() and s[2] in "/\\":
        print("[exptable2graph] ignored Windows drive path on this OS")
        return None
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def stats_with_std(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    out: Dict[str, Any] = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "runs": [float(x) for x in values],
    }
    return out


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim_per_head: int,
        out_dim: int,
        *,
        heads1: int = 8,
        heads2: int = 1,
        dropout: float = 0.6,
    ):
        super().__init__()
        if heads2 != 1:
            raise ValueError("NC GAT needs heads2=1")
        hid_concat = hidden_dim_per_head * heads1
        self.conv1 = GATConv(in_dim, hidden_dim_per_head, heads=heads1, dropout=dropout)
        self.conv2 = GATConv(hid_concat, out_dim, heads=heads2, concat=False, dropout=dropout)
        self.ff_dropout = float(min(dropout, 0.5))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.ff_dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


def train_step_nc(
    model,
    data,
    optimizer,
    ce_class_weight: Optional[torch.Tensor] = None,
):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    y = data.y.view(-1).long()
    m = data.train_mask & (y >= 0)
    if not bool(m.any()):
        return 0.0
    if not torch.isfinite(out[m]).all():
        optimizer.zero_grad()
        return float("nan")
    loss = F.cross_entropy(out[m], y[m], weight=ce_class_weight)
    if not torch.isfinite(loss):
        optimizer.zero_grad()
        return float("nan")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def _split_metrics_nc(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    m = mask & (y >= 0)
    if not bool(m.any()):
        return {"acc": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
    y_true = y[m].detach().cpu().numpy()
    y_hat = pred[m].detach().cpu().numpy()
    return {
        "acc": float(accuracy_score(y_true, y_hat)),
        "f1_macro": float(f1_score(y_true, y_hat, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_hat, average="micro", zero_division=0)),
    }


@torch.no_grad()
def evaluate_nc_detailed(model, data) -> Dict[str, Dict[str, float]]:
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    return {
        "train": _split_metrics_nc(pred, data.y, data.train_mask),
        "val": _split_metrics_nc(pred, data.y, data.val_mask),
        "test": _split_metrics_nc(pred, data.y, data.test_mask),
    }


def run_nc(
    model_name: str,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    opt_hp: Dict[str, float],
    nc_gat_spec: Optional[Dict[str, Any]] = None,
    nc_ce_class_weight: str = "none",
) -> Dict[str, Any]:
    assert data is not None and dataset_meta is not None
    set_seed(seed)

    ce_w: Optional[torch.Tensor] = None
    if (nc_ce_class_weight or "none").strip().lower() == "balanced":
        ce_w = _nc_balanced_ce_class_weight(data, dataset_meta.num_classes, device)

    if model_name == "gcn":
        model = GCN(dataset_meta.num_features, 64, dataset_meta.num_classes)
    elif model_name == "gat":
        gkw = _gat_nc_kwargs_from_config(nc_gat_spec)
        model = GAT(
            dataset_meta.num_features,
            gkw["hidden_dim_per_head"],
            dataset_meta.num_classes,
            heads1=gkw["heads1"],
            heads2=gkw["heads2"],
            dropout=gkw["dropout"],
        )
    elif model_name == "mlp":
        model = MLP(dataset_meta.num_features, 64, dataset_meta.num_classes)
    else:
        raise ValueError(model_name)

    model = model.to(device)
    optimizer = _adam_for_baseline(model_name, model.parameters(), opt_hp)

    best_val = -1.0
    best_state: Optional[dict] = None
    best_test_snapshot: Optional[Dict[str, float]] = None
    bad = 0

    val_history: List[float] = []
    train_loss_history: List[float] = []

    t_train0 = time.perf_counter()
    epochs_ran = 0

    for ep in range(max_epochs):
        tl = train_step_nc(model, data, optimizer, ce_class_weight=ce_w)
        train_loss_history.append(tl)
        metrics = evaluate_nc_detailed(model, data)
        val_acc = metrics["val"]["acc"]
        val_history.append(val_acc)
        epochs_ran = ep + 1

        if val_acc > best_val + min_delta:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_test_snapshot = metrics["test"].copy()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    total_train_sec = time.perf_counter() - t_train0
    sec_per_epoch = total_train_sec / max(epochs_ran, 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    t_test0 = time.perf_counter()
    final_metrics = evaluate_nc_detailed(model, data)
    total_test_sec = time.perf_counter() - t_test0

    test_use = best_test_snapshot if best_test_snapshot is not None else final_metrics["test"]

    return {
        "test_acc": test_use["acc"],
        "test_f1_macro": test_use["f1_macro"],
        "test_f1_micro": test_use["f1_micro"],
        "total_train_sec": total_train_sec,
        "sec_per_epoch": sec_per_epoch,
        "total_test_sec": total_test_sec,
        "epochs": epochs_ran,
        "curves": {
            "val_acc": val_history,
            "train_loss": train_loss_history,
        },
    }


_LP_DECODE_L2NORM = True


def configure_lp_decoder(*, l2_normalize_before_dot: bool) -> None:
    global _LP_DECODE_L2NORM
    _LP_DECODE_L2NORM = l2_normalize_before_dot


def decode_link_logits(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    zd = F.normalize(z, p=2, dim=-1, eps=1e-8) if _LP_DECODE_L2NORM else z
    return (zd[edge_label_index[0]] * zd[edge_label_index[1]]).sum(dim=-1)


class GCNLinkEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, emb_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATLinkEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_per_head: int,
        emb_dim: int,
        *,
        heads1: int = 8,
        heads2: int = 1,
        dropout: float = 0.6,
    ):
        super().__init__()
        if heads2 != 1:
            raise ValueError("GATLinkEncoder needs heads2=1")
        hid_concat = hidden_per_head * heads1
        self.conv1 = GATConv(in_dim, hidden_per_head, heads=heads1, dropout=dropout)
        self.conv2 = GATConv(hid_concat, emb_dim, heads=heads2, concat=False, dropout=dropout)
        self.ff_dropout = float(min(dropout, 0.5))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.ff_dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class MLPLinkEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x, edge_index=None):
        return self.net(x)


@torch.no_grad()
def link_classification_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    *,
    logit_threshold: Optional[float] = None,
) -> Dict[str, float]:
    if y.numel() == 0:
        return {"auc": 0.0, "ap": 0.0, "acc": 0.0, "f1_macro": 0.0, "f1_micro": 0.0}
    prob = logits.sigmoid().cpu().numpy().reshape(-1)
    lg = logits.detach().float().cpu().numpy().reshape(-1)
    yy = y.float().cpu().numpy().astype(int).reshape(-1)
    if logit_threshold is None:
        pred = (prob >= 0.5).astype(int)
    else:
        t = float(logit_threshold)
        pred = (lg >= t).astype(int)
    auc = 0.5
    ap = 0.0
    if len(np.unique(yy)) >= 2:
        auc = float(roc_auc_score(yy, prob))
        ap = float(average_precision_score(yy, prob))
    return {
        "auc": auc,
        "ap": ap,
        "acc": float(accuracy_score(yy, pred)),
        "f1_macro": float(f1_score(yy, pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(yy, pred, average="micro", zero_division=0)),
    }


def _lp_best_logit_threshold_val_macro_f1(val_logits: torch.Tensor, val_y: torch.Tensor) -> float:
    lg = val_logits.detach().float().cpu().numpy().reshape(-1)
    yy = val_y.detach().float().cpu().numpy().astype(int).reshape(-1)
    if yy.size == 0 or len(np.unique(yy)) < 2:
        return 0.0
    xs = np.sort(np.unique(lg))
    if xs.size == 0:
        return 0.0
    cands: List[float] = [float(xs[0] - 1e-7)]
    for i in range(int(xs.size) - 1):
        cands.append(float(0.5 * (xs[i] + xs[i + 1])))
    cands.append(float(xs[-1] + 1e-7))
    best_f1 = -1.0
    best_t = 0.0
    for t in cands:
        pred = (lg >= t).astype(int)
        f1m = float(f1_score(yy, pred, average="macro", zero_division=0))
        if f1m > best_f1 + 1e-15:
            best_f1 = f1m
            best_t = t
        elif abs(f1m - best_f1) <= 1e-15 and t < best_t:
            best_t = t
    return float(best_t)


def train_step_lp(model, feat, train_data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model(feat, train_data.edge_index)
    logits = decode_link_logits(z, train_data.edge_label_index)
    loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label.float())
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate_lp_detailed(model, feat, edge_index_gnn, edge_label_index, edge_label):
    model.eval()
    z = model(feat, edge_index_gnn)
    logits = decode_link_logits(z, edge_label_index)
    return link_classification_metrics(logits, edge_label)


@torch.no_grad()
def print_lp_diagnostics(
    model: nn.Module,
    feat: torch.Tensor,
    tag: str,
    edge_index_gnn: torch.Tensor,
    edge_label_index: torch.Tensor,
    edge_label: torch.Tensor,
) -> None:
    model.eval()
    z = model(feat, edge_index_gnn)
    logits = decode_link_logits(z, edge_label_index)
    prob = logits.sigmoid().cpu().numpy().ravel()
    y = edge_label.float().cpu().numpy().astype(int).ravel()
    pos = prob[y == 1]
    neg = prob[y == 0]
    lg = logits.detach().cpu().numpy().ravel()
    dec = "cosine(L2norm)" if _LP_DECODE_L2NORM else "raw dot"
    print(f"  [lp {tag}] n={len(y)} dec={dec}")
    if len(pos) > 0:
        print(
            f"    prob | y=1: mean={pos.mean():.4f} std={pos.std():.4f} "
            f"min={pos.min():.4f} max={pos.max():.4f}"
        )
    if len(neg) > 0:
        print(
            f"    prob | y=0: mean={neg.mean():.4f} std={neg.std():.4f} "
            f"min={neg.min():.4f} max={neg.max():.4f}"
        )
    if len(pos) > 0 and len(neg) > 0:
        print(f"    pos-neg={pos.mean() - neg.mean():.4f}")
    print(f"    logits: min={lg.min():.4f} max={lg.max():.4f} mean={lg.mean():.4f}")
    auc_p = auc_n = 0.5
    if len(np.unique(y)) >= 2:
        auc_p = float(roc_auc_score(y, prob))
        auc_n = float(roc_auc_score(y, -prob))
    ap_s = 0.0
    if len(np.unique(y)) >= 2:
        ap_s = float(average_precision_score(y, prob))
    print(f"    AUC(y, prob)={auc_p:.4f}  AP={ap_s:.4f}  AUC(y, -prob)={auc_n:.4f}")
    acc05 = float(((prob >= 0.5).astype(int) == y).mean())
    print(f"    acc@0.5={acc05:.4f}")


def run_lp(
    model_name: str,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    opt_hp: Dict[str, float],
    lp_diagnostics: bool = False,
    lp_flip_edge_labels: bool = False,
    lp_cfg_section: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    assert data is not None and dataset_meta is not None
    set_seed(seed)

    if getattr(dataset_meta, "lp_precomputed_splits", None) is not None:
        train_d, val_d, test_d = dataset_meta.lp_precomputed_splits
        train_d = train_d.to(device)
        val_d = val_d.to(device)
        test_d = test_d.to(device)
    else:
        try:
            split = RandomLinkSplit(
                num_val=0.05,
                num_test=0.1,
                is_undirected=True,
                add_negative_train_samples=True,
                neg_sampling_ratio=1.0,
            )
        except TypeError:
            split = RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True)
        train_d, val_d, test_d = split(data.clone())

    if lp_flip_edge_labels:
        for d in (train_d, val_d, test_d):
            d.edge_label = 1.0 - d.edge_label.float()

    in_dim = dataset_meta.num_features
    emb_dim = 64

    if model_name == "gcn":
        model = GCNLinkEncoder(in_dim, 64, emb_dim)
    elif model_name == "gat":
        lp_c = lp_cfg_section or {}
        gh = int(lp_c.get("gat_hidden_dim_per_head", 8))
        gheads = int(lp_c.get("gat_heads", 8))
        gdo = float(lp_c.get("dropout_gat", 0.6))
        model = GATLinkEncoder(in_dim, gh, emb_dim, heads1=gheads, heads2=1, dropout=gdo)
    elif model_name == "mlp":
        model = MLPLinkEncoder(in_dim, 64, emb_dim)
    else:
        raise ValueError(model_name)

    model = model.to(device)
    optimizer = _adam_for_baseline(model_name, model.parameters(), opt_hp)
    feat = data.x

    best_val_auc = -1.0
    best_test_snapshot: Optional[Dict[str, float]] = None
    best_state: Optional[dict] = None
    bad = 0

    val_history: List[float] = []
    train_loss_history: List[float] = []
    test_m: Dict[str, float] = {
        "auc": 0.5,
        "ap": 0.0,
        "acc": 0.0,
        "f1_macro": 0.0,
        "f1_micro": 0.0,
    }

    t_train0 = time.perf_counter()
    epochs_ran = 0

    for _ in range(max_epochs):
        tl = train_step_lp(model, feat, train_d, optimizer)
        train_loss_history.append(tl)
        val_m = evaluate_lp_detailed(
            model, feat, val_d.edge_index, val_d.edge_label_index, val_d.edge_label
        )
        test_m = evaluate_lp_detailed(
            model, feat, test_d.edge_index, test_d.edge_label_index, test_d.edge_label
        )
        val_auc = val_m["auc"]
        val_history.append(val_auc)
        epochs_ran += 1

        if val_auc > best_val_auc + min_delta:
            best_val_auc = val_auc
            best_test_snapshot = test_m.copy()
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    total_train_sec = time.perf_counter() - t_train0
    sec_per_epoch = total_train_sec / max(epochs_ran, 1)

    if best_state is not None:
        model.load_state_dict(best_state)

    t_test0 = time.perf_counter()
    _ = evaluate_lp_detailed(
        model, feat, test_d.edge_index, test_d.edge_label_index, test_d.edge_label
    )
    total_test_sec = time.perf_counter() - t_test0

    test_use = dict(best_test_snapshot if best_test_snapshot is not None else test_m)
    model.eval()
    with torch.no_grad():
        z_val = model(feat, val_d.edge_index)
        val_logits = decode_link_logits(z_val, val_d.edge_label_index)
        t_star = _lp_best_logit_threshold_val_macro_f1(val_logits, val_d.edge_label)
        z_te = model(feat, test_d.edge_index)
        test_logits = decode_link_logits(z_te, test_d.edge_label_index)
    tuned = link_classification_metrics(
        test_logits, test_d.edge_label, logit_threshold=t_star
    )
    test_use["acc"] = tuned["acc"]
    test_use["f1_macro"] = tuned["f1_macro"]
    test_use["f1_micro"] = tuned["f1_micro"]

    if lp_diagnostics and seed == 0:
        print(f"\n[lp diag] {model_name} seed={seed}")
        print_lp_diagnostics(model, feat, "val", val_d.edge_index, val_d.edge_label_index, val_d.edge_label)
        print_lp_diagnostics(model, feat, "test", test_d.edge_index, test_d.edge_label_index, test_d.edge_label)

    return {
        "test_acc": test_use["acc"],
        "test_f1_macro": test_use["f1_macro"],
        "test_f1_micro": test_use["f1_micro"],
        "test_auc": test_use["auc"],
        "test_ap": test_use["ap"],
        "lp_val_logit_threshold": t_star,
        "lp_acc_f1_protocol": "val_macro_f1_argmax_then_test",
        "total_train_sec": total_train_sec,
        "sec_per_epoch": sec_per_epoch,
        "total_test_sec": total_test_sec,
        "epochs": epochs_ran,
        "curves": {
            "val_auc": val_history,
            "train_loss": train_loss_history,
        },
    }


def experiment(
    model_name: str,
    seeds: List[int],
    task: str,
    max_epochs: int,
    patience: int,
    min_delta: float,
    opt_hp: Dict[str, float],
    lp_diagnostics: bool = False,
    lp_flip_edge_labels: bool = False,
    nc_gat_spec: Optional[Dict[str, Any]] = None,
    lp_cfg_section: Optional[Dict[str, Any]] = None,
    nc_ce_class_weight: str = "none",
) -> Dict[str, Any]:
    test_accs: List[float] = []
    f1_macros: List[float] = []
    f1_micros: List[float] = []
    test_aucs: List[float] = []
    test_aps: List[float] = []
    lp_val_logit_thresholds: List[float] = []
    train_times: List[float] = []
    epoch_times: List[float] = []
    test_times: List[float] = []
    epochs_list: List[float] = []
    curves_by_seed: Dict[str, Any] = {}

    for seed in seeds:
        if task == "nc":
            r = run_nc(
                model_name,
                seed,
                max_epochs,
                patience,
                min_delta,
                opt_hp,
                nc_gat_spec=nc_gat_spec,
                nc_ce_class_weight=nc_ce_class_weight,
            )
            test_accs.append(r["test_acc"])
            f1_macros.append(r["test_f1_macro"])
            f1_micros.append(r["test_f1_micro"])
        else:
            r = run_lp(
                model_name,
                seed,
                max_epochs,
                patience,
                min_delta,
                opt_hp,
                lp_diagnostics=lp_diagnostics,
                lp_flip_edge_labels=lp_flip_edge_labels,
                lp_cfg_section=lp_cfg_section,
            )
            test_accs.append(r["test_acc"])
            f1_macros.append(r["test_f1_macro"])
            f1_micros.append(r["test_f1_micro"])
            test_aucs.append(r["test_auc"])
            test_aps.append(r["test_ap"])
            lp_val_logit_thresholds.append(float(r.get("lp_val_logit_threshold", 0.0)))

        train_times.append(r["total_train_sec"])
        epoch_times.append(r["sec_per_epoch"])
        test_times.append(r["total_test_sec"])
        epochs_list.append(float(r["epochs"]))
        curves_by_seed[f"seed_{seed}"] = r["curves"]

    out: Dict[str, Any] = {
        "test_acc": stats_with_std(test_accs),
        "test_f1_macro": stats_with_std(f1_macros),
        "test_f1_micro": stats_with_std(f1_micros),
    }
    if task == "lp":
        out["test_auc"] = stats_with_std(test_aucs)
        out["test_ap"] = stats_with_std(test_aps)
        out["lp_val_logit_threshold"] = stats_with_std(lp_val_logit_thresholds)
        out["lp_acc_f1_protocol"] = "val_macro_f1_argmax_then_test"
    out["total_train_sec"] = stats_with_std(train_times)
    out["sec_per_epoch"] = stats_with_std(epoch_times)
    out["total_test_sec"] = stats_with_std(test_times)
    out["epochs"] = stats_with_std(epochs_list)
    out["convergence_curves"] = curves_by_seed
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--task", type=str, choices=["nc", "lp"], default="nc")
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--runs", type=int, default=0)
    p.add_argument("--data-root", type=str, default="")
    p.add_argument("--exptable2graph-root", type=str, default="")
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--min-delta", type=float, default=-1.0)
    p.add_argument("--split-seed", type=int, default=-1)
    p.add_argument("--nc-lr", type=float, default=None)
    p.add_argument("--nc-wd", type=float, default=None)
    p.add_argument("--nc-wd-mlp", type=float, default=None)
    p.add_argument("--nc-ce-class-weight", type=str, default=None, choices=["none", "balanced"])
    p.add_argument("--nc-gat-dropout", type=float, default=None)
    p.add_argument("--nc-gat-heads", type=int, default=None)
    p.add_argument("--nc-gat-hidden-per-head", type=int, default=None)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--lp-diagnostics", action="store_true")
    p.add_argument("--lp-flip-edge-labels", action="store_true")
    p.add_argument("--lp-raw-dot-product", action="store_true")
    p.add_argument("--device", type=str, default="auto")
    return p


def _parse_seeds(s: str, runs: int, fallback: List[int]) -> List[int]:
    s = (s or "").strip()
    if s:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if runs > 0:
        return list(range(runs))
    return list(fallback)


if __name__ == "__main__":
    args = _build_parser().parse_args()
    device = resolve_device(args.device)
    configure_lp_decoder(l2_normalize_before_dot=not args.lp_raw_dot_product)
    cfg = load_json_config(Path(args.config))
    ds_name = args.dataset.lower().strip()

    data_root = resolve_path(args.data_root or cfg.get("data_root") or "data")
    ex_cli = (args.exptable2graph_root or "").strip()
    ex_cfg = (cfg.get("exptable2graph_root") or "").strip()
    ex_root = resolve_exptable2graph_root(ex_cli) if ex_cli else resolve_exptable2graph_root(ex_cfg)

    nc_es = (cfg.get("nc") or {}).get("early_stopping") or {}
    lp_es = (cfg.get("lp") or {}).get("early_stopping") or {}
    es = nc_es if args.task == "nc" else lp_es

    max_epochs = args.max_epochs or int(es.get("max_epochs", 1000))
    patience = args.patience or int(es.get("patience", 100))
    min_delta = args.min_delta if args.min_delta >= 0 else float(es.get("min_delta", 1e-4))

    split_seed = args.split_seed if args.split_seed >= 0 else int(cfg.get("split_seed", 42))
    seeds_cfg = cfg.get("seeds") or [0, 1, 2, 3, 4]
    seeds = _parse_seeds(args.seeds, args.runs, [int(x) for x in seeds_cfg])

    models_order: List[str] = list(cfg.get("models_order") or ["mlp", "gcn", "gat"])

    opt_hp_nc = _optimizer_hp_from_config(cfg.get("nc") or {})
    opt_hp_lp = _optimizer_hp_from_config(cfg.get("lp") or {})
    opt_hp = opt_hp_nc if args.task == "nc" else opt_hp_lp
    if args.task == "nc":
        if args.nc_lr is not None:
            opt_hp["lr"] = float(args.nc_lr)
        if args.nc_wd is not None:
            opt_hp["weight_decay_graph"] = float(args.nc_wd)
        if args.nc_wd_mlp is not None:
            opt_hp["weight_decay_mlp"] = float(args.nc_wd_mlp)

    print("config", Path(args.config).resolve())
    print("data_root", data_root)
    if ex_root is not None:
        print("exptable2graph_root", ex_root)
    print("seeds", seeds)

    data, dataset_meta = load_dataset(
        ds_name,
        data_root,
        split_seed=split_seed,
        exptable2graph_root=ex_root,
        task=args.task,
    )
    data = data.to(device)
    print("device", device, torch.cuda.get_device_name(device) if device.type == "cuda" else "")
    print("data", dataset_meta.source, data.num_nodes, data.num_edges // 2)
    print(
        "run",
        args.task,
        f"epochs<={max_epochs}",
        f"patience={patience}",
        f"lr={opt_hp['lr']}",
        f"wd_g={opt_hp['weight_decay_graph']}",
        f"wd_mlp={opt_hp['weight_decay_mlp']}",
    )
    nc_gat_spec: Dict[str, Any] = dict((cfg.get("nc") or {}).get("gat") or {})
    if args.task == "nc":
        if args.nc_gat_dropout is not None:
            nc_gat_spec["dropout"] = float(args.nc_gat_dropout)
        if args.nc_gat_heads is not None:
            nc_gat_spec["num_heads_layer1"] = int(args.nc_gat_heads)
        if args.nc_gat_hidden_per_head is not None:
            nc_gat_spec["hidden_dim_per_head"] = int(args.nc_gat_hidden_per_head)
    lp_cfg_section = cfg.get("lp") or {}
    ce_cli = getattr(args, "nc_ce_class_weight", None)
    ce_cfg = str((cfg.get("nc") or {}).get("ce_class_weight", "none")).strip().lower()
    if ce_cli is not None:
        nc_ce_mode = str(ce_cli).strip().lower()
    else:
        nc_ce_mode = ce_cfg if ce_cfg in ("none", "balanced") else "none"
    if nc_ce_mode not in ("none", "balanced"):
        nc_ce_mode = "none"
    summary: Dict[str, Any] = {
        "device": str(device),
        "dataset": ds_name,
        "task": args.task,
        "split_seed": split_seed,
        "seeds": seeds,
        "data_root": str(data_root),
        "exptable2graph_root": str(ex_root) if ex_root else "",
        "data_source": dataset_meta.source,
        "config_path": str(Path(args.config).resolve()),
        "config": cfg,
        "early_stopping_used": {
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
        },
        "optimizer_hp": {
            "lr": opt_hp["lr"],
            "weight_decay_graph_models": opt_hp["weight_decay_graph"],
            "weight_decay_mlp": opt_hp["weight_decay_mlp"],
        },
        "models": {},
    }
    if args.task == "nc":
        summary["nc_ce_class_weight"] = nc_ce_mode
        summary["nc_gat_spec_effective"] = {
            k: nc_gat_spec.get(k)
            for k in ("dropout", "num_heads_layer1", "num_heads_layer2", "hidden_dim_per_head")
            if k in nc_gat_spec
        }
    if args.task == "lp":
        summary["lp_acc_f1_protocol"] = "val_macro_f1_argmax_then_test"

    for model in models_order:
        summary["models"][model] = experiment(
            model,
            seeds=seeds,
            task=args.task,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            opt_hp=opt_hp,
            lp_diagnostics=args.lp_diagnostics,
            lp_flip_edge_labels=args.lp_flip_edge_labels,
            nc_gat_spec=nc_gat_spec,
            lp_cfg_section=lp_cfg_section,
            nc_ce_class_weight=nc_ce_mode,
        )
        m = summary["models"][model]
        if args.task == "lp":
            print(
                f"{model.upper()} | AUC {m['test_auc']['mean']:.4f}±{m['test_auc']['std']:.4f} | "
                f"AP {m['test_ap']['mean']:.4f}±{m['test_ap']['std']:.4f} | "
                f"ACC {m['test_acc']['mean']:.4f}±{m['test_acc']['std']:.4f} | "
                f"F1m {m['test_f1_macro']['mean']:.4f}±{m['test_f1_macro']['std']:.4f} | "
                f"F1μ {m['test_f1_micro']['mean']:.4f}±{m['test_f1_micro']['std']:.4f}"
            )
        else:
            print(
                f"{model.upper()} | ACC {m['test_acc']['mean']:.4f}±{m['test_acc']['std']:.4f} | "
                f"F1m {m['test_f1_macro']['mean']:.4f}±{m['test_f1_macro']['std']:.4f} | "
                f"F1μ {m['test_f1_micro']['mean']:.4f}±{m['test_f1_micro']['std']:.4f}"
            )

    suffix = f"{ds_name}_{args.task}"
    out_path = Path(args.out) if args.out else data_root / f"baseline_{suffix}_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("out", out_path)
