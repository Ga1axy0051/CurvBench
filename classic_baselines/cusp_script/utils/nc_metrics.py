# -*- coding: utf-8 -*-
"""Node classification metrics: accuracy and macro/micro/weighted F1."""
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def nc_split_metrics(preds: np.ndarray, labels: np.ndarray, mask) -> dict:
    """mask: bool tensor or ndarray, length num_nodes."""
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
        }
    p = preds[idx]
    y = labels[idx]
    valid = y >= 0
    if not np.all(valid):
        p, y = p[valid], y[valid]
    if y.size == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
        }
    return {
        "accuracy": float(accuracy_score(y, p)),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y, p, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
    }
