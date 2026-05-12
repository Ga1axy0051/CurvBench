import torch
import torch.nn.functional as F
import numpy as np
from downstream.adapter import GraphGlueAdapter
from sklearn.metrics import roc_auc_score, f1_score


def _forward_pass(model, data):
    pred, holo_loss, curv_loss = model(data)
    return pred, holo_loss, curv_loss


def _flatten_metric_inputs(preds_list, trues_list):
    preds = np.concatenate(preds_list, axis=-1).reshape(-1)
    trues = np.concatenate(trues_list, axis=-1).reshape(-1)
    return preds, trues


def _compute_metrics(preds_list, trues_list, metric: str = "acc"):
    preds, trues = _flatten_metric_inputs(preds_list, trues_list)
    if metric == "acc":
        metric = np.sum(preds == trues) / len(preds)
    elif metric == "auc":
        metric = roc_auc_score(trues, preds)
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return metric


def _compute_macro_f1(preds_list, trues_list):
    preds, trues = _flatten_metric_inputs(preds_list, trues_list)
    return f1_score(trues, preds, average="macro", zero_division=0)


def train_step(loader,
               optimizer,
               model: GraphGlueAdapter,
               device,
               label_attr='y',
               metric="acc"):
    model.train()
    total_loss = 0.0
    total_task_loss = 0.0
    total_holo_loss = 0.0
    total_curv_loss = 0.0
    preds_list = []
    trues_list = []

    for i, data in enumerate(loader):
        optimizer.zero_grad()
        data = data.to(device)

        pred, holo_loss, curv_loss = _forward_pass(model, data)

        label = getattr(data, label_attr)

        task_loss = F.cross_entropy(pred, label)
        loss = task_loss + holo_loss + curv_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_task_loss += task_loss.item()
        total_holo_loss += holo_loss.item()
        total_curv_loss += curv_loss.item()
        preds_list.append(pred.detach().cpu().numpy().argmax(-1))
        trues_list.append(label.detach().cpu().numpy())

    acc = _compute_metrics(preds_list, trues_list, metric)
    avg_loss = total_loss / len(loader)
    avg_task_loss = total_task_loss / len(loader)
    avg_holo_loss = total_holo_loss / len(loader)
    avg_curv_loss = total_curv_loss / len(loader)
    return avg_loss, avg_task_loss, acc, avg_holo_loss, avg_curv_loss


def eval_step(loader,
              model: GraphGlueAdapter,
              device,
              label_attr='y',
              metric="acc",
              return_macro_f1: bool = False):
    model.eval()
    total_loss = 0.0
    total_task_loss = 0.0
    total_holo_loss = 0.0
    total_curv_loss = 0.0
    preds_list = []
    trues_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)
            pred, holo_loss, curv_loss = _forward_pass(model, data)

            label = getattr(data, label_attr)

            task_loss = F.cross_entropy(pred, label)
            loss = task_loss + holo_loss + curv_loss
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_holo_loss += holo_loss.item()
            total_curv_loss += curv_loss.item()
            preds_list.append(pred.detach().cpu().numpy().argmax(-1))
            trues_list.append(label.detach().cpu().numpy())

    acc = _compute_metrics(preds_list, trues_list, metric)
    avg_loss = total_loss / len(loader)
    avg_task_loss = total_task_loss / len(loader)
    avg_holo_loss = total_holo_loss / len(loader)
    avg_curv_loss = total_curv_loss / len(loader)
    if return_macro_f1:
        macro_f1 = _compute_macro_f1(preds_list, trues_list)
        return avg_loss, avg_task_loss, acc, avg_holo_loss, avg_curv_loss, macro_f1
    return avg_loss, avg_task_loss, acc, avg_holo_loss, avg_curv_loss