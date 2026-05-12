import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import networkx as nx
import random


def cal_accuracy(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    correct = (preds == trues).sum()
    return correct / len(trues)


def cal_F1(preds, trues):
    preds = torch.argmax(preds, dim=-1)
    weighted_f1 = f1_score(trues, preds, average='weighted')
    macro_f1 = f1_score(trues, preds, average='macro')
    return weighted_f1, macro_f1


def cal_AUC_AP(scores, trues):
    auc = roc_auc_score(trues, scores)
    ap = average_precision_score(trues, scores)
    return auc, ap

def cal_shortest_dis(edge_index):
    dis_shortest = {}
    edge_index_ = edge_index.detach().cpu().numpy().astype(int)
    for i in range(edge_index_.shape[1]):
        src = edge_index_[0][i]
        dst = edge_index_[1][i]
        dis_shortest[(src, dst)] = 1.0
        dis_shortest[(dst, src)] = 1.0
    return dis_shortest

