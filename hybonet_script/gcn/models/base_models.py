"""Base model class."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import MarginLoss


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
            if self.manifold.name == 'Lorentz':
                x = self.manifold.expmap0(x)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError


class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        self.margin = args.margin
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output[idx]

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        if self.manifold_name == 'Lorentz':
            correct = output.gather(1, data['labels'][idx].unsqueeze(-1))
            loss = F.relu(self.margin - correct + output).mean()
        else:
            loss = F.cross_entropy(output, data['labels'][idx], self.weights)
        preds = output.max(1)[1].detach().cpu().numpy()
        lab = data['labels'][idx].detach().cpu().numpy()
        acc = float(accuracy_score(lab, preds))
        macro_f1 = float(f1_score(lab, preds, average='macro', zero_division=0))
        micro_f1 = float(f1_score(lab, preds, average='micro', zero_division=0))
        metrics = {
            'loss': loss,
            'acc': acc,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'f1': micro_f1,
        }
        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1, 'macro_f1': -1, 'micro_f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.loss = MarginLoss(args.margin)

    def decode(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        return -sqdist

    def compute_metrics(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        preds = torch.stack([pos_scores, neg_scores], dim=-1)
        loss = self.loss(preds)

        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        ps = pos_scores.data.numpy()
        ns = neg_scores.data.numpy()
        n = min(len(ps), len(ns))
        ps, ns = ps[:n], ns[:n]
        all_s = np.concatenate([ps, ns])
        tau = float(np.median(all_s))
        y_true = np.array([1] * n + [0] * n, dtype=np.int64)
        y_pred = (all_s > tau).astype(np.int64)
        acc_lp = float(accuracy_score(y_true, y_pred))
        macro_lp = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        micro_lp = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        pair_acc = float((ps > ns).mean())
        metrics = {
            'loss': loss,
            'roc': roc,
            'ap': ap,
            'acc': acc_lp,
            'macro_f1': macro_lp,
            'micro_f1': micro_lp,
            'f1': micro_lp,
            'pair_acc': pair_acc,
            'score_threshold_median': tau,
        }
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1, 'acc': -1, 'macro_f1': -1, 'micro_f1': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])

