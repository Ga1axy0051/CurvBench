import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from backbone import GNNClassifier
from data_factory import load_data, mask_edges
from logger import create_logger
from models import Experts, FermiDiracDecoder, Gating, Sampler
from utils import cal_AUC_AP, cal_F1, cal_accuracy, cal_shortest_dis
from geoopt.optim import RiemannianAdam


class Exp:
    def __init__(self, configs):
        self.configs = configs
        if torch.cuda.is_available() and configs.gpu >= 0:
            self.device = torch.device(f'cuda:{configs.gpu}')
            torch.cuda.set_device(configs.gpu)
        else:
            self.device = torch.device('cpu')

    def checkpoint_path(self):
        if not self.configs.save_dir:
            return None
        return os.path.join(self.configs.save_dir, 'checkpoint_last.pt')

    def model_path(self):
        if not self.configs.save_dir:
            return None
        return os.path.join(self.configs.save_dir, 'model.pth')

    def save_checkpoint(self, payload):
        path = self.checkpoint_path()
        if path is None:
            return
        torch.save(payload, path)

    def load_checkpoint(self):
        path = self.checkpoint_path()
        if path is None or not os.path.exists(path):
            return None
        return torch.load(path, map_location=self.device)

    def move_optimizer_state_to_device(self, optimizer):
        if self.device.type == 'cpu':
            return
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def train(self):
        logger = create_logger(self.configs.log_path)
        features, in_features, labels, edge_index, neg_edge, masks, n_classes = load_data(
            self.configs.root_path,
            self.configs.dataset,
            downstream_task=self.configs.downstream_task,
            split_seed=self.configs.split_seed,
        )
        edge_index = edge_index.long()
        neg_edge = neg_edge.long()
        edge_index_cpu = edge_index.cpu()
        neg_edge_cpu = neg_edge.cpu()
        features = features.to(self.device)
        labels = labels.to(self.device)
        self.masks = masks
        self.in_features = in_features
        self.configs.in_features = in_features
        self.n_classes = n_classes
        self.labels = labels
        self.edge_index = edge_index_cpu.to(self.device)
        self.neg_edge = neg_edge_cpu.to(self.device)
        self.edge_index_cpu = edge_index_cpu
        self.neg_edge_cpu = neg_edge_cpu
        self.features = features
        self.dis_shortest = cal_shortest_dis(self.edge_index_cpu)

        val_prop = 0.05
        test_prop = 0.1
        pos_edges_cpu, neg_edges_cpu = mask_edges(self.edge_index_cpu, self.neg_edge_cpu, val_prop, test_prop)
        self.pos_edges = tuple(edge.to(self.device) for edge in pos_edges_cpu)
        self.neg_edges = tuple(edge.to(self.device) for edge in neg_edges_cpu)
        self.subgraph_sampler = Sampler(
            method='ego',
            sample_hop=self.configs.sample_hop,
            dataset=self.configs.dataset,
            configs=self.configs,
        )

        checkpoint_state = self.load_checkpoint() if int(getattr(self.configs, 'resume', 1)) else None
        if checkpoint_state and checkpoint_state.get('completed'):
            logger.info('Checkpoint already marked complete. Skip training.')
            return

        if self.configs.downstream_task == 'NC':
            accs = []
            wf1s = []
            mf1s = []
            val_accs = []
            val_wf1s = []
            val_mf1s = []
        elif self.configs.downstream_task == 'LP':
            aucs = []
            aps = []
            val_aucs = []
            val_aps = []
        else:
            raise NotImplementedError

        if self.configs.downstream_task == 'LP':
            self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(
                self.features, self.pos_edges[0].cpu(), 'LP'
            )
        else:
            self.subgraph_feature, self.subgraph_edge_index, self.subgraph_batch = self.subgraph_sampler.sample(
                self.features, self.edge_index_cpu, 'NC'
            )

        if self.configs.exp_iters != 1:
            logger.warning('Checkpoint resume is designed for exp_iters=1. Current run will proceed without multi-iter recovery guarantees.')

        for exp_iter in range(self.configs.exp_iters):
            logger.info(f'\ntrain iters {exp_iter}')

            model = Experts(
                init_curvs=self.configs.init_curvs,
                in_dim=in_features,
                hidden_dim=self.configs.hidden_features,
                out_dim=self.configs.embed_features,
                learnable=True,
                num_factors_cls=self.configs.num_factors_cls,
            ).to(self.device)
            model_gating = Gating(
                in_dim=in_features,
                hidden_dim=self.configs.hidden_features,
                out_dim=self.configs.embed_features,
                num_experts=self.configs.num_factors,
                configs=self.configs,
            ).to(self.device)

            logger.info('--------------------------Training Start-------------------------')
            if self.configs.downstream_task == 'NC':
                skip_lp_warmup_nc = bool(int(getattr(self.configs, 'skip_lp_warmup_nc', 1)))
                if skip_lp_warmup_nc:
                    logger.info('Skipping LP warmup for NC by configuration.')
                    if self.checkpoint_path():
                        self.save_checkpoint({
                            'stage': 'cls',
                            'epoch': -1,
                            'completed': False,
                            'model_state_dict': model.state_dict(),
                            'model_gating_state_dict': model_gating.state_dict(),
                        })
                    checkpoint_state = self.load_checkpoint()
                elif checkpoint_state and checkpoint_state.get('stage') == 'cls':
                    logger.info('Skipping LP warmup because checkpoint already reached cls stage.')
                else:
                    self.train_lp(
                        model,
                        model_gating,
                        self.pos_edges,
                        self.neg_edges,
                        logger,
                        checkpoint_state=checkpoint_state,
                        stage_name='lp_for_nc',
                    )
                    self.save_checkpoint({
                        'stage': 'cls',
                        'epoch': -1,
                        'completed': False,
                        'model_state_dict': model.state_dict(),
                        'model_gating_state_dict': model_gating.state_dict(),
                    })
                    checkpoint_state = self.load_checkpoint()

                metrics, model_cls = self.train_cls(
                    model,
                    model_gating,
                    logger,
                    checkpoint_state=checkpoint_state,
                    stage_name='cls',
                )
                logger.info(f"best_epoch={metrics['best_epoch']}")
                logger.info(f"test_accuracy={metrics['test_acc'] * 100: .2f}%")
                logger.info(
                    f"micro_f1={metrics['test_acc'] * 100: .2f}%, "
                    f"weighted_f1={metrics['test_weighted_f1'] * 100: .2f}%, "
                    f"macro_f1={metrics['test_macro_f1'] * 100: .2f}%"
                )
                accs.append(metrics['test_acc'])
                wf1s.append(metrics['test_weighted_f1'])
                mf1s.append(metrics['test_macro_f1'])
                val_accs.append(metrics['best_val_acc'])
                val_wf1s.append(metrics['best_val_weighted_f1'])
                val_mf1s.append(metrics['best_val_macro_f1'])
                self.save_checkpoint({
                    'stage': 'done',
                    'epoch': metrics['best_epoch'],
                    'completed': True,
                    'final_metrics': metrics,
                    'task': 'NC',
                    'model_state_dict': model.state_dict(),
                    'model_gating_state_dict': model_gating.state_dict(),
                    'model_cls_state_dict': model_cls.state_dict(),
                })
                if self.model_path():
                    torch.save({
                        'task': 'NC',
                        'model_state_dict': model.state_dict(),
                        'model_gating_state_dict': model_gating.state_dict(),
                        'model_cls_state_dict': model_cls.state_dict(),
                    }, self.model_path())
            elif self.configs.downstream_task == 'LP':
                metrics = self.train_lp(
                    model,
                    model_gating,
                    self.pos_edges,
                    self.neg_edges,
                    logger,
                    checkpoint_state=checkpoint_state,
                    stage_name='lp',
                )
                logger.info(f"best_epoch={metrics['best_epoch']}")
                logger.info(f"test_auc={metrics['test_auc'] * 100: .2f}%, test_ap={metrics['test_ap'] * 100: .2f}%")
                aucs.append(metrics['test_auc'])
                aps.append(metrics['test_ap'])
                val_aucs.append(metrics['best_val_auc'])
                val_aps.append(metrics['best_val_ap'])
                self.save_checkpoint({
                    'stage': 'done',
                    'epoch': metrics['best_epoch'],
                    'completed': True,
                    'final_metrics': metrics,
                    'task': 'LP',
                    'model_state_dict': model.state_dict(),
                    'model_gating_state_dict': model_gating.state_dict(),
                })
                if self.model_path():
                    torch.save({
                        'task': 'LP',
                        'model_state_dict': model.state_dict(),
                        'model_gating_state_dict': model_gating.state_dict(),
                    }, self.model_path())

            checkpoint_state = None

        if self.configs.downstream_task == 'NC':
            mean_test_acc = float(np.mean(accs))
            std_test_acc = float(np.std(accs))
            mean_test_wf1 = float(np.mean(wf1s))
            std_test_wf1 = float(np.std(wf1s))
            mean_test_mf1 = float(np.mean(mf1s))
            std_test_mf1 = float(np.std(mf1s))
            mean_val_acc = float(np.mean(val_accs))
            mean_val_wf1 = float(np.mean(val_wf1s))
            mean_val_mf1 = float(np.mean(val_mf1s))
            logger.info('----NC Task----')
            logger.info(f'test acc: {mean_test_acc}~{std_test_acc}')
            logger.info(f'test weighted-f1: {mean_test_wf1}~{std_test_wf1}')
            logger.info(f'test macro-f1: {mean_test_mf1}~{std_test_mf1}')
            logger.info(
                'FINAL_SUMMARY '
                f'task=NC dataset={self.configs.dataset} '
                f'val_acc={mean_val_acc:.6f} val_micro_f1={mean_val_acc:.6f} '
                f'val_macro_f1={mean_val_mf1:.6f} val_weighted_f1={mean_val_wf1:.6f} '
                f'val_wf1={mean_val_wf1:.6f} val_mf1={mean_val_mf1:.6f} '
                f'test_acc={mean_test_acc:.6f} test_micro_f1={mean_test_acc:.6f} '
                f'test_macro_f1={mean_test_mf1:.6f} test_weighted_f1={mean_test_wf1:.6f} '
                f'test_wf1={mean_test_wf1:.6f} test_mf1={mean_test_mf1:.6f}'
            )
        else:
            mean_test_auc = float(np.mean(aucs))
            std_test_auc = float(np.std(aucs))
            mean_test_ap = float(np.mean(aps))
            std_test_ap = float(np.std(aps))
            mean_val_auc = float(np.mean(val_aucs))
            mean_val_ap = float(np.mean(val_aps))
            logger.info('----LP Task----')
            logger.info(f'test AUC: {mean_test_auc}~{std_test_auc}')
            logger.info(f'test AP: {mean_test_ap}~{std_test_ap}')
            logger.info(
                'FINAL_SUMMARY '
                f'task=LP dataset={self.configs.dataset} '
                f'val_auc={mean_val_auc:.6f} val_ap={mean_val_ap:.6f} '
                f'test_auc={mean_test_auc:.6f} test_ap={mean_test_ap:.6f}'
            )

    def cal_cls_loss(self, model, edge_index, mask, features, labels):
        out = model(features, edge_index)
        loss = F.cross_entropy(out[mask], labels[mask])
        acc = cal_accuracy(out[mask], labels[mask])
        weighted_f1, macro_f1 = cal_F1(out[mask].detach().cpu(), labels[mask].detach().cpu())
        return loss, acc, weighted_f1, macro_f1

    def train_cls(self, model, model_gating, logger, checkpoint_state=None, stage_name='cls'):
        self.configs.coef_dis = 0.0
        d = self.configs.num_factors_cls * self.configs.embed_features
        model_cls = GNNClassifier(
            backbone=self.configs.backbone,
            n_layers=max(1, self.configs.n_layers),
            in_features=self.in_features + d,
            hidden_features=self.configs.hidden_features_cls,
            out_features=self.n_classes,
            n_heads=self.configs.n_heads,
            drop_edge=self.configs.drop_edge_cls,
            drop_node=self.configs.drop_cls,
        ).to(self.device)
        optimizer_cls = torch.optim.Adam(
            model_cls.parameters(), lr=self.configs.lr_cls, weight_decay=self.configs.w_decay_cls
        )
        r_optim = RiemannianAdam(
            model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100
        )
        optimizer_gating = torch.optim.Adam(
            model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating
        )
        best_acc = -1.0
        best_weighted_f1 = 0.0
        best_macro_f1 = -1.0
        best_epoch = 0
        best_test_acc = 0.0
        best_test_weighted_f1 = 0.0
        best_test_macro_f1 = 0.0
        early_stop_count = 0
        start_epoch = 0

        if checkpoint_state and checkpoint_state.get('stage') == stage_name:
            if checkpoint_state.get('model_state_dict'):
                model.load_state_dict(checkpoint_state['model_state_dict'])
            if checkpoint_state.get('model_gating_state_dict'):
                model_gating.load_state_dict(checkpoint_state['model_gating_state_dict'])
            if checkpoint_state.get('model_cls_state_dict'):
                model_cls.load_state_dict(checkpoint_state['model_cls_state_dict'])
            if checkpoint_state.get('optimizer_cls_state_dict'):
                optimizer_cls.load_state_dict(checkpoint_state['optimizer_cls_state_dict'])
                self.move_optimizer_state_to_device(optimizer_cls)
            if checkpoint_state.get('r_optim_state_dict'):
                r_optim.load_state_dict(checkpoint_state['r_optim_state_dict'])
                self.move_optimizer_state_to_device(r_optim)
            if checkpoint_state.get('optimizer_gating_state_dict'):
                optimizer_gating.load_state_dict(checkpoint_state['optimizer_gating_state_dict'])
                self.move_optimizer_state_to_device(optimizer_gating)
            best_acc = float(checkpoint_state.get('best_val_acc', best_acc))
            best_weighted_f1 = float(checkpoint_state.get('best_val_weighted_f1', best_weighted_f1))
            best_macro_f1 = float(checkpoint_state.get('best_val_macro_f1', best_macro_f1))
            best_epoch = int(checkpoint_state.get('best_epoch', best_epoch))
            best_test_acc = float(checkpoint_state.get('best_test_acc', best_test_acc))
            best_test_weighted_f1 = float(checkpoint_state.get('best_test_weighted_f1', best_test_weighted_f1))
            best_test_macro_f1 = float(checkpoint_state.get('best_test_macro_f1', best_test_macro_f1))
            early_stop_count = int(checkpoint_state.get('early_stop_count', early_stop_count))
            start_epoch = int(checkpoint_state.get('epoch', -1)) + 1
            if int(checkpoint_state.get('epoch', -1)) < 0:
                if bool(int(getattr(self.configs, 'skip_lp_warmup_nc', 1))):
                    logger.info(f'Starting cls stage from epoch {start_epoch} without LP warmup.')
                else:
                    logger.info(f'Starting cls stage from epoch {start_epoch} after LP warmup.')
            else:
                logger.info(f'Resumed cls stage from epoch {start_epoch}.')

        for epoch in range(start_epoch, self.configs.epochs_cls + 1):
            now_time = time.time()
            model_cls.train()
            model.train()
            model_gating.train()
            optimizer_cls.zero_grad()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()

            embeddings = model.encode(self.features, self.edge_index, self.configs.dataset)
            embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            experts_weight, loss_distortion = model_gating(
                self.subgraph_feature,
                self.subgraph_edge_index,
                self.subgraph_batch,
                embeddings,
                self.dis_shortest,
                self.configs.embed_features,
                self.edge_index,
            )
            experts_weight = torch.nan_to_num(experts_weight, nan=0.0, posinf=0.0, neginf=0.0)
            loss_distortion = torch.nan_to_num(loss_distortion, nan=0.0, posinf=0.0, neginf=0.0)
            experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
            embeddings = embeddings * experts_weight
            features = torch.concat([self.features, embeddings], -1)
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            loss, acc, weighted_f1, macro_f1 = self.cal_cls_loss(
                model_cls, self.edge_index, self.masks[0], features, self.labels
            )
            loss = loss + self.configs.coef_dis * loss_distortion
            if not torch.isfinite(loss):
                raise ValueError('non-finite classification loss encountered during training')

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_cls.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(model_gating.parameters(), max_norm=1.0)
            optimizer_cls.step()
            r_optim.step()
            optimizer_gating.step()
            logger.info(f'Epoch {epoch}: train_loss={loss.item()}, train_accuracy={acc}, time={time.time() - now_time}')

            if epoch % self.configs.eval_freq == 0:
                model_cls.eval()
                model.eval()
                model_gating.eval()

                with torch.no_grad():
                    embeddings = model.encode(self.features, self.edge_index)
                    embeddings = torch.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                    experts_weight = model_gating(
                        self.subgraph_feature,
                        self.subgraph_edge_index,
                        self.subgraph_batch,
                    )
                    experts_weight = torch.nan_to_num(experts_weight, nan=0.0, posinf=0.0, neginf=0.0)
                    experts_weight = experts_weight.repeat_interleave(self.configs.embed_features, dim=1)
                    embeddings = embeddings * experts_weight
                    features = torch.concat([self.features, embeddings], -1)
                    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    _, acc, weighted_f1, macro_f1 = self.cal_cls_loss(
                        model_cls, self.edge_index, self.masks[1], features, self.labels
                    )
                    logger.info(f'Epoch {epoch}: val_accuracy={acc}, val_wf1={weighted_f1}, val_mf1={macro_f1}')

                    if macro_f1 > best_macro_f1:
                        best_acc = float(acc)
                        best_weighted_f1 = float(weighted_f1)
                        best_macro_f1 = float(macro_f1)
                        best_epoch = epoch
                        early_stop_count = 0
                        _, test_acc, test_weighted_f1, test_macro_f1 = self.cal_cls_loss(
                            model_cls, self.edge_index, self.masks[2], features, self.labels
                        )
                        best_test_acc = float(test_acc)
                        best_test_weighted_f1 = float(test_weighted_f1)
                        best_test_macro_f1 = float(test_macro_f1)
                    else:
                        early_stop_count += 1
                        if early_stop_count > self.configs.patience_cls:
                            break
                    if epoch < self.configs.min_epoch_cls:
                        early_stop_count = 0

            self.save_checkpoint({
                'stage': stage_name,
                'epoch': epoch,
                'completed': False,
                'model_state_dict': model.state_dict(),
                'model_gating_state_dict': model_gating.state_dict(),
                'model_cls_state_dict': model_cls.state_dict(),
                'optimizer_cls_state_dict': optimizer_cls.state_dict(),
                'r_optim_state_dict': r_optim.state_dict(),
                'optimizer_gating_state_dict': optimizer_gating.state_dict(),
                'best_val_acc': best_acc,
                'best_val_weighted_f1': best_weighted_f1,
                'best_val_macro_f1': best_macro_f1,
                'best_epoch': best_epoch,
                'best_test_acc': best_test_acc,
                'best_test_weighted_f1': best_test_weighted_f1,
                'best_test_macro_f1': best_test_macro_f1,
                'early_stop_count': early_stop_count,
            })

        return {
            'best_val_acc': best_acc,
            'best_val_weighted_f1': best_weighted_f1,
            'best_val_macro_f1': best_macro_f1,
            'test_acc': best_test_acc,
            'test_weighted_f1': best_test_weighted_f1,
            'test_macro_f1': best_test_macro_f1,
            'best_epoch': best_epoch,
        }, model_cls

    def cal_lp_loss(self, embeddings, experts_weight, decoder, pos_edges, neg_edges):
        pos_diff = (embeddings[pos_edges[0]] - embeddings[pos_edges[1]]) ** 2
        pos_diff = pos_diff.reshape(
            pos_diff.shape[0], pos_diff.shape[1] // self.configs.embed_features, self.configs.embed_features
        ).sum(dim=2)
        pos_weights = F.softmax(experts_weight[pos_edges[0]] * experts_weight[pos_edges[1]], dim=1)
        pos_scores = decoder(torch.sum(pos_diff * pos_weights, -1))

        neg_diff = (embeddings[neg_edges[0]] - embeddings[neg_edges[1]]) ** 2
        neg_diff = neg_diff.reshape(
            neg_diff.shape[0], neg_diff.shape[1] // self.configs.embed_features, self.configs.embed_features
        ).sum(dim=2)
        neg_weights = F.softmax(experts_weight[neg_edges[0]] * experts_weight[neg_edges[1]], dim=1)
        neg_scores = decoder(torch.sum(neg_diff * neg_weights, -1))

        # Keep the LP warmup numerically stable on difficult datasets such as PTE.
        pos_scores = torch.nan_to_num(pos_scores, nan=0.5, posinf=0.99, neginf=0.01).clamp(0.01, 0.99)
        neg_scores = torch.nan_to_num(neg_scores, nan=0.5, posinf=0.99, neginf=0.01).clamp(0.01, 0.99)

        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + F.binary_cross_entropy(
            neg_scores, torch.zeros_like(neg_scores)
        )
        label = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.detach().cpu().numpy()) + list(neg_scores.detach().cpu().numpy())
        auc, ap = cal_AUC_AP(preds, label)
        return loss, auc, ap

    def train_lp(self, model, model_gating, pos_edges, neg_edges, logger, checkpoint_state=None, stage_name='lp'):
        r_optim = RiemannianAdam(
            model.parameters(), lr=self.configs.lr_Riemann, weight_decay=self.configs.w_decay, stabilize=100
        )
        optimizer_gating = torch.optim.Adam(
            model_gating.parameters(), lr=self.configs.lr_gating, weight_decay=self.configs.w_decay_gating
        )

        decoder = FermiDiracDecoder(self.configs.r, self.configs.t).to(self.device)
        best_ap = -1.0
        best_auc = -1.0
        best_epoch = 0
        best_test_auc = 0.0
        best_test_ap = 0.0
        early_stop_count = 0
        start_epoch = 0

        if checkpoint_state and checkpoint_state.get('stage') == stage_name:
            if checkpoint_state.get('model_state_dict'):
                model.load_state_dict(checkpoint_state['model_state_dict'])
            if checkpoint_state.get('model_gating_state_dict'):
                model_gating.load_state_dict(checkpoint_state['model_gating_state_dict'])
            if checkpoint_state.get('decoder_state_dict'):
                decoder.load_state_dict(checkpoint_state['decoder_state_dict'])
            if checkpoint_state.get('r_optim_state_dict'):
                r_optim.load_state_dict(checkpoint_state['r_optim_state_dict'])
                self.move_optimizer_state_to_device(r_optim)
            if checkpoint_state.get('optimizer_gating_state_dict'):
                optimizer_gating.load_state_dict(checkpoint_state['optimizer_gating_state_dict'])
                self.move_optimizer_state_to_device(optimizer_gating)
            best_ap = float(checkpoint_state.get('best_val_ap', best_ap))
            best_auc = float(checkpoint_state.get('best_val_auc', best_auc))
            best_epoch = int(checkpoint_state.get('best_epoch', best_epoch))
            best_test_auc = float(checkpoint_state.get('best_test_auc', best_test_auc))
            best_test_ap = float(checkpoint_state.get('best_test_ap', best_test_ap))
            early_stop_count = int(checkpoint_state.get('early_stop_count', early_stop_count))
            start_epoch = int(checkpoint_state.get('epoch', -1)) + 1
            logger.info(f'Resumed {stage_name} stage from epoch {start_epoch}.')

        for epoch in range(start_epoch, self.configs.epochs_lp + 1):
            start_time = time.time()
            model.train()
            model_gating.train()
            r_optim.zero_grad()
            optimizer_gating.zero_grad()

            embeddings = model(self.features, pos_edges[0])
            experts_weight, loss_distortion = model_gating(
                self.subgraph_feature,
                self.subgraph_edge_index,
                self.subgraph_batch,
                embeddings,
                self.dis_shortest,
                self.configs.embed_features,
                pos_edges[0],
            )

            neg_edge_train = neg_edges[0][:, np.random.randint(0, neg_edges[0].shape[1], pos_edges[0].shape[1])]
            loss, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[0], neg_edge_train)
            loss = loss + self.configs.coef_dis * loss_distortion
            loss.backward()
            r_optim.step()
            optimizer_gating.step()
            logger.info(f'Epoch {epoch}: train_loss={loss.item()}, train_AUC={auc}, train_AP={ap}, time={time.time() - start_time}')

            if epoch % self.configs.eval_freq == 0:
                model.eval()
                model_gating.eval()
                with torch.no_grad():
                    embeddings = model(self.features, pos_edges[0])
                    experts_weight = model_gating(
                        self.subgraph_feature,
                        self.subgraph_edge_index,
                        self.subgraph_batch,
                    )
                    _, auc, ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[1], neg_edges[1])
                    logger.info(f'Epoch {epoch}: val_AUC={auc}, val_AP={ap}')
                    if ap > best_ap:
                        best_ap = float(ap)
                        best_auc = float(auc)
                        best_epoch = epoch
                        early_stop_count = 0
                        _, test_auc, test_ap = self.cal_lp_loss(embeddings, experts_weight, decoder, pos_edges[2], neg_edges[2])
                        best_test_auc = float(test_auc)
                        best_test_ap = float(test_ap)
                    else:
                        early_stop_count += 1
                        if early_stop_count > self.configs.patience_lp:
                            break
                    if epoch < self.configs.min_epoch_lp:
                        early_stop_count = 0

            self.save_checkpoint({
                'stage': stage_name,
                'epoch': epoch,
                'completed': False,
                'model_state_dict': model.state_dict(),
                'model_gating_state_dict': model_gating.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'r_optim_state_dict': r_optim.state_dict(),
                'optimizer_gating_state_dict': optimizer_gating.state_dict(),
                'best_val_auc': best_auc,
                'best_val_ap': best_ap,
                'best_epoch': best_epoch,
                'best_test_auc': best_test_auc,
                'best_test_ap': best_test_ap,
                'early_stop_count': early_stop_count,
            })

        return {
            'best_val_auc': best_auc,
            'best_val_ap': best_ap,
            'test_auc': best_test_auc,
            'test_ap': best_test_ap,
            'best_epoch': best_epoch,
        }
