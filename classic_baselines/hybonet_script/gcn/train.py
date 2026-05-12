from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
from optim import RiemannianAdam, RiemannianSGD
import os
import pickle
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

from geoopt import ManifoldParameter


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def _experiment_dir(repo_root, args):
    """logs/experiments/<dataset>/<task>/seed_<seed>/ (matches suite layout)."""
    return os.path.join(
        repo_root, 'logs', 'experiments',
        str(args.dataset), str(args.task),
        'seed_{}'.format(int(args.seed)))


def _build_experiment_config(args):
    """Structured hyperparams for experiment_config.json (no graph blobs)."""
    def _b(x):
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if x in (True, False):
            return bool(x)
        try:
            return int(x)
        except (TypeError, ValueError):
            pass
        try:
            return float(x)
        except (TypeError, ValueError):
            return str(x)

    act = getattr(args, 'act', None)
    if isinstance(act, str) and act.lower() in ('none', 'null'):
        act = None
    elif act is not None and not isinstance(act, str):
        act = str(act)
    return {
        'training': {
            'lr': _b(getattr(args, 'lr', None)),
            'epochs': _b(getattr(args, 'epochs', None)),
            'optimizer': getattr(args, 'optimizer', None),
            'dropout': _b(getattr(args, 'dropout', None)),
            'weight_decay': _b(getattr(args, 'weight_decay', None)),
            'grad_clip': _b(getattr(args, 'grad_clip', None)),
            'patience': _b(getattr(args, 'patience', None)),
            'min_epochs': _b(getattr(args, 'min_epochs', None)),
            'seed': int(args.seed),
            'split_seed': int(getattr(args, 'split_seed', args.seed)),
            'log_freq': _b(getattr(args, 'log_freq', None)),
            'eval_freq': _b(getattr(args, 'eval_freq', None)),
            'gamma': _b(getattr(args, 'gamma', None)),
            'lr_reduce_freq': _b(getattr(args, 'lr_reduce_freq', None)),
        },
        'model': {
            'model': getattr(args, 'model', None),
            'num_layers': _b(getattr(args, 'num_layers', None)),
            'dim': _b(getattr(args, 'dim', None)),
            'manifold': getattr(args, 'manifold', None),
            'bias': _b(getattr(args, 'bias', None)),
            'act': act,
            'margin': _b(getattr(args, 'margin', None)),
            'c': _b(getattr(args, 'c', None)) if getattr(args, 'c', None) is not None else None,
            'use_att': _b(getattr(args, 'use_att', None)),
            'local_agg': _b(getattr(args, 'local_agg', None)),
            'n_heads': _b(getattr(args, 'n_heads', None)),
            'alpha': _b(getattr(args, 'alpha', None)),
        },
        'data': {
            'dataset': str(args.dataset),
            'task': str(args.task),
            'use_feats': _b(getattr(args, 'use_feats', None)),
            'normalize_feats': _b(getattr(args, 'normalize_feats', None)),
            'normalize_adj': _b(getattr(args, 'normalize_adj', None)),
            'val_prop': _b(getattr(args, 'val_prop', None)),
            'test_prop': _b(getattr(args, 'test_prop', None)),
            'n_nodes': getattr(args, 'n_nodes', None),
            'feat_dim': getattr(args, 'feat_dim', None),
            'n_classes': getattr(args, 'n_classes', None),
        },
        'extra': {
            'double_precision': _b(getattr(args, 'double_precision', None)),
            'cuda': _b(getattr(args, 'cuda', None)),
            'pretrained_embeddings': getattr(args, 'pretrained_embeddings', None),
            'pos_weight': _b(getattr(args, 'pos_weight', None)),
        },
        'software': {'torch': torch.__version__},
    }


def _curve_record(train_metrics, val_metrics, epoch, last_val_time):
    rec = {
        'epoch': epoch + 1,
        'train_loss': _to_float(train_metrics['loss']),
        'val_eval_time_sec': float(last_val_time),
    }
    for k, v in val_metrics.items():
        out_key = 'val_loss' if k == 'loss' else 'val_{}'.format(k)
        if torch.is_tensor(v):
            if v.numel() == 1:
                rec[out_key] = _to_float(v)
        elif isinstance(v, (float, int, np.floating)):
            rec[out_key] = float(v)
    return rec


def _args_json_safe(args):
    """vars(args) minus non-JSON fields."""
    d = {}
    for k, v in vars(args).items():
        if k == 'data':
            continue
        if callable(v):
            continue
        try:
            json.dumps(v)
            d[k] = v
        except (TypeError, ValueError):
            d[k] = str(v)
    return d


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    use_curve = int(getattr(args, 'save_curve', 0))
    use_ec = int(getattr(args, 'always_save_config', 0))
    experiment_dir = None
    if use_curve or use_ec:
        experiment_dir = _experiment_dir(repo_root, args)
        os.makedirs(experiment_dir, exist_ok=True)

    save_dir = None
    if args.save:
        if args.save_dir:
            save_dir = args.save_dir
        elif experiment_dir is not None:
            save_dir = experiment_dir
        else:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            log_root = os.environ.get('LOG_DIR')
            if not log_root:
                log_root = os.path.join(repo_root, 'logs')
            models_dir = os.path.join(log_root, args.task, date)
            save_dir = get_dir_name(models_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                logging.StreamHandler()])
    elif experiment_dir is not None:
        save_dir = experiment_dir
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                logging.StreamHandler()])
    else:
        logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])

    convergence_curve = []

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    if experiment_dir:
        logging.info('Experiment artifacts directory: {}'.format(experiment_dir))

    # Load data
    data_root = os.environ.get('DATAPATH')
    if not data_root:
        data_root = os.path.join(repo_root, 'data')
    # telecom_*: default data/telecom or TELECOM_DATAPATH
    if args.dataset in ('telecom_nc', 'telecom_lp'):
        telecom_dir = os.environ.get('TELECOM_DATAPATH')
        if telecom_dir:
            datapath = telecom_dir
        else:
            datapath = os.path.join(repo_root, 'data', 'telecom')
        logging.info('Using telecom data directory: {}'.format(datapath))
    elif args.dataset in ('cs_phds_nc', 'cs_phds_lp'):
        # data/cs_phds/*_ready or CS_PHDS_DATAPATH (parent of *_ready folders)
        cs_phds_base = os.environ.get('CS_PHDS_DATAPATH')
        if not cs_phds_base:
            cs_phds_base = os.path.join(data_root, 'cs_phds')
        sub = 'cs_phds_nc_ready' if args.dataset == 'cs_phds_nc' else 'cs_phds_lp_ready'
        datapath = os.path.join(cs_phds_base, sub)
        logging.info('Using cs_phds data directory: {}'.format(datapath))
    else:
        datapath = os.path.join(data_root, args.dataset)
    data = load_data(args, datapath)
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        # Remap label ids to 0..C-1 on supervised nodes only (exptable unused-class fix).
        idx_sup = data['idx_train'] + data['idx_val'] + data['idx_test']
        if len(idx_sup) > 0:
            ys = data['labels'][idx_sup]
            cmin = int(ys.min().item())
            cmax = int(ys.max().item())
            data['labels'] = data['labels'] - cmin
            args.n_classes = cmax - cmin + 1
        else:
            args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    else:
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        if args.task == 'lp':
            Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    no_decay = ['bias', 'scale']
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in model.named_parameters()
            if p.requires_grad and not any(
                nd in n
                for nd in no_decay) and not isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [
            p for n, p in model.named_parameters() if p.requires_grad and any(
                nd in n
                for nd in no_decay) or isinstance(p, ManifoldParameter)
        ],
        'weight_decay':
        0.0
    }]
    if args.optimizer == 'radam':
        optimizer = RiemannianAdam(params=optimizer_grouped_parameters,
                                   lr=args.lr,
                                   stabilize=10)
    elif args.optimizer == 'rsgd':
        optimizer = RiemannianSGD(params=optimizer_grouped_parameters,
                                  lr=args.lr,
                                  stabilize=10)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=int(
                                                       args.lr_reduce_freq),
                                                   gamma=float(args.gamma))
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    model = model.to(args.device)
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(args.device)
    logging.info(f"Total number of parameters: {tot_params}")
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    # Timing & best-epoch tracking
    cum_val_time = 0.0
    cum_test_time = 0.0
    n_val_evals = 0
    n_test_runs = 0
    last_val_time = 0.0
    last_test_time = 0.0
    best_epoch = -1
    epochs_run = 0
    for epoch in range(args.epochs):
        epochs_run = epoch + 1
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        train_metrics['loss'].backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join([
                'Epoch: {:04d}'.format(epoch + 1),
                'lr: {}'.format(lr_scheduler.get_last_lr()),
                format_metrics(train_metrics, 'train'),
                'time: {:.4f}s'.format(time.time() - t)
            ]))
        with torch.no_grad():
            if (epoch + 1) % args.eval_freq == 0:
                model.eval()
                t_val0 = time.time()
                embeddings = model.encode(data['features'],
                                          data['adj_train_norm'])
                val_metrics = model.compute_metrics(embeddings, data, 'val')
                last_val_time = time.time() - t_val0
                cum_val_time += last_val_time
                n_val_evals += 1
                if use_curve:
                    convergence_curve.append(
                        _curve_record(train_metrics, val_metrics, epoch, last_val_time))
                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join([
                        'Epoch: {:04d}'.format(epoch + 1),
                        format_metrics(val_metrics, 'val'),
                        'val_eval_time: {:.4f}s'.format(last_val_time)
                    ]))
                if model.has_improved(best_val_metrics, val_metrics):
                    t_test0 = time.time()
                    best_test_metrics = model.compute_metrics(
                        embeddings, data, 'test')
                    last_test_time = time.time() - t_test0
                    cum_test_time += last_test_time
                    n_test_runs += 1
                    best_emb = embeddings.cpu()
                    best_epoch = epoch + 1
                    if args.save:
                        np.save(os.path.join(save_dir, 'embeddings.npy'),
                                best_emb.detach().numpy())
                    best_val_metrics = val_metrics
                    counter = 0
                    if (epoch + 1) % args.log_freq == 0:
                        logging.info(
                            'Epoch: {:04d} test_eval_time: {:.4f}s'.format(
                                epoch + 1, last_test_time))
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break

    wall_train_s = time.time() - t_total
    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(wall_train_s))
    if not best_test_metrics:
        model.eval()
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        t_test0 = time.time()
        best_test_metrics = model.compute_metrics(best_emb, data, 'test')
        last_test_time = time.time() - t_test0
        cum_test_time += last_test_time
        n_test_runs += 1
    # --- Best checkpoint & timing summary ---
    v_loss = _to_float(best_val_metrics['loss']) if 'loss' in best_val_metrics else float('nan')
    v_acc = _to_float(best_val_metrics['acc']) if 'acc' in best_val_metrics else float('nan')
    t_loss = _to_float(best_test_metrics['loss']) if 'loss' in best_test_metrics else float('nan')

    if args.task == 'nc':
        logging.info(
            "Best epoch: {} , Best val loss: {} , val acc: {}".format(
                best_epoch, v_loss, v_acc))
        idx_te = data['idx_test']
        with torch.no_grad():
            model.eval()
            h = model.encode(data['features'], data['adj_train_norm'])
            out = model.decode(h, data['adj_train_norm'], idx_te)
            pred = out.max(1)[1].detach().cpu().numpy()
            y = data['labels'][idx_te].detach().cpu().numpy()
        try:
            macro_f1 = f1_score(y, pred, average='macro', zero_division=0)
            micro_f1 = f1_score(y, pred, average='micro', zero_division=0)
        except TypeError:
            macro_f1 = f1_score(y, pred, average='macro')
            micro_f1 = f1_score(y, pred, average='micro')
        logging.info(
            "Test loss: {} macro {:.8f} micro {:.8f} | last Val eval time: {:.4f}s | last Test eval time: {:.4f}s".format(
                t_loss, macro_f1, micro_f1, last_val_time, last_test_time))
    else:
        # lp: roc / ap
        roc = _to_float(best_test_metrics.get('roc', float('nan')))
        ap = _to_float(best_test_metrics.get('ap', float('nan')))
        logging.info(
            "Best epoch: {} , Best val loss: {} , val roc: {:.4f} , val ap: {:.4f}".format(
                best_epoch, v_loss,
                _to_float(best_val_metrics.get('roc', float('nan'))),
                _to_float(best_val_metrics.get('ap', float('nan')))))
        logging.info(
            "Test loss: {} roc {:.8f} ap {:.8f} | last Val eval time: {:.4f}s | last Test eval time: {:.4f}s".format(
                t_loss, roc, ap, last_val_time, last_test_time))

    logging.info("--- Timing summary ---")
    logging.info(
        "Total wall-clock training time: {:.2f}s ({} epochs)".format(
            wall_train_s, epochs_run))
    if n_val_evals > 0:
        logging.info(
            "Cumulative Val eval time: {:.2f}s (avg {:.4f}s / eval)".format(
                cum_val_time, cum_val_time / n_val_evals))
    if n_test_runs > 0:
        logging.info(
            "Cumulative Test eval time: {:.2f}s ({} test runs, only when val improved)".format(
                cum_test_time, n_test_runs))

    logging.info(" ".join(
        ["Val set results:",
         format_metrics(best_val_metrics, 'val')]))
    logging.info(" ".join(
        ["Test set results:",
         format_metrics(best_test_metrics, 'test')]))

    out_dir = experiment_dir if experiment_dir is not None else save_dir
    if out_dir is not None and use_curve:
        with open(os.path.join(out_dir, 'convergence_curve.json'), 'w', encoding='utf-8') as f:
            json.dump(convergence_curve, f, indent=2)
    if out_dir is not None and use_ec:
        with open(os.path.join(out_dir, 'experiment_config.json'), 'w', encoding='utf-8') as f:
            json.dump(_build_experiment_config(args), f, indent=2, ensure_ascii=False)

    row = {
        'seed': int(args.seed),
        'split_seed': int(getattr(args, 'split_seed', args.seed)),
        'best_epoch': int(best_epoch),
        'total_train_time_sec': float(wall_train_s),
        'time_per_epoch_sec': float(wall_train_s) / max(int(epochs_run), 1),
        'cum_test_time_sec': float(cum_test_time),
    }
    if args.task == 'nc':
        row['test_acc'] = _to_float(best_test_metrics.get('acc', float('nan')))
        row['test_macro_f1'] = _to_float(best_test_metrics.get('macro_f1', float('nan')))
        row['test_micro_f1'] = _to_float(best_test_metrics.get('micro_f1', float('nan')))
    else:
        row['test_roc'] = _to_float(best_test_metrics.get('roc', float('nan')))
        row['test_ap'] = _to_float(best_test_metrics.get('ap', float('nan')))
        row['test_acc'] = _to_float(best_test_metrics.get('acc', float('nan')))
        row['test_macro_f1'] = _to_float(best_test_metrics.get('macro_f1', float('nan')))
        row['test_micro_f1'] = _to_float(best_test_metrics.get('micro_f1', float('nan')))
        row['test_pair_acc'] = _to_float(best_test_metrics.get('pair_acc', float('nan')))

    if out_dir is not None:
        with open(os.path.join(out_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(row, f, indent=2)

    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'),
                best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(),
                        open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(_args_json_safe(args), f, indent=2, default=str)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")
    elif out_dir is not None and use_ec:
        with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(_args_json_safe(args), f, indent=2, default=str)

    return row


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
