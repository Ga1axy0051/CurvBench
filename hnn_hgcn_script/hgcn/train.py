from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from sklearn.metrics import accuracy_score, f1_score
from config import parser
from models.base_models import NCModel, LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics

if os.environ.get('HGCN_DETECT_ANOMALY') == '1':
    torch.autograd.set_detect_anomaly(True)


def ensure_runtime_env():
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ.setdefault('DATAPATH', os.path.join(project_root, 'data'))
    os.environ.setdefault('LOG_DIR', os.path.join(project_root, 'logs'))


def scalarize_metric_value(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.generic):
        return float(value)
    return float(value) if isinstance(value, (int, float)) else value


def normalize_metrics(metrics):
    if metrics is None:
        return None
    return {key: scalarize_metric_value(value) for key, value in metrics.items()}


def summarize_nc_split(model, data, embeddings, split):
    idx = data[f'idx_{split}']
    with torch.no_grad():
        output = model.decode(embeddings, data['adj_train_norm'], idx)
        preds = torch.argmax(output, dim=1).detach().cpu().numpy()
        labels = data['labels'][idx].detach().cpu().numpy()

    acc = float(accuracy_score(labels, preds))
    micro_f1 = float(f1_score(labels, preds, average='micro', zero_division=0))
    macro_f1 = float(f1_score(labels, preds, average='macro', zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, average='weighted', zero_division=0))
    return {
        f'{split}_acc': acc,
        f'{split}_micro_f1': micro_f1,
        f'{split}_macro_f1': macro_f1,
        f'{split}_weighted_f1': weighted_f1,
        f'{split}_wf1': weighted_f1,
        f'{split}_mf1': macro_f1,
    }


def format_nc_final_summary(dataset, val_summary, test_summary):
    ordered_keys = [
        'val_acc',
        'val_micro_f1',
        'val_macro_f1',
        'val_weighted_f1',
        'val_wf1',
        'val_mf1',
        'test_acc',
        'test_micro_f1',
        'test_macro_f1',
        'test_weighted_f1',
        'test_wf1',
        'test_mf1',
    ]
    payload = {'task': 'nc', 'dataset': dataset}
    payload.update(val_summary or {})
    payload.update(test_summary or {})
    return 'FINAL_SUMMARY ' + ' '.join(
        f'{key}={payload[key]:.6f}' if isinstance(payload[key], float) else f'{key}={payload[key]}'
        for key in ['task', 'dataset', *ordered_keys]
        if key in payload
    )


def has_valid_best_metrics(model, best_val_metrics):
    return bool(best_val_metrics) and model.has_improved(model.init_metric_dict(), best_val_metrics)


def move_optimizer_state_to_device(optimizer, device):
    if str(device) == 'cpu':
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def checkpoint_path(save_dir):
    return os.path.join(save_dir, 'checkpoint_last.pt')


def save_checkpoint(
    save_dir,
    epoch,
    model,
    optimizer,
    lr_scheduler,
    counter,
    best_val_metrics,
    best_test_metrics,
    best_val_summary,
    best_test_summary,
    best_emb,
    completed,
):
    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'counter': int(counter),
            'best_val_metrics': best_val_metrics,
            'best_test_metrics': best_test_metrics,
            'best_val_summary': best_val_summary,
            'best_test_summary': best_test_summary,
            'best_emb': best_emb.detach().cpu() if torch.is_tensor(best_emb) else best_emb,
            'completed': bool(completed),
        },
        checkpoint_path(save_dir),
    )


def load_checkpoint(save_dir, model, optimizer, lr_scheduler, device):
    path = checkpoint_path(save_dir)
    if not os.path.exists(path):
        return None
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    lr_scheduler.load_state_dict(state['scheduler_state_dict'])
    move_optimizer_state_to_device(optimizer, device)
    return state


def train(args):
    ensure_runtime_env()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    save_dir = None
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                logging.StreamHandler(),
            ],
        )

    logging.info(f'Using: {args.device}')
    logging.info('Using seed {}.'.format(args.seed))

    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
    if args.task == 'nc':
        Model = NCModel
        args.n_classes = int(data['labels'].max() + 1)
        logging.info(f'Num classes: {args.n_classes}')
    elif args.task == 'lp':
        args.nb_false_edges = len(data['train_edges_false'])
        args.nb_edges = len(data['train_edges'])
        Model = LPModel
    else:
        raise ValueError(f'Unsupported task: {args.task}')

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma),
    )
    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f'Total number of parameters: {tot_params}')
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)

    t_total = time.time()
    counter = 0
    best_val_metrics = normalize_metrics(model.init_metric_dict())
    best_test_metrics = None
    best_val_summary = None
    best_test_summary = None
    best_emb = None
    start_epoch = 0
    completed_from_checkpoint = False

    if args.save and int(getattr(args, 'resume', 1)) and save_dir:
        state = load_checkpoint(save_dir, model, optimizer, lr_scheduler, args.device)
        if state is not None:
            start_epoch = int(state.get('epoch', -1)) + 1
            counter = int(state.get('counter', 0))
            best_val_metrics = state.get('best_val_metrics') or best_val_metrics
            best_test_metrics = state.get('best_test_metrics')
            best_val_summary = state.get('best_val_summary')
            best_test_summary = state.get('best_test_summary')
            best_emb = state.get('best_emb')
            completed_from_checkpoint = bool(state.get('completed', False))
            logging.info(f'Resumed from checkpoint at epoch {start_epoch}.')

    if not completed_from_checkpoint:
        recovered_from_instability = False
        for epoch in range(start_epoch, args.epochs):
            t = time.time()
            try:
                model.train()
                optimizer.zero_grad()
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                train_metrics = model.compute_metrics(embeddings, data, 'train')
                train_metrics['loss'].backward()
                if args.grad_clip is not None:
                    max_norm = float(args.grad_clip)
                    all_params = list(model.parameters())
                    for param in all_params:
                        torch.nn.utils.clip_grad_norm_(param, max_norm)
                optimizer.step()
                lr_scheduler.step()
                if (epoch + 1) % args.log_freq == 0:
                    logging.info(' '.join([
                        'Epoch: {:04d}'.format(epoch + 1),
                        'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                        format_metrics(train_metrics, 'train'),
                        'time: {:.4f}s'.format(time.time() - t),
                    ]))

                if (epoch + 1) % args.eval_freq == 0:
                    model.eval()
                    embeddings = model.encode(data['features'], data['adj_train_norm'])
                    val_metrics = model.compute_metrics(embeddings, data, 'val')
                    val_metrics_state = normalize_metrics(val_metrics)
                    if (epoch + 1) % args.log_freq == 0:
                        logging.info(' '.join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
                    if model.has_improved(best_val_metrics, val_metrics_state):
                        test_metrics_raw, _, _ = model.compute_metrics(embeddings, data, 'test')
                        best_test_metrics = normalize_metrics(test_metrics_raw)
                        if args.task == 'nc':
                            best_val_summary = summarize_nc_split(model, data, embeddings, 'val')
                            best_test_summary = summarize_nc_split(model, data, embeddings, 'test')
                        best_emb = embeddings.detach().cpu()
                        if args.save:
                            np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.numpy())
                        best_val_metrics = val_metrics_state
                        counter = 0
                    else:
                        counter += 1
                        if counter == args.patience and epoch > args.min_epochs:
                            logging.info('Early stopping')
                            if args.save:
                                save_checkpoint(
                                    save_dir,
                                    epoch,
                                    model,
                                    optimizer,
                                    lr_scheduler,
                                    counter,
                                    best_val_metrics,
                                    best_test_metrics,
                                    best_val_summary,
                                    best_test_summary,
                                    best_emb,
                                    completed=False,
                                )
                            break

                if args.save:
                    save_checkpoint(
                        save_dir,
                        epoch,
                        model,
                        optimizer,
                        lr_scheduler,
                        counter,
                        best_val_metrics,
                        best_test_metrics,
                        best_val_summary,
                        best_test_summary,
                        best_emb,
                        completed=False,
                    )
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                if has_valid_best_metrics(model, best_val_metrics):
                    recovered_from_instability = True
                    logging.warning(
                        'Numerical instability at epoch %d after valid validation metrics were found. '
                        'Keeping best checkpoint metrics and stopping early. Error: %s',
                        epoch + 1,
                        exc,
                    )
                    break
                raise

    logging.info('Optimization Finished!')
    logging.info('Total time elapsed: {:.4f}s'.format(time.time() - t_total))
    if not has_valid_best_metrics(model, best_val_metrics):
        raise RuntimeError('Training finished without any valid validation metrics.')
    if not best_test_metrics:
        model.eval()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        best_emb = embeddings.detach().cpu()
        test_metrics_raw, _, _ = model.compute_metrics(embeddings, data, 'test')
        best_test_metrics = normalize_metrics(test_metrics_raw)
        if args.task == 'nc':
            best_val_summary = summarize_nc_split(model, data, embeddings, 'val')
            best_test_summary = summarize_nc_split(model, data, embeddings, 'test')
    elif args.task == 'nc' and (best_val_summary is None or best_test_summary is None):
        model.eval()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        best_val_summary = summarize_nc_split(model, data, embeddings, 'val')
        best_test_summary = summarize_nc_split(model, data, embeddings, 'test')
    logging.info(' '.join(['Val set results:', format_metrics(best_val_metrics, 'val')]))
    logging.info(' '.join(['Test set results:', format_metrics(best_test_metrics, 'test')]))
    if args.task == 'nc':
        logging.info(format_nc_final_summary(args.dataset, best_val_summary, best_test_summary))
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        save_checkpoint(
            save_dir,
            args.epochs - 1,
            model,
            optimizer,
            lr_scheduler,
            counter,
            best_val_metrics,
            best_test_metrics,
            best_val_summary,
            best_test_summary,
            best_emb,
            completed=True,
        )
        logging.info(f'Saved model in {save_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
