import argparse
import copy
import json
import os
import random

import numpy as np
import torch

from exp import Exp
from logger import create_logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_p100_cuda_compat_error(exc: Exception) -> bool:
    message = str(exc)
    keywords = [
        'no kernel image is available for execution on the device',
        'cudaerrornokernelimagefordevice',
        'acceleratorerror',
    ]
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in keywords)


parser = argparse.ArgumentParser(description='')

# Experiment settings
parser.add_argument('--downstream_task', type=str, default='NC', choices=['NC', 'LP'])
parser.add_argument('--dataset', type=str)
parser.add_argument('--root_path', type=str, default='./datasets')
parser.add_argument('--in_features', type=int)
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--exp_iters', type=int, default=10)
parser.add_argument('--version', type=str, default='Train')
parser.add_argument('--log_path', type=str)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=3047)
parser.add_argument('--split_seed', type=int, default=3047)

# Riemannian Embeds
parser.add_argument('--num_factors', type=int, default=5)
parser.add_argument('--init_curvs', type=float, nargs='+', default=[-3, -1, 0, 1, 3])
parser.add_argument('--backbone', type=str, default='gcn', choices=['gcn', 'gat', 'sage'])
parser.add_argument('--hidden_features', type=int, default=64)
parser.add_argument('--embed_features', type=int, default=32, help='dimensions of graph embedding')
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--lr_Riemann', type=float, default=0.01)
parser.add_argument('--w_decay', type=float, default=5e-4)
parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')

# Gating
parser.add_argument('--sample_hop', type=int, nargs='+', default=[2, 3])
parser.add_argument('--lr_gating', type=float, default=0.01)
parser.add_argument('--w_decay_gating', type=float, default=5e-4)
parser.add_argument('--coef_dis', type=float, default=0.1)

# Link Prediction
parser.add_argument('--epochs_lp', type=int, default=5000)
parser.add_argument('--patience_lp', type=int, default=100)
parser.add_argument('--min_epoch_lp', type=int, default=200)
parser.add_argument('--t', type=float, default=1.0, help='for Fermi-Dirac decoder')
parser.add_argument('--r', type=float, default=2.0, help='Fermi-Dirac decoder')

# Node Classification
parser.add_argument('--drop_cls', type=float, default=0.0)
parser.add_argument('--drop_edge_cls', type=float, default=0.0)
parser.add_argument('--hidden_features_cls', type=int, default=32)
parser.add_argument('--num_factors_cls', type=int, default=5)
parser.add_argument('--lr_cls', type=float, default=0.01)
parser.add_argument('--w_decay_cls', type=float, default=5e-4)
parser.add_argument('--epochs_cls', type=int, default=5000)
parser.add_argument('--patience_cls', type=int, default=100)
parser.add_argument('--min_epoch_cls', type=int, default=200)

# GPU
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--force_cpu', action='store_true')
parser.add_argument('--resume', type=int, default=1)
parser.add_argument('--skip_lp_warmup_nc', type=int, default=1)

configs = parser.parse_args()
configs.num_factors = len(configs.init_curvs)
configs.num_factors_cls = configs.num_factors

set_seed(configs.seed)

if configs.force_cpu:
    configs.gpu = -1

if configs.save_dir:
    os.makedirs(configs.save_dir, exist_ok=True)
    configs.log_path = configs.log_path or os.path.join(configs.save_dir, 'log.txt')
else:
    results_dir = os.path.join('./results', configs.version)
    os.makedirs(results_dir, exist_ok=True)
    configs.log_path = configs.log_path or os.path.join(
        results_dir,
        f'{configs.downstream_task}_{configs.backbone}_{configs.dataset}.log',
    )

logger = create_logger(configs.log_path)
logger.info(configs)

if configs.save_dir:
    with open(os.path.join(configs.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(configs), f, indent=2, ensure_ascii=False)

try:
    exp = Exp(configs)
    exp.train()
except Exception as exc:
    if configs.force_cpu or configs.gpu < 0 or not is_p100_cuda_compat_error(exc):
        raise

    logger.warning('Detected CUDA kernel compatibility issue on current GPU. Falling back to CPU.')
    logger.warning(str(exc))
    fallback_configs = copy.deepcopy(configs)
    fallback_configs.gpu = -1
    fallback_configs.force_cpu = True

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    set_seed(fallback_configs.seed)
    exp = Exp(fallback_configs)
    exp.train()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
