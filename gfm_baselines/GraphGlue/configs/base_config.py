from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from argparse import ArgumentParser
import yaml
import os


@dataclass
class ModelConfig:
    """Shared pretraining datasets"""
    pretrain_single_graph_data: List[str] = None
    pretrain_multi_graph_data: List[str] = None
    root: str = "./datasets"
    k_hops: int = 2
    num_neighbors: Optional[List[int]] = None
    """For Node2Vec, data like KG that without node features"""
    nv_dim: int = 128
    nv_batch_size: int = 128
    nv_walk_length: int = 20
    nv_context_size: int = 10
    nv_lr: float = 0.01
    nv_walks_per_node: int = 10
    nv_p: float = 1.0
    nv_q: float = 1.0
    nv_num_epochs: int = 100

    """Shared model architecture configuration"""

    n_layers: int = 2
    in_dim: int = 128
    hid_dim: int = 256
    att_dim: int = 512
    num_generators: int = 32
    bias: bool = True
    act_str: str = "gelu"

    conv_name: str = "gcn"
    normalize: bool = True
    norm_str: str = "layer_norm"

    temperature: float = 1.0
    ema_alpha: float = 0.99
    geo_regular_coef: float = 0.1

    knn: int = 5

    """Shared Loader"""
    num_workers: int = 2
    seed: int = 42


def add_model_config(parser: ArgumentParser):
    """Add shared model architecture arguments"""
    group = parser.add_argument_group("Model Architecture")
    parser.add_argument("--root", type=str, default="./datasets", help="Root directory for datasets.")
    group.add_argument('--pretrain_single_graph_data', type=str, nargs='+',
                       default=["ogbn-arxiv", "Reddit", "FB15k_237"],
                       help='node-level pretraining datasets, including local datasets under root/<dataset_name>/')
    group.add_argument('--pretrain_multi_graph_data', type=str, nargs='*',
                       default=["PROTEINS", "HIV"],
                       help='graph-level pretraining datasets; pass an empty list for node-only pretraining')
    parser.add_argument('--k_hops', type=int, default=2,
                        help='subgraph sample hops <= len(num_neighbors)')
    parser.add_argument('--num_neighbors', type=int, nargs="+", default=[10, 10],
                        help='maximum number of nodes per graph')

    # Node2Vec parameters (for KGs without node features)
    parser.add_argument('--nv_dim', type=int, default=64,
                        help='dimension of node2vec embedding')
    parser.add_argument('--nv_batch_size', type=int, default=128,
                        help='Batch size for Node2Vec training (default: 128)')
    parser.add_argument('--nv_walk_length', type=int, default=20,
                        help='Length of random walks in Node2Vec (default: 20)')
    parser.add_argument('--nv_context_size', type=int, default=10,
                        help='Context size for context-target prediction (default: 10)')
    parser.add_argument('--nv_lr', type=float, default=0.01,
                        help='Learning rate for Node2Vec optimizer (default: 0.01)')
    parser.add_argument('--nv_walks_per_node', type=int, default=10,
                        help='Number of random walks per node (default: 10)')
    parser.add_argument('--nv_p', type=float, default=1.0,
                        help='Return parameter in Node2Vec (default: 1.0)')
    parser.add_argument('--nv_q', type=float, default=1.0,
                        help='In-out parameter in Node2Vec (default: 1.0)')
    parser.add_argument('--nv_num_epochs', type=int, default=100,
                        help='Number of epochs to train Node2Vec (default: 100)')

    # model configurations
    group.add_argument('--n_layers', type=int, default=2,
                       help='Number of GNN layers')
    group.add_argument('--in_dim', type=int, default=128,
                       help='Input feature dimension')
    group.add_argument('--hid_dim', type=int, default=512,
                       help='Hidden dimension')
    group.add_argument('--att_dim', type=int, default=512,
                       help='Attention dimension (if used)')
    group.add_argument('--num_generators', type=int, default=32,
                       help='Number of generators in FM')
    group.add_argument('--conv_name', type=str, default='gcn', choices=['gcn', 'sage', 'gin'],
                       help='GNN layer type')
    group.add_argument('--act_str', type=str, default='relu',
                       help='Activation function')
    group.add_argument('--normalize', action='store_true',
                       help='Whether to normalize adjacency matrix')
    group.add_argument('--bias', action='store_false',
                       help='Whether to add bias term')
    group.add_argument('--norm_str', type=str, default='layer_norm', choices=['layer_norm', 'batch_norm'],
                       help="Normalization type")
    group.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature')
    group.add_argument('--ema_alpha', type=float, default=0.99,
                        help='Exponential moving average coefficient')
    group.add_argument('--geo_regular_coef', type=float, default=0.1,
                       help='Regularization coefficient of PT')

    parser.add_argument('--knn', type=int, default=5,
                        help='KNN graph connections for inter-graph loss')

    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible training and few-shot splits')
    return parser


def save_config_to_yaml(config: ModelConfig, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    exclude_keys = {'resume_checkpoint', 'resume_temp_checkpoint'}
    filtered_config = {k: v for k, v in config.__dict__.items() if k not in exclude_keys}
    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(filtered_config, f, default_flow_style=False, indent=2, sort_keys=False)
    print(f"Config saved to {filepath}")


def load_config_from_yaml(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Config file not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)