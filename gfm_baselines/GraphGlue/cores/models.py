import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_mean
from cores.layers import NormModule, FeedForwardLayer, GNNLayer
from cores.loss_funcs import ContrastiveLoss, ManifoldGlueLoss
from utils.math import matrix_log_diag, matrix_exp_diag, parallel_translation, diagonal_metric, knn_graphs
from typing import List, Optional, Dict, Tuple, Any, Mapping
import re

EPS = 1e-6


class SparsePerturbation(nn.Module):
    def __init__(self, num_generators, hid_dim, att_dim):
        super(SparsePerturbation, self).__init__()
        self.num_generators = num_generators
        self.generators = nn.Parameter(torch.empty(num_generators, hid_dim))
        nn.init.orthogonal_(self.generators.data)
        self.att_proj = nn.Linear(hid_dim, att_dim)

    def forward(self, x, edge_index, edge_weight, batch, batch_size, knn: int):
        """

        :param x: [N, d]
        :param edge_index: [2, E]
        :param edge_weight: [E,]
        :param batch: [N]
        :param batch_size: mini-batch size
        :param knn: knn for sparsify

        :return: List[Data]
        """
        M = self.num_generators
        B = batch_size

        N = x.shape[0]
        weights = torch.sigmoid(self.att_proj(self.generators).repeat(B, 1) @ self.att_proj(x).t())  # [BM, N]
        knn_edge_index, add_edge_weight = knn_graphs(weights, knn, return_weight=True)
        add_edge_src, add_edge_dst = knn_edge_index[0], knn_edge_index[1]

        add_edge_index = torch.stack([add_edge_src + N, add_edge_dst], dim=0)
        add_edge_index, add_edge_weight = to_undirected(add_edge_index, add_edge_weight, num_nodes=N + B * M)
        new_edge_index = torch.concat([edge_index, add_edge_index], dim=-1)
        new_edge_weight = torch.concat([edge_weight, add_edge_weight], dim=-1)

        # [1,...,N, M1,...,MM, ..., M1,...,MM]
        xp = torch.concat([x, self.generators.repeat(B, 1)], dim=0)  # [N + BM, d]
        add_batch = torch.arange(B, B + B * M, device=x.device)  # [BM]
        new_batch = torch.concat([batch, add_batch], dim=0)  # [N + BM]

        aug_graph = Data(x=xp,
                         edge_index=new_edge_index,
                         edge_weight=new_edge_weight,
                         batch=new_batch,
                         batch_size=batch_size)

        return aug_graph


class PooLedSubgraphGNN(nn.Module):
    def __init__(self, conv_name: str, n_layers: int,
                 in_dim: int, hid_dim: int,
                 normalize: bool = True, bias: bool = True,
                 norm_str: str = "ln", act_str: str = "relu", drop=0.1):
        super().__init__()
        self.convs = nn.ModuleList([
            GNNLayer(conv_name, in_dim, hid_dim,
                     normalize, bias, norm_str, act_str, drop)
        ])
        for _ in range(n_layers - 1):
            self.convs.append(
                GNNLayer(conv_name, hid_dim, hid_dim,
                         normalize, bias, norm_str, act_str, drop))
        self.out_norm = NormModule(norm_str, hid_dim)
        self.out_fc = FeedForwardLayer(hid_dim, hid_dim, hid_dim, bias, act_str, drop)

    def forward(self, graph):
        for conv in self.convs:
            x = conv(graph.x, graph.edge_index, graph.edge_weight)
        x = self.out_norm(x)
        x = global_mean_pool(x, graph.batch)  # [B + BM, d]
        return x


class GraphGlue(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_generators = configs.num_generators
        self.input_lin = nn.Linear(configs.in_dim, configs.hid_dim)
        self.graph_perturb = SparsePerturbation(configs.num_generators,
                                                configs.hid_dim,
                                                configs.att_dim)
        self.encoder = PooLedSubgraphGNN(configs.conv_name, configs.n_layers,
                                         configs.hid_dim, configs.hid_dim,
                                         configs.normalize, configs.bias,
                                         configs.norm_str, configs.act_str, configs.drop)
        datasets_list = configs.pretrain_single_graph_data + configs.pretrain_multi_graph_data
        self.prototype_manager = RiemannianPrototypeManager(datasets_list, configs.hid_dim,
                                                            configs.num_generators,
                                                            configs.ema_alpha,
                                                            configs.temperature)
        self.contra_loss = ContrastiveLoss(configs.temperature)
        self.geo_loss = ManifoldGlueLoss(configs.geo_regular_coef)
        self.knn = configs.knn

    def forward(self, graph: Data):
        """

        :param graph: 1) Feature dimension is unified. 2) BatchData

        :return: node/graph embedding, tangent vectors [torch.Tensor, torch.Tensor] with shape [N, d] [N, M, d]
        """

        x = graph.x.clone()
        B = graph.batch_size

        x = self.input_lin(x)
        aug_graph = self.graph_perturb(x, graph.edge_index, graph.edge_weight, graph.batch, B, self.knn)
        z = self.encoder(aug_graph)
        z_center = z[: B]  # [B, d]
        z_aug = z[B:].reshape(B, -1, z.shape[-1])  # [B, M, d]
        z_tan = z_aug - z_center.unsqueeze(1)  # [B, M, d]
        tan_norm = torch.norm(z_tan, p=2, dim=-1, keepdim=True)  # [B, M, 1]
        z_tan = torch.linalg.qr(z_tan.transpose(-1, -2))[0].transpose(-1, -2) * tan_norm
        return z_center, z_tan

    def local_struct_loss(self, z, z_tan):
        cl_loss = self.contra_loss(z, z.unsqueeze(1) + z_tan)
        return cl_loss

    def manifold_gluing_loss(self, z_tan, triple_paths):
        """

        :param z_tan: [N, M, d]
        :param triple_paths: [3, num_paths]

        :return: loss for each graph batch or all datasets
        """
        if triple_paths.numel() > 0:
            metric = diagonal_metric(z_tan)  # [N, M]
            holo_loss, curv_loss = self.gluing_loss_from_metric(metric, triple_paths)
            geo_loss = holo_loss + curv_loss
        else:
            geo_loss = torch.zeros(1, device=z_tan.device, dtype=z_tan.dtype, requires_grad=True).squeeze()

        return geo_loss

    def gluing_loss_from_metric(self, metric, triple_paths):
        vi, vj, vk = triple_paths[0], triple_paths[1], triple_paths[2]
        metric_i, metric_j, metric_k = metric[vi], metric[vj], metric[vk]  # [T, M]
        metrics = torch.stack([metric_i, metric_j, metric_k], dim=0)  # [3, T, M]
        src_indices = torch.tensor([0, 1, 0], device=metrics.device)
        dst_indices = torch.tensor([1, 2, 2], device=metrics.device)
        pt_matrix = parallel_translation(metrics[src_indices], metrics[dst_indices])  # [3, T, M]

        log_r_matrix_ij = matrix_log_diag(metric_i) - matrix_log_diag(metric_j)  # [T]
        log_r_matrix_jk = matrix_log_diag(metric_j) - matrix_log_diag(metric_k)
        log_r_matrix = torch.stack([log_r_matrix_ij, log_r_matrix_jk], dim=0)  # [2, T]

        holo_loss, curv_loss = self.geo_loss(pt_matrix, log_r_matrix)
        return holo_loss, curv_loss

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True,
                        assign: bool = False):
        proto_z_pattern = re.compile(r'^prototype_manager\.proto_z_(?!tan_)([a-zA-Z0-9_]+)$')
        proto_metric_pattern = re.compile(r'^prototype_manager\.proto_metric_([a-zA-Z0-9_]+)$')
        datasets_to_register = set()

        for key in state_dict.keys():
            match_z = proto_z_pattern.match(key)
            match_metric = proto_metric_pattern.match(key)

            if match_z:
                datasets_to_register.add(match_z.group(1))  # e.g., 'Computers'
            elif match_metric:
                datasets_to_register.add(match_metric.group(1))  # e.g., 'Computers'

        for safe_name in datasets_to_register:
            z_key = f'prototype_manager.proto_z_{safe_name}'
            tan_key = f'prototype_manager.proto_metric_{safe_name}'

            if not hasattr(self.prototype_manager, f'proto_z_{safe_name}'):
                z_tensor = state_dict[z_key].clone()
                self.prototype_manager.register_buffer(f'proto_z_{safe_name}', z_tensor)

            if not hasattr(self.prototype_manager, f'proto_metric_{safe_name}'):
                tan_tensor = state_dict[tan_key].clone()
                self.prototype_manager.register_buffer(f'proto_metric_{safe_name}', tan_tensor)

        super().load_state_dict(state_dict, strict=strict)
        self.prototype_manager.rebuild_cache_from_buffers()

    @torch.no_grad()
    def update_prototype(self, z: torch.Tensor, z_tan: torch.Tensor, data_name_map: torch.Tensor):
        z = z.detach()
        z_tan = z_tan.detach()
        self.prototype_manager.update_prototype(z, z_tan, data_name_map)

    def get_all_prototypes(self):
        return self.prototype_manager.get_all_prototypes()

    def prototype_loss(self, z: torch.Tensor, data_name_map: torch.Tensor):
        return self.prototype_manager.loss(z, data_name_map)

    def frozen(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfrozen(self):
        for param in self.parameters():
            param.requires_grad_(True)

    @staticmethod
    def knn_graph(h: torch.Tensor,
                  top_k,
                  is_cross: bool = False,
                  data_name_map=None,
                  return_weight: bool = False,
                  is_to_undirected: bool = False):
        """
        Construct symmetric KNN graph (undirected, no self-loops).

        :param h: [N, D] node features
        :param top_k: number of neighbors (excluding self)
        :param return_weight: whether to return edge weights

        :return: edge_index [2, E], edge_weight [E] (optional)
        """
        N = h.shape[0]
        if top_k >= N:
            top_k = N - 1

        similarity = h @ h.t()  # [N, N]
        if is_cross:
            assert data_name_map is not None, "data_name_map must be provided"
            group_i = data_name_map.unsqueeze(1)  # (N, 1)
            group_j = data_name_map.unsqueeze(0)  # (1, N)
            mask = group_i != group_j  # (N, N)

            sim_masked = similarity.clone()
            sim_masked[~mask] = float('-inf')
            return knn_graphs(sim_masked, top_k, return_weight=return_weight, is_to_undirected=is_to_undirected)
        else:
            return knn_graphs(similarity, top_k, return_weight=return_weight, is_to_undirected=is_to_undirected)


class RiemannianPrototypeManager(nn.Module):
    """
    EMA Riemannian prototype manager and updater.

    Manages per-dataset prototypes (z and z_tan) with EMA updates.
    Supports contrastive loss between node embeddings and prototypes.
    """

    def __init__(self, datasets_list: List[str],
                 hid_dim: int,
                 num_generators: int,
                 ema_alpha: float = 0.99,
                 temperature: float = 1.0):
        super().__init__()
        self.datasets_list = [re.sub(r'[^a-zA-Z0-9_]', '_', name) for name in datasets_list]
        self.hid_dim = hid_dim
        self.num_generators = num_generators
        self.ema_alpha = ema_alpha
        self.temperature = temperature

        # Runtime caches (not saved in state_dict)
        self._proto_z_dict: Dict[str, torch.Tensor] = {}  # dataset_name -> tensor (on device)
        self._proto_metric_dict: Dict[str, torch.Tensor] = {}  # dataset_name -> tensor (on device)
        self.prototype_keys: List[str] = []  # ordered list of dataset names

        # For safety: keep a mapping from sanitized name to original
        self._sanitized_to_original: Dict[str, str] = {}

    @torch.no_grad()
    def update_prototype(self, z: torch.Tensor, z_tan: torch.Tensor, data_name_map: torch.Tensor):
        """
        Update or initialize prototype for a dataset using EMA.
        """
        dataset_idx = torch.unique(data_name_map).cpu().numpy()
        metric = diagonal_metric(z_tan)  # [N, M]
        log_metric = matrix_log_diag(metric)  # [N, M]
        z_mean = scatter_mean(z, data_name_map, dim=0)
        log_metric_mean = scatter_mean(log_metric, data_name_map, dim=0)  # [K, M]
        for i, dataset_name in enumerate([self.datasets_list[idx] for idx in dataset_idx]):
            if dataset_name not in self._proto_z_dict:
                self._register_new_prototype(dataset_name, z_mean[i], log_metric_mean[i])
            else:
                alpha = self.ema_alpha
                proto_z = self._proto_z_dict[dataset_name]
                proto_metric = self._proto_metric_dict[dataset_name]
                log_new_metric = alpha * matrix_log_diag(proto_metric) + (1 - alpha) * log_metric_mean[i]

                # In-place EMA update
                proto_z.copy_(alpha * proto_z + (1 - alpha) * z_mean[i])
                proto_metric.copy_(matrix_exp_diag(log_new_metric))

    def _register_new_prototype(self, dataset_name: str, z_mean: torch.Tensor, log_metric_mean: torch.Tensor):
        """
        Register a new prototype as buffer and update caches.
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', dataset_name)

        # Clone and detach
        p_z = z_mean.detach().clone()
        p_metric = matrix_exp_diag(log_metric_mean.detach().clone())

        # Register as persistent buffers
        self.register_buffer(f'proto_z_{safe_name}', p_z)
        self.register_buffer(f'proto_metric_{safe_name}', p_metric)

        # Cache original name -> tensor
        # Note: getattr is safe here because we just registered it
        self._proto_z_dict[dataset_name] = getattr(self, f'proto_z_{safe_name}')
        self._proto_metric_dict[dataset_name] = getattr(self, f'proto_metric_{safe_name}')

        if dataset_name not in self.prototype_keys:
            self.prototype_keys.append(dataset_name)
            self._sanitized_to_original[safe_name] = dataset_name

    def get_prototype(self, dataset_name: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get prototype for a specific dataset.
        """
        return (
            self._proto_z_dict.get(dataset_name),
            self._proto_metric_dict.get(dataset_name)
        )

    def get_all_prototypes(self) -> Tuple[List[str], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get all prototypes as stacked tensors.
        Returns:
            (names, all_z: [K, d], all_metric: [K, M, M])
        """
        names = [name for name in self.datasets_list if name in self.prototype_keys]
        all_z = torch.stack([self._proto_z_dict[name] for name in names], dim=0)
        all_metric = torch.stack([self._proto_metric_dict[name] for name in names], dim=0)
        return names, all_z, all_metric

    def loss(self, z: torch.Tensor, data_name_map: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between node embeddings and all prototypes.
        """
        names, all_proto_z, _ = self.get_all_prototypes()

        all_proto_z = all_proto_z.to(z.device)
        sim = torch.mm(z, all_proto_z.t()) / self.temperature

        loss = F.cross_entropy(sim, data_name_map)
        return loss

    @torch.no_grad()
    def rebuild_cache_from_buffers(self):
        """
        Rebuild _proto_z_dict and prototype_keys from registered buffers.
        Called by parent module after loading state_dict.
        """
        self._proto_z_dict.clear()
        self._proto_metric_dict.clear()
        self.prototype_keys.clear()
        self._sanitized_to_original.clear()

        prefix = 'proto_z_'
        tan_prefix = 'proto_metric_'

        for name, buffer in self.named_buffers():
            if name.startswith(prefix) and not name.startswith(tan_prefix):
                safe_name = name[len(prefix):]
                try:
                    # Try to recover original name from previous mapping
                    original_name = self._sanitized_to_original.get(safe_name, safe_name)
                except:
                    original_name = safe_name  # fallback

                # Get both buffers
                z_buf = getattr(self, name)
                tan_name = f'{tan_prefix}{safe_name}'
                if hasattr(self, tan_name):
                    z_tan_buf = getattr(self, tan_name)
                else:
                    raise RuntimeError(f"Missing tangent prototype buffer: {tan_name}")

                self._proto_z_dict[original_name] = z_buf
                self._proto_metric_dict[original_name] = z_tan_buf
                if original_name not in self.prototype_keys:
                    self.prototype_keys.append(original_name)
                    self._sanitized_to_original[safe_name] = original_name

    def extra_repr(self) -> str:
        return (f"datasets_list={self.datasets_list}, hid_dim={self.hid_dim}, num_generators={self.num_generators}, "
                f"ema_alpha={self.ema_alpha}, temperature={self.temperature}")