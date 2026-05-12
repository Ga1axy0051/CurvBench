import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected

from cores.models import GraphGlue
from torch_geometric.data import Data
from cores.layers import ActivateModule
from utils import search_triangles
from utils.math import diagonal_metric, matrix_log_diag, knn_graphs


class GraphGlueAdapter(nn.Module):
    def __init__(self, configs,
                 feature_dim,
                 pretrained_model: GraphGlue,
                 task_type: str,
                 num_cls: int):
        """

        :param configs: PretrainConfig
        :param feature_dim:
        :param pretrained_model:
        :param task_type: [node_cls, graph_cls, edge_cls]
        :param num_cls: classes number
        """
        super(GraphGlueAdapter, self).__init__()
        assert task_type in ["node_cls", "graph_cls", "link_cls"], "the task type must be one of [node_cls, graph_cls, link_cls]"
        self.configs = configs
        self.input_lin = nn.Linear(feature_dim, configs.in_dim)
        self.pretrained_model = pretrained_model
        self.pretrained_model.frozen()
        self.prompt_z = nn.Parameter(torch.empty(configs.hid_dim, configs.hid_dim))
        nn.init.orthogonal_(self.prompt_z.data)

        num_datasets = len(configs.pretrain_single_graph_data) + len(configs.pretrain_multi_graph_data)
        self.gated_func = nn.Sequential(
            nn.Linear(configs.hid_dim, configs.hid_dim, bias=configs.bias),
            nn.Dropout(configs.drop),
            ActivateModule(configs.act_str),
            nn.Linear(configs.hid_dim, num_datasets, bias=configs.bias),
        )

        self.align_coef = configs.align_coef
        self.align_knn = configs.align_knn
        self.align_samples = configs.align_samples
        self.head = ADAPTERS[task_type](configs.hid_dim + 2 * configs.num_generators, num_cls, configs.drop)

    def forward(self, graph: Data):
        graph.x = self.input_lin(graph.x)
        z, z_tan = self.pretrained_model(graph)

        z_adapt = z @ self.prompt_z
        z_tan_adapt = z_tan @ self.prompt_z

        metric_adapt = diagonal_metric(z_tan_adapt)
        log_metric_adapt = matrix_log_diag(metric_adapt) # [*, M]

        _, proto_z, proto_metric = self.pretrained_model.get_all_prototypes() # [K, M]
        weights = self.gated_func(z_adapt).softmax(-1)    # [*, K]
        log_metric_align = weights @ proto_metric   # [*, M]
        z_log_metric_adapt = torch.concat([z_adapt, log_metric_adapt, log_metric_align], dim=-1)

        holo_loss, curv_loss = self.geometric_transfer_metric(z_adapt, metric_adapt, proto_z, proto_metric)
        pred = self.head(z_log_metric_adapt, graph)
        return pred, self.align_coef * holo_loss, self.align_coef *  curv_loss

    def geometric_transfer_metric(self, z, metric, z_proto, proto_metric):
        N = z.shape[0]
        K = z_proto.shape[0]
        weights = z @ z_proto.t()   # [N, K]
        knn_edge_index, _ = knn_graphs(weights, self.align_knn, return_weight=True, is_to_undirected=False)
        src, dst = knn_edge_index[0], knn_edge_index[1]
        dst += N
        knn_edge_index = to_undirected(torch.stack([src, dst], dim=0), num_nodes=N + K)
        proto_idx = torch.arange(K).to(z.device) + N
        proto_src = proto_idx.unsqueeze(1).expand(-1, K).reshape(-1)
        proto_dst = proto_idx.unsqueeze(0).expand(K, -1).reshape(-1)
        mask = proto_src != proto_dst
        proto_edge_index = torch.stack([proto_src[mask], proto_dst[mask]], dim=0)  # shape: [2, K*(K-1)]
        edge_index = torch.concat([knn_edge_index, proto_edge_index], dim=-1)
        paths = search_triangles(edge_index, num_path_samples=self.align_samples)
        holo_loss, curv_loss = self.pretrained_model.gluing_loss_from_metric(torch.concat([metric, proto_metric], dim=0), paths[0])
        return holo_loss, curv_loss


class GraphClassificationHead(nn.Module):
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(GraphClassificationHead, self).__init__()
        self.head = nn.Linear(hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        return self.head(z)


class LinkClassificationHead(nn.Module):
    """
    For knowledge graph link prediction (edge classification / triple scoring)
    Using dot product or bilinear scoring.
    """
    def __init__(self, hid_dim: int, num_classes: int, drop: float = 0.2):
        super(LinkClassificationHead, self).__init__()
        self.score_fn = nn.Bilinear(hid_dim, hid_dim, num_classes)
        self.drop = nn.Dropout(drop)

    def forward(self, z: torch.Tensor, graph: Data):
        z = self.drop(z)
        src_emb = z[::2]
        dst_emb = z[1::2]
        src_emb = F.normalize(src_emb, p=2, dim=1)
        dst_emb = F.normalize(dst_emb, p=2, dim=1)
        return self.score_fn(src_emb, dst_emb)


ADAPTERS = {
    'node_cls': GraphClassificationHead,
    'graph_cls': GraphClassificationHead,
    'link_cls': LinkClassificationHead,
}