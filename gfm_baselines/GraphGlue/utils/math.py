import torch
from torch_geometric.utils import to_undirected, remove_self_loops

EPS = 1e-6
MAX = 50


def diagonal_metric(basis: torch.Tensor) -> torch.Tensor:
    """

    :param basis: The basis vectors with shape [*, M ,d]
    :return: diagonal metric: [*, M]
    """
    return torch.sum(basis * basis, dim=-1)


def matrix_log_diag(diag_G: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix logarithm for a batch of diagonal positive definite matrices.
    Uses eigenvalue decomposition for numerical stability.

    Args:
        diag_G: Tensor of shape [*, M], diagonal positive definite.

    Returns:
        Log(G): Tensor of same shape as G.
    """
    return torch.log(diag_G.clamp(min=EPS))


def matrix_exp_diag(diag_G: torch.Tensor) -> torch.Tensor:
    """
    Compute the matrix logarithm for a batch of diagonal positive definite matrices.
    Uses eigenvalue decomposition for numerical stability.

    Args:
        diag_G: Tensor of shape [*, M], diagonal positive definite.

    Returns:
        Log(G): Tensor of same shape as G.
    """
    return torch.exp(diag_G.clamp(max=MAX))


def log_volume(basis):
    """
    Log volume of metric tensor w.r.t. the standard basis.
    :param basis: [*, M, d]
    :return: [*]
    """
    diag_G = diagonal_metric(basis)
    log_vol = torch.sum(torch.log(diag_G.clamp(min=EPS)), dim=-1)
    return log_vol


def log_volume_ratio(basis_src, basis_dst):
    """
    Volume ratio between two tangent spaces to estimate Ricci Curvature.
    :param basis_src: [*, M, d]
    :param basis_dst: [*, M, d]

    :return: log ratio: torch.Tensor
    """
    log_vol_src, log_vol_dst = log_volume(basis_src), log_volume(basis_dst)
    log_ratio = log_vol_src - log_vol_dst
    return log_ratio


def parallel_translation(G_i: torch.Tensor, G_j: torch.Tensor) -> torch.Tensor:
    """
    Compute the optimal isometric parallel transport map P such that:
        P^T @ Gj @ P = Gi
    Just for diagonal metric tensor

    Args:
        G_i (torch.Tensor): Metric tensor at node i, shape (*, M)
        G_j (torch.Tensor): Metric tensor at node j, shape (*, M)

    Returns:
        P (torch.Tensor): Optimal parallel transport map, shape (*, M)
    """
    P = torch.sqrt(G_i / G_j.clamp(min=EPS))  # [*, M]
    return P


def knn_graphs(dense_adj: torch.Tensor, top_k: int, dim=-1, return_weight=False, is_to_undirected=False):
    """
    Construct KNN graph for dense adjacency matrix with weights.
    """
    N = dense_adj.shape[0]
    device = dense_adj.device
    topk_vals, topk_indices = dense_adj.topk(k=top_k + 1, dim=dim)
    topk_indices = topk_indices[:, 1:]  # [N, top_k]
    topk_vals = topk_vals[:, 1:]

    row = torch.arange(N, device=device).unsqueeze(1).expand(N, top_k)  # [N, top_k]
    col = topk_indices  # [N, top_k]

    edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)  # [2, N * top_k]

    if is_to_undirected:
        edge_index = to_undirected(edge_index, num_nodes=N)

    if return_weight:
        row, col = edge_index
        edge_weight = dense_adj[row, col]
    else:
        edge_weight = None

    return edge_index, edge_weight


def diag_metric_logmap(G_i, G_j):
    """
    Compute the diagonal metric logmap P: \log_{G_i}(G_j)
    :param G_i: [*, M]
    :param G_j:[*, M]
    :return: tangent vector at G_i
    """
    mid_term = matrix_log_diag(G_j) - matrix_log_diag(G_i)
    return G_i * mid_term