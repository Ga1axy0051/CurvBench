from __future__ import annotations
from typing import Optional, Literal
import torch
from tqdm import tqdm


@torch.no_grad()
def sectional_curvature_gpu(
    adjacency_matrix: torch.Tensor,
    distance_matrix: torch.Tensor,
    *,
    relative: bool = True,
    device: Optional[torch.device | str] = None,
    mode: Literal["fast", "strict"] = "fast",   #  新增：切换计算模式
    m_chunk_size: int = 4,                      # strict 模式的节点批次
    pair_chunk_size: int = 2048,                # (b,c) 对分块大小
    show_progress: bool = True,
) -> torch.Tensor:
    """
    GPU 加速版截面曲率估计，支持两种模式：
      - mode='fast'   ：轻量快速版（适合中小图，显存低）
      - mode='strict' ：优化内存访问与累计精度（大图或实验统计）

    公式：
      ξ(a,b,c;m) = [ d(a,m)^2 + d(b,c)^2/4 - (d(a,b)^2 + d(a,c)^2)/2 ] / (2 d(a,m))

    参数
    ----
    adjacency_matrix : torch.Tensor(n, n)
        图的邻接矩阵（二值或 bool）
    distance_matrix : torch.Tensor(n, n)
        成对距离矩阵（最短路距离或欧氏距离）
    relative : bool
        是否按 max(D) 归一化（与原版一致）
    device : str | torch.device
        计算设备；默认为输入张量所在设备
    mode : {'fast', 'strict'}
        选择快速或严格模式
    m_chunk_size : int
        strict 模式下节点批大小（1 最稳，>1 更快）
    pair_chunk_size : int
        邻居对 (b,c) 的分块大小（显存足可调大）
    show_progress : bool
        是否显示进度条

    返回
    ----
    node_curvatures : torch.Tensor(n,)
        每个节点 m 的平均截面曲率
    """
    A = adjacency_matrix
    D = distance_matrix
    assert A.ndim == 2 and D.ndim == 2 and A.shape == D.shape, "A, D 必须同型二维方阵"
    n = A.shape[0]

    # ---------------------- 设备放置 ----------------------
    if device is None:
        device = A.device
    else:
        device = torch.device(device)
        A = A.to(device, non_blocking=True)
        D = D.to(device, non_blocking=True)

    node_curv = torch.zeros(n, dtype=torch.float32, device=device)
    all_a = torch.arange(n, device=device)

    # ======================================================
    #  模式一：FAST —— 简洁快速版
    # ======================================================
    if mode == "fast":
        neighbor_lists = [torch.nonzero(A[i] > 0, as_tuple=False).flatten() for i in range(n)]
        iterator = tqdm(range(n), desc="Computing node curvatures (fast)", ncols=90) if show_progress else range(n)

        for m in iterator:
            neighbors = neighbor_lists[m]
            if neighbors.numel() < 2:
                continue

            pairs = torch.combinations(neighbors, r=2)
            am = D[:, m]
            denom = 2.0 * am
            valid_a_mask = (all_a != m) & (denom > 1e-12) & torch.isfinite(denom)
            if not valid_a_mask.any():
                continue

            am2 = am[valid_a_mask] ** 2
            denom = denom[valid_a_mask]

            sum_over_pairs = 0.0
            counted_pairs = 0

            for p_start in range(0, pairs.shape[0], pair_chunk_size):
                p_end = min(p_start + pair_chunk_size, pairs.shape[0])
                chunk = pairs[p_start:p_end]
                b_idx, c_idx = chunk[:, 0], chunk[:, 1]

                Dab2 = (D[valid_a_mask][:, b_idx] ** 2)
                Dac2 = (D[valid_a_mask][:, c_idx] ** 2)
                Dbc2_quarter = (D[b_idx, c_idx] ** 2) * 0.25
                numer = am2[:, None] + Dbc2_quarter[None, :] - 0.5 * (Dab2 + Dac2)
                curv_mat = numer / (denom[:, None])
                per_pair_mean = curv_mat.mean(dim=0)

                sum_over_pairs += per_pair_mean.sum()
                counted_pairs += per_pair_mean.numel()

                del Dab2, Dac2, numer, curv_mat, per_pair_mean

            if counted_pairs > 0:
                node_curv[m] = (sum_over_pairs / counted_pairs).float()

    # ======================================================
    #  模式二：STRICT —— 高精度并行优化版
    # ======================================================
    elif mode == "strict":
        m_indices = torch.arange(n, device=device)
        m_steps = range(0, n, m_chunk_size)
        if show_progress:
            m_steps = tqdm(m_steps, total=(n + m_chunk_size - 1) // m_chunk_size,
                           desc="Computing node curvatures (strict)", ncols=90)

        for m_start in m_steps:
            m_end = min(m_start + m_chunk_size, n)
            for m in m_indices[m_start:m_end]:
                nbrs = torch.nonzero(A[m] > 0, as_tuple=False).flatten()
                deg = nbrs.numel()
                if deg < 2:
                    node_curv[m] = 0.0
                    continue

                pairs = torch.combinations(nbrs, r=2)
                am = D[:, m]
                denom = 2.0 * am
                valid_a_mask = (all_a != m) & (denom != 0)
                if not valid_a_mask.any():
                    node_curv[m] = 0.0
                    continue

                am2 = (am ** 2)[valid_a_mask]
                denom_valid = denom[valid_a_mask]
                D_valid = D[valid_a_mask, :]

                sum_over_pairs = torch.zeros((), dtype=torch.float64, device=device)
                num_pairs_total = 0

                for p_start in range(0, pairs.shape[0], pair_chunk_size):
                    p_end = min(p_start + pair_chunk_size, pairs.shape[0])
                    chunk = pairs[p_start:p_end]
                    b_idx, c_idx = chunk[:, 0], chunk[:, 1]

                    Dab2 = (D_valid[:, b_idx] ** 2)
                    Dac2 = (D_valid[:, c_idx] ** 2)
                    Dbc2_quarter = (D[b_idx, c_idx] ** 2) * 0.25

                    numer = am2[:, None] + Dbc2_quarter[None, :] - 0.5 * (Dab2 + Dac2)
                    curv_mat = numer / (denom_valid[:, None])
                    per_pair_mean = curv_mat.mean(dim=0).to(torch.float64)

                    sum_over_pairs += per_pair_mean.sum()
                    num_pairs_total += per_pair_mean.numel()

                    del Dab2, Dac2, numer, curv_mat, per_pair_mean

                node_curv[m] = (sum_over_pairs / max(num_pairs_total, 1)).to(torch.float32)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'fast' or 'strict'.")

    # ======================================================
    #  归一化 + CUDA 同步
    # ======================================================
    if relative:
        # 检查是否存在 inf
        has_inf = torch.isinf(D).any()

        if has_inf:
            # 只取有限值的最大值 —— 但不构造巨大 mask（避免 nonzero 限制）
            # 方案：逐块扫描最大 finite 值（流式，不占大内存）
            maxD = float("-inf")
            chunk = 5000

            for start in range(0, D.size(0), chunk):
                end = min(start + chunk, D.size(0))
                block = D[start:end]

                # 局部 finite 最大值（不会创建大 mask）
                finite_block = block[~torch.isinf(block)]
                if finite_block.numel() > 0:
                    local_max = finite_block.max().item()
                    maxD = max(maxD, local_max)

            if maxD == float("-inf"):
                raise RuntimeError("APSP 距离全是 inf，图可能完全不连通。")

        else:
            # 没有 inf，直接取最大值
            maxD = torch.max(D).item()
        if maxD > 0:
            node_curv = node_curv / maxD

    if device.type == "cuda":
        torch.cuda.synchronize()

    return node_curv