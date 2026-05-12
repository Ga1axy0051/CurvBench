import torch
import time
import gc
from torch_geometric.utils import is_undirected, to_undirected, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes


def search_triangles(
    edge_index,
    num_path_samples: int = 1000,
    path_sample_times: int = 1,
    return_relabel_mapping: bool = False,
    sampling_strategy: str = 'weighted',  # 'weighted' or 'uniform'
    verbose: bool = False
):
    """We only search the edge pairs instead of search full triangles"""
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index)
    device = edge_index.device

    if verbose:
        E = edge_index.size(1)
        N = maybe_num_nodes(edge_index)
        print(f"[INFO] Graph Scale: {N} nodes, {E} edges")
        start_time = time.time()

    with torch.no_grad():
        row, col = edge_index
        num_nodes = maybe_num_nodes(edge_index)

        deg = degree(row, num_nodes=num_nodes, dtype=torch.long)
        ptr = torch.cat([torch.zeros(1, dtype=torch.long, device=device), deg.cumsum(0)])
        idx = torch.argsort(row)
        col_sorted = col[idx]

        candidate_mask = deg >= 2
        candidate_nodes = torch.where(candidate_mask)[0]
        candidate_deg = deg[candidate_mask]

        if len(candidate_nodes) == 0:
            empty_shape = (path_sample_times, 3, 0)
            result = torch.empty(empty_shape, dtype=torch.long, device=device)
            if return_relabel_mapping:
                return result, torch.empty(0, dtype=torch.long, device=device), {}
            return result

        if sampling_strategy == 'weighted':
            sample_weights = candidate_deg.float()
            sample_weights = sample_weights / sample_weights.sum()
        else:
            sample_weights = None

        all_sampled_paths = []

        for sample_round in range(path_sample_times):
            if num_path_samples <= 0:
                all_sampled_paths.append(torch.empty((3, 0), dtype=torch.long, device=device))
                continue

            sampled_paths = torch.empty((num_path_samples, 3), dtype=torch.long, device=device)
            filled = 0
            attempts = 0
            max_attempts = num_path_samples * 10

            batch_multiplier = 5

            while filled < num_path_samples and attempts < max_attempts:
                remaining = num_path_samples - filled
                dynamic_batch = min(4096, max(256, remaining * batch_multiplier))

                j_indices = torch.multinomial(
                    sample_weights if sample_weights is not None else torch.ones(len(candidate_nodes), device=device),
                    dynamic_batch,
                    replacement=True
                )
                j_nodes = candidate_nodes[j_indices]

                i_nodes, k_nodes = sample_two_neighbors(ptr, col_sorted, j_nodes, device)

                valid_mask = (i_nodes != k_nodes) & (i_nodes != j_nodes) & (k_nodes != j_nodes)

                if valid_mask.any():
                    valid_i = i_nodes[valid_mask]
                    valid_j = j_nodes[valid_mask]
                    valid_k = k_nodes[valid_mask]
                    new_paths = torch.stack([valid_i, valid_j, valid_k], dim=1)  # [V, 3]

                    take = min(new_paths.size(0), num_path_samples - filled)
                    sampled_paths[filled:filled + take] = new_paths[:take]
                    filled += take

                attempts += 1

                if attempts % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            if filled == 0:
                final_paths = torch.empty((0, 3), dtype=torch.long, device=device).t()
            else:
                final_paths = sampled_paths[:filled]
                if filled < num_path_samples:
                    repeat_times = (num_path_samples // filled) + 1
                    final_paths = final_paths.repeat(repeat_times, 1)[:num_path_samples]
                final_paths = final_paths.t().contiguous()

            all_sampled_paths.append(final_paths)

        if path_sample_times == 1:
            result = all_sampled_paths[0].unsqueeze(0)
        else:
            result = torch.stack(all_sampled_paths, dim=0)

        if verbose:
            end_time = time.time()
            print(f"Time Cost: {end_time - start_time:.4f} 秒")

        if not return_relabel_mapping:
            return result

        relabeled_list = []
        mappings = []
        inverse_mappings = []

        for t in range(path_sample_times):
            path_flat = result[t].t()
            if path_flat.size(0) == 0:
                relabeled_list.append(result[t])
                mappings.append(torch.empty(0, dtype=torch.long, device=device))
                inverse_mappings.append(torch.tensor([], dtype=torch.long, device=device))
                continue

            unique_nodes, inverse_flat = torch.unique(path_flat, return_inverse=True)
            relabeled_path = inverse_flat.view_as(path_flat).t().contiguous()
            relabeled_list.append(relabeled_path)

            inv_map = torch.full((unique_nodes.max().item() + 1,), -1, dtype=torch.long, device=device)
            inv_map[unique_nodes] = torch.arange(len(unique_nodes), device=device)

            mappings.append(unique_nodes)
            inverse_mappings.append(inv_map)

        relabeled_paths = torch.stack(relabeled_list, dim=0)

        if path_sample_times == 1:
            return relabeled_paths.squeeze(0), mappings[0], inverse_mappings[0]
        else:
            return relabeled_paths, mappings, inverse_mappings


def sample_two_neighbors(ptr, col_sorted, j_nodes, device):
    j_ptr_start = ptr[j_nodes]
    j_ptr_end = ptr[j_nodes + 1]
    j_deg = j_ptr_end - j_ptr_start

    j_deg = torch.clamp(j_deg, min=1)

    rand1 = (torch.rand(j_nodes.size(0), device=device) * j_deg).long() + j_ptr_start
    rand2 = (torch.rand(j_nodes.size(0), device=device) * j_deg).long() + j_ptr_start

    same_mask = rand1 == rand2
    retry_limit = 3
    retries = 0
    while same_mask.any() and retries < retry_limit:
        rand2[same_mask] = (torch.rand(same_mask.sum(), device=device) * j_deg[same_mask]).long() + j_ptr_start[same_mask]
        same_mask = rand1 == rand2
        retries += 1

    i_nodes = col_sorted[rand1]
    k_nodes = col_sorted[rand2]
    return i_nodes, k_nodes