import numpy as np
import torch


def unify_feature_dimension(
        x,
        uni_dim: int,
        center: bool = True
):
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # [n] -> [n, 1]

    num_nodes, original_dim = x.shape
    device = x.device

    if num_nodes == 0 or original_dim == 0:
        return torch.zeros((num_nodes, uni_dim), device=device, dtype=torch.float)

    x = x.float()

    if center:
        x = x - x.mean(dim=0, keepdim=True)

    if original_dim >= uni_dim:
        U, S, Vt = torch.svd(x, some=True)  # U: [n, min(n,d)], S: [min(n,d)]
        k = min(U.shape[1], uni_dim)
        U_k = U[:, :k]  # [n, k]
        S_k = S[:k]  # [k]
        x_reduced = U_k * S_k  # [n, k]

        if k < uni_dim:
            padding = torch.zeros((num_nodes, uni_dim - k), device=device, dtype=torch.float)
            x_reduced = torch.cat([x_reduced, padding], dim=1)  # [n, uni_dim]

    else:
        U, S, Vt = torch.svd(x, some=True)
        k = U.shape[1]  # min(n, original_dim)
        x_reduced = U * S  # [n, k]

        if k < uni_dim:
            padding = torch.zeros((num_nodes, uni_dim - k), device=device, dtype=torch.float)
            x_reduced = torch.cat([x_reduced, padding], dim=1)  # [n, uni_dim]
        else:
            x_reduced = x_reduced[:, :uni_dim]

    x_reduced = torch.nan_to_num(x_reduced, nan=0.0, posinf=0.0, neginf=0.0)

    return x_reduced


def graph_few_shot_splits(dataset, k_shot, num_val, num_splits):
    if hasattr(dataset, "labels"):
        labels = dataset.labels.numpy().tolist()
    else:
        labels = [data.y.item() for data in dataset]

    num_classes = len(set(labels))
    num_graphs = len(dataset)

    label_to_indices = [[] for _ in range(num_classes)]
    for idx, y in enumerate(labels):
        label_to_indices[y].append(idx)

    for y in range(num_classes):
        if len(label_to_indices[y]) < k_shot:
            raise ValueError(f"Class {y} has only {len(label_to_indices[y])} graphs, but k_shot={k_shot}")

    label_to_indices_np = [np.array(indices) for indices in label_to_indices]

    train_masks, val_masks, test_masks = [], [], []

    for split_id in range(num_splits):
        train_indices = []
        remaining_indices = []

        for y in range(num_classes):
            indices = label_to_indices_np[y].copy()
            np.random.shuffle(indices)
            train_indices.extend(indices[:k_shot].tolist())
            remaining_indices.extend(indices[k_shot:].tolist())

        remaining_indices = np.array(remaining_indices)
        np.random.shuffle(remaining_indices)
        val_size = int(len(remaining_indices) * num_val)
        val_indices = remaining_indices[:val_size]
        test_indices = remaining_indices[val_size:]

        train_mask = torch.zeros(num_graphs, dtype=torch.bool)
        val_mask = torch.zeros(num_graphs, dtype=torch.bool)
        test_mask = torch.zeros(num_graphs, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

        if split_id == 0:
            print(f"Total graphs: {num_graphs}")
            print(f"Train (support): {len(train_indices)} graphs ({k_shot} per class)")
            print(f"Val: {len(val_indices)} graphs")
            print(f"Test: {len(test_indices)} graphs")
            print(f"Val ratio in remaining: {len(val_indices) / (len(val_indices) + len(test_indices)):.2f}")

    train_mask = torch.stack(train_masks, dim=1)
    val_mask = torch.stack(val_masks, dim=1)
    test_mask = torch.stack(test_masks, dim=1)

    return train_mask, val_mask, test_mask


def link_k_shot_split(data, k_shot, num_splits, num_val=0.1, num_way=10):
    """

    :param data: PyG-style data object with edge_index, edge_type
    :param k_shot: int, number of training samples per selected relation
    :param num_splits: int, number of random splits to generate
    :param num_val: float, ratio of validation set from remaining edges (after k-shot)
    :param num_way: int, number of relations to sample for few-shot (default=10)
    :return: train_mask, val_mask, test_mask — each of shape [num_edges, num_splits]
    """

    edge_index = data.edge_index  # [2, num_edges]
    edge_type = data.edge_type    # [num_edges,]
    num_edges = edge_index.size(1)
    num_relations = int(edge_type.max().item() + 1)

    train_masks = []
    val_masks = []
    test_masks = []

    all_relations = []
    for rel in range(num_relations):
        if (edge_type == rel).any():
            all_relations.append(rel)
    all_relations = torch.tensor(all_relations)

    selected_relations_list = []

    for _ in range(num_splits):
        if len(all_relations) < num_way:
            raise ValueError(f"Not enough relations ({len(all_relations)}) to sample {num_way}-way.")
        perm = torch.randperm(len(all_relations))[:num_way]
        selected_relations = all_relations[perm]  # [num_way]
        selected_relations_list.append(selected_relations)

        train_indices = []
        val_indices = []
        test_indices = []

        for rel in range(num_relations):
            if rel not in selected_relations:
                continue

            rel_mask = (edge_type == rel)
            rel_indices = rel_mask.nonzero(as_tuple=False).view(-1)  # [num_rel_edges]
            num_rel_edges = rel_indices.size(0)

            if num_rel_edges < k_shot + 2:
                continue

            perm_rel = torch.randperm(num_rel_edges)
            rel_indices_shuffled = rel_indices[perm_rel]

            k = min(k_shot, num_rel_edges)
            train_indices.append(rel_indices_shuffled[:k])

            remaining_indices = rel_indices_shuffled[k:]
            num_remaining = remaining_indices.size(0)

            if num_remaining == 0:
                val_split = torch.empty(0, dtype=torch.long)
                test_split = torch.empty(0, dtype=torch.long)
            else:
                val_size = int(num_remaining * num_val)
                val_split = remaining_indices[:val_size]
                test_split = remaining_indices[val_size:]

            val_indices.append(val_split)
            test_indices.append(test_split)

        train_idx = torch.cat(train_indices) if len(train_indices) > 0 else torch.empty(0, dtype=torch.long)
        val_idx = torch.cat(val_indices) if len(val_indices) > 0 else torch.empty(0, dtype=torch.long)
        test_idx = torch.cat(test_indices) if len(test_indices) > 0 else torch.empty(0, dtype=torch.long)

        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)

    # Stack masks for all splits: [num_edges, num_splits]
    train_mask = torch.stack(train_masks, dim=1)
    val_mask = torch.stack(val_masks, dim=1)
    test_mask = torch.stack(test_masks, dim=1)

    return train_mask, val_mask, test_mask, selected_relations_list