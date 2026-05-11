import os
import sys
import torch
from torch.utils.data import random_split, Subset


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.append(SRC_ROOT)

from data.supervised import induced_graphs
from data.utils import preprocess, iterate_datasets


def build_fewshot_cache(dataset, few_shot, seed=777, node_feature_dim=100, cache_dir='storage/.cache'):
    cache_dir = os.path.join(ROOT, cache_dir, dataset)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{few_shot}_shot_nf{node_feature_dim}_s{seed}.pt')

    if os.path.exists(cache_path):
        print(f'[skip] exists: {cache_path}')
        return cache_path

    data = preprocess(
        next(iterate_datasets(dataset, cache_dir=os.path.join(ROOT, 'storage/.cache'))),
        node_feature_dim=node_feature_dim
    )
    num_classes = torch.unique(data.y).size(0)
    train_dict_list = {key.item(): [] for key in torch.unique(data.y)}
    val_test_list = []
    target_graph_list = induced_graphs(data)

    for index, graph in enumerate(target_graph_list):
        i_class = graph.y
        if len(train_dict_list[i_class]) >= few_shot:
            val_test_list.append(graph)
        else:
            train_dict_list[i_class].append(index)

    all_indices = []
    for _, indice_list in train_dict_list.items():
        all_indices += indice_list

    train_set = Subset(target_graph_list, all_indices)
    val_set, test_set = random_split(val_test_list, [0.1, 0.9], torch.Generator().manual_seed(seed))

    results = [
        {
            'train': train_set,
            'val': val_set,
            'test': test_set,
        },
        num_classes
    ]

    torch.save(results, cache_path)
    print(f'[saved] {cache_path}')
    return cache_path


if __name__ == '__main__':
    for shot in [1, 5]:
        build_fewshot_cache('cs_phds', shot)
