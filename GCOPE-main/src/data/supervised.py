from fastargs.decorators import param
import torch


def induced_graphs(data, smallest_size=10, largest_size=30):

    from torch_geometric.utils import subgraph, k_hop_subgraph
    from torch_geometric.data import Data
    import numpy as np

    induced_graph_list = []
    total_node_num = data.x.size(0)

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index
                                            ,relabel_nodes=True)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            #pos_nodes = torch.argwhere(data.y == int(current_label)) 
            pos_nodes = torch.where(data.y == current_label)[0]
            #candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            #candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            #subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])
            candidate_nodes = torch.from_numpy(
                np.setdiff1d(
                    pos_nodes.cpu().numpy(),
                    subset.cpu().numpy()
                )
            ).long()
            if len(candidate_nodes) > 0:
                perm = torch.randperm(candidate_nodes.shape[0])
                candidate_nodes = candidate_nodes[perm[:need_node_num]]

                subset = torch.cat([subset, candidate_nodes])
            subset = torch.unique(subset)
           
            

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))
        # print("="*80)
        # print("data.num_nodes =", data.num_nodes)
        # print("x shape =", data.x.shape)
        # print("y shape =", data.y.shape)
        # print("edge max =", data.edge_index.max().item())
        # print("subset max =", subset.max().item())
        # print("subset min =", subset.min().item())
        # print("subset len =", subset.shape[0])
        # print("="*80)
        subset = subset[subset < data.num_nodes]
        subset = subset.long()


        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True
                                     ,num_nodes=data.num_nodes)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)
        induced_graph_list.append(induced_graph)
        if(index%1000==0):
            print('生成的第{}/{}张子图数据'.format(index,total_node_num))
    
    print('生成了{}/{}张子图数据'.format(index,total_node_num))
    return induced_graph_list


@param('data.seed')
@param('general.cache_dir')
@param('general.few_shot')
@param('data.node_feature_dim')
def get_supervised_data(dataset, ratios, seed, cache_dir, few_shot, node_feature_dim):
    
    import os
    cache_dir = os.path.join(cache_dir, dataset)
    os.makedirs(cache_dir, exist_ok=True)


    if(few_shot == 0):
        cache_path = os.path.join(
            cache_dir,
            ','.join([f'{r:f}' for r in ratios]) + f'_nf{node_feature_dim}_s{seed}' + '.pt'
        )

        # import pdb
        # pdb.set_trace()

        if os.path.exists(cache_path):
            return torch.load(cache_path)

        from .utils import preprocess, iterate_datasets

        data = preprocess(next(iterate_datasets(dataset)))
            
        num_classes = torch.unique(data.y).size(0)
        target_graph_list = induced_graphs(data)
        print("target_graph_list len =", len(target_graph_list))
        print("num_classes =", num_classes)
        from torch.utils.data import random_split
        train_set, val_set, test_set = random_split(target_graph_list, ratios, torch.Generator().manual_seed(seed))

    else:
        cache_path = os.path.join(
            cache_dir + f'/{few_shot}_shot' + f'_nf{node_feature_dim}_s{seed}' + '.pt'
        )
        print("supervised-81c",cache_path)
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        print("supervised-84")
        from .utils import preprocess, iterate_datasets
        if dataset !='penn94':
            data = preprocess(next(iterate_datasets(dataset)))
            print("supervised-87",dataset,data)
        if dataset =='penn94':
            data = preprocess(next(iterate_datasets(dataset)))
            data.y[data.y==-1]=2
        print("supervised-93-Afterchange",data.y[100:200])
        num_classes = torch.unique(data.y).size(0)
        train_dict_list = {key.item():[] for key in torch.unique(data.y)}
        val_test_list = []
        target_graph_list = induced_graphs(data)

        from torch.utils.data import random_split, Subset

        for index, graph in enumerate(target_graph_list):
            
            i_class = graph.y

            if( len(train_dict_list[i_class]) >= few_shot):
                val_test_list.append(graph)
            else:
                train_dict_list[i_class].append(index)
        
        all_indices = []
        for i_class, indice_list in train_dict_list.items():
            all_indices+=indice_list

        train_set = Subset(target_graph_list, all_indices)

        val_set, test_set = random_split(val_test_list, [0.1,0.9], torch.Generator().manual_seed(seed))
        
    # import pdb
    # pdb.set_trace()

    results = [
    {
        'train': train_set,
        'val': val_set,
        'test': test_set,
    }, 
        num_classes
    ]

    # save to cache
    torch.save(results, cache_path)

    return results
