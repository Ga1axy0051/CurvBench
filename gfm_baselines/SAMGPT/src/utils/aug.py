import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np


def aug_random_mask(input_feature, drop_percent=0.2):
    
    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):

    percent = drop_percent / 2
    row_idx, col_idx = input_adj.nonzero()

    index_list = []
    for i in range(len(row_idx)):
        index_list.append((row_idx[i], col_idx[i]))

    processed_edges = set()
    single_index_list = []
    for i in index_list:
        if (i[1], i[0]) not in processed_edges:
            single_index_list.append(i)
            processed_edges.add(i)
            processed_edges.add((i[1], i[0]))
    
    edge_num = int(len(row_idx) / 2)    
    add_drop_num = int(edge_num * percent / 2) 
    aug_adj = copy.deepcopy(input_adj.todense().tolist())

    edge_idx = [i for i in range(edge_num)]
    drop_idx = random.sample(edge_idx, add_drop_num)

    
    for i in drop_idx:
        aug_adj[single_index_list[i][0]][single_index_list[i][1]] = 0
        aug_adj[single_index_list[i][1]][single_index_list[i][0]] = 0
    

    #above finish drop edges
    node_num = input_adj.shape[0]
    l = [(i, j) for i in range(node_num) for j in range(i)]
    add_list = random.sample(l, add_drop_num)

    for i in add_list:
        
        aug_adj[i[0]][i[1]] = 1
        aug_adj[i[1]][i[0]] = 1
    
    aug_adj = np.matrix(aug_adj)
    aug_adj = sp.csr_matrix(aug_adj)
    return aug_adj


def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):
    
    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):
        
        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()
        
        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break

    
    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj





def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out

from utils import process

def build_aug(adj, feature, sparse, drop_percent):
    
    aug_adj1edge = aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
    aug_adj2edge = aug_random_edge(adj, drop_percent=drop_percent)
    aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
    aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    if sparse:
        adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
        aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)
    else:
        adj = torch.FloatTensor(adj.todense() )
        aug_adj1edge = torch.FloatTensor(aug_adj1edge.todense() )
        aug_adj2edge = torch.FloatTensor(aug_adj2edge.todense() )

    nb_nodes = feature.shape[0]
    idx = np.random.permutation(nb_nodes)
    shuf_fts = feature[idx, :]

    lbl_1 = torch.ones(1, nb_nodes)
    lbl_2 = torch.zeros(1, nb_nodes)
    lbl = torch.cat((lbl_1, lbl_2), dim=1)#.squeeze(0)
    return torch.stack([feature, shuf_fts, feature.detach(), feature.detach()]), torch.stack([adj, aug_adj1edge, aug_adj2edge]), lbl
    
