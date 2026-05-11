import warnings
import pickle as pkl
import sys, os
from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit,WikipediaNetwork,Actor,WebKB,Flickr

import scipy.sparse as sp
import torch
import numpy as np

import torch as th
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F

# from sklearn import datasets
# from sklearn.preprocessing import LabelBinarizer, scale
# from sklearn.model_selection import train_test_split
# from ogb.nodeproppred import DglNodePropPredDataset
# import copy
import dgl
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.simplefilter("ignore")


#邻接矩阵加自环
def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new

#邻接矩阵移除自环
def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse.FloatTensor(adj.indices()[:, diag_index], adj.values()[diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_

#将adj所有边权设为1
def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return th.sparse.FloatTensor(index, th.ones(len(index[0])), adj.shape)

#解析一个包含索引的文本文件，并将索引作为整数列表返回。
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

#将一个 SciPy 的稀疏矩阵 sparse_mx 转换为 PyTorch 的稀疏张量 
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

#将PyTorch的稀疏张量转换成DGL图
def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx=torch_sparse_mx.to_sparse()
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0,:], indices[1,:]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device=torch_sparse_mx.device)
    dgl_graph.edata['w'] = values.detach().to(torch_sparse_mx.device)
    return dgl_graph

#将DGL图转换成PyTorch的稀疏张量
def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    num_nodes = dgl_graph.number_of_nodes()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values, ([num_nodes,num_nodes]))
    return torch_sparse_mx

#将DGL图转换成PyTorch的稠密张量
def dgl_graph_to_torch_dense(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    num_nodes = dgl_graph.number_of_nodes()
    dense_mx = torch.zeros(num_nodes, num_nodes)
    dense_mx[rows_, cols_] = values
    return dense_mx


# def load_citation_network(dataset_str, sparse=None):
#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))
#
#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)
#
#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range - min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range - min(test_idx_range), :] = ty
#         ty = ty_extended
#
#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
#     if not sparse:
#         adj = np.array(adj.todense(),dtype='float32')
#     else:
#         adj = sparse_mx_to_torch_sparse_tensor(adj)
#
#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]
#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y) + 500)
#
#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])
#
#     features = torch.FloatTensor(features.todense())
#     labels = torch.LongTensor(labels)
#     train_mask = torch.BoolTensor(train_mask)
#     val_mask = torch.BoolTensor(val_mask)
#     test_mask = torch.BoolTensor(test_mask)
#
#     nfeats = features.shape[1]
#     for i in range(labels.shape[0]):
#         sum_ = torch.sum(labels[i])
#         if sum_ != 1:
#             labels[i] = torch.tensor([1, 0, 0, 0, 0, 0])
#     labels = (labels == 1).nonzero()[:, 1]
#     nclasses = torch.max(labels).item() + 1
#
#     return features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adj

def load_dblp():
    path = "data/dblp/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')
    feat = F.normalize(feat,dim=1,p=2)
    # adjs = torch.load(path+'adj.pt')
    # adjs = [adj.to_sparse().coalesce() for adj in adjs]
    adjs = torch.load('data/dblp/robust/random/9/adjs_aug.pt')

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    print(adjs)
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]

    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs



def load_mag():

    path = "data__/mag-4/"
    label = th.load(path+'label.pt').long()
    # label = F.one_hot(label, num_classes=4)

    feat = th.load(path+'feat.pt').float()
    nnodes = len(feat)
    adj_1 = th.load(path+'pap.pt').coalesce()
    adj_2 = th.load(path+'pp.pt').coalesce()
    adjs = [adj_1, adj_2]
    adjs = [sparse_tensor_add_self_loop(adj).coalesce() for adj in adjs]
    adjs = [adj_values_one(adj) for adj in adjs]
    print(adjs)

    train_index = torch.load(path+'train_index.pt')
    val_index = torch.load(path+'val_index.pt')
    test_index = torch.load(path+'test_index.pt')

    train_mask = torch.zeros(nnodes, dtype=torch.int64)
    train_mask[train_index] = 1
    val_mask = torch.zeros(nnodes, dtype=torch.int64)
    val_mask[val_index] = 1
    test_mask = torch.zeros(nnodes, dtype=torch.int64)
    test_mask[test_index] = 1

    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    nclasses = torch.max(label).item() + 1

    adj_f = (adjs[0] + adjs[1]).coalesce()
    adj_f = torch.sparse.FloatTensor(adj_f.indices(), torch.ones(len(adj_f.indices()[0])), adj_f.shape).coalesce()
    # adj_f = adjs[0] + adjs[1]
    print(adj_f)
    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adj_f

def load_acm():
    path = "./data/acm/processed/"
    label = torch.load(path + "label.pt").long()
    nclasses = int(label.max() + 1)
    feat = torch.load(path+'features.pt')

    p = 0.8  # Set your desired probability here
    mask = torch.rand(feat.shape) <= p
    feat[mask] = float(0)

    adjs = torch.load(path+'adj.pt')
    adjs = [adj.to_sparse().coalesce() for adj in adjs]
    # adjs = torch.load('data/acm/robust/random/9/adjs_remove.pt')
    # adjs = [adj.coalesce() for adj in adjs]

    train_mask = torch.load(path+'train_mask.pt')
    val_mask = torch.load(path+'val_mask.pt')
    test_mask = torch.load(path+'test_mask.pt')

    adjs = remove_self_loop(adjs)
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]
    print(adjs)
    adjs = [adj_values_one(adj).coalesce().to_dense() for adj in adjs]


    return feat, feat.shape[1], label, nclasses, train_mask, val_mask, test_mask, adjs


def load_cora(dataset):
    if dataset in ['Cora','Citeseer','Pubmed']:
        dataset = Planetoid(root='data', name=dataset)
    elif dataset in ['Chameleon','Squirrel']:
        dataset = WikipediaNetwork(root='data',name=dataset)
    elif dataset in ['Cornell']:
        dataset = WebKB(root='data', name='Cornell') 
    data = dataset[0]

    # 特征预处理
    features = data.x.float()
    features = F.normalize(features, p=2, dim=1) # 对特征进行归一化

    # 标签预处理
    labels = data.y.long()
    nclasses = labels.max().item() + 1

    # 邻接矩阵处理  
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]), (num_nodes, num_nodes)).to_dense()
    # 如果需要自环，则添加:
    adj = adj + torch.eye(num_nodes)


    # 训练集验证集测试集掩码，Planetoid数据集应该已经包含了这些信息
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    return features, features.shape[1], labels, nclasses, train_mask, val_mask, test_mask, [adj]  # 注意这里只有一个邻接矩阵



def load_data(args):
    data = load_cora(args)
    return data

def add_noise_edges(graph, rho):
    # 获取原始图的节点数量
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # 生成全连接图的边索引
    full_src, full_dst = np.meshgrid(np.arange(num_nodes), np.arange(num_nodes))
    full_src = full_src.flatten()
    full_dst = full_dst.flatten()

    # 删除已有的边和自环
    src, dst = graph.edges()
    existing_edges = set(zip(src.numpy(), dst.numpy()))
    full_edges = set(zip(full_src, full_dst))
    full_edges -= existing_edges
    full_edges = [(s, d) for s, d in full_edges if s != d]

    # 计算要添加的噪声边数量
    num_noise_edges = int(num_edges * rho)

    # 随机抽取剩余的边作为新边
    noise_edges_indices = np.random.choice(len(full_edges), num_noise_edges, replace=False)
    noise_edges = [full_edges[i] for i in noise_edges_indices]

    # 将新边添加到图中
    noise_src, noise_dst = zip(*noise_edges)
    graph.add_edges(noise_src, noise_dst)

    return graph


# # 随机加边
# path = "./data/Cora/processed/"

# feat = torch.load(path + 'features.pt')
# feat = F.normalize(feat, p=2, dim=1)
# adjs = torch.load(path + 'adj.pt')
# adjs = [adj.to_sparse().coalesce() for adj in adjs]
# adjs = remove_self_loop(adjs)
# adjs = [adj_values_one(adj).coalesce() for adj in adjs]

# for dr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#     g_0 = torch_sparse_to_dgl_graph(adjs[0])
#     g_1 = torch_sparse_to_dgl_graph(adjs[1])
#     g_2 = torch_sparse_to_dgl_graph(adjs[2])
#     g_0_aug = add_noise_edges(g_0, dr)
#     g_1_aug = add_noise_edges(g_1, dr)
#     g_2_aug = add_noise_edges(g_2, dr)
#     adjs_aug = [dgl_graph_to_torch_sparse(g_0), dgl_graph_to_torch_sparse(g_1), dgl_graph_to_torch_sparse(g_2)]
#     print(adjs_aug)
#     torch.save(adjs_aug, "./data/yelp/robust/random/"+str(int(dr*10))+'/adjs_aug.pt')


# # 随机删边
# for dr in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
#     g_0 = torch_sparse_to_dgl_graph(adjs[0])
#     g_1 = torch_sparse_to_dgl_graph(adjs[1])
#     g_2 = torch_sparse_to_dgl_graph(adjs[2])
#     g_list = [g_0, g_1, g_2]

#     num_edges_to_remove = [int(dr * g.number_of_edges()) for g in g_list]

#     # 随机选择要删除的边的索引
#     edges_to_remove = [np.random.choice(g_list[i].number_of_edges(), num_edges_to_remove[i], replace=False) for i in [0,1,2]]

#     # 删除选定的边
#     g_0.remove_edges(edges_to_remove[0])
#     g_1.remove_edges(edges_to_remove[1])
#     g_2.remove_edges(edges_to_remove[2])
#     adjs_aug = [dgl_graph_to_torch_sparse(g_0), dgl_graph_to_torch_sparse(g_1),dgl_graph_to_torch_sparse(g_2)]
#     print(adjs_aug)
#     torch.save(adjs_aug, "./data/yelp/robust/random/"+str(int(dr*10))+'/adjs_remove.pt')



# def load_data(args):
#     data = load_acm()
#     return data

# for dr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     # 加载数据
#     data = load_data(None)
#     features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adjs = data
#     g = torch_sparse_to_dgl_graph(adjs[0])

#     # 添加噪声边
#     g_aug = add_noise_edges(g, dr)
#     adjs_aug = [dgl_graph_to_torch_sparse(g_aug)]
#     directory = f"./data/cora/robust/random/{int(dr * 10)}/"
#     os.makedirs(directory, exist_ok=True)
#     torch.save(adjs_aug,  os.path.join(directory, 'adjs_aug.pt'))

#     # 随机删边
#     num_edges_to_remove = int(dr * g.number_of_edges())
#     edges_to_remove = np.random.choice(g.number_of_edges(), num_edges_to_remove, replace=False)
#     g.remove_edges(edges_to_remove)
#     adjs_remove = [dgl_graph_to_torch_sparse(g)]
#     torch.save(adjs_remove, os.path.join(directory, 'adjs_remove.pt'))

def attack_adj(dataset,mode,rate):
    """mode = aug or remove or meta
    rate = 0.1....0.9
    """
    if mode !='meta':
        data = load_data(dataset)
        features, nfeats, labels, nclasses, train_mask, val_mask, test_mask, adjs = data
        g = torch_sparse_to_dgl_graph(adjs[0])
    if mode =='aug':
        g_aug = add_noise_edges(g, rate)
        adjs_aug = dgl_graph_to_torch_sparse(g_aug)
        #print("adjs_aug:",g_aug)
        adjs_aug=torch_sparse_to_scipy(adjs_aug)
        #print("adjs_aug:",adjs_aug)
        return adjs_aug
    elif mode == 'remove':
        num_edges_to_remove = int(rate * g.number_of_edges())
        edges_to_remove = np.random.choice(g.number_of_edges(), num_edges_to_remove, replace=False)
        g.remove_edges(edges_to_remove)
        adjs_remove = dgl_graph_to_torch_sparse(g)
        #print("adjs_remove:",adjs_remove)
        adjs_remove=torch_sparse_to_scipy(adjs_remove)
        #print("adjs_remove:",adjs_remove)
        return adjs_remove
    elif mode == 'meta':
        """"rate=0.05,0.10,0.15,0.20,0.25"""
        adj = torch.load('/data/guoquanjiang/WS/MDGPT/model_node/data/meta_attack/'+str(100*rate)+dataset+'perturbed_adj.pt')
        return adj

    
def dense_adj_to_coo(dense_adj):
    coords = np.argwhere(dense_adj)
    rows, cols = coords[:, 0], coords[:, 1]
    values = dense_adj[rows, cols]
    coo = sp.coo_matrix((values, (rows, cols)), shape=dense_adj.shape)
    return coo


def torch_sparse_to_scipy(torch_sparse_matrix):
    """
    将 PyTorch 稀疏矩阵转换为 SciPy 稀疏矩阵。

    参数:
    torch_sparse_matrix -- PyTorch 稀疏矩阵 (torch.sparse.FloatTensor 或 torch.sparse_coo_tensor)

    返回值:
    scipy_sparse_matrix -- SciPy 稀疏矩阵 (scipy.sparse.coo_matrix)
    """
    # 合并重复索引（如果有的话）
    torch_sparse_matrix = torch_sparse_matrix.coalesce()

    # 获取行索引、列索引和对应的值
    row_indices = torch_sparse_matrix.indices()[0].numpy()
    col_indices = torch_sparse_matrix.indices()[1].numpy()
    data = torch_sparse_matrix.values().numpy()

    # 确定矩阵的形状
    size = torch_sparse_matrix.size()

    # 创建 SciPy 稀疏矩阵
    scipy_sparse_matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=size)

    return scipy_sparse_matrix