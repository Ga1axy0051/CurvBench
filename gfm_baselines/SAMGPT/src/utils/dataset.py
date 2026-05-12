import torch
import matplotlib.pyplot as plt
from torch_geometric.datasets import WebKB, Planetoid, Amazon, Coauthor, WikipediaNetwork, Reddit, \
    Flickr, PPI, Yelp, Twitch, Actor, KarateClub, FacebookPagePage, LastFMAsia
 #BitcoinOTC 

from torch_geometric.utils import degree
from ogb.nodeproppred import PygNodePropPredDataset
#from ogb.lsc import MAG240MDataset
from torch_geometric.utils import to_networkx, degree
import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    coalesce
)
import numpy as np
import scipy.sparse as sp

import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data

class DiseaseDataset(InMemoryDataset):
    def __init__(self, root='data'):
        super().__init__(root)
        self.data, self.slices = self.load_data()

    def load_data(self):
        base_dir = os.path.join(self.root, "Disease")

        edge_path = os.path.join(base_dir, "disease_nc.edges.csv")
        feat_path = os.path.join(base_dir, "disease_nc.feats.npz")
        label_path = os.path.join(base_dir, "disease_nc.labels.npy")

        # =====================================================
        # 1. edges
        # =====================================================
        edge_df = pd.read_csv(edge_path, header=None)

        edge_index = torch.tensor(
            edge_df.values.T,
            dtype=torch.long
        )

        # 转无向图
        edge_index = to_undirected(edge_index)

        # 去自环
        edge_index, _ = remove_self_loops(edge_index)

        # 去重排序
        edge_index = coalesce(edge_index)

        # =====================================================
        # 2. features
        # =====================================================
        x_sparse = sp.load_npz(feat_path)

        # 当前规模可直接 dense
        x = torch.tensor(
            x_sparse.toarray(),
            dtype=torch.float
        )

        # =====================================================
        # 3. labels
        # =====================================================
        labels = np.load(label_path).astype(np.int64)

        y = torch.tensor(labels, dtype=torch.long)

        # 保险处理 one-hot
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )

        return self.collate([data])
class TelecomDataset(InMemoryDataset):
    #y label是独热编码，故从三维降低为一个数
    def __init__(self, root='data'):
        super().__init__(root)
        self.data, self.slices = self.load_data()

    def load_data(self):
        path = os.path.join(self.root, 'telecom', 'telecom_graph.pt')
        data = torch.load(path, map_location='cpu')

        # one-hot -> class index
        if len(data.y.shape) > 1:
            data.y = torch.argmax(data.y, dim=1)

        return self.collate([data])

class AirportDataset(InMemoryDataset):
    def __init__(self, root='data'):
        super().__init__(root)
        self.data, self.slices = self.load_data()

    def load_data(self):
        """
        读取论文处理好的 Airport PyG 数据，并包含越界侦察与修复
        """
        path = os.path.join(self.root, 'Airport', 'HGCN_Airport_PyG.pt')
        data = torch.load(path, map_location='cpu')

        # ===== 1. 保证 feature 是 float，并获取真实的节点数量 =====
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.float()
            num_nodes = data.x.size(0)
        else:
            # 如果没有节点特征，尝试从 data.num_nodes 获取
            num_nodes = data.num_nodes

        # ===== 2. 侦察边越界问题 =====
        if hasattr(data, 'edge_index') and data.edge_index is not None:
            max_idx = data.edge_index.max().item()
            min_idx = data.edge_index.min().item()
            
            # 如果最大索引 >= 节点总数，或者最小索引 < 0，说明存在越界
            if max_idx >= num_nodes or min_idx < 0:
                print(f"[警告] 侦察到 Airport 数据集存在边越界问题！")
                print(f" - 真实节点总数 (num_nodes): {num_nodes}")
                print(f" - 边索引最大值: {max_idx}")
                print(f" - 边索引最小值: {min_idx}")
                print(f" - 正在自动清洗越界边...")
                
                # ===== 3. 修复：创建掩码过滤越界边 =====
                # 必须满足: 起点 < num_nodes 且 终点 < num_nodes 且 起点 >= 0 且 终点 >= 0
                valid_edge_mask = (
                    (data.edge_index[0] < num_nodes) & 
                    (data.edge_index[1] < num_nodes) & 
                    (data.edge_index[0] >= 0) & 
                    (data.edge_index[1] >= 0)
                )
                
                # 更新 edge_index
                data.edge_index = data.edge_index[:, valid_edge_mask]
                
                # 同步过滤边属性 (edge_attr) 或边权重 (edge_weight) - 这一步非常重要，否则边和属性数量会对不上
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    data.edge_attr = data.edge_attr[valid_edge_mask]
                if hasattr(data, 'edge_weight') and data.edge_weight is not None:
                    data.edge_weight = data.edge_weight[valid_edge_mask]
                    
                print(f"[完成] 清洗后边数量: {data.edge_index.size(1)}")
            #else:
                #print(f"[正常] Airport 数据集边索引健康，最大索引: {max_idx}, 节点数: {num_nodes}")

        # ===== 4. 保证 label 格式 =====
        if hasattr(data, 'y') and data.y is not None:
            if data.y.dim() > 1:
                data.y = torch.argmax(data.y, dim=1)
            data.y = data.y.long()
            
            # ===== [新增] 动态标签平移，修复负数标签越界 =====
            min_label = data.y.min().item()
            if min_label < 0:
                print(f"[数据清洗] 侦察到负数标签 (min={min_label})，正在将所有标签自动平移 {-min_label} 变为从 0 开始...")
                data.y = data.y - min_label

        # 确保 data 对象的 num_nodes 属性准确无误
        data.num_nodes = num_nodes

        return self.collate([data])


class CSPhDsDataset(InMemoryDataset):
    def __init__(self, root='data'):
        super().__init__(root)
        self.data, self.slices = self.load_data()

    def load_data(self):
        base_dir = os.path.join(self.root, 'cs_phds', 'cs_phds_nc_ready')

        adj_path = os.path.join(base_dir, 'adj.pt')
        feat_path = os.path.join(base_dir, 'feats.pt')
        label_path = os.path.join(base_dir, 'labels.pt')
        adj = torch.load(adj_path, map_location='cpu').float()
        x = torch.load(feat_path, map_location='cpu').float()
        y = torch.load(label_path, map_location='cpu')

        if x.dim() == 1:
            x = x.unsqueeze(1)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)

        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
        y = y.long()

        min_label = y.min().item()
        if min_label < 0:
            y = y - min_label

        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous().long()
        edge_index = to_undirected(edge_index)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)

        data = Data(
            x=x,
            edge_index=edge_index,
            y=y
        )
        data.num_nodes = x.size(0)

        return self.collate([data])



WebKB_datasets = ['Texas', 'Cornell', 'Wisconsin']
Planetoid_datasets = ['Cora', 'Citeseer', 'Pubmed']
Amazon_datasets = ['Photo', 'Computers']
Coauthor_datasets = ['CS', 'Physics']
WikipediaNetwork_datasets = ['chameleon', 'squirrel']
Reddit_datasets = ['Reddit']
OGB_datasets = ['ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100M', 'ogbn-mag']

Flickr_datasets = ['Flickr']
PPI_datasets = ['PPI']
Yelp_datasets = ['Yelp']
Twitch_datasets = ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
Actor_datasets = ['Actor']
KarateClub_datasets = ['KarateClub']
FacebookPagePage_datasets = ['FacebookPagePage']
LastFMAsia_datasets = ['LastFMAsia']
CSPhDs_datasets = ['cs_phds', 'CSPhDs']
#BitcoinOTC_datasets = ['BitcoinOTC']
#MAG240MDatasets = ['MAG240MDataset']

def load_dataset(name, path='./data'):
    if name in Planetoid_datasets:
        dataset = Planetoid(root=path, name=name)
    elif name in Amazon_datasets:
        dataset = Amazon(root=path, name=name)
    elif name in Coauthor_datasets:
        dataset = Coauthor(root=path, name=name)
    elif name in WebKB_datasets:
        dataset = WebKB(root=path, name=name)
    elif name in WikipediaNetwork_datasets:
        dataset = WikipediaNetwork(root=path, name=name)
    elif name in Reddit_datasets:
        dataset = Reddit(root=f'{path}/Reddit')
    elif name in OGB_datasets:
        dataset = PygNodePropPredDataset(root=path, name=name)
    elif name in Flickr_datasets:
        dataset = Flickr(root=f'{path}/Flickr')
    elif name in PPI_datasets:
        dataset = PPI(root=f'{path}/PPI')
    elif name in Yelp_datasets:
        dataset = Yelp(f'{path}/Yelp')
    elif name in Twitch_datasets:
        dataset = Twitch(root=path,name=name)
    elif name in Actor_datasets:
        dataset = Actor(root=f'{path}/Actor')
    elif name in KarateClub_datasets:
        dataset = KarateClub()
    elif name in FacebookPagePage_datasets:
        dataset = FacebookPagePage(root=f'{path}/Facebook')
    elif name in LastFMAsia_datasets:
        dataset = LastFMAsia(root=f'{path}/LastFMAsia')
    elif name in CSPhDs_datasets:
        dataset = CSPhDsDataset(root='/data/home/2022080136001/jhupload')
    elif name in ['Disease']:
        dataset = DiseaseDataset(root='/data/home/2022080136001/jhupload/data')
    elif name in ['Telecom']:
        dataset = TelecomDataset(root='/data/home/2022080136001/jhupload/data')
    elif name in ['Airport']:
        dataset = AirportDataset(root='/data/home/2022080136001/jhupload/data')
    #elif name in BitcoinOTC_datasets:
    #    dataset = BitcoinOTC(root=f'{path}/BitcoinOTC')
    #elif name in MAG240MDatasets:
    #    dataset = MAG240MDataset(root=f'{path}/MAG240MDataset')
    else:
        raise ValueError(f"Unknown dataset name: {name}")
    #print(f'{name}: {dataset[0].num_nodes}')
    return dataset

def analyze_dataset(name, graph = False):
    data_ = load_dataset(name)
    #print(len(data_))
    data = data_[0]
    deg = degree(data.edge_index[0], data.num_nodes)
    average_degree = deg.mean().item()
    print(f'\n{name}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}')
    G = to_networkx(data, to_undirected=True)

    print('Clustering Coefficient:')
    global_clustering_coefficient = nx.transitivity(G) 
    print(f'Global Clustering Coefficient: {global_clustering_coefficient:.4f}')

    average_clustering_coefficient = nx.average_clustering(G)
    print(f'Average Clustering Coefficient: {average_clustering_coefficient:.4f}')

    shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))

    path_lengths = []
    for node, lengths in shortest_path_lengths.items():
        path_lengths.extend(lengths.values())
    
    network_diameter = max(path_lengths)
    print(f'Network Diameter:')

    average_shortest_path_length = np.mean(path_lengths)
    print(f'Average Shortest Path Length: {average_shortest_path_length:.4f}')

    percentile_90_shortest_path_length = np.percentile(path_lengths, 90)
    print(f'90th Percentile Shortest Path Length: {percentile_90_shortest_path_length:.4f}')
    if graph:
        plt.figure()
        plt.hist(deg.cpu().numpy(), bins=range(int(deg.min()), int(deg.max()) + 1), edgecolor='gray')
        plt.title(f"Degree Distribution of {name}")
        plt.xlabel("Degree")
        plt.ylabel("Frequency")
        plt.show()
    

def analyze_dataset_multi(name, graph = False):
    data_ = load_dataset(name)
    #print(len(data_))
    for i, data in enumerate(data_):
        deg = degree(data.edge_index[0], data.num_nodes)
        average_degree = deg.mean().item()
        print(f'{name}_{i+1}: avg_degree:{average_degree:.4f}, num_nodes:{data.num_nodes}, num_edges:{data.num_edges}, num_classes:{data_.num_classes}, num_features:{data_.num_features}')
        if graph:
            plt.figure()
            plt.hist(deg.cpu().numpy(), bins=range(int(deg.min()), int(deg.max()) + 1), edgecolor='gray')
            plt.title(f"Degree Distribution of {name}")
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.show()

if __name__ == '__main__':
    selected_datasets = [#'Texas', 'Cornell', 'Wisconsin', 
                        #'Cornell',
                        'Cora', 'Citeseer', 'Pubmed', 
                        'Photo',
                        'Computers', 
                        #'CS', 'Physics', 
                        #'chameleon', 'squirrel', 
                        #'Reddit',
                        #'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins', #'ogbn-mag',
                        #'Flickr', 
                        #'PPI', 
                        #'Yelp', 
                        #'Actor',
                        #'ES',
                        #'DE', 'EN', 'ES', 
                        #'FR', 'PT', 'RU',
                        #'KarateClub',
                        'FacebookPagePage', 'LastFMAsia', 
                        # #'BitcoinOTC'
                        #'MAG240MDataset'
                        ]
    for dataset in selected_datasets:
        analyze_dataset(dataset)
        #analyze_dataset_multi(dataset)
