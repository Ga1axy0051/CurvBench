from copy import deepcopy
import torch
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, WebKB, Amazon, WikipediaNetwork,Actor,Flickr,Reddit,LINKXDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree, add_self_loops
from fastargs.decorators import param

import sys; import os; sys.path.append("/data/hxz/WXY/CurvBench")
try:
    from parquet_loader import load_parquet_as_pyg
except Exception as e:
    print(f"Error loading parquet_loader: {e}")

import math
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
        data.y = data.y.long()
        data.x = data.x.float()

        num_nodes = min(data.x.size(0), data.y.size(0))

        data.x = data.x[:num_nodes]
        data.y = data.y[:num_nodes]

        mask = (
            (data.edge_index[0] >= 0) &
            (data.edge_index[0] < num_nodes) &
            (data.edge_index[1] >= 0) &
            (data.edge_index[1] < num_nodes)
        )

        data.edge_index = data.edge_index[:, mask]
        data.num_nodes = num_nodes
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


def x_padding(data, out_dim):
    
    assert data.x.size(-1) <= out_dim
    
    incremental_dimension = out_dim - data.x.size(-1)
    zero_features = torch.zeros((data.x.size(0), incremental_dimension), dtype=data.x.dtype, device=data.x.device)
    data.x = torch.cat([data.x, zero_features], dim=-1)

    return data


def x_svd(data, out_dim):
    
    assert data.x.size(-1) >= out_dim

    reduction = SVDFeatureReduction(out_dim)
    return reduction(data)


@param('general.cache_dir')
def iterate_datasets(data_names, cache_dir):
    if isinstance(data_names, str):
        data_names = [data_names]
    for data_name in data_names:
        data = load_parquet_as_pyg(data_name)
        yield data

@param('general.cache_dir')
def iterate_dataset_feature_tokens(data_names, cache_dir):
    if isinstance(data_names, str):
        data_names = [data_names]
    for data_name in data_names:
        data = load_parquet_as_pyg(data_name)
        yield data

# including projection operation, SVD
@param('data.node_feature_dim')
def preprocess(data, node_feature_dim):


     # ===== 修复 edge 越界 =====
    num_nodes = data.x.shape[0]

    mask = (
        (data.edge_index[0] >= 0) &
        (data.edge_index[0] < num_nodes) &
        (data.edge_index[1] >= 0) &
        (data.edge_index[1] < num_nodes)
    )

    data.edge_index = data.edge_index[:, mask]
    data.num_nodes = num_nodes


    """上游对数据预处理"""
    if hasattr(data, 'train_mask'):
        del data.train_mask
    if hasattr(data, 'val_mask'):
        del data.val_mask
    if hasattr(data, 'test_mask'):
        del data.test_mask



    if node_feature_dim <= 0:
        edge_index_with_loops = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.x = degree(edge_index_with_loops[1]).reshape((-1,1))
        
    
    else:
        # import pdb
        # pdb.set_trace()        
        if data.x.size(-1) > node_feature_dim:
            data = x_svd(data, node_feature_dim)
        elif data.x.size(-1) < node_feature_dim:
            data = x_padding(data, node_feature_dim)
        else:
            pass
    
    return data

# For prompting
def loss_contrastive_learning(x1, x2):
    # T = 0.1
    T = 0.5
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)
    
    sim_matrix = torch.einsum('ik,jk->ij', x1+1e-7, x2+1e-7) / torch.einsum('i,j->ij', x1_abs+1e-7, x2_abs+1e-7)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')
    
    sim_matrix = torch.exp(sim_matrix / T)
    
    if(True in sim_matrix.isnan()):
        print('Emerging nan value')    
    
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    if(True in pos_sim.isnan()):
        print('Emerging nan value')

    loss = pos_sim / ((sim_matrix.sum(dim=1) - pos_sim) + 1e-4)
    loss = - torch.log(loss).mean()
    if math.isnan(loss.item()):
        print("The value is NaN.")

    return loss

# used in pre_train.py
@param('general.reconstruct')
def gen_ran_output(data, simgrace, reconstruct):
    vice_model = deepcopy(simgrace)

    for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), simgrace.named_parameters()):
        if vice_name.split('.')[0] == 'projection_head':
            vice_model_param.data = param.data
        else:
            vice_model_param.data = param.data + 0.1 * torch.normal(0, torch.ones_like(
                param.data) * param.data.std())
    if(reconstruct==0.0):
    
        zj = vice_model.forward_cl(data)

        return zj
    
    else:
    
        zj, hj = vice_model.forward_cl(data)

        return zj, hj
