from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import time
from utils import Calbound
from models import LogReg
from preprompt_3_28 import PrePrompt,pca_compression
import preprompt as preprompt
from utils import process
import pdb
import aug
import os
import tqdm
import argparse
from downprompt_3_28 import downprompt,prefeatureprompt
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("MDGFM")
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


def pca_compression_fixed_dim(seq, k):
    seq = np.asarray(seq)
    max_k = min(seq.shape[0], seq.shape[1], k)

    if max_k < k:
        print(f"[数据处理] 当前特征维度不足 {k}，PCA 先压到 {max_k} 维，再补零到 {k} 维。")

    seq = pca_compression(seq, k=max_k)

    if max_k < k:
        pad = np.zeros((seq.shape[0], k - max_k), dtype=seq.dtype)
        seq = np.concatenate([seq, pad], axis=1)

    return seq


parser.add_argument('--dataset', type=str, default="Chameleon", help='data')
parser.add_argument('--drop_percent', type=float, default=0.5, help='drop percent')

parser.add_argument('--lr', type=float, default=0.02, help='pretrain lr')
parser.add_argument('--downstreamlr', type=float, default=0.003, help='downstream lr')
parser.add_argument('--epochs', type=int, default=60, help='epoch')
parser.add_argument('--shot_num', type=int, default=1, help='shotnum')

parser.add_argument('--seed', type=int, default=1024, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--save_name', type=str, default='model_add_node_lay3_computers.pkl', help='save ckpt name')
parser.add_argument('--val_name', type=str, default='noval_graphcl_BZR.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
args = parser.parse_args()

print(args)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit,Actor,WikipediaNetwork,WebKB,Flickr
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
print('-' * 100)
global_start_time = time.perf_counter()

batch_size = 128
nb_epochs = args.epochs
patience = 500
lr_list=args.lr
l2_coef = 0.0001
drop_prob = 0.5
hid_units = 256
sparse = True
useMLP =False
LP = False
shot_num=args.shot_num
# Pubmed need to be 100
testnum = 10000
downstreamlrlist = args.downstreamlr
nonlinearity = 'prelu' 
dataset = args.dataset
device = torch.device("cuda")
best = 1e9
firstbest = 0

dataset1 = Planetoid(root='data', name='Cora')                                                                                               
loader1 = DataLoader(dataset1)
dataset2 = Planetoid(root='data', name='Pubmed')                                                                                         
loader2 = DataLoader(dataset2)
dataset3 = Planetoid(root='data', name='Citeseer')
loader3 = DataLoader(dataset3)
dataset4 = Actor(root='data/Actor')  
loader4 = DataLoader(dataset4)
dataset5 = WebKB(root='data', name='Cornell') 
loader5 = DataLoader(dataset5)
dataset6 = DiseaseDataset(root='/MDGFM-main/MDGFM-main/data')
loader6 = DataLoader(dataset6)

dataset7 = TelecomDataset(root='MDGFM-main/MDGFM-main/data')
loader7 = DataLoader(dataset7)

dataset8 = AirportDataset(root='MDGFM-main/MDGFM-main/data')
loader8 = DataLoader(dataset8)

dataset9 = CSPhDsDataset(root='')
loader9 = DataLoader(dataset9)


cnt_wait = 0
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
unify_dim = 50
a=args.save_name
n_=0
for lr in [lr_list]:
    time_=time.localtime()
    n_+=1
    best = 1e9
    firstbest = 0
    args.save_name = str(time_)+a
    for step, (data1,data2,data3,data4,data5,data6,data7,data8) in enumerate(zip(loader1,loader2,loader3,loader4,loader5,loader6,loader7,loader8)):
        
        features11,adj1= process.process_tu(data1,data1.x.shape[1])
        features22,adj2= process.process_tu(data2,data2.x.shape[1])
        features33,adj3= process.process_tu(data3,data3.x.shape[1])
        features44,adj4= process.process_tu(data4,data4.x.shape[1])
        features55,adj5= process.process_tu(data5,data5.x.shape[1])
        features66,adj6= process.process_tu(data6,data6.x.shape[1])
        pre_i=6
        if args.dataset=='Cora' or args.dataset=='Disease':
            features11,adj1= process.process_tu(data7,data7.x.shape[1])
            #features11,adj1= process.process_tu(data8,data8.x.shape[1])
            features66,adj6= process.process_tu(data8,data8.x.shape[1])
            pre_i=0
        elif args.dataset=='Pubmed' or args.dataset=='Cornell':
            features22,adj2= process.process_tu(data7,data7.x.shape[1])
            #features22,adj2= process.process_tu(data8,data8.x.shape[1])
            features55,adj5= process.process_tu(data8,data8.x.shape[1])
            pre_i=1
        elif args.dataset=='Citeseer' or args.dataset=='Telecom':
            features33,adj3= process.process_tu(data8,data8.x.shape[1])
            pre_i=2
        elif args.dataset=='Actor' or args.dataset=='Airport':
            features44,adj4= process.process_tu(data7,data7.x.shape[1])
            #features44,adj4= process.process_tu(data6,data6.x.shape[1])
            pre_i=3
        features1 = pca_compression(features11,k=unify_dim)
        features2 = pca_compression(features22,k=unify_dim)
        features3 = pca_compression(features33,k=unify_dim)
        features4 = pca_compression(features44,k=unify_dim)
        features5 = pca_compression(features55,k=unify_dim)
        features6 = pca_compression(features66,k=unify_dim)
        
        features1 = torch.FloatTensor(features1).cuda()
        features2 = torch.FloatTensor(features2).cuda()
        features3 = torch.FloatTensor(features3).cuda()
        features4 = torch.FloatTensor(features4).cuda()
        features5 = torch.FloatTensor(features5).cuda()
        features6 = torch.FloatTensor(features6).cuda()
        
        adj = process.combine_dataset(adj1,adj2,adj3,adj4,adj5)
        negative_sample = preprompt.prompt_pretrain_sample(adj,50)

    adj2 = process.normalize_adj(adj2 + sp.eye(adj2.shape[0]))
    adj1 = process.normalize_adj(adj1 + sp.eye(adj1.shape[0]))
    adj3 = process.normalize_adj(adj3 + sp.eye(adj3.shape[0]))
    adj4 = process.normalize_adj(adj4 + sp.eye(adj4.shape[0]))
    adj5 = process.normalize_adj(adj5 + sp.eye(adj5.shape[0]))
    adj6 = process.normalize_adj(adj6 + sp.eye(adj6.shape[0]))
    if sparse:
        sp_adj1 = process.sparse_mx_to_torch_sparse_tensor(adj1)
        sp_adj2 = process.sparse_mx_to_torch_sparse_tensor(adj2)
        sp_adj3 = process.sparse_mx_to_torch_sparse_tensor(adj3)
        sp_adj4 = process.sparse_mx_to_torch_sparse_tensor(adj4)
        sp_adj5 = process.sparse_mx_to_torch_sparse_tensor(adj5)
        sp_adj6 = process.sparse_mx_to_torch_sparse_tensor(adj6)
        
    model = PrePrompt(unify_dim, hid_units, nonlinearity, negative_sample,3,0.1,args.combinetype)
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    if torch.cuda.is_available():
        model = model.cuda()
        features1 = features1.cuda()
        features2 = features2.cuda()
        features3 = features3.cuda()
        features4 = features4.cuda()
        features5 = features5.cuda()
        features6 = features6.cuda()

        if sparse:
            sp_adj1 = sp_adj1.cuda()
            sp_adj2 = sp_adj2.cuda()
            sp_adj3 = sp_adj3.cuda()
            sp_adj4 = sp_adj4.cuda()
            sp_adj5 = sp_adj5.cuda()
            sp_adj6 = sp_adj6.cuda()
            

    epoch_train_start = time.time()
    real_epoch_num = 0
    for epoch in range(nb_epochs):
        torch.cuda.empty_cache()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        loss = 0
        regloss = 0
        model.train()
        optimiser.zero_grad()
        loss = model( features1,features2,features3,features4,features5,features6,
                    sp_adj1 if sparse else adj1, sp_adj2 if sparse else adj2,sp_adj3 if sparse else adj3,sp_adj4 if sparse else adj4,sp_adj5 if sparse else adj5,sp_adj6 if sparse else adj6,
                    sparse, None, None, None,pre_i)
        loss.backward()
        optimiser.step()
        print('Loss:[{:.4f}]'.format(loss.item()))
        if loss < best:
            firstbest = 1
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), args.save_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            break
        print('Loading {}th epoch'.format(best_t))
        real_epoch_num += 1
    epoch_train_end = time.time()
    pure_train_time = epoch_train_end - epoch_train_start
    train_time_per_epoch = pure_train_time / real_epoch_num

    pretrain_end_time = time.perf_counter()
    pretrain_elapsed = pretrain_end_time - global_start_time
    print("=" * 100)
    print(f"Pretrain finished!")
    print(f"Pretrain elapsed time: {pretrain_elapsed:.2f} seconds")
    print(f"Pretrain elapsed time: {pretrain_elapsed/60:.2f} minutes")
    print("=" * 100)
    model = PrePrompt(unify_dim, hid_units, nonlinearity,1,3,0.1,args.combinetype)

    print('#'*50)
    print('Downastream dataset is ',args.dataset)
    downk=15
    if args.dataset == 'Cora' or args.dataset =='Citeseer' or args.dataset =='Pubmed':
        dataset = Planetoid(root='data', name=args.dataset)                                                                                         
        downk=30
        if args.dataset =='Pubmed':
            testnum=100
    if args.dataset == 'Chameleon' or args.dataset == 'Squirrel':
        dataset=WikipediaNetwork(root='data',name=args.dataset)
        downk=15
    if args.dataset == 'Cornell':
        dataset=WebKB(root='data', name=args.dataset)
        downk=15
    if args.dataset == 'Actor':
        dataset=Actor(root='data/Actor')
        downk=15
    if args.dataset == 'Disease':
        dataset=DiseaseDataset(root="")
        downk=15
    if args.dataset == 'Telecom':
        dataset=TelecomDataset(root="")
        downk=15
    if args.dataset == 'Airport':
        dataset=AirportDataset(root='')
        downk=15
    if args.dataset == 'cs_phds':
        dataset=CSPhDsDataset(root='')
        downk=15

    print(args.dataset)
    loader = DataLoader(dataset)
    for data in loader:
        print(data)
        if isinstance(data, list):
            data = data[0]
        features,adj= process.process_tu(data,data.x.shape[1])
        features = pca_compression_fixed_dim(features, k=unify_dim)
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
        sp_adj = sp_adj.cuda()
        features = torch.FloatTensor(features).cuda()
        print(features.shape)
        ln=data.y.shape[0]-testnum
        if ln<0:
            ln=0
        idx_test = range(ln,data.y.shape[0])
        labels = data.y
        labels_cuda = labels.cuda()
        data=np.array(data.y)
        np.unique(data)
        nb_classes=len(np.unique(data))
        print(nb_classes)
        
    model = model.cuda()

    model.load_state_dict(torch.load(args.save_name))

    embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None,LP)
    acclist = torch.FloatTensor(100,).cuda()

    for downstreamlr in [downstreamlrlist]:
        
        print(labels.shape)
        fixed_idx_test = torch.arange(ln, labels.shape[0], device=labels_cuda.device)
        use_episode_holdout = (ln == 0)
        if use_episode_holdout:
            print("[downstream] testnum >= num_nodes, use per-episode held-out nodes as test set.")
        else:
            test_lbls = labels_cuda[fixed_idx_test]
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        tot_macro = 0
        tot_micro = 0

        macro_f1s = []
        micro_f1s = []

        print('-' * 100)
        for shotnum in range(shot_num,shot_num+1):
            tot = torch.zeros(1)
            tot = tot.cuda()
            tot_macro = 0
            tot_micro = 0

            accs = []
            macro_f1s = []
            micro_f1s = []
            cnt_wait = 0
            best = 1e9
            best_t = 0
            print("shotnum",shotnum)
            for i in tqdm(range(50)):
                log = downprompt(model.texttoken1.weight.detach(),model.texttoken2.weight.detach(),model.texttoken3.weight.detach(),model.texttoken4.weight.detach(),model.texttoken5.weight.detach(),model.texttoken6.weight.detach(),
                                model.sumtext.weight.detach(),
                                model.pretext1.weight.detach(),model.pretext2.weight.detach(),model.pretext3.weight.detach(),model.pretext4.weight.detach(),model.pretext5.weight.detach(),model.pretext6.weight.detach(),
                                model.balancetoken1.weight.detach(),model.balancetoken2.weight.detach(),model.balancetoken3.weight.detach(),model.balancetoken4.weight.detach(),model.balancetoken5.weight.detach(),model.balancetoken6.weight.detach(),
                                hid_units, nb_classes,args.combinetype,unify_dim).cuda()
                idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
                pretrain_embs = embeds[0, idx_train]
                train_lbls = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()

                if use_episode_holdout:
                    train_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels_cuda.device)
                    train_mask[idx_train] = True
                    idx_test_episode = (~train_mask).nonzero(as_tuple=True)[0]
                    test_embs = embeds[0, idx_test_episode]
                    test_lbls = labels_cuda[idx_test_episode]
                else:
                    idx_test_episode = fixed_idx_test
                    test_embs = embeds[0, fixed_idx_test]

                opt = torch.optim.Adam([
                    {'params': log.parameters()}
                ], lr=downstreamlr)
                log = log.cuda()
                best = 1e9
                pat_steps = 0
                best_acc = torch.zeros(1)
                best_acc = best_acc.cuda()
            
                for _ in range(400):
                    log.train()
                    opt.zero_grad()
                    logits = log(features,sp_adj,sparse,model.gcn,idx_train,pretrain_embs,downk,train_lbls,1).float().cuda()
                    loss = xent(logits, train_lbls)
                    if loss < best:
                        best = loss
                        cnt_wait = 0
                    else:
                        cnt_wait += 1
                    if cnt_wait == patience:
                        print('Early stopping!')
                        break
                    
                    loss.backward(retain_graph=True)
                    opt.step()
                logits = log(features,sp_adj,sparse,model.gcn,idx_test_episode,test_embs,downk)

                preds = torch.argmax(logits, dim=1).cuda()
                acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]


                # ===== 转 cpu numpy =====
                y_true = test_lbls.detach().cpu().numpy()
                y_pred = preds.cpu().numpy()

                # ===== 计算指标 =====
                
                macro_f1 = f1_score(y_true, y_pred, average='macro')
                micro_f1 = f1_score(y_true, y_pred, average='micro')

                accs.append(acc * 100)
                macro_f1s.append(macro_f1 * 100)
                micro_f1s.append(micro_f1 * 100)

                print('ACC       : [{:.4f}]'.format(acc))
                print('Macro-F1  : [{:.4f}]'.format(macro_f1))
                print('Micro-F1  : [{:.4f}]'.format(micro_f1))

                tot += acc.item()
                tot_macro += macro_f1
                tot_micro += micro_f1
            print('-' * 100)
            print('-' * 100)

            # 修正后的打印部分
            eval_count = len(accs)
            avg_acc = (tot / eval_count).item()
            avg_macro = tot_macro / eval_count
            avg_micro = tot_micro / eval_count

            print(f'Average ACC      : [{avg_acc:.4f}]')
            print(f'Average Macro-F1 : [{avg_macro:.4f}]')
            print(f'Average Micro-F1 : [{avg_micro:.4f}]')

            accs = torch.tensor(accs)
            macro_f1s = torch.tensor(macro_f1s)
            micro_f1s = torch.tensor(micro_f1s)

            print('ACC Mean   : [{:.4f}]'.format(accs.mean().item()))
            print('ACC Std    : [{:.4f}]'.format(accs.std().item()))

            print('Macro Mean : [{:.4f}]'.format(macro_f1s.mean().item()))
            print('Macro Std  : [{:.4f}]'.format(macro_f1s.std().item()))

            print('Micro Mean : [{:.4f}]'.format(micro_f1s.mean().item()))
            print('Micro Std  : [{:.4f}]'.format(micro_f1s.std().item()))

            print('-' * 100)

            row = [
                'Final:',
                "lr", lr,
                "downstreamlr", downstreamlr,
                "nb_epochs", nb_epochs,
                "hid_units", hid_units,
                "ACC_mean", accs.mean().item(),
                "ACC_std", accs.std().item(),
                "MacroF1_mean", macro_f1s.mean().item(),
                "MacroF1_std", macro_f1s.std().item(),
                "MicroF1_mean", micro_f1s.mean().item(),
                "MicroF1_std", micro_f1s.std().item()
            ]

            #row = ['Final:',"lr",lr,"downstreamlr",downstreamlr,"nb_epochs",nb_epochs,hid_units,accs.mean().item(),accs.std().item()]
            out = open("data/ICML25_{}_fewshot.csv".format(args.dataset.lower()), "a", newline="")
            csv_writer = csv.writer(out, dialect="excel")
            csv_writer.writerow(row)
    final_end_time = time.perf_counter()
    downstream_elapsed = final_end_time - pretrain_end_time
    total_elapsed = final_end_time - global_start_time

    print("=" * 100)
    print("All training finished!")
    print(f"Downstream elapsed time: {downstream_elapsed:.2f} seconds")
    print(f"Pretrain elapsed time: {pretrain_elapsed:.2f} seconds")
    print(f"Train Time / Epoch        : {train_time_per_epoch:.4f} sec")
    
    print(f"Downstream elapsed time: {downstream_elapsed/60:.2f} minutes")
    print(f"Total elapsed time: {total_elapsed:.2f} seconds")
    print(f"Total elapsed time: {total_elapsed/60:.2f} minutes")
    print("=" * 100)
