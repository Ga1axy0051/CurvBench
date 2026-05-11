from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random
import torch.nn.functional as F
from torch_geometric.utils import (
    to_undirected,
    remove_self_loops,
    coalesce
)
from models import LogReg
from preprompt import PrePrompt,pca_compression
import preprompt
from utils import process
import pdb
import aug
import os
import tqdm
import argparse
from robust import attack_adj
from downprompt import downprompt,prefeatureprompt
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("MDGPT")
import torch.nn.functional as F
#"Cornell", "Pubmed", "Cora", "Disease", "Citeseer", "Telecom", "Airport", "Actor"
parser.add_argument('--dataset', type=str, default="cs_phds", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--save_name', type=str, default='model_add_node_lay3_computers.pkl', help='save ckpt name')
parser.add_argument('--val_name', type=str, default='noval_graphcl_BZR.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
# parser.add_argument('--local_rank', type=str, help='local rank for dist')      
args = parser.parse_args()

# world_size = torch.cuda.device_count()
# local_rank = args.local_rank
# dist_backend = 'nccl'
# dist.init_process_group(backend=dist_backend)
print('-' * 100)
print(args)
print('-' * 100)

# dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)
import time
pretrain_start = time.time()

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.set_device(int(local_rank))

from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit,Actor,WikipediaNetwork,WebKB,Flickr,LINKXDataset
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
# training params
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
        print(f"[data] PCA compresses to {max_k} dims first, then pads to {k}.")

    seq = pca_compression(seq, k=max_k)

    if max_k < k:
        pad = np.zeros((seq.shape[0], k - max_k), dtype=seq.dtype)
        seq = np.concatenate([seq, pad], axis=1)

    return seq

# idx_train = torch.load("data/fewshot/0/idx.pt").type(torch.long).cuda()
batch_size = 100
nb_epochs = 200
patience = 50
lr = 0.0001
downstreamlrlist = [0.003]
l2_coef = 0.0001
drop_prob = 0.0
hid_units = 256
negetive_sample_num=100
test_num=321
sparse = True
useMLP =False
class_num = 2
LP = False


nonlinearity = 'prelu'  # special name to separate parameters
dataset = args.dataset
device = torch.device("cuda")


best = 1e9
firstbest = 0

    # lr = 0.01
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
dataset6 = DiseaseDataset(root='/')
loader6 = DataLoader(dataset6)

dataset7 = TelecomDataset(root='/')
loader7 = DataLoader(dataset7)

dataset8 = CSPhDsDataset(root='')
loader8 = DataLoader(dataset8)


cnt_wait = 0
mode='meta'
b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
unify_dim =50
#"Cornell", "Pubmed", "Cora", "Disease", "Citeseer", "Telecom", "cs_phd", "Actor"
#5  2  1  6  3  7  8  4
for step, (data1,data2,data3,data4,data5,data6) in enumerate(zip(loader1,loader6,loader3,loader7,loader8,loader4)): # data1 is not used
    #print(step,data1,data2)
    features11,adj1= process.process_tu(data1,data1.x.shape[1])
    
    features22,adj2= process.process_tu(data2,data2.x.shape[1])
    #adj2 = attack_adj('cora',mode,0.05)
    features33,adj3= process.process_tu(data3,data3.x.shape[1])
    #adj3 = attack_adj('citeseer',mode,0.05)
    features44,adj4= process.process_tu(data4,data4.x.shape[1])
    #adj4 = attack_adj('pubmed',mode,0.05)
    features55,adj5= process.process_tu(data5,data5.x.shape[1])
    #adj5 = attack_adj('Chameleon',mode,0.05)
    features66,adj6= process.process_tu(data6,data6.x.shape[1])
    #adj6 = attack_adj('Squirrel',mode,0.05)

#        # features1 = model.texttoken1(torch.FloatTensor(features11).cuda())
#        # features2 = model.texttoken2(torch.FloatTensor(features22).cuda())
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
#        # print("texttoken1",model.texttoken1.weight)
#        setshape1 = adj1.shape[0]
#        setshape2 = adj2.shape[0]
    adj = process.combine_dataset(adj1,adj2,adj3,adj4,adj5,adj6)
    # print(adj)
    # print(adj.shape)
    negetive_sample = preprompt.prompt_pretrain_sample(adj,negetive_sample_num)
    # negetive_sample2 = preprompt.prompt_pretrain_sample(adj2,50)
    # negetive_sample3 = preprompt.prompt_pretrain_sample(adj3,50)
    # negetive_sample4 = preprompt.prompt_pretrain_sample(adj4,50)
    # negetive_sample5 = preprompt.prompt_pretrain_sample(adj5,50)
    # negetive_sample = torch.cat((torch.Tensor(negetive_sample2).cuda(),torch.Tensor(negetive_sample3).cuda(),torch.Tensor(negetive_sample4).cuda(),torch.Tensor(negetive_sample5).cuda()),dim=0).cuda()
adj1 = process.normalize_adj(adj1 + sp.eye(adj1.shape[0]))
adj2 = process.normalize_adj(adj2 + sp.eye(adj2.shape[0]))
adj3 = process.normalize_adj(adj3 + sp.eye(adj3.shape[0]))
adj4 = process.normalize_adj(adj4 + sp.eye(adj4.shape[0]))
adj5 = process.normalize_adj(adj5 + sp.eye(adj5.shape[0]))
adj6 = process.normalize_adj(adj6 + sp.eye(adj6.shape[0]))
if sparse:
    sp_adj1 = process.sparse_mx_to_torch_sparse_tensor(adj1)
    sp_adj4 = process.sparse_mx_to_torch_sparse_tensor(adj4)
    sp_adj2 = process.sparse_mx_to_torch_sparse_tensor(adj2)
    sp_adj3 = process.sparse_mx_to_torch_sparse_tensor(adj3)
    sp_adj5 = process.sparse_mx_to_torch_sparse_tensor(adj5)
    sp_adj6 = process.sparse_mx_to_torch_sparse_tensor(adj6)
    
# else:
#     adj1 = adj1.todense()
#     adj2 = adj2.todense()
# if not sparse:
#     adj1 = torch.FloatTensor(adj1[np.newaxis])
#     adj2 = torch.FloatTensor(adj2[np.newaxis])
model = PrePrompt(unify_dim, hid_units, nonlinearity,negetive_sample,3,0.1,args.combinetype)
    
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
if torch.cuda.is_available():
    print('Using CUDA')
    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    model = model.cuda()
    features1 = features1.cuda()
    features4 = features4.cuda()
    features2 = features2.cuda()
    features3 = features3.cuda()
    features5 = features5.cuda()
    features6 = features6.cuda()
    if sparse:
        sp_adj1 = sp_adj1.cuda()
        sp_adj4 = sp_adj4.cuda()
        sp_adj2 = sp_adj2.cuda()
        sp_adj3 = sp_adj3.cuda()
        sp_adj5 = sp_adj5.cuda()
        sp_adj6 = sp_adj6.cuda()
    # else:
    #     adj1 = adj1.cuda()
    #     adj2 = adj2.cuda()
    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()
epoch_train_start = time.time()
real_epoch_num = 0
for epoch in range(nb_epochs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loss = 0
    regloss = 0
    # model.texttoken1.reset_parameters()
    # model.texttoken2.reset_parameters()
    # best = 1e9
    model.train()
    optimiser.zero_grad()
    loss = model( features1,features2,features3,features4,features5,features6,sp_adj1 if sparse else adj1,
                sp_adj2 if sparse else adj2,sp_adj3 if sparse else adj3,sp_adj4 if sparse else adj4,sp_adj5 if sparse else adj5,sp_adj6 if sparse else adj6,
                sparse, None, None, None)
    loss.backward()
    optimiser.step()
    # loss.backward()
    # optimiser.step()
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
model = PrePrompt(unify_dim, hid_units, nonlinearity,1,3,0.1,args.combinetype)
pretrain_end = time.time()
pretrain_time = pretrain_end - pretrain_start


print('#'*50)
print('Downastream dataset is ',args.dataset)

# if args.dataset == 'Cora' or args.dataset =='Citeseer' or args.dataset =='Pubmed':
#     dataset = Planetoid(root='data', name=args.dataset)                                                                                         
# if args.dataset == 'Computers' or args.dataset =='Photo':
#     dataset = Amazon(root='data', name=args.dataset) 
# if args.dataset == 'Reddit':
#     dataset = Reddit(root='data/Reddit') 

if args.dataset == 'Cora' or args.dataset =='Citeseer' or args.dataset =='Pubmed':
    dataset = Planetoid(root='data', name=args.dataset)                                                                                         
    if args.dataset =='Pubmed':
        testnum=100
if args.dataset == 'Chameleon' or args.dataset == 'Squirrel':
    dataset=WikipediaNetwork(root='data',name=args.dataset)

if args.dataset == 'Cornell':
    dataset=WebKB(root='data', name=args.dataset)

if args.dataset == 'Actor':
    dataset=Actor(root='data/Actor')
if args.dataset == 'Disease':
    dataset=DiseaseDataset(root='')
if args.dataset == 'Telecom':
    dataset=TelecomDataset(root='')
if args.dataset == 'Airport':
    dataset=AirportDataset(root='')
if args.dataset == 'cs_phds':
    dataset=CSPhDsDataset(root='')

if args.dataset == 'Cornell' or args.dataset == 'Wisconsin':
    dataset=WebKB(root='data', name=args.dataset)
if args.dataset == 'Penn94':
        dataset=LINKXDataset(root='data/Penn',name='penn94')

print(dataset)
loader = DataLoader(dataset)
for data in loader:
    print(data)
    testnum=50
    features,adj= process.process_tu(data,data.x.shape[1])
    #adj=attack_adj('Cornell',mode,0.05)
    print('process done')
    features = pca_compression_fixed_dim(features, k=unify_dim)
    print('pca')
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    print("adj")
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_adj = sp_adj.cuda()
    features = torch.FloatTensor(features).cuda()
    labels = data.y
    labels_cuda = labels.cuda()
    data=np.array(data.y)

    np.unique(data)

    nb_classes=len(np.unique(data))
    print(nb_classes)

    if args.dataset == 'cs_phds':
        idx_test = torch.arange(labels.shape[0], device=labels_cuda.device)
    else:
        random_number = random.randint(0, 20)
        a=filter(lambda x: x < 2*testnum+random_number and int(labels[x]) != -1, range(random_number,random_number+2*testnum))
        idx_test=list(a)
        print("idx_test2",idx_test)
    

model = model.cuda()

model.load_state_dict(torch.load(args.save_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None,LP)

acclist = torch.FloatTensor(100,).cuda()


print("=" * 80)
print(f"Upstream Pretraining Time: {pretrain_time:.2f} sec")
print("=" * 80)

downstream_start = time.time()
for downstreamlr in downstreamlrlist:
    
    print(labels.shape)
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
    f1_micro_list = []
    f1_macro_list = []
    print('-' * 100)

    for shotnum in range(1,2):
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        cnt_wait = 0
        best = 1e9
        best_t = 0
        print("shotnum",shotnum)
        for i in tqdm(range(50)):
            log = downprompt(model.texttoken1.weight.detach(),model.texttoken2.weight.detach(),model.texttoken3.weight.detach(),model.texttoken4.weight.detach(),model.texttoken5.weight.detach(),model.texttoken6.weight.detach(),
                             model.pretext1.weight.detach(),model.pretext2.weight.detach(),model.pretext3.weight.detach(),model.pretext4.weight.detach(),model.pretext5.weight.detach(),model.pretext6.weight.detach(),
                             hid_units, nb_classes,args.combinetype,unify_dim).cuda()
    

            idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
            pretrain_embs = embeds[0, idx_train]
            train_lbls = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()
            if args.dataset == 'cs_phds':
                train_mask = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels_cuda.device)
                train_mask[idx_train] = True
                idx_test_episode = (~train_mask).nonzero(as_tuple=True)[0]
                test_embs = embeds[0, idx_test_episode]
                test_lbls = labels_cuda[idx_test_episode]
            else:
                idx_test_episode = idx_test
                test_embs = embeds[0, idx_test]
                test_lbls = labels[idx_test].cuda()
            # print("true",i,train_lbls)
            # opt = torch.optim.Adam(log.parameters(),downstreamprompt.parameters(),lr=0.01, weight_decay=0.0)
            opt = torch.optim.Adam([
                {'params': log.parameters()}
            ], lr=downstreamlr)
            # opt = torch.optim.Adam(log.parameters(), lr=downstreamlr)
            log = log.cuda()
            best = 1e9
            pat_steps = 0
            best_acc = torch.zeros(1)
            best_acc = best_acc.cuda()
            for _ in range(400):
                log.train()
                opt.zero_grad()
                # print(features)
                # preval_embs = embeds[0, idx_val]
                
                # logits = log(pretrain_embs)
                logits = log(features,sp_adj,sparse,model.gcn,idx_train,pretrain_embs,train_lbls,1).float().cuda()
                loss = xent(logits, train_lbls)


                
                # print('loss' ,loss)
                # print("predict=",torch.argmax(logits, dim=1))
                # print("lbels",train_lbls)
                # print("predict=",torch.argmax(logits, dim=1))
                # print("train acc",torch.sum(torch.argmax(logits, dim=1)== train_lbls).float() / train_lbls.shape[0])
                if loss < best:
                    best = loss
                    # best_t = epoch
                    cnt_wait = 0
                    # torch.save(model.state_dict(), args.save_name)
                else:
                    cnt_wait += 1
                if cnt_wait == patience:
                    print('Early stopping!')
                    break
                
                loss.backward(retain_graph=True)
                opt.step()
            logits = log(features,sp_adj,sparse,model.gcn,idx_test_episode,test_embs)
            # print("logits",logits)
            # print(log.a)
            preds = torch.argmax(logits, dim=1).cuda()
            # print('preds:',preds)
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)
            # ===== 新增 F1 =====
            preds_np = preds.detach().cpu().numpy()
            labels_np = test_lbls.detach().cpu().numpy()

            f1_micro = f1_score(labels_np, preds_np, average='micro')
            f1_macro = f1_score(labels_np, preds_np, average='macro')

            f1_micro_list.append(f1_micro * 100)
            f1_macro_list.append(f1_macro * 100)

            print('acc:[{:.4f}]  micro_f1:[{:.4f}]  macro_f1:[{:.4f}]'.format(
                acc.item(), f1_micro, f1_macro
            ))
            tot += acc
        print('-' * 100)
        print('Average accuracy:[{:.4f}]'.format(tot.item() / 50))
        accs = torch.stack(accs)

        f1_micro_tensor = torch.tensor(f1_micro_list)
        f1_macro_tensor = torch.tensor(f1_macro_list)

        print('ACC       : {:.2f} ± {:.2f}'.format(
            accs.mean().item(),
            accs.std().item()
        ))

        print('Micro-F1  : {:.2f} ± {:.2f}'.format(
            f1_micro_tensor.mean().item(),
            f1_micro_tensor.std().item()
        ))

        print('Macro-F1  : {:.2f} ± {:.2f}'.format(
            f1_macro_tensor.mean().item(),
            f1_macro_tensor.std().item()
        ))


        print('-' * 100)
        row = ["execute:",shotnum,lr,downstreamlr,hid_units,accs.mean().item(),accs.std().item()]
        out = open("data/NIPS24_{}_fewshot.csv".format(args.dataset.lower()), "a", newline="")
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
downstream_end = time.time()
downstream_time = downstream_end - downstream_start

print("=" * 80)
print(f"Upstream Pretraining Time: {pretrain_time:.2f} sec")
print(f"Downstream Few-shot Time: {downstream_time:.2f} sec")
print(f"Train Time / Epoch        : {train_time_per_epoch:.2f} sec")
print("=" * 80)
