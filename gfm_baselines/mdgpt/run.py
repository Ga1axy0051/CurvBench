from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from utils import Calbound
from models import LogReg
from runpreprompt import PrePrompt,pca_compression
import runpreprompt
from utils import process
import pdb
import aug
import os
import tqdm
import argparse
from downprompt15 import downprompt,prefeatureprompt
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("MDGPT")
import torch.nn.functional as F
parser.add_argument('--dataset', type=str, default="Chameleon", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.5, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--save_name', type=str, default='model_add_node_lay3_computers.pkl', help='save ckpt name')
parser.add_argument('--val_name', type=str, default='noval_graphcl_BZR.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
# parser.add_argument('--local_rank', type=str, help='local rank for dist')      
args = parser.parse_args()

print(args)


# dataset = args.dataset
aug_type = args.aug_type
# drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit,Actor,WikipediaNetwork,WebKB
print('-' * 100)
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
print('-' * 100)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
print('-' * 100)
print("checkpoint")
batch_size = 128
nb_epochs = 200
patience = 80
lr = 0.001
l2_coef = 0.0001
drop_prob = 0.5
hid_units = 256
sparse = True
useMLP =False
class_num = 2
LP = False
downstreamlrlist = [0.001]

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


dataset4 = WikipediaNetwork(root='data',name='Chameleon')
loader4 = DataLoader(dataset4)

dataset5 = WebKB(root='data', name='Wisconsin')
loader5 = DataLoader(dataset5)

dataset6 = WikipediaNetwork(root='data', name='Squirrel')
loader6 = DataLoader(dataset6)

dataset7 = Actor(root='data/Actor')
loader7 = DataLoader(dataset7)

dataset8 = WebKB(root='data', name='Cornell')
loader8 = DataLoader(dataset8)

cnt_wait = 0

b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()

unify_dim = 50

for step, (data1,data2,data3,data4,data5,data6,data7,data8) in enumerate(zip(loader1,loader2,loader3,loader8,loader5,loader6,loader7,loader4)):

    print(step,'\n',data2)
    features11,adj1= process.process_tu(data1,data1.x.shape[1])
    features22,adj2= process.process_tu(data2,data2.x.shape[1])
    features33,adj3= process.process_tu(data3,data3.x.shape[1])
    features44,adj4= process.process_tu(data4,data4.x.shape[1])
    features55,adj5= process.process_tu(data5,data5.x.shape[1])
    features66,adj6= process.process_tu(data6,data6.x.shape[1])
    features77,adj7= process.process_tu(data7,data7.x.shape[1])

    features1 = pca_compression(features11,k=unify_dim)
    features2 = pca_compression(features22,k=unify_dim)
    features3 = pca_compression(features33,k=unify_dim)
    features4 = pca_compression(features44,k=unify_dim)
    features5 = pca_compression(features55,k=unify_dim)
    features6 = pca_compression(features66,k=unify_dim)
    features7 = pca_compression(features77,k=unify_dim)
    
    features1 = torch.FloatTensor(features1).cuda()
    features2 = torch.FloatTensor(features2).cuda()
    features3 = torch.FloatTensor(features3).cuda()
    features4 = torch.FloatTensor(features4).cuda()
    features5 = torch.FloatTensor(features5).cuda()
    features6 = torch.FloatTensor(features6).cuda()
    features7 = torch.FloatTensor(features7).cuda()
    
    adj = process.combine_dataset(adj1,adj2,adj3,adj4,adj5,adj6,adj7)
    negative_sample = runpreprompt.prompt_pretrain_sample(adj,50)

adj1 = process.normalize_adj(adj1 + sp.eye(adj1.shape[0]))
adj2 = process.normalize_adj(adj2 + sp.eye(adj2.shape[0]))
adj3 = process.normalize_adj(adj3 + sp.eye(adj3.shape[0]))
adj4 = process.normalize_adj(adj4 + sp.eye(adj4.shape[0]))
adj5 = process.normalize_adj(adj5 + sp.eye(adj5.shape[0]))
adj6 = process.normalize_adj(adj6 + sp.eye(adj6.shape[0]))
adj7 = process.normalize_adj(adj7 + sp.eye(adj7.shape[0]))
if sparse:
    sp_adj1 = process.sparse_mx_to_torch_sparse_tensor(adj1)
    sp_adj2 = process.sparse_mx_to_torch_sparse_tensor(adj2)
    sp_adj3 = process.sparse_mx_to_torch_sparse_tensor(adj3)
    sp_adj4 = process.sparse_mx_to_torch_sparse_tensor(adj4)
    sp_adj5 = process.sparse_mx_to_torch_sparse_tensor(adj5)
    sp_adj6 = process.sparse_mx_to_torch_sparse_tensor(adj6)
    sp_adj7 = process.sparse_mx_to_torch_sparse_tensor(adj7)
    
model = PrePrompt(unify_dim, hid_units, nonlinearity, negative_sample,3,0.1,args.combinetype)
    
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
if torch.cuda.is_available():
    print('Using CUDA')
    # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    model = model.cuda()
    features1 = features1.cuda()
    features2 = features2.cuda()
    features3 = features3.cuda()
    features4 = features4.cuda()
    features5 = features5.cuda()
    features6 = features6.cuda()
    features7 = features7.cuda()

    if sparse:
        sp_adj1 = sp_adj1.cuda()
        sp_adj2 = sp_adj2.cuda()
        sp_adj3 = sp_adj3.cuda()
        sp_adj4 = sp_adj4.cuda()
        sp_adj5 = sp_adj5.cuda()
        sp_adj6 = sp_adj6.cuda()
        sp_adj7 = sp_adj7.cuda()
        


for epoch in range(nb_epochs):
    torch.cuda.empty_cache()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loss = 0
    regloss = 0

    # best = 1e9
    model.train()
    optimiser.zero_grad()
    loss = model( features1,features2,features3,features4,features5,features6,features7,
                sp_adj1 if sparse else adj1, sp_adj2 if sparse else adj2,sp_adj3 if sparse else adj3,sp_adj4 if sparse else adj4,sp_adj5 if sparse else adj5,
                sp_adj6 if sparse else adj6, sp_adj7 if sparse else adj7,
                sparse, None, None, None)
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


model = PrePrompt(unify_dim, hid_units, nonlinearity,1,3,0.1,args.combinetype)

print('#'*50)
print('Downastream dataset is ',args.dataset)

if args.dataset == 'Cora' or args.dataset =='Citeseer' or args.dataset =='Pubmed':
    dataset = Planetoid(root='data', name=args.dataset)                                                                                         
if args.dataset == 'Computers' or args.dataset =='Photo':
    dataset = Amazon(root='data', name=args.dataset) 
if args.dataset == 'Reddit':
    dataset = Reddit(root='data/Reddit') 
if args.dataset == 'Chameleon' or args.dataset == 'Squirrel':
    dataset=WikipediaNetwork(root='data',name=args.dataset)
if args.dataset == 'Actor':
    dataset=WikipediaNetwork(root='data/Actor')
    
if args.dataset == 'Cornell' or args.dataset == 'Wisconsin':
    dataset=WebKB(root='data', name=args.dataset)

print(dataset)


loader = DataLoader(dataset)
for data in loader:
    print(data)
    features,adj= process.process_tu(data,data.x.shape[1])
    print('process done')
    features = pca_compression(features,k=unify_dim)
    print('pca')
    adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    print("adj")
    sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_adj = sp_adj.cuda()
    features = torch.FloatTensor(features).cuda()
    print(features.shape)
    idx_test = range(data.y.shape[0]-100,data.y.shape[0])
    labels = data.y
    data=np.array(data.y)

    np.unique(data)

    nb_classes=len(np.unique(data))
    print(nb_classes)
    

model = model.cuda()

model.load_state_dict(torch.load(args.save_name))

embeds, _ = model.embed(features, sp_adj if sparse else adj, sparse, None,LP)
acclist = torch.FloatTensor(100,).cuda()



for downstreamlr in downstreamlrlist:
    
    print(labels.shape)
    # val_lbls = torch.argmax(labels[idx_val.cpu()], dim=1).cuda()
    test_lbls = labels[idx_test].cuda()
    tot = torch.zeros(1)
    tot = tot.cuda()
    accs = []
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
            log = downprompt(model.texttoken1.weight.detach(),model.texttoken2.weight.detach(),model.texttoken3.weight.detach(),model.texttoken4.weight.detach(),model.texttoken5.weight.detach(),model.texttoken6.weight.detach(),model.texttoken7.weight.detach(),model.sumtext.weight.detach(),
                            model.pretext1.weight.detach(),model.pretext2.weight.detach(),model.pretext3.weight.detach(),model.pretext4.weight.detach(),model.pretext5.weight.detach(),model.pretext6.weight.detach(),model.pretext7.weight.detach(),hid_units, nb_classes,args.combinetype,unify_dim).cuda()
            idx_train = torch.load("data/fewshot_{}/{}-shot_{}/{}/idx.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).cuda()
            pretrain_embs = embeds[0, idx_train]
            test_embs = embeds[0, idx_test]
            train_lbls = torch.load("data/fewshot_{}/{}-shot_{}/{}/labels.pt".format(args.dataset.lower(),shotnum,args.dataset.lower(),i)).type(torch.long).squeeze().cuda()
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
                logits = log(features,sp_adj,sparse,model.gcn,idx_train,pretrain_embs,train_lbls,1).float().cuda()
                loss = xent(logits, train_lbls)

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
            logits = log(features,sp_adj,sparse,model.gcn,idx_test,test_embs)
            preds = torch.argmax(logits, dim=1).cuda()
            acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            accs.append(acc * 100)
            print('acc:[{:.4f}]'.format(acc))
            tot += acc
        print('-' * 100)
        print('15:','Average accuracy:[{:.4f}]'.format(tot.item() / 50))
        accs = torch.stack(accs)
        print('Mean:[{:.4f}]'.format(accs.mean().item()))
        print('Std :[{:.4f}]'.format(accs.std().item()))
        print('-' * 100)
        row = ['15:',shotnum,lr,"downstreamlrlist",downstreamlrlist,nb_epochs,downstreamlr,hid_units,accs.mean().item(),accs.std().item()]
        out = open("data/NIPS24_{}_fewshot.csv".format(args.dataset.lower()), "a", newline="")
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
