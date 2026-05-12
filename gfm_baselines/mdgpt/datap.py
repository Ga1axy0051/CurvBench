from unittest import loader
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import f1_score
import random

from models import LogReg
from preprompt import PrePrompt,pca_compression
import preprompt
from utils import process
import pdb
import aug
import os
import tqdm
import argparse
from downprompt import downprompt,prefeatureprompt
import csv
from tqdm import tqdm
parser = argparse.ArgumentParser("MDGPT")

import torch.nn.functional as F
parser.add_argument('--dataset', type=str, default="Cora", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--save_name', type=str, default='model_add_node_lay3_computers.pkl', help='save ckpt name')
parser.add_argument('--val_name', type=str, default='noval_graphcl_BZR.pkl', help='save val')
parser.add_argument('--combinetype', type=str, default='mul', help='the type of text combining')
# parser.add_argument('--local_rank', type=str, help='local rank for dist')      
args = parser.parse_args()


import torch
import torch.nn as nn
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.cuda.set_device(int(local_rank))

from torch_geometric.datasets import TUDataset,Planetoid,Amazon,Coauthor,Reddit,WikipediaNetwork,Actor
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


dataset1 = Planetoid(root='data', name='Cora')                                                                                               
loader1 = DataLoader(dataset1)

dataset2 = Planetoid(root='data', name='Citeseer')                                                                                         
loader2 = DataLoader(dataset2)

dataset3 = Planetoid(root='data', name='Pubmed')                                                                                         
loader3 = DataLoader(dataset3)



dataset4 = Amazon(root='data',name='Photo')
loader4 = DataLoader(dataset4)

dataset5 = Amazon(root='data',name='Computers')

loader5 = DataLoader(dataset5)


dataset6 = Coauthor(root='data',name='CS')
loader6 = DataLoader(dataset6)

dataset7= WikipediaNetwork(root='data', name='Chameleon')
loader7 = DataLoader(dataset7)

dataset8= WikipediaNetwork(root='data', name='Squirrel')
loader8 = DataLoader(dataset8)

dataset9= Actor(root='data/Actor')
loader9 = DataLoader(dataset9)


for step, (data1,data2,data3,data4,data5,data6,data7,data8,data9) in enumerate(zip(loader1,loader2,loader3,loader4,loader5,loader6,loader7,loader8,loader9)):
    
    print("data2:",data2)
    #print("data2[0]:",data2[0])

    print("data7:",data7)
    #print("data7[0]:",data7[0])

    print("data8:",data8)
   # print("data8[0]:",data8[0])
    print("data9:",data9)
    #print("data9[0]:",data9[0])
    data7_batch=data7[0]
    
    #features11,adj1= process.process_tu(data1,data1.x.shape[1])
    #features22,adj2= process.process_tu(data2,data2.x.shape[1])
    #features33,adj3= process.process_tu(data3,data3.x.shape[1])
    #features44,adj4= process.process_tu(data4,data4.x.shape[1])
    #features55,adj5= process.process_tu(data5,data5.x.shape[1])
    #features66,adj6= process.process_tu(data6,data6.x.shape[1])
    features77,adj7= process.process_tu(data7,data7.x.shape[1])
    features88,adj8= process.process_tu(data8,data8.x.shape[1])
    features99,adj9= process.process_tu(data9,data9.x.shape[1])

    #features1 = pca_compression(features11,k=50)
    #features2 = pca_compression(features22,k=50)
    #features3 = pca_compression(features33,k=50)
    #features4 = pca_compression(features44,k=50)
    #features5 = pca_compression(features55,k=50)
    #features6 = pca_compression(features66,k=50)
    features7 = pca_compression(features77,k=50)
    features8 = pca_compression(features88,k=50)
    features9 = pca_compression(features99,k=50)





    #features2 = torch.FloatTensor(features2).cuda()
    
    adj = process.combine_dataset(adj7,adj8,adj9)
    #("adj:",adj)
    print("adj.shape:",adj.shape)
    negetive_sample = preprompt.prompt_pretrain_sample(adj,50)
    print(negetive_sample)
    