import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv 
from torch_geometric.utils import dense_to_sparse
from torch_sparse import spmm
class GAT(nn.Module):
    def __init__(self, in_ft, out_ft, nheads=2, concat=True, dropout=0.6, alpha=0.2, bias=True):
        super(GAT, self).__init__()
        self.gat = GATConv(in_ft, out_ft, heads=nheads, concat=concat, dropout=dropout, bias=bias)
        self.act = nn.PReLU()  
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        x = input[0]  
        adj = input[1]

        return self.dropout(self.act(self.gat(x, adj)))
