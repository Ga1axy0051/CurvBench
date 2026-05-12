import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GAT
from layers.prompt import *
class GatLayers(torch.nn.Module):
    def __init__(self, n_in, n_h, num_layers_num, dropout=0.6, heads=2):
        super(GatLayers, self).__init__()

        self.act = torch.nn.PReLU() 
        self.num_layers_num = num_layers_num
        self.g_net, self.bns = self.create_net(n_in, int(n_h/heads), num_layers_num, heads)
        #self.g_net, self.bns = self.create_net(n_in, n_h, num_layers_num, heads)
        self.heads = heads
        self.dropout = torch.nn.Dropout(p=dropout)

    def create_net(self, input_dim, hidden_dim, num_layers, heads):
        self.convs = torch.nn.ModuleList() 
        self.bns = torch.nn.ModuleList()  

        for i in range(num_layers):
            if i:
                conv = GAT(hidden_dim * heads, hidden_dim, heads)
                #conv = GAT(input_dim, hidden_dim, heads)
            else:
                conv = GAT(input_dim, hidden_dim, heads)
            
            bn = torch.nn.BatchNorm1d(hidden_dim * heads)  
            #bn = torch.nn.BatchNorm1d(hidden_dim) 
            
            self.convs.append(conv)
            self.bns.append(bn)
        
        return self.convs, self.bns

    def forward(self, seq, adj, sparse=True, LP=False, prompt_layers=None):
        graph_output = torch.squeeze(seq, dim=0) 
        xs = []  
        if prompt_layers:
            assert len(prompt_layers) == self.num_layers_num

        for i in range(self.num_layers_num):
            input=(graph_output,adj)
            if i:
                graph_output = self.convs[i](input) + graph_output
            else:
                graph_output = self.convs[i](input)
            
            if prompt_layers:
                graph_output = prompt_layers[i](graph_output)
            
            if LP:
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                
            xs.append(graph_output)

        return graph_output.unsqueeze(dim=0) 

