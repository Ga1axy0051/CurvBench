import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_h = 4):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(n_in, n_h) ,
                                 nn.ReLU(), 
                                 nn.Linear(n_h, n_out))
    def forward(self, x):
        return torch.mean(self.mlp(x), dim=0)
