class PCNet(nn.Module):
    def __init__(self, in_channels, out_channels, config, task='nc'):
        super().__init__()
        self.config = config
        self.hidden = config['hidden_channels']
        self.dropout = config['dropout']
        self.K, self.N, self.t = config['pc_K'], config['pc_N'], config['pc_t']
        self.p, self.eta = config['pc_p'], config['pc_eta']
        
        final_out = self.hidden if task == 'lp' else out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, final_out)
        )
        self.theta_0 = nn.Parameter(torch.ones(1, 1))
        self.thetas = nn.Parameter(torch.ones(self.K, 1) * 0.1) 
        C = get_pc_coefficients(self.K, self.N, self.t)
        self.register_buffer('C', C)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        h = self.mlp(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        L_index, L_weight = get_generalized_laplacian(edge_index, num_nodes, self.eta, self.p)
        L_sparse = torch.sparse_coo_tensor(L_index, L_weight, torch.Size([num_nodes, num_nodes])).to(x.device)
        
        T_list = [h]
        current_T = h
        for n in range(1, self.N + 1):
            neg_L_X = -torch.sparse.mm(L_sparse, current_T)
            current_T = neg_L_X / n
            T_list.append(current_T)
            
        T_tensor = torch.stack(T_list, dim=0) 
        inner_sum = torch.einsum('kn,nvc->kvc', self.C, T_tensor) 
        final_sum = torch.sum(self.thetas.unsqueeze(2) * inner_sum, dim=0) 
        Z = self.theta_0 * h + final_sum
        return Z