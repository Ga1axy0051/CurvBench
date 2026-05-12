class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, out_channels, config, task='nc'):
        super().__init__()
        self.config = config
        self.conv1 = SAGEConv(in_channels, config['hidden_channels'])
        final_out = config['hidden_channels'] if task == 'lp' else out_channels
        self.conv2 = SAGEConv(config['hidden_channels'], final_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.config['dropout'], training=self.training)
        x = self.conv2(x, edge_index)
        return x