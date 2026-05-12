from torch_geometric.nn import Node2Vec
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data
import torch
from data.data_process import unify_feature_dimension, link_k_shot_split


class FlattenLabels(BaseTransform):
    def __init__(self):
        super().__init__()

    def forward(self, data):
        data.y = data.y.reshape(-1)
        return data


class UnifyFeatureDims(BaseTransform):
    def __init__(self, uni_dim: int, mode='random'):
        super().__init__()
        assert mode in ['random', 'svd'], "mode must be 'random' or 'svd'!"
        self.uni_dim = uni_dim
        self.mode = mode

    def forward(self, data: Data):
        d = data.x.shape[-1]
        x = data.x.float()
        if self.mode == 'random':
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)
            data.x = x @ torch.randn(d, self.uni_dim)
        else:
            data.x = unify_feature_dimension(x, self.uni_dim)
        return data


class FewShotLinkSplit(BaseTransform):
    def __init__(self, k_shot, num_splits, num_val=0.1):
        super().__init__()
        self.k_shot = k_shot
        self.num_splits = num_splits
        self.num_val = num_val
        self.selected_relations_list = None

    def forward(self, data):
        train_mask, val_mask, test_mask, selected_relations_list = link_k_shot_split(data,
                                                            self.k_shot, self.num_splits,
                                                            self.num_val)
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        self.selected_relations_list = selected_relations_list
        return data


class Node2VecEmbedding(BaseTransform):
    def __init__(self, embed_dim=128, batch_size=128, walk_length=20, context_size=10, lr=0.01,
                 walks_per_node=10, p=1.0, q=1.0, num_epochs=100, device=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.walk_length = walk_length
        self.context_size = context_size
        self.lr = lr
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_epochs = num_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, data):
        edge_index = data.edge_index

        model = Node2Vec(
            edge_index,
            num_nodes=data.num_nodes,
            embedding_dim=self.embed_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            sparse=True
        ).to(self.device)

        optimizer = torch.optim.SparseAdam(model.parameters(), lr=self.lr)

        model.train()
        for epoch in range(1, self.num_epochs + 1):
            total_loss = 0
            loader = model.loader(batch_size=self.batch_size, shuffle=True)
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 50 == 0:
                print(f'Node2Vec Epoch {epoch}, Loss: {total_loss / len(loader):.4f}')

        data.x = model.embedding.weight.detach().cpu()

        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.embed_dim}, epochs={self.num_epochs})'