import torch
from cores.models import SparsePerturbation, PooLedSubgraphGNN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader
from data.data_process import search_triangles, unify_feature_dimension, RenameFromRootedEgoNets
from torch_geometric.transforms import RootedEgoNets, Compose


if __name__ == '__main__':
    ptgb = SparsePerturbation(2, 8, 32)
    gnn = PooLedSubgraphGNN("gcn", 2, 8, 16, bias=True, norm_str="none", act_str="relu", drop=0.1)
    x = torch.randn(5, 16)
    edge_index = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 3], [2, 4], [3, 2], [3, 4], [4, 2], [4, 3]]).t().contiguous()
    graph = Data(x, edge_index, edge_weight=torch.ones_like(edge_index[0]))
    graph.x = unify_feature_dimension(graph.x, 8)
    loader = NeighborLoader(graph, batch_size=2, shuffle=False, num_neighbors=[-1], disjoint=False, transform=Compose([RootedEgoNets(1), RenameFromRootedEgoNets()]))
    z_tans = []
    for graph in loader:
        paths = search_triangles(graph.origin_edge_index)
        aug_graphs = ptgb(graph.x, graph.edge_index, graph.edge_weight, graph.batch, graph.batch_graph_nums)
        z_tan = []
        for aug_graph in aug_graphs:
            tan = gnn(aug_graph.x, aug_graph.edge_index, aug_graph.edge_weight, aug_graph.batch)
            z_tan.append(tan)
        z_tan = torch.stack(z_tan, dim=1)
        z_tans.append(z_tan)
    z_tans = torch.cat(z_tans, dim=0)
    print(z_tans.shape)