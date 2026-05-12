from data.data_process import link_k_shot_split
from data.data_loader import load_few_shot_link_graph_data, load_pretrain_single_graph_data
from configs.pretrain_config import parse_pretrain_config
from torch_geometric.sampler import BaseSampler

if __name__ == '__main__':
    configs = parse_pretrain_config()
    data_name = "FB15k_237"
    configs.root = "../datasets"
    data = load_pretrain_single_graph_data(configs, data_name)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for batch in loader:
    #     pass