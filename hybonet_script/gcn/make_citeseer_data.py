import os
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
import networkx as nx

# Planetoid-style pickles live here
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'citeseer')
DATASET = 'citeseer'

def load_raw_planetoid(data_dir, name):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for n in names:
        with open(os.path.join(data_dir, f'ind.{name}.{n}'), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    return objects  # x, y, tx, ty, allx, ally, graph


def main():
    x, y, tx, ty, allx, ally, graph = load_raw_planetoid(DATA_DIR, DATASET)

    # Build adjacency; nodes 0..N-1
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    num_nodes = adj.shape[0]

    # Features from allx; pad rows to num_nodes if needed
    features = allx.tocsr()
    if features.shape[0] < num_nodes:
        pad_rows = num_nodes - features.shape[0]
        features = sp.vstack(
            [features, sp.csr_matrix((pad_rows, features.shape[1]))],
            format='csr'
        )
    elif features.shape[0] > num_nodes:
        features = features[:num_nodes]

    # Labels: argmax of one-hot, padded to num_nodes
    labels_onehot = ally
    if labels_onehot.shape[0] < num_nodes:
        pad_rows = num_nodes - labels_onehot.shape[0]
        labels_onehot = np.vstack(
            [labels_onehot, np.zeros((pad_rows, labels_onehot.shape[1]))]
        )
    elif labels_onehot.shape[0] > num_nodes:
        labels_onehot = labels_onehot[:num_nodes]

    labels = np.argmax(labels_onehot, axis=1).astype(np.int64)

    out_dir = DATA_DIR
    os.makedirs(out_dir, exist_ok=True)

    # edges.csv
    edges_path = os.path.join(out_dir, f'{DATASET}.edges.csv')
    with open(edges_path, 'w') as f:
        coo = adj.tocoo()
        for i, j in zip(coo.row, coo.col):
            if i <= j:  # undirected, one entry per pair
                f.write(f'{i},{j}\n')
    print(f'Wrote edges to {edges_path}')

    # feats.npz
    feats_path = os.path.join(out_dir, f'{DATASET}.feats.npz')
    sp.save_npz(feats_path, features)
    print(f'Wrote features to {feats_path}, shape={features.shape}')

    # labels.npy
    labels_path = os.path.join(out_dir, f'{DATASET}.labels.npy')
    np.save(labels_path, labels)
    print(f'Wrote labels to {labels_path}, shape={labels.shape}')

if __name__ == '__main__':
    main()