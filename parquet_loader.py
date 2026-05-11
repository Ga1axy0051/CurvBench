import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch

def _get_parquet_path(dataset_name):
    # Depending on where the code is run, datasets is usually at root/datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "datasets", dataset_name, "unified_data.parquet")

def load_parquet_as_pyg(dataset_name):
    from torch_geometric.data import Data
    try:
        df = pd.read_parquet(_get_parquet_path(dataset_name))
        row = df.iloc[0]
        x = torch.tensor(np.stack(row['x']).astype(np.float32))
        edge_index = torch.tensor(np.stack(row['edge_index']).astype(np.int64))
        y = torch.tensor(row['y'].astype(np.int64)) if 'y' in row else torch.zeros(x.size(0), dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        for mask_name in ['train_mask', 'val_mask', 'test_mask']:
            if mask_name in row:
                setattr(data, mask_name, torch.tensor(row[mask_name].astype(bool)))
        return data
    except Exception as e:
        print("Failed to load parquet:", e)
        return None

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(list(classes))}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def load_parquet_as_legacy_kipf(dataset_name):
    """
    Returns adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    which is the format output by typical Kipf GCN load_data implementations.
    """
    data = load_parquet_as_pyg(dataset_name)
    if data is None:
        raise FileNotFoundError("Could not find Parquet data for " + dataset_name)
    
    features = sp.lil_matrix(data.x.numpy())
    
    # building adjacency matrix
    edges = data.edge_index.numpy().T
    N = data.x.size(0)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    labels = encode_onehot(data.y.numpy())
    
    idx_train = torch.where(data.train_mask)[0].numpy()
    idx_val = torch.where(data.val_mask)[0].numpy()
    idx_test = torch.where(data.test_mask)[0].numpy()
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_parquet_as_hgcn(dataset_name):
    """
    Returns a dictionary for HGCN/QGCN format load_data.
    """
    data = load_parquet_as_pyg(dataset_name)
    if data is None:
        raise FileNotFoundError("Could not find Parquet data for " + dataset_name)
        
    features = sp.lil_matrix(data.x.numpy())
    
    # building adjacency matrix
    edges = data.edge_index.numpy().T
    N = data.x.size(0)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(N, N), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    idx_train = torch.where(data.train_mask)[0].tolist()
    idx_val = torch.where(data.val_mask)[0].tolist()
    idx_test = torch.where(data.test_mask)[0].tolist()
    labels = torch.LongTensor(data.y.numpy())
    
    return {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_val': idx_val, 'idx_test': idx_test}


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)

