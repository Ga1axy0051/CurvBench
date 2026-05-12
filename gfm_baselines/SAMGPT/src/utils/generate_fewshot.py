import os
import torch
import random
import argparse
from torch_geometric.datasets import Twitch,Flickr,FacebookPagePage,Coauthor
from dataset import *
from process import *
import scipy.sparse as sp
import numpy as np

def save_sample(data, path):
    os.makedirs(path, exist_ok=True)
    torch.save(data['idx'], os.path.join(path, 'idx.pt'))
    torch.save(data['labels'], os.path.join(path, 'labels.pt'))
    if 'batch' in data:
        torch.save(data['batch'], os.path.join(path, 'batch.pt'))

def generate_fewshot_samples(labels, num_shots, num_samples):
    fewshot_samples = []
    unique_labels = torch.unique(labels)
    for _ in range(num_samples):
        samples = []
        for label in unique_labels:
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(label_indices) < num_shots:
                continue 
            selected_indices = random.sample(label_indices.tolist(), num_shots)
            samples.extend(selected_indices)
        fewshot_samples.append({
            'idx': torch.tensor(samples),
            'labels': labels[samples]
        })
    return fewshot_samples

def create_folders(base_path, dataset_name, num_shots=10, num_samples=100):
    for shot in range(1, num_shots + 1):
        for i in range(1, num_samples + 1):
            os.makedirs(os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i)), exist_ok=True)

def save_fewshot_data(dataset_name, num_shots=10, num_samples=100, path = './data'):
    print(f'Generating node_data for {dataset_name}')
    dataset = load_dataset(dataset_name, path)
    data = dataset[0]
    labels = data.y

    base_path = os.path.join(path, f'fewshot_{dataset_name.lower()}')
    create_folders(base_path, dataset_name, num_shots, num_samples)

    for shot in range(1, num_shots + 1):
        samples = generate_fewshot_samples(labels, shot, num_samples)
        for i, sample in enumerate(samples):
            sample_path = os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i))
            save_sample(sample, sample_path)

def generate_fewshot_samples_graph(dataset_name, shotnum, num_samples, path='./data'):
    data = load_dataset(dataset_name, path)[0]
    features, adj = process_tu(data,data['x'].shape[1])
    samples = []
    for i in range(num_samples):
        idx_train = torch.load(f"{path}/fewshot_{dataset_name.lower()}/{shotnum}-shot_{dataset_name.lower()}/{i}/idx.pt").type(torch.long)
        lbl_train = torch.load(f"{path}/fewshot_{dataset_name.lower()}/{shotnum}-shot_{dataset_name.lower()}/{i}/labels.pt").type(torch.long)
        subgraph = build_subgraph(adj, idx_train)
        samples.append(
            {
            'idx' : subgraph['idx'],
            'batch' : subgraph['batch'],
            'labels' : lbl_train,
            }
        )
    return samples


def save_fewshot_graph_data(dataset_name, num_shots=10, num_samples=100):
    print(f'Generating graph_data for {dataset_name}')
    path = './data'
    base_path = os.path.join(path, f'fewshot_{dataset_name.lower()}_graph')
    create_folders(base_path, dataset_name, num_shots, num_samples)

    for shot in range(1, num_shots + 1):
        samples = generate_fewshot_samples_graph(dataset_name, shot, num_samples, path)
        for i, sample in enumerate(samples):
            sample_path = os.path.join(base_path, f'{shot}-shot_{dataset_name.lower()}', str(i))
            #batch = torch.arange(len(sample['idx']))
            #sample['batch'] = batch
            save_sample(sample, sample_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cs_phds')
    parser.add_argument('--num_shots', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--path', type=str, default='./data')
    parser.add_argument('--node', type=int, default=1)
    parser.add_argument('--graph', type=int, default=1)
    args = parser.parse_args()

    if args.node:
        save_fewshot_data(args.dataset, num_shots=args.num_shots, num_samples=args.num_samples, path=args.path)
    if args.graph:
        save_fewshot_graph_data(args.dataset, num_shots=args.num_shots, num_samples=args.num_samples)
