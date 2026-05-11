import argparse
import os
import sys
import json
import time
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from torch_geometric.utils import to_dense_adj

# Allow importing parquet_loader from parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parquet_loader import load_parquet_as_pyg

# Import the local sectional_curvature function
try:
    from sectional_curvature_strict_fast import sectional_curvature_gpu
except ImportError:
    # fallback if needed
    pass

def main():
    parser = argparse.ArgumentParser(description="Calculate Curvature using Parquet datasets")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    args, unknown = parser.parse_known_args()

    dataset_name = args.dataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading {dataset_name} using parquet_loader...")
    data = load_parquet_as_pyg(dataset_name)
    if data is None:
        print(f"Error: Failed to load dataset {dataset_name}")
        sys.exit(1)

    features = data.x
    labels = data.y
    adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]

    print("Data loaded successfully:")
    print(f"Nodes: {adj.shape[0]}")
    print(f"Features: {features.shape[1]}")
    print(f"Classes: {len(torch.unique(labels)) if labels is not None else 'None'}\n")

    # Compute pairwise Euclidean distance for geometry/curvature
    with torch.no_grad():
        try:
            print("Computing pairwise distance matrix...")
            dists = torch.cdist(features, features)
        except RuntimeError as e:
            print(f"OOM error when computing distance matrix: {e}")
            sys.exit(1)

    n_nodes = adj.shape[0]
    mode = "fast" if n_nodes <= 5000 else "strict"
    print(f"Selected computation mode: {mode.upper()} (nodes={n_nodes})\n")

    adj = adj.to(device)
    dists = dists.to(device)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("curvature_results", dataset_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Starting sectional curvature calculation for {dataset_name}...\n")
    start_time = time.time()

    try:
        curvatures = sectional_curvature_gpu(
            adj,
            dists,
            device=device,
            mode=mode,
            pair_chunk_size=2048,
            relative=True,
            show_progress=True,
        )
    except RuntimeError as e:
        if "CUDA" in str(e):
            print("\nGPU OOM, switching to CPU...")
            torch.cuda.empty_cache()
            adj = adj.cpu()
            dists = dists.cpu()
            curvatures = sectional_curvature_gpu(
                adj, dists, device="cpu", mode=mode, relative=True, show_progress=True
            )
        else:
            raise e

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed / 60:.2f} minutes\n")

    curvatures_cpu = curvatures.detach().cpu()
    torch.save(curvatures_cpu, os.path.join(save_dir, "curvatures.pt"))

    summary = {
        "dataset": dataset_name,
        "mode": mode,
        "nodes": n_nodes,
        "mean": float(curvatures_cpu.mean().item()),
        "min": float(curvatures_cpu.min().item()),
        "max": float(curvatures_cpu.max().item()),
        "device": str(device),
        "elapsed_minutes": round(elapsed / 60, 2),
        "timestamp": timestamp,
    }
    with open(os.path.join(save_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Mean Curvature:", summary["mean"])
    print("Min:", summary["min"])
    print("Max:", summary["max"])
    print(f"Results saved to: {save_dir}\n")

    # Plotting curvature distribution
    plt.figure(figsize=(8, 5))
    plt.hist(curvatures_cpu.numpy(), bins=30, color="skyblue", edgecolor="black")
    plt.title(f"{dataset_name.upper()} node curvature ({mode.upper()} GPU)")
    plt.xlabel("Sectional Curvature")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curvature_distribution.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()

