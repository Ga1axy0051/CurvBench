"""# Dataloaders Submodule.

The dataloaders module allows users to load datasets from Manify's datasets repo [on Hugging Face](https://huggingface.co/manify).

We provide a summary of the data types available, and their original sources, here.

Earlier versions of Manify included scripts to process raw data, which we have replaced with a single, centralized Hugging Face repo and the function `load_hf`. For transparency, we have preserved the data generation code in [the Dataset-Generation branch of Manify](https://github.com/pchlenski/manify/tree/Dataset-Generation).

| Dataset | Task | Distance Matrix | Features | Labels | Adjacency Matrix | Source/Citation |
|---------|------|----------------|----------|--------|-----------------|-----------------|
| cities | none | ✅ | ❌ | ❌ | ❌ | [Network Repository: Cities](https://networkrepository.com/Cities.php) |
| cs_phds | regression | ✅ | ❌ | ✅ | ✅ | [Network Repository: CS PhDs](https://networkrepository.com/CSphd.php) |
| polblogs | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Polblogs](https://networkrepository.com/polblogs.php) |
| polbooks | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Polbooks](https://networkrepository.com/polbooks.php) |
| cora | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Cora](https://networkrepository.com/cora.php) |
| citeseer | classification | ✅ | ❌ | ✅ | ✅ | [Network Repository: Citeseer](https://networkrepository.com/citeseer.php) |
| karate_club | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Karate](https://networkrepository.com/karate.php) |
| lesmis | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Lesmis](https://networkrepository.com/lesmis.php) |
| adjnoun | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Adjnoun](https://networkrepository.com/adjnoun.php) |
| football | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Football](https://networkrepository.com/football.php) |
| dolphins | none | ✅ | ❌ | ❌ | ✅ | [Network Repository: Dolphins](https://networkrepository.com/dolphins.php) |
| blood_cells | classification | ❌ | ✅ | ✅ | ❌ | See datasets from Zheng et al (2017): Massively parallel digital transcriptional profiling of single cells.<br>- [CD8+ Cytotoxic T-cells](https://www.10xgenomics.com/datasets/cd-8-plus-cytotoxic-t-cells-1-standard-1-1-0)<br>- [CD8+/CD45RA+ Naive Cytotoxic T Cells](https://www.10xgenomics.com/datasets/cd-8-plus-cd-45-r-aplus-naive-cytotoxic-t-cells-1-standard-1-1-0)<br>- [CD56+ Natural Killer Cells](https://www.10xgenomics.com/datasets/cd-56-plus-natural-killer-cells-1-standard-1-1-0)<br>- [CD4+ Helper T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-helper-t-cells-1-standard-1-1-0)<br>- [CD4+/CD45RO+ Memory T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-oplus-memory-t-cells-1-standard-1-1-0)<br>- [CD4+/CD45RA+/CD25- Naive T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-45-r-aplus-cd-25-naive-t-cells-1-standard-1-1-0)<br>- [CD4+/CD25+ Regulatory T Cells](https://www.10xgenomics.com/datasets/cd-4-plus-cd-25-plus-regulatory-t-cells-1-standard-1-1-0)<br>- [CD34+ Cells](https://www.10xgenomics.com/datasets/cd-34-plus-cells-1-standard-1-1-0)<br>- [CD19+ B Cells](https://www.10xgenomics.com/datasets/cd-19-plus-b-cells-1-standard-1-1-0)<br>- [CD14+ Monocytes](https://www.10xgenomics.com/datasets/cd-14-plus-monocytes-1-standard-1-1-0) |
| lymphoma | classification | ❌ | ✅ | ✅ | ❌ | See datasets from 10x Genomics:<br>- [Hodgkin's Lymphoma](https://www.10xgenomics.com/datasets/hodgkins-lymphoma-dissociated-tumor-targeted-immunology-panel-3-1-standard-4-0-0)<br>- [Healthy Donor PBMCs](https://www.10xgenomics.com/datasets/pbm-cs-from-a-healthy-donor-targeted-compare-immunology-panel-3-1-standard-4-0-0) |
| cifar_100 | classification | ❌ | ✅ | ✅ | ❌ | [Hugging Face Datasets: CIFAR-100](https://huggingface.co/datasets/uoft-cs/cifar100) |
| mnist | classification | ❌ | ✅ | ✅ | ❌ | [Hugging Face Datasets: MNIST](https://huggingface.co/datasets/ylecun/mnist) |
| temperature | regression | ❌ | ✅ | ✅ | ❌ | [Citation] |
| landmasses | classification | ❌ | ✅ | ✅ | ❌ | Generated using [basemap.is_land](https://matplotlib.org/basemap/stable/api/basemap_api.html#mpl_toolkits.basemap.Basemap.is_land) |
| neuron_33 | classification | ❌ | ✅ | ✅ | ❌ | [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/623474400) |
| neuron_46 | classification | ❌ | ✅ | ✅ | ❌ | [Allen Brain Atlas](https://celltypes.brain-map.org/experiment/electrophysiology/623474400) |
| traffic | regression | ❌ | ✅ | ✅ | ❌ | [Kaggle: Traffic Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset) |
| qiita | none | ✅ | ✅ | ❌ | ❌ | [NeuroSEED Git Repo](https://github.com/gcorso/NeuroSEED) |
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
import numpy as np
from datasets import load_dataset
from torch_geometric.datasets import Amazon


if TYPE_CHECKING:
    from jaxtyping import Float, Real


def load_hf(
    name: str, namespace: str = "manify"
) -> tuple[
    Float[torch.Tensor, "n_points ..."] | None,  # features
    Float[torch.Tensor, "n_points n_points"] | None,  # pairwise dists
    Float[torch.Tensor, "n_points n_points"] | None,  # adjacency
    Real[torch.Tensor, "n_points"] | None,  # labels
]:
    """
    Load a dataset from HuggingFace Hub at {namespace}/{name}, or from PyG if name='pubmed'.
    """
    # ======================================================================================
    #  1. web-Google (SNAP) ———— 新增的分支（你需要的）
    # ======================================================================================
    if name.lower().replace("_", "").replace("-", "") in ["webgoogle", "webgoogle"]:
        import networkx as nx
        import time

        SNAP_PATH = "/home/guoquanjiang/WXY/benchmark_datasets/web-Google/web-Google.txt"
        print(f"📘 Loading SNAP web-Google from: {SNAP_PATH}")

        t0 = time.time()

        # ---------- 读 SNAP 边 ----------
        G = nx.DiGraph()
        with open(SNAP_PATH, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                u, v = map(int, line.split())
                G.add_edge(u, v)

        print(f" Loaded directed graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # ---------- 转成无向图 ----------
        H = G.to_undirected()
        nodes = list(H.nodes())
        node_idx = {v: i for i, v in enumerate(nodes)}
        n = len(nodes)

        print(f" Converting to adjacency dense FP16 matrix, size = {n}x{n} ... (≈ {n*n*2/1024/1024/1024:.2f} GB)")

        # ---------- adjacency dense ----------
        adj = torch.zeros((n, n), dtype=torch.float16)
        for u, v in H.edges():
            i, j = node_idx[u], node_idx[v]
            adj[i, j] = 1
            adj[j, i] = 1

        # ---------- shortest path distances ----------
        print(" Computing all-pairs shortest path distance matrix (APSP)...")
        print(" 这一步最耗时（可能 1~3 小时），请耐心等待。")

        dists = torch.full((n, n), float("inf"), dtype=torch.float32)

        for i, src in enumerate(nodes):
            lengths = nx.single_source_shortest_path_length(H, src)
            for tgt, d in lengths.items():
                dists[i, node_idx[tgt]] = float(d)

        # inf → finite
        max_finite = torch.max(dists[dists < float("inf")])
        dists[dists == float("inf")] = max_finite * 2

        print(f" APSP done in {time.time() - t0:.2f} seconds\n")

        # ---- web-Google 没有 features / labels ----
        features = None
        labels = None

        return features, dists, adj, labels


    # ✅ 新增分支：PubMed 数据集
    if name.lower() == "pubmed":
        print("📘 Loading PubMed dataset using PyTorch Geometric ...")
        from torch_geometric.datasets import Planetoid
        from torch_geometric.utils import to_dense_adj
        import time

        start_time = time.time()
        dataset = Planetoid(root="data/PubMed", name="PubMed")
        data = dataset[0]

        features = data.x
        labels = data.y
        adj = to_dense_adj(data.edge_index)[0]

        print(f"✅ Loaded raw PubMed tensors: features {features.shape}, adj {adj.shape}, labels {labels.shape}")

        # 计算 pairwise 欧式距离矩阵
        with torch.no_grad():
            try:
                print(" Computing pairwise distance matrix...")
                dists = torch.cdist(features, features)
            except RuntimeError:
                subset = 1000
                print(f" 内存不足，抽样前 {subset} 个节点计算距离矩阵")
                features = features[:subset]
                labels = labels[:subset]
                adj = adj[:subset, :subset]
                dists = torch.cdist(features, features)

        elapsed = time.time() - start_time
        print(f" PubMed dataset loaded in {elapsed:.2f} seconds")
        print(f"节点数: {features.shape[0]}, 特征维度: {features.shape[1]}, 类别数: {len(torch.unique(labels))}\n")

        return features, dists, adj, labels
    
    #COMPUTERS dataset
    if name.lower() == "computers":
        print("📘 Loading Amazon Computers dataset using PyTorch Geometric ...")
        dataset = Amazon(root="data/Computers", name="Computers")
        data = dataset[0]

    # adjacency matrix
        adj = torch.zeros((data.num_nodes, data.num_nodes), dtype=torch.float32)
        edges = data.edge_index
        adj[edges[0], edges[1]] = 1
        adj[edges[1], edges[0]] = 1  # 无向图

    # 计算 pairwise 距离矩阵（简单版：用特征欧氏距离）
        print("Computing pairwise distance matrix (features-based)...")
        features = data.x
        dists = torch.cdist(features, features, p=2)

        features = features.float()
        labels = data.y.long()
        print(" Amazon Computers dataset loaded successfully!")
        return features, dists, adj, labels
    
    

    # 🟦 单独分支：处理 Amazon Photo 数据集
    # ======================================================
    if name == "photo":
        print("📘 Loading Amazon Photo dataset using PyTorch Geometric ...")

        dataset = Amazon(root="./data", name="Photo")  # 注意首字母大写
        data = dataset[0]

        # 构建稠密邻接矩阵
        adj = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones(data.edge_index.shape[1]),
            (data.num_nodes, data.num_nodes)
        ).to_dense()

        # 特征欧氏距离矩阵（计算量较大，可考虑只近似或采样）
        dists = torch.cdist(data.x.float(), data.x.float())

        print(f"✅ Loaded Photo dataset: {data.num_nodes} nodes, {data.num_features} features, {data.y.unique().numel()} classes.")
        return data.x, dists, adj, data.y

    

    # 🌐 WordNet Hypernym Graph (Poincaré Embeddings version)

    if name.lower() in ["wordnet", "wordnet_poincare"]:
        print(" Loading WordNet hypernym graph (poincaré version) ...")

        import os

        path = "./manify/data/wordnet/wordnet_direct_graph.pt"   # 你自己生成的那个文件
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"找不到 {path}\n"
                "请运行 build_wordnet_graph.py 来生成 wordnet_graph.pt"
            )

        data = torch.load(path)

        features = data["features"]        # [N, 1]
        adj = data["adj_sparse"]           # 稀疏邻接矩阵
        labels = data["labels"]            # None
        dists = data["dists"]              # None

        print(f"✔ WordNet loaded: nodes={features.shape[0]}, feature_dim={features.shape[1]}")
        print(f"✔ adjacency nnz = {adj._nnz()} (sparse)")

        return features, dists, adj, labels
    
    if name.lower() in ["telecom", "telegraph", "telecom_graph"]:
        print(" Loading telecom_graph.pt ...")

        import torch
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        import os
        import time

        path = "/home/guoquanjiang/WXY/manify/data/telecom/telecom_graph.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(f" 未找到文件：{path}")

        t0 = time.time()

        data = torch.load(path, map_location="cpu")

        # TeleGraph 文件字段通常包含：
        #   - data["edge_index"]
        #   - data["x"] (可能有，也可能没有)
        #   - data["y"] (一般没有)
        edge_index = data["edge_index"]
        num_nodes = int(edge_index.max()) + 1

        print(f"  nodes={num_nodes}, edges={edge_index.shape[1]}")

        # ------------------------------
        # adjacency matrix (dense)
        # ------------------------------
        print("  Building dense adjacency matrix ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

        # ------------------------------
        # APSP shortest-path matrix
        # ------------------------------
        print("  Computing APSP distance matrix (NetworkX)...")
        print("   注意：这个步骤可能需要几十秒~数分钟")

        G = nx.Graph()
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)

        dists = torch.full((num_nodes, num_nodes), float("inf"), dtype=torch.float32)

        for i in range(num_nodes):
            sp = nx.single_source_shortest_path_length(G, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        # inf → large finite
        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        # ------------------------------
        # features / labels
        # ------------------------------
        features = data["x"].float() if "x" in data and data["x"] is not None else None
        labels = data["y"].long() if "y" in data and data["y"] is not None else None

        print(f"✔ Telecom loaded in {time.time() - t0:.2f} seconds\n")

        return features, dists, adj, labels
    
        # ======================================================================================
    #  Higgs Retweet Network — Small Version (dense adj + APSP)
    # ======================================================================================
    if name.lower() in ["higgs", "higgs_retweet", "higgs_retweet_network", "twitter"]:
        print(" Loading Higgs Retweet Network (dense version) ...")

        import os
        import torch
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        import time

        path = "/home/guoquanjiang/WXY/manify/data/twitter/higgs-retweet_network.edgelist"
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 Higgs edgelist 文件: {path}")

        # 1) Load as networkx graph
        print(" Reading edge list ...")
        G = nx.read_edgelist(path, nodetype=int, data=False)

        num_nodes = max(G.nodes()) + 1
        num_edges = G.number_of_edges()
        print(f"✔ Loaded: {num_nodes} nodes, {num_edges} edges")


        # 2) convert to edge_index
        edges = torch.tensor(list(G.edges()), dtype=torch.long)
        edges = torch.cat([edges, edges[:, [1, 0]]], dim=0)
        edge_index = edges.t().contiguous()

        # 3) Dense adjacency
        print("📦 Building dense adjacency ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

        # 4) APSP shortest-path distances
        print("⏳ Computing APSP via NetworkX...")
        t0 = time.time()
        dists = torch.full((num_nodes, num_nodes), float('inf'))

        for i in range(num_nodes):
            sp = nx.single_source_shortest_path_length(G, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        print(f"✔ APSP done in {time.time() - t0:.2f} seconds.")

        # Higgs 没有 feature / label
        return None, dists, adj, None
    
    # ======================================================================================
    #  Higgs Retweet Subgraph — 用于曲率实验（dense adj + APSP）
    # ======================================================================================
    if name.lower() in ["twitter10k", "higgs_sub10k"]:
        print(" Loading Higgs subgraph (dense version) ...")

        import os
        import torch
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        import time

        path = "/home/guoquanjiang/WXY/manify/data/twitter/higgs_sub10k.edgelist"
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到 Higgs 子图文件: {path}")

        print(" Reading subgraph edge list ...")
        G = nx.read_edgelist(path, nodetype=int, data=False)

        # 重新映射节点 ID 到 0..n-1，避免 ID 稀疏导致矩阵很大
        nodes = list(G.nodes())
        node_id = {v: i for i, v in enumerate(nodes)}
        num_nodes = len(nodes)

        edges = []
        for u, v in G.edges():
            edges.append([node_id[u], node_id[v]])
            edges.append([node_id[v], node_id[u]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        print(f"✔ Subgraph: {num_nodes} nodes, {edge_index.shape[1]//2} undirected edges")

        # dense adjacency
        print(" Building dense adjacency ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

           # APSP shortest-path distances
        print(" Computing APSP via NetworkX ...")
        t0 = time.time()

        import numpy as np
        from tqdm import tqdm   # ← tqdm

        dists = torch.full((num_nodes, num_nodes), float('inf'), dtype=torch.float32)

        # 用映射后的索引 BFS
        G_remap = nx.Graph()
        G_remap.add_edges_from([(node_id[u], node_id[v]) for u, v in G.edges()])

        # 加 tqdm 进度条
        for i in tqdm(range(num_nodes), desc="APSP (BFS from each node)", ncols=100):
            sp = nx.single_source_shortest_path_length(G_remap, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        print(f"✔ APSP done in {time.time() - t0:.2f} seconds.")


        # Higgs 子图没有 feature / label
        features = None
        labels = None

        return features, dists, adj, labels
    
    # ======================================================================================
    #  Rocketfuel ISP Network — AS7018 (AT&T) with tqdm
    # ======================================================================================
    # ======================================================================================
    #  Rocketfuel ISP Network — AS7018 (FORCE APSP, STRICT MODE)
    # ======================================================================================
    if name.lower() in ["as7018", "rocketfuel_7018", "att7018"]:
        print("📘 Loading Rocketfuel AS7018 (r1) graph — FORCE APSP")

        import os
        import torch
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        from tqdm import tqdm
        import time

        EDGE_PATH = "/home/guoquanjiang/WXY/benchmark_datasets/rocket7018IPL/7018_edgelist.txt"
        if not os.path.exists(EDGE_PATH):
            raise FileNotFoundError(f"未找到 AS7018 edgelist: {EDGE_PATH}")

        t0 = time.time()

        # --------------------------------------------------
        # 1) Load graph
        # --------------------------------------------------
        print(" Reading edgelist ...")
        G = nx.read_edgelist(EDGE_PATH, nodetype=int, data=False)

        nodes = list(G.nodes())
        node_id = {v: i for i, v in enumerate(nodes)}
        num_nodes = len(nodes)

        # --------------------------------------------------
        # 2) Build edge_index
        # --------------------------------------------------
        edges = []
        for u, v in tqdm(G.edges(), desc=" Building edge_index", ncols=100):
            edges.append([node_id[u], node_id[v]])
            edges.append([node_id[v], node_id[u]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        print(f"✔ AS7018 graph: nodes={num_nodes}, edges={edge_index.shape[1]//2}")

        # --------------------------------------------------
        # 3) Dense adjacency
        # --------------------------------------------------
        print(" Building dense adjacency matrix ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

        # --------------------------------------------------
        # 4) APSP via BFS (THIS IS THE EXPENSIVE PART)
        # --------------------------------------------------
        print("  Computing APSP via NetworkX BFS ...")

        # 重建 remapped graph（BFS 用）
        G_remap = nx.Graph()
        G_remap.add_edges_from(
            [(node_id[u], node_id[v]) for u, v in G.edges()]
        )

        dists = torch.full(
            (num_nodes, num_nodes),
            float("inf"),
            dtype=torch.float32
        )

        for i in tqdm(range(num_nodes),
                    desc="APSP (BFS from each node)",
                    ncols=100):
            sp = nx.single_source_shortest_path_length(G_remap, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        # inf → finite（manify 兼容）
        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        print(f"✔ APSP done in {time.time() - t0:.2f} seconds")

        # --------------------------------------------------
        # 5) No features / labels
        # --------------------------------------------------
        features = None
        labels = None

        print("✔ AS7018 loaded with FULL APSP\n")

        return features, dists, adj, labels
    
    # ======================================================================================
    #  roadNet-CA BFS subgraph — 20K nodes (dense, for sectional curvature)
    # ======================================================================================
    if name.lower() in ["roadnet_ca_20k", "roadnetca20k"]:
        print("📘 Loading roadNet-CA BFS 20K subgraph (DENSE) ...")

        import os
        import torch
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        from tqdm import tqdm
        import time

        EDGE_PATH = "/home/guoquanjiang/WXY/benchmark_datasets/roadNetCA/roadNet-CA_bfs20k.edgelist"
        if not os.path.exists(EDGE_PATH):
            raise FileNotFoundError(f"未找到文件: {EDGE_PATH}")

        t0 = time.time()

        # 1) load graph
        print(" Reading edgelist ...")
        G = nx.read_edgelist(EDGE_PATH, nodetype=int, data=False)

        # 2) remap node ids
        nodes = list(G.nodes())
        node_id = {v: i for i, v in enumerate(nodes)}
        num_nodes = len(nodes)

        edges = []
        for u, v in tqdm(G.edges(), desc=" Building edge_index", ncols=100):
            edges.append([node_id[u], node_id[v]])
            edges.append([node_id[v], node_id[u]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        print(f"✔ roadNet-CA-20K: nodes={num_nodes}, edges={edge_index.shape[1]//2}")

        # 3) dense adjacency
        print(" Building dense adjacency matrix ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

        features = None
        dists = None
        labels = None

        print(f"✔ roadNet-CA-20K loaded in {time.time() - t0:.2f} seconds\n")
        # --------------------------------------------------
        # 4) FORCE APSP (STRICT MODE)
        # --------------------------------------------------
        print("  Computing APSP via NetworkX BFS  ...")

        import networkx as nx
        from tqdm import tqdm

        # 用 remap 后的 id 构建 Graph
        G_remap = nx.Graph()
        G_remap.add_edges_from(
            [(node_id[u], node_id[v]) for u, v in G.edges()]
        )

        # 分配距离矩阵
        dists = torch.full(
            (num_nodes, num_nodes),
            float("inf"),
            dtype=torch.float32
        )

        # BFS from each node
        for i in tqdm(range(num_nodes),
                    desc="APSP (BFS from each node)",
                    ncols=100):
            sp = nx.single_source_shortest_path_length(G_remap, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        # inf → finite（manify / STRICT 需要）
        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        return features, dists, adj, labels
    
   # ======================================================================================
    #  roadNet-CA BFS subgraph — 100K nodes (DENSE + FORCE APSP, STRICT MODE)
    # ======================================================================================
    if name.lower() in ["roadnet_ca_100k", "roadnetca100k"]:
        print("📘 Loading roadNet-CA BFS 100K subgraph (DENSE + FORCE APSP) ...")

        import os
        import torch
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        from tqdm import tqdm
        import time

        EDGE_PATH = "/home/guoquanjiang/WXY/benchmark_datasets/roadNetCA/roadNet-CA_bfs100k.edgelist"
        if not os.path.exists(EDGE_PATH):
            raise FileNotFoundError(f"未找到文件: {EDGE_PATH}")

        t0 = time.time()

        # --------------------------------------------------
        # 1) Load edgelist
        # --------------------------------------------------
        print(" Reading edgelist ...")
        G = nx.read_edgelist(EDGE_PATH, nodetype=int, data=False)

        # --------------------------------------------------
        # 2) Remap node ids → 0..N-1
        # --------------------------------------------------
        nodes = list(G.nodes())
        node_id = {v: i for i, v in enumerate(nodes)}
        num_nodes = len(nodes)

        edges = []
        for u, v in tqdm(G.edges(), desc=" Building edge_index", ncols=100):
            edges.append([node_id[u], node_id[v]])
            edges.append([node_id[v], node_id[u]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        print(f"✔ roadNet-CA-100K: nodes={num_nodes}, edges={edge_index.shape[1]//2}")

        # --------------------------------------------------
        # 3) Dense adjacency
        # --------------------------------------------------
        print(" Building HUGE dense adjacency matrix ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

        # --------------------------------------------------
        # 4) FORCE APSP via NetworkX BFS
        # --------------------------------------------------
        print("  Computing APSP via NetworkX BFS  ...")

        G_remap = nx.Graph()
        G_remap.add_edges_from(
            [(node_id[u], node_id[v]) for u, v in G.edges()]
        )

        dists = torch.full(
            (num_nodes, num_nodes),
            float("inf"),
            dtype=torch.float32
        )

        for i in tqdm(range(num_nodes),
                    desc="APSP (BFS from each node)",
                    ncols=100):
            sp = nx.single_source_shortest_path_length(G_remap, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        print(f"✔ APSP done in {time.time() - t0:.2f} seconds")

        # --------------------------------------------------
        # 5) No features / labels
        # --------------------------------------------------
        features = None
        labels = None

        print("✔ roadNet-CA-100K loaded with FULL APSP\n")
        return features, dists, adj, labels
    
    # ======================================================================================
    #  Airport Network (OpenFlights) — local processed graph
    # ======================================================================================
    if name.lower() in ["airport", "airports", "openflights", "airport-raw"]:
            print("📘 Loading Airport-Raw network (7,543 nodes local graph) ...")

            import os
            import torch
            import networkx as nx
            from torch_geometric.utils import to_dense_adj
            from tqdm import tqdm
            import time

            BASE_DIR = "D:\\Github\\manify\\manify\\data\\Airport"
            
            # 🌟 核心优化：直接读取我们验证过的 7543 节点全量 PyG 文件
            # 这样比解析 airport_alldata.p (DataFrame) 更安全、更快捷
            GRAPH_PATH = os.path.join(BASE_DIR, "HGCN_Airport_PyG.pt")

            if not os.path.exists(GRAPH_PATH):
                raise FileNotFoundError(f"未找到 {GRAPH_PATH}，请确保文件存在。")

            t0 = time.time()

            # --------------------------------------------------
            # 1) Load PyG Data directly
            # --------------------------------------------------
            data = torch.load(GRAPH_PATH, weights_only=False)
            if hasattr(data, 'cpu'):
                data = data.cpu()
                
            num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
            edge_index = data.edge_index

            print(f"✔ Loaded Airport-Raw: nodes={num_nodes}, edges={edge_index.size(1)}")

            # --------------------------------------------------
            # 2) Dense adjacency
            # --------------------------------------------------
            print("📦 Building dense adjacency matrix ...")
            adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

            # --------------------------------------------------
            # 3) APSP (Build NetworkX graph on the fly for BFS)
            # --------------------------------------------------
            print("⏳ Computing APSP via NetworkX BFS ...")
            print("   （如果只是 manifold fitting，把这段注释掉）")
            
            # 将 PyG 的 edge_index 瞬间转换为 NetworkX 图以计算最短路
            G_remap = nx.Graph()
            G_remap.add_nodes_from(range(num_nodes))
            G_remap.add_edges_from(edge_index.t().tolist())

            dists = torch.full(
                (num_nodes, num_nodes),
                float("inf"),
                dtype=torch.float32
            )

            for i in tqdm(range(num_nodes),
                          desc="APSP (BFS from each node)",
                          ncols=100):
                sp = nx.single_source_shortest_path_length(G_remap, i)
                for j, d in sp.items():
                    dists[i, j] = float(d)

            max_finite = dists[dists < float("inf")].max()
            dists[dists == float("inf")] = max_finite * 2

            # --------------------------------------------------
            # 4) Features / Labels (遵循你原代码设为 None 的逻辑)
            # --------------------------------------------------
            features = None
            labels = None

            print(f"✔ Airport-Raw loaded in {time.time() - t0:.2f} seconds\n")

            return features, dists, adj, labels
    
    # ======================================================================================
    #  Disease Network (HGCN format) — NC / LP
    # ======================================================================================
    if name.lower() in ["disease", "disease_nc", "disease_lp"]:
        print(" Loading Disease network (HGCN format) ...")

        import os
        import torch
        import pandas as pd
        import networkx as nx
        from torch_geometric.utils import to_dense_adj
        from tqdm import tqdm
        import time

        BASE_DIR = "/home/guoquanjiang/WXY/manify/data/disease"

        EDGE_PATH = os.path.join(BASE_DIR, "disease_nc.edges.csv")

        if not os.path.exists(EDGE_PATH):
            raise FileNotFoundError(f"未找到 edge.csv: {EDGE_PATH}")

        t0 = time.time()

        # --------------------------------------------------
        # 1) Load edge list
        # --------------------------------------------------
        print(" Reading edge.csv ...")
        df = pd.read_csv(EDGE_PATH)

        G = nx.Graph()
        for _, row in df.iterrows():
            u = int(row[0])
            v = int(row[1])
            if u != v:
                G.add_edge(u, v)

        print(f"✔ Raw graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        # --------------------------------------------------
        # 2) Largest connected component（非常重要）
        # --------------------------------------------------
        lcc = max(nx.connected_components(G), key=len)
        G = G.subgraph(lcc).copy()

        print(f"✔ LCC graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

        # --------------------------------------------------
        # 3) Remap node ids → 0..N-1
        # --------------------------------------------------
        nodes = list(G.nodes())
        node_id = {v: i for i, v in enumerate(nodes)}
        num_nodes = len(nodes)

        edges = []
        for u, v in tqdm(G.edges(), desc=" Building edge_index", ncols=100):
            edges.append([node_id[u], node_id[v]])
            edges.append([node_id[v], node_id[u]])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # --------------------------------------------------
        # 4) Dense adjacency
        # --------------------------------------------------
        print(" Building dense adjacency matrix ...")
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).float()

        # --------------------------------------------------
        # 5) APSP (STRICT MODE)
        # --------------------------------------------------
        print(" Computing APSP via NetworkX BFS ...")

        G_remap = nx.Graph()
        G_remap.add_edges_from(
            [(node_id[u], node_id[v]) for u, v in G.edges()]
        )

        dists = torch.full(
            (num_nodes, num_nodes),
            float("inf"),
            dtype=torch.float32
        )

        for i in tqdm(range(num_nodes),
                    desc="APSP (BFS from each node)",
                    ncols=100):
            sp = nx.single_source_shortest_path_length(G_remap, i)
            for j, d in sp.items():
                dists[i, j] = float(d)

        max_finite = dists[dists < float("inf")].max()
        dists[dists == float("inf")] = max_finite * 2

        # --------------------------------------------------
        # 6) features / labels（曲率实验不需要）
        # --------------------------------------------------
        features = None
        labels = None

        print(f"✔ Disease loaded in {time.time() - t0:.2f} seconds\n")

        return features, dists, adj, labels




    
    # ✅ 原始逻辑（Hugging Face 数据集）
    ds = load_dataset(f"{namespace}/{name}")
    data = ds.get("train", ds)  # use "train" split if available, else the only split
    row = data[0]

    def to_tensor(key: str, dtype: torch.dtype) -> torch.Tensor | None:
        vals = row.get(key, [])
        if not vals:
            return None
        return torch.tensor(vals, dtype=dtype)

    dists = to_tensor("distances", torch.float32)
    feats = to_tensor("features", torch.float32)
    adj = to_tensor("adjacency", torch.float32)

    cls_ls = row.get("classification_labels", [])
    reg_ls = row.get("regression_labels", [])
    if cls_ls:
        labels = torch.tensor(cls_ls, dtype=torch.int64)
    elif reg_ls:
        labels = torch.tensor(reg_ls, dtype=torch.float32)
    else:
        labels = None

    return feats, dists, adj, labels