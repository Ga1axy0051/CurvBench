import os
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, scatter

import warnings
warnings.filterwarnings("ignore")

# ================= 1. 全局配置 =================
MODELS_TO_RUN = ["GraphSAGE", "PCNet"]

CONFIG = {
    "nc_data_root": r"D:\Github\benchmark_exp\data\cs_phds_nc_ready",
    "lp_data_root": r"D:\Github\benchmark_exp\data\cs_phds_lp_ready",
    "num_runs": 5,           
    "epochs": 200,
    "hidden_channels": 64,
    "num_layers": 2,         
    "lr": 0.01,
    "weight_decay": 5e-4,
    "dropout": 0.5,
    "activation": "relu",
    
    "pc_K": 5, "pc_N": 10, "pc_t": 1.5, "pc_p": 2.0, "pc_eta": 0.5,           
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
ROOT_EXP_DIR = Path(f"./cs_phds_final_{TIMESTAMP}")
ROOT_EXP_DIR.mkdir(parents=True, exist_ok=True)

# ================= 2. 数学底层与模型架构 =================
def get_generalized_laplacian(edge_index, num_nodes, eta=0.5, p=2.0):
    device = edge_index.device
    edge_weight = torch.ones(edge_index.size(1), device=device)
    edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes)
    row, col = edge_index
    deg = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow(-eta)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    norm_weight = deg_inv[row] * edge_weight * deg_inv[col]
    L_weight = -norm_weight
    self_loop_mask = row == col
    L_weight[self_loop_mask] = (p - 1.0) + L_weight[self_loop_mask]
    return edge_index, L_weight

def get_pc_coefficients(K, N, t):
    C = torch.zeros(K, N + 1)
    for k_idx in range(K):
        gamma = k_idx + 1  
        C[k_idx, 0] = 1.0
        if N >= 1: C[k_idx, 1] = gamma - t
        for n in range(2, N + 1):
            C[k_idx, n] = (gamma - n - t + 1) * C[k_idx, n-1] - (n - 1) * t * C[k_idx, n-2]
    return C

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
        self.register_buffer('C', get_pc_coefficients(self.K, self.N, self.t))

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        h = F.dropout(self.mlp(x), p=self.dropout, training=self.training)
        
        L_index, L_weight = get_generalized_laplacian(edge_index, num_nodes, self.eta, self.p)
        L_sparse = torch.sparse_coo_tensor(L_index, L_weight, torch.Size([num_nodes, num_nodes])).to(x.device)
        
        T_list = [h]
        current_T = h
        for n in range(1, self.N + 1):
            current_T = -torch.sparse.mm(L_sparse, current_T) / n
            T_list.append(current_T)
            
        T_tensor = torch.stack(T_list, dim=0) 
        inner_sum = torch.einsum('kn,nvc->kvc', self.C, T_tensor) 
        return self.theta_0 * h + torch.sum(self.thetas.unsqueeze(2) * inner_sum, dim=0)

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, out_channels, config, task='nc'):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, config['hidden_channels'])
        self.conv2 = SAGEConv(config['hidden_channels'], config['hidden_channels'] if task == 'lp' else out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

class LinkPredictor(nn.Module):
    def forward(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

# ================= 3. 数据组装与解析引擎 =================
def load_adj_to_edge_index(adj_tensor):
    """智能转换各种格式的邻接矩阵为 edge_index"""
    if hasattr(adj_tensor, 'is_sparse') and adj_tensor.is_sparse:
        return adj_tensor._indices()
    elif adj_tensor.dim() == 2 and adj_tensor.size(0) == adj_tensor.size(1):
        return adj_tensor.nonzero(as_tuple=False).t().contiguous()
    return adj_tensor # 假设已经是 [2, E] 格式

def get_cs_phds_nc_data():
    nc_dir = Path(CONFIG['nc_data_root'])
    if not nc_dir.exists(): return None
    
    x = torch.load(nc_dir / 'feats.pt', weights_only=False)
    y = torch.load(nc_dir / 'labels.pt', weights_only=False)
    adj = torch.load(nc_dir / 'adj.pt', weights_only=False)
    
    edge_index = load_adj_to_edge_index(adj)
    data = Data(x=x, edge_index=edge_index, y=y)
    # NC 没有官方 splits，我们使用标准的 RandomNodeSplit
    data = T.RandomNodeSplit(num_val=0.1, num_test=0.2)(data)
    return data

def get_cs_phds_lp_data():
    lp_dir = Path(CONFIG['lp_data_root'])
    if not lp_dir.exists(): return None
    
    x = torch.load(lp_dir / 'feats.pt', weights_only=False)
    adj_train = torch.load(lp_dir / 'adj_train.pt', weights_only=False)
    train_edge_index = load_adj_to_edge_index(adj_train)
    
    splits = torch.load(lp_dir / 'splits.pt', weights_only=False)
    return x, train_edge_index, splits

# ================= 4. 任务执行流 =================
def run_node_classification(data, config, model_name):
    data = data.to(config['device'])
    data.x = torch.nan_to_num(data.x, nan=0.0)
    
    num_classes = int(data.y.max()) + 1
    model = (PCNet if model_name == "PCNet" else GraphSAGEModel)(data.num_features, num_classes, config, task='nc').to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    mask_train = data.train_mask[:, 0] if data.train_mask.dim() > 1 else data.train_mask
    mask_test = data.test_mask[:, 0] if data.test_mask.dim() > 1 else data.test_mask

    train_start_time = time.time()
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[mask_train], data.y[mask_train])
        loss.backward()
        optimizer.step()
    train_time = time.time() - train_start_time

    test_start_time = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)[mask_test].cpu().numpy()
        y_true = data.y[mask_test].cpu().numpy()
    test_time = time.time() - test_start_time
    
    return {
        "acc": accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average='macro'),
        "micro_f1": f1_score(y_true, preds, average='micro'),
        "train_time": train_time, "test_time": test_time
    }

def run_link_prediction(x, train_edge_index, splits, config, model_name):
    x = torch.nan_to_num(x.to(config['device']), nan=0.0)
    train_edge_index = train_edge_index.to(config['device'])
    
    model = (PCNet if model_name == "PCNet" else GraphSAGEModel)(x.size(1), None, config, task='lp').to(config['device'])
    predictor = LinkPredictor().to(config['device'])
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=config['lr'])
    
    # 解析官方 splits
    try:
        if 'test' in splits and isinstance(splits['test'], dict):
            test_pos = splits['test']['pos'].to(config['device']) if 'pos' in splits['test'] else splits['test']['edge'].to(config['device'])
            test_neg = splits['test']['neg'].to(config['device']) if 'neg' in splits['test'] else splits['test']['edge_neg'].to(config['device'])
        else:
            test_pos = splits.get('test_pos', splits.get('test_edges')).to(config['device'])
            test_neg = splits.get('test_neg', splits.get('test_edges_false')).to(config['device'])
            
        y_true = torch.cat([torch.ones(test_pos.size(1)), torch.zeros(test_neg.size(1))], dim=0).numpy()
        test_edge_label_index = torch.cat([test_pos, test_neg], dim=1)
    except Exception as e:
        print(f"  ⚠️ 解析官方 splits 失败 ({e})，将回退到安全模式。")
        return None

    # 训练
    train_start_time = time.time()
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        z = model(x, train_edge_index)
        
        neg_edges = torch.randint(0, x.size(0), train_edge_index.size(), dtype=torch.long, device=config['device'])
        train_labels = torch.cat([torch.ones(train_edge_index.size(1)), torch.zeros(neg_edges.size(1))], dim=0).to(config['device'])
        train_edges_all = torch.cat([train_edge_index, neg_edges], dim=1)
        
        out = predictor(z, train_edges_all)
        loss = F.binary_cross_entropy_with_logits(out, train_labels)
        loss.backward()
        optimizer.step()
    train_time = time.time() - train_start_time

    # 测试
    test_start_time = time.time()
    model.eval()
    with torch.no_grad():
        z = torch.nan_to_num(model(x, train_edge_index), nan=0.0) 
        out_np = torch.nan_to_num(predictor(z, test_edge_label_index).sigmoid(), nan=0.5).cpu().numpy()
        
        try:
            auc = roc_auc_score(y_true, out_np)
            ap = average_precision_score(y_true, out_np)
        except ValueError:
            auc, ap = 0.5, 0.5
            
    test_time = time.time() - test_start_time

    return {"auc": auc, "ap": ap, "train_time": train_time, "test_time": test_time}

# ================= 5. 主控流程 =================
def main():
    print("🚀 开始 CS PhDs 数据集专属测试...")
    
    print("📦 正在手工拼装 NC 和 LP 数据...")
    data_nc = get_cs_phds_nc_data()
    lp_data_tuple = get_cs_phds_lp_data()
    
    if not data_nc and not lp_data_tuple:
        print("❌ 错误：未找到 NC 或 LP 数据，请检查路径。")
        return

    for m_name in MODELS_TO_RUN:
        print(f"\n" + "="*40)
        print(f"🤖 当前模型: {m_name}")
        print("="*40)
        
        model_exp_dir = ROOT_EXP_DIR / m_name
        model_exp_dir.mkdir(parents=True, exist_ok=True)
        run_metrics = []
        
        for run in range(CONFIG['num_runs']):
            nc_res, lp_res = None, None
            
            if data_nc:
                nc_res = run_node_classification(data_nc, CONFIG, m_name)
            if lp_data_tuple:
                x_lp, train_edge_index, splits = lp_data_tuple
                lp_res = run_link_prediction(x_lp, train_edge_index, splits, CONFIG, m_name)
            
            total_train_time = (nc_res['train_time'] if nc_res else 0) + (lp_res['train_time'] if lp_res else 0)
            total_test_time = (nc_res['test_time'] if nc_res else 0) + (lp_res['test_time'] if lp_res else 0)
            
            # 🚀 新增：计算 per epoch time
            effective_epochs = (CONFIG['epochs'] if nc_res else 0) + (CONFIG['epochs'] if lp_res else 0)
            train_time_per_epoch = total_train_time / effective_epochs if effective_epochs > 0 else 0.0
            
            run_metrics.append({
                "NC_Acc": nc_res['acc'] if nc_res else np.nan,
                "NC_MacroF1": nc_res['macro_f1'] if nc_res else np.nan,
                "NC_MicroF1": nc_res['micro_f1'] if nc_res else np.nan,
                "LP_AUC": lp_res['auc'] if lp_res else np.nan,
                "LP_AP": lp_res['ap'] if lp_res else np.nan,
                "Total_Train_Time": total_train_time,
                "Train_Time_Per_Epoch": train_time_per_epoch,  # 🚀 添加到指标字典中
                "Total_Test_Time": total_test_time
            })
            
        # 统计结果 (pd.DataFrame 会自动计算所有列的均值和标准差)
        df_run = pd.DataFrame(run_metrics)
        summary = {"Dataset": "CS_PhDs"}
        for col in df_run.columns:
            summary[f"{col}_mean"] = df_run[col].mean(skipna=True)
            summary[f"{col}_std"] = df_run[col].std(skipna=True)
            
        print(f"  ✅ [NC] Acc: {summary['NC_Acc_mean']:.4f} ± {summary['NC_Acc_std']:.4f}")
        print(f"  ✅ [LP] AP: {summary['LP_AP_mean']:.4f} ± {summary['LP_AP_std']:.4f} | AUC: {summary['LP_AUC_mean']:.4f} ± {summary['LP_AUC_std']:.4f}")
        print(f"  ⏳ [Time] Total Train: {summary['Total_Train_Time_mean']:.2f}s ± {summary['Total_Train_Time_std']:.2f}s | Train/Epoch: {summary['Train_Time_Per_Epoch_mean']:.6f}s ± {summary['Train_Time_Per_Epoch_std']:.6f}s") # 🚀 打印出单轮耗时及方差
        
        # 保存 Markdown
        with open(model_exp_dir / f"Report_{m_name}_CS_PhDs.md", "w", encoding="utf-8") as f:
            f.write(f"# CS PhDs 实验报告 - {m_name}\n\n")
            # 🚀 在表头中加入 Train/Epoch(s)
            f.write("| Dataset | NC Acc | NC Mac-F1 | NC Mic-F1 | LP AUC | LP AP | Total Train(s) | Train/Epoch(s) | Total Test(s) |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            
            def fmt(col):
                if pd.notna(summary.get(f"{col}_mean")):
                    return f"{summary[f'{col}_mean']:.4f} ± {summary[f'{col}_std']:.4f}"
                return "N/A"
            # 🚀 在写入行数据时增加对 Train_Time_Per_Epoch 的调用
            f.write(f"| **CS_PhDs** | {fmt('NC_Acc')} | {fmt('NC_MacroF1')} | {fmt('NC_MicroF1')} | {fmt('LP_AUC')} | {fmt('LP_AP')} | {fmt('Total_Train_Time')} | {fmt('Train_Time_Per_Epoch')} | {fmt('Total_Test_Time')} |\n")

    print(f"\n🎉 完美！CS PhDs 测试完成，报告保存在: {ROOT_EXP_DIR}")

if __name__ == "__main__":
    main()