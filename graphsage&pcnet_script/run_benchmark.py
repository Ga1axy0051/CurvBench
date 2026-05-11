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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T
from torch_geometric.utils import add_remaining_self_loops, scatter
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

import warnings
warnings.filterwarnings("ignore")

# ================= 1. 全局配置 (双模型自动流转) =================
MODELS_TO_RUN = ["GraphSAGE", "PCNet"]

CONFIG = {
    "data_root": r"D:\Github\benchmark_exp\data",
    "num_runs": 5,           
    "epochs": 200,
    "hidden_channels": 64,
    "num_layers": 2,         
    "lr": 0.01,
    "weight_decay": 5e-4,
    "dropout": 0.5,
    "activation": "relu",
    
    "pc_K": 5,               
    "pc_N": 10,              
    "pc_t": 1.5,             
    "pc_p": 2.0,             
    "pc_eta": 0.5,           
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
ROOT_EXP_DIR = Path(f"./full_benchmark_{TIMESTAMP}")
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
        if N >= 1:
            C[k_idx, 1] = gamma - t
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
        C = get_pc_coefficients(self.K, self.N, self.t)
        self.register_buffer('C', C)

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        h = self.mlp(x)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        L_index, L_weight = get_generalized_laplacian(edge_index, num_nodes, self.eta, self.p)
        L_sparse = torch.sparse_coo_tensor(L_index, L_weight, torch.Size([num_nodes, num_nodes])).to(x.device)
        
        T_list = [h]
        current_T = h
        for n in range(1, self.N + 1):
            neg_L_X = -torch.sparse.mm(L_sparse, current_T)
            current_T = neg_L_X / n
            T_list.append(current_T)
            
        T_tensor = torch.stack(T_list, dim=0) 
        inner_sum = torch.einsum('kn,nvc->kvc', self.C, T_tensor) 
        final_sum = torch.sum(self.thetas.unsqueeze(2) * inner_sum, dim=0) 
        Z = self.theta_0 * h + final_sum
        return Z

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

class LinkPredictor(nn.Module):
    def forward(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

def instantiate_model(data, out_channels, config, model_name, task):
    if model_name == "PCNet":
        return PCNet(data.num_features, out_channels, config, task=task).to(config['device'])
    elif model_name == "GraphSAGE":
        return GraphSAGEModel(data.num_features, out_channels, config, task=task).to(config['device'])
    else:
        raise ValueError(f"❌ 未知模型: {model_name}")

# ================= 3. 任务执行流 =================
def run_node_classification(data, config, model_name):
    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1.0, neginf=-1.0)
    if not hasattr(data, 'y') or data.y is None: return None

    num_classes = int(data.y.max()) + 1
    model = instantiate_model(data, num_classes, config, model_name, task='nc')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    def get_single_mask(mask):
        if hasattr(mask, 'dim') and mask.dim() > 1: return mask[:, 0]
        return mask

    train_mask = get_single_mask(data.train_mask)
    val_mask = get_single_mask(data.val_mask)
    test_mask = get_single_mask(data.test_mask)

    stats = {"loss": [], "acc": []}
    
    train_start_time = time.time()
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        mask = train_mask & (data.y != -1)
        
        if mask.sum() == 0: return None
            
        loss = F.cross_entropy(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_valid = val_mask & (data.y != -1)
            val_acc = accuracy_score(data.y[val_valid].cpu(), pred[val_valid].cpu()) if val_valid.sum() > 0 else 0.0
        stats["loss"].append(loss.item())
        stats["acc"].append(val_acc)
    train_time = time.time() - train_start_time

    test_start_time = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        test_valid = test_mask & (data.y != -1)
        if test_valid.sum() == 0: return None
        preds = logits.argmax(dim=1)[test_valid].cpu().numpy()
        y_true = data.y[test_valid].cpu().numpy()
    test_time = time.time() - test_start_time
    
    return {
        "acc": accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average='macro'),
        "micro_f1": f1_score(y_true, preds, average='micro'),
        "train_time": train_time,
        "test_time": test_time,
        "stats": stats
    }

def run_link_prediction(data, config, model_name):
    if not hasattr(data, 'edge_index') or data.edge_index is None or data.edge_index.size(1) < 20:
        return None

    data.x = torch.nan_to_num(data.x, nan=0.0, posinf=1.0, neginf=-1.0)

    try:
        split = T.RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True, add_negative_train_samples=True)
        train_data, val_data, test_data = split(data)
    except Exception:
        return None
    
    model = instantiate_model(train_data, None, config, model_name, task='lp')
    predictor = LinkPredictor().to(config['device'])
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=config['lr'])
    
    train_start_time = time.time()
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        z = model(train_data.x, train_data.edge_index)
        out = predictor(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(out, train_data.edge_label)
        loss.backward()
        optimizer.step()
    train_time = time.time() - train_start_time

    test_start_time = time.time()
    model.eval()
    with torch.no_grad():
        z = model(test_data.x, test_data.edge_index)
        z = torch.nan_to_num(z, nan=0.0) 
        out = predictor(z, test_data.edge_label_index).sigmoid()
        
        out_np = torch.nan_to_num(out, nan=0.5).cpu().numpy()
        y_true = test_data.edge_label.cpu().numpy()
        
        try:
            auc = roc_auc_score(y_true, out_np)
            ap = average_precision_score(y_true, out_np) # 👈 直接计算 AP
        except ValueError:
            auc, ap = 0.5, 0.5 # 如果全图全是 NaN，退化为随机猜测水平 0.5
            
    test_time = time.time() - test_start_time

    return {
        "auc": auc, 
        "ap": ap, # 👈 返回 AP
        "train_time": train_time,
        "test_time": test_time
    }

# ================= 4. 报告生成 =================
def generate_markdown_report(results_table, model_name, exp_dir):
    md_path = exp_dir / f"Report_{model_name}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark 实验报告 - {model_name}\n\n")
        headers = [
            "Dataset", "NC Acc", "NC Mac-F1", "NC Mic-F1", 
            "LP AUC", "LP AP", 
            "Total Train(s)", "Epoch(s)", "Total Test(s)"
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join([":---"] * len(headers)) + "|\n")
        
        for row in results_table:
            dataset = row['Dataset']
            def fmt(col):
                if f"{col}_mean" in row and pd.notna(row[f"{col}_mean"]):
                    return f"{row[f'{col}_mean']:.4f} ± {row[f'{col}_std']:.4f}"
                return "N/A"
            f.write(f"| **{dataset}** | {fmt('NC_Acc')} | {fmt('NC_MacroF1')} | {fmt('NC_MicroF1')} | {fmt('LP_AUC')} | {fmt('LP_AP')} | {fmt('Total_Train_Time')} | {fmt('Train_Time_Per_Epoch')} | {fmt('Total_Test_Time')} |\n")

def save_intermediate_txt(dataset_name, summary, model_name, exp_dir):
    txt_path = exp_dir / f"{model_name}_{dataset_name}_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== {model_name} - {dataset_name} 实验报告 ===\n")
        for k, v in summary.items():
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")

# ================= 5. 主控流程 =================
def main():
    
    import sys; import os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from parquet_loader import load_parquet_as_pyg
    except Exception as e:
        print(f'Error loading parquet_loader: {e}')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, default='nc')
    args, _ = parser.parse_known_args()

    with open(ROOT_EXP_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=4)

    for m_name in MODELS_TO_RUN:
        print(f"\n" + "="*50)
        print(f"🚀 正在启动模型流: {m_name}")
        print("="*50)
        
        model_exp_dir = ROOT_EXP_DIR / m_name
        model_exp_dir.mkdir(parents=True, exist_ok=True)
        results_table = []
        
        
        dataset_name = args.dataset
        print(f"\\n[{m_name}] >>>> 正在测试: {dataset_name} <<<<")
        try:
            data = load_parquet_as_pyg(dataset_name).to(CONFIG['device'])
        except Exception as e:
            print(f"❌ 读取失败: {e}")

                
            run_metrics = []
            
            for run in range(CONFIG['num_runs']):
                nc_res = run_node_classification(data, CONFIG, m_name)
                lp_res = run_link_prediction(data, CONFIG, m_name)
                
                nc_train = nc_res['train_time'] if nc_res else 0
                nc_test = nc_res['test_time'] if nc_res else 0
                lp_train = lp_res['train_time'] if lp_res else 0
                lp_test = lp_res['test_time'] if lp_res else 0
                
                total_train_time = nc_train + lp_train
                total_test_time = nc_test + lp_test
                effective_epochs = (CONFIG['epochs'] if nc_res else 0) + (CONFIG['epochs'] if lp_res else 0)
                train_time_per_epoch = total_train_time / effective_epochs if effective_epochs > 0 else 0
                
                run_metrics.append({
                    "NC_Acc": nc_res['acc'] if nc_res else np.nan,
                    "NC_MacroF1": nc_res['macro_f1'] if nc_res else np.nan,
                    "NC_MicroF1": nc_res['micro_f1'] if nc_res else np.nan,
                    "LP_AUC": lp_res['auc'] if lp_res else np.nan,
                    "LP_AP": lp_res['ap'] if lp_res else np.nan,
                    "Total_Train_Time": total_train_time,
                    "Train_Time_Per_Epoch": train_time_per_epoch,
                    "Total_Test_Time": total_test_time
                })
                
                if run == 0 and nc_res:
                    plt.figure(figsize=(10, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(nc_res['stats']['loss'], color='blue')
                    plt.title(f"{dataset_name} Loss")
                    plt.subplot(1, 2, 2)
                    plt.plot(nc_res['stats']['acc'], color='green')
                    plt.title(f"{dataset_name} Acc")
                    plt.tight_layout()
                    plt.savefig(model_exp_dir / f"{dataset_name}_convergence.png")
                    plt.close()

            df_run = pd.DataFrame(run_metrics)
            summary = {"Dataset": dataset_name}
            for col in df_run.columns:
                summary[f"{col}_mean"] = df_run[col].mean(skipna=True)
                summary[f"{col}_std"] = df_run[col].std(skipna=True)
            
            results_table.append(summary)
            save_intermediate_txt(dataset_name, summary, m_name, model_exp_dir)
            print(f"  ✅ 完成! 数据集 {dataset_name} 的中间结果已存盘。")

        if results_table:
            pd.DataFrame(results_table).to_csv(model_exp_dir / f"Metrics_{m_name}.csv", index=False)
            generate_markdown_report(results_table, m_name, model_exp_dir)
            print(f"🎉 [{m_name}] 实验全部结束！")

    print(f"\n🏆 所有模型全量实验圆满完成！总目录: {ROOT_EXP_DIR}")

if __name__ == "__main__":
    main()