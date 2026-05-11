import torch
import time
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

from manify.utils.dataloaders import load_hf
from manify.curvature_estimation.sectional_curvature_strict_fast import sectional_curvature_gpu


# =========================================================
# 自动选择设备
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
    print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB\n")


# =========================================================
# 选择数据集
# =========================================================
#dataset_name = "computers"   #  换数据集
#dataset_name = "pubmed"
#dataset_name = "cora"
#dataset_name = "citeseer"
#dataset_name = "photo"
#dataset_name = "cs"
#dataset_name = "polblogs"
#dataset_name = "polbooks"
#dataset_name = "wordnet"   # WordNet 数据集
#dataset_name = "telecom"
#dataset_name = "twitter"
#dataset_name = "twitter10k"   # Higgs 子图数据集
#dataset_name = "rocketfuel_7018"
#dataset_name = "roadnetca20k"
#dataset_name = "roadnetca100k"
dataset_name = "airport"   # Airport 数据集
#dataset_name = "disease"   # Disease 网络数据集
# =========================================================
# 加载数据集
# =========================================================
features, dists, adj, labels = load_hf(dataset_name)

if features is None:
    print(" Warning: features 为 None，使用单位矩阵代替。")
    features = torch.eye(adj.shape[0])

print(" 数据加载成功：")
print(f"节点数: {adj.shape[0]}")
print(f"特征维度: {features.shape[1]}")
if labels is not None:
    print(f"类别数: {len(torch.unique(labels))}\n")
else:
    print("类别数: None（此数据集无标签）\n")



# =========================================================
# 自动选择模式
# =========================================================
n_nodes = adj.shape[0]
if n_nodes <= 5000:
    mode = "fast"
else:
    mode = "strict"

print(f" 自动选择计算模式：{mode.upper()}  （节点数 = {n_nodes}）\n")


# =========================================================
# 移动数据到设备
# =========================================================
adj = adj.to(device)
dists = dists.to(device)


# =========================================================
# 输出目录准备
# =========================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
save_dir = os.path.join("curvature_results", dataset_name, timestamp)
os.makedirs(save_dir, exist_ok=True)


# =========================================================
# 曲率计算
# =========================================================
print(f" 开始计算 {dataset_name} 数据集的截面曲率...\n")
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
        print("\n GPU 显存不足，自动切换到 CPU 重新计算。")
        torch.cuda.empty_cache()
        adj = adj.cpu()
        dists = dists.cpu()
        curvatures = sectional_curvature_gpu(
            adj,
            dists,
            device="cpu",
            mode=mode,
            relative=True,
            show_progress=True,
        )
    else:
        raise e

elapsed = time.time() - start_time
print(f"\n 曲率计算总耗时: {elapsed / 60:.2f} 分钟\n")


# =========================================================
# 结果保存与统计
# =========================================================
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

print(" 平均曲率:", summary["mean"])
print(" 最小值:", summary["min"])
print(" 最大值:", summary["max"])
print(f" 结果已保存到: {save_dir}\n")


# =========================================================
# 绘制曲率分布直方图
# =========================================================
plt.figure(figsize=(8, 5))
plt.hist(curvatures_cpu.numpy(), bins=30, color="skyblue", edgecolor="black")
plt.title(f"{dataset_name.upper()} node curvature ({mode.upper()} GPU)")
plt.xlabel("Sectional Curvature")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "curvature_distribution.png"), dpi=300)
plt.show()
