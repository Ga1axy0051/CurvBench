# CurvBench (曲率图基准测试框架)

CurvBench 是一个致力于评估并测试图神经网络及图基础模型在不同图数据集上的几何性质（如截面曲率）与有效性的统一基准测试框架。本项目涵盖了从传统的 GNN 模型到前沿的预训练大模型 (GFM) 的标准测试集合，并且附带了完整的自动化脚本以测算模型的下游表现和数据集属性。

## 📁 项目结构

项目已被精简且完全模块化，分离了基础测试与基础图大语言模型测试的代码依赖。

```
CurvBench/
├── Cal_curv/               # 自动图曲率分布计算专用分析模块 (支持 GPU 快速计算 / 严格计算)。
├── classic_baselines/      # 经典图神经网络（传统 GNN 和改进 GNN 系列）。
│   ├── cusp_script/        # Cusp (CVPR 2023)
│   ├── graphmore_script/   # GraphMoRe
│   ├── graphsage&pcnet_script/ # GraphSAGE 与 PCNet 基线
│   ├── hat_script/         # HAT
│   ├── hnn_hgcn_script/    # HGCN (双曲 GCN)
│   ├── hybonet_script/     # HyboNet (双曲 GNN 相关)
│   ├── mlp_gcn_gat_script/ # 核心三剑客 (MLP, GCN, GAT)
│   └── qgcn_script/        # QGCN (量子态 GCN 相关)
├── gfm_baselines/          # 图基础模型以及基于少样本推断（Few-Shot）系列的大模型。
│   ├── GCOPE-main/         # GCOPE (预训练结构)
│   ├── mdgfm/              # MDGFM
│   ├── mdgpt/              # MDGPT
│   └── SAMGPT/             # SAMGPT (新融合)
├── datasets/               # 统一挂载的 parquet 格式下游数据集 (从 HF Hub 下载并存储在这)。
├── scripts/                # 全部 Bash 批量自动化测试及调用脚本。
├── main.py                 # 所有模型的唯一、统一中心调度器入口。
├── parquet_loader.py       # 负责将中心挂载的 Dataset 无缝转换喂给所有的不同模型架构。
└── environment.yml         # CurvBench 统合的基础 Python/Conda 运行环境清单。
```

## 📊 托管数据集列表

总计 15 个统一的标准图数据集：
- **引文网络字典:** `cora`, `citeseer`, `PubMed`
- **异构与专业图谱:** `Actor`, `Airport`, `Carcinogenesis_data`, `cornell`, `cs_phds_lp_ready`, `cs_phds_nc_ready`, `Disease`, `f1`, `Hepatitis_std_data`, `PTE`, `telecom`, `Toxicology_data`

## 🚀 统一入口与单独模型调用

整个项目的设计模式为**万物皆通过 `main.py` 调用**。你可以针对特定的一个模型（或者特定的一种 GFM），对任意兼容的数据集进行测试：

```bash
# 激活环境
conda activate curvbench

# 测算图截面几何曲率
python main.py --model cal_curv --dataset cora

# 测试经典 Baseline (例如 GCN、GraphSAGE，pcnet)
python main.py --model graphsage --dataset Actor --task nc
python main.py --model pcnet --dataset cora --task nc
python main.py --model hat --dataset citeseer

# 测试包含 少样本 (Few-Shot) 的图大模型 (例如 5-shot)
python main.py --model gcope --dataset PubMed --shot_num 5
python main.py --model mdgpt --dataset telecom --shot_num 1
python main.py --model samgpt --dataset f1 --shot_num 1
```

**所有可选的模型 `--model` 标识符包括：**
- **经典类**: `mlp_gcn_gat`, `cusp`, `hat`, `hgcn`, `hybonet`, `qgcn`, `graphmore`, `graphsage`, `pcnet`
- **图基大模型**: `gcope`, `mdgfm`, `mdgpt`, `samgpt`
- **专属组件**: `cal_curv`

## ⚙️ 一键评估脚本

我们已针对不同的大模块编写了可以直接测试核心数据的 `*_test.sh`（只测试 3 个快速数据集）与遍历全量数据的 `*_full.sh` 测试环境：

```bash
cd scripts/

# 测算 15 个数据集曲率分布并在 logs 生成 png 柱状图
bash cal_curv_full.sh

# 测试所有的传统基准模型
bash classic_test.sh     # 只测试三大网
nohup bash classic_full.sh &   # 后台跑完全部数据集

# 测试所有 GFM 模型 (脚本内部覆盖了 1-shot 和 5-shot)
bash gfm_test.sh
```

由于运行日志与 `__pycache__` 等产生的杂项会被 `.gitignore` 过滤，你可以随时使用 `git status` 并提交修改。
