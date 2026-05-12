# SA²GFM

**Structure-Aware Semantic Augmentation for Robust Graph Foundation Models**

Official PyTorch implementation of **SA²GFM** (*SA²GFM: Enhancing Robust Graph Foundation Models with Structure-Aware Semantic Augmentation*).

**Dataset (Hugging Face):** [aboutime233/SA2GFM](https://huggingface.co/datasets/aboutime233/SA2GFM)

---

## Abstract

Graph foundation models (GFMs) must generalize across domains while remaining stable under noise and adversarial perturbations. **SA²GFM** pre-trains **per-domain expert** encoders with **structure-aware semantic augmentation** (community structure → text prompts → language-model features fused with graph signals) and a **self-supervised information bottleneck** objective. At fine-tuning time, a **mixture-of-experts router** (with a **null expert** for negative transfer) combines source experts on the target graph, together with **hierarchical structure optimization** on the target topology. This release contains end-to-end scripts for community detection, feature enhancement, few-shot splits, single-graph pretraining, MoE downstream fine-tuning, and attack data generation.

---

## Table of contents

- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Repository layout](#repository-layout)
- [Data preparation](#data-preparation)
- [End-to-end pipeline](#end-to-end-pipeline)
- [YAML Config Usage (`configs/`)](#yaml-config-usage-configs)
- [Expert–target pairing](#experttarget-pairing)
- [Module reference](#module-reference)
- [Attacks and `downstream`](#attacks-and-downstream)

---

## Highlights

- **Structure-aware pretraining**: Hierarchical communities → textual prompts → BERT; fused with structural SVD features (`enhanced_x_64` / `enhanced_x`).
- **Self-supervised information bottleneck (SS-IB)**: Contrastive consistency + KL-style compression for denoised, transferable representations.
- **MoE fine-tuning**: Adaptive routing over pretrained experts; **null expert** when no source domain is useful.
- **Target structure refinement**: Lightweight intra-/inter-cluster optimization for robustness to structural perturbations.
- **Reproducible attack pipeline**: Targeted (e.g., Nettack chain) and random feature/structure corruptions; outputs wired to downstream evaluation.

All artifact paths resolve through **`SA2GFM_DATA_ROOT`** and [`attack_data_gen/lib/paths.py`](attack_data_gen/lib/paths.py) (no hard-coded user directories).

---

## Requirements

- **Python** 3.8+ recommended  
- **PyTorch** ≥ 1.12, **PyTorch Geometric** ≥ 2.0, **torch-scatter**  
- **NetworkX** (community detection; Louvain needs NetworkX ≥ 3.2)  
- **transformers** / **Hugging Face** (default: `bert-base-uncased` for feature enhancement)  
- **SwanLab** (optional logging; disable flags where noted below)

Install **per stage** from each module’s `requirements.txt` (versions may differ slightly across components).

---

## Installation

```bash
git clone <REPOSITORY_URL> SA2GFM
cd SA2GFM
```

Set the data root (see [Data preparation](#data-preparation)):

```bash
export SA2GFM_DATA_ROOT=/path/to/your/data
```

Example: install downstream dependencies only:

```bash
cd downstream && pip install -r requirements.txt && cd ..
```

Repeat for `community_detection`, `node_feature_enhance`, `few_shot_gen`, `pretrain`, and `attack_data_gen` as needed.

---

## Repository layout

| Path | Role |
|------|------|
| [`community_detection/`](community_detection/) | `ori/*.pt` → `communities/{dataset}_communities.pt` |
| [`node_feature_enhance/`](node_feature_enhance/) | BERT + structure branch → **`enhanced_x_64`** |
| [`few_shot_gen/`](few_shot_gen/) | Few-shot `split_*.pt` under `few_shot/{dataset}/{k}shot/` |
| [`pretrain/`](pretrain/) | Single-graph expert pretraining → `save_model/{name}.pt` |
| [`downstream/`](downstream/) | MoE fine-tuning and evaluation |
| [`attack_data_gen/`](attack_data_gen/) | Targeted / random attacks → `outputs/` |
| [`data/`](data/) | Optional local layout: `data/ori/{dataset}.pt`, … |

**Default data root** when `SA2GFM_DATA_ROOT` is unset: if the repo lives under `.../test_pretrain/SA2GFM` and `.../test_pretrain/data/ori/` exists, use `.../test_pretrain/data`; otherwise use this repo’s `data/`. See path helpers in the codebase for details.

---

## Data preparation

Graph tensors and related assets are available on Hugging Face Datasets: **[aboutime233/SA2GFM](https://huggingface.co/datasets/aboutime233/SA2GFM)**. Download or `git clone` the dataset and arrange files so they match the layout below (e.g. `ori/{dataset}.pt` under your `SA2GFM_DATA_ROOT`).

Under `$SA2GFM_DATA_ROOT`:

| Path | Contents |
|------|----------|
| `ori/{dataset}.pt` | PyG `Data`: `x`, `edge_index`, `y`; pretrain/downstream require **`enhanced_x_64`** (or `enhanced_x`) |
| `communities/{dataset}_communities.pt` | `{"communities": list[list[int]], ...}` |
| `few_shot/{dataset}/{k}shot/split_{i}.pt` | `{"indices", "labels"}` |
| `save_model/{name}.pt` | `JointContrastiveModel` checkpoint per source graph |

Standalone checkout: you may place graphs under `data/ori/{dataset}.pt` and point `SA2GFM_DATA_ROOT` at `data/` or the parent that contains `ori/`, `communities/`, etc.

### Feature file naming (important)

`node_feature_enhance` **by default** may write `ori/{dataset}_enhanced_x64.pt`, while **pretrain** and **downstream** read **`ori/{dataset}.pt`**. Either:

- write directly to the canonical name, e.g.  
  `bash node_feature_enhance/scripts/run_build.sh cora --device cuda --output "$SA2GFM_DATA_ROOT/ori/cora.pt"`  
  (back up the original first), or  
- copy: `cp ori/{dataset}_enhanced_x64.pt ori/{dataset}.pt`

Every **source** graph that should appear under `save_model/` must carry enhanced fields in `ori/{source}.pt`.

### Few-shot split count

**All datasets use the same count:** generate **20** splits per dataset and shot setting (`split_0.pt` … `split_19.pt`). **`downstream --num-splits`** defaults to **20** when set to `-1`. `few_shot_gen/scripts/run_all_default.sh` calls `run_generate.sh` with **`--n-splits 20`** for each dataset × (1-shot, 5-shot). Increase `--n-splits` / `--num-splits` only if you intentionally need more runs.

---

## End-to-end pipeline

| Step | Directory | Description |
|------|-----------|-------------|
| 1 | `community_detection/` | Community partitions from `ori/*.pt` |
| 2 | `node_feature_enhance/` | Build **`enhanced_x_64`** (SVD + BERT templates) |
| 3 | `few_shot_gen/` | k-shot splits; default reserves last **1000** nodes for test |
| 4 | `pretrain/` | One expert per **source** graph → `save_model/{source}.pt` |
| 5 | `downstream/` | MoE on **target**: train `--epochs` per split, **one** test eval per split, aggregate **mean / std** |

**Downstream protocol** (see `downstream/`): for each few-shot split, optimize on training nodes only; evaluate once on the fixed test band (last 1000 nodes); report mean / std / min / max over splits. Default `--epochs 1`; **`--num-splits -1` → 20** splits for every dataset (including arxiv). **Transductive** setting should be stated explicitly in the paper.

---

## YAML Config Usage (`configs/`)

The `configs/` directory at the repository root contains reusable YAML configurations (for example, `curvature_fold1_pretrain.yaml`). The unified entrypoint is `main.py`, and configurations are loaded via `--config_load_path`:

```bash
# Pretraining (load YAML)
python main.py --run_type pretrain --config_load_path configs/curvature_fold1_pretrain.yaml

# Downstream adaptation (the same YAML can also be used for adapt)
python main.py --run_type adapt --config_load_path configs/curvature_fold1_pretrain.yaml
```

Rules:

- `--config_load_path` points to the YAML file path (relative paths are resolved from the current working directory).
- CLI arguments have higher priority than YAML values. If the same key appears in both, the CLI value is used.
- If `fold_name` is not explicitly set in YAML, the YAML filename (without extension) is used automatically.
- You can override key parameters after loading YAML, for example:

```bash
python main.py \
  --run_type pretrain \
  --config_load_path configs/curvature_fold1_pretrain.yaml \
  --gpu 0 \
  --pretrain_nb_epochs 3000 \
  --no_swanlab
```

Recommended practice:

- For new experiments, copy any YAML under `configs/` and modify fields such as `source_datasets`, `target_datasets`, and training hyperparameters.
- Keep `root` as a relative path (e.g., `./datasets`) or align it with your `SA2GFM_DATA_ROOT` to avoid path confusion.

---

## Expert–target pairing

MoE on target `T` loads experts from `get_pretrain_datasets(T)` in [`downstream/lib/config.py`](downstream/lib/config.py); **`T` is excluded** from its own expert list. Pretrain each listed graph to `save_model/{name}.pt`. Helper: `pretrain/scripts/run_experts_for_downstream.sh <T>`.

| Target `T` | Source experts `name` |
|------------|------------------------|
| cora | citeseer, pubmed, P-home, wikics |
| citeseer | cora, pubmed, P-home, wikics |
| pubmed | cora, citeseer, P-home, wikics |
| P-tech | cora, citeseer, pubmed, P-home, wikics |
| P-home | cora, citeseer, pubmed, wikics |
| wikics | cora, citeseer, pubmed, P-home |
| arxiv | P-home, P-tech, wikics |

---

## Module reference

Commands assume the **repository root** as the current working directory.

### `community_detection/`

**Output:** `communities/{dataset}_communities.pt` (disjoint `communities`; optional `meta`).  
**Methods** (`--method`): `louvain`, `greedy_modularity`, `label_propagation`.

```bash
cd community_detection && pip install -r requirements.txt
export SA2GFM_DATA_ROOT=/path/to/data
python pipeline/01_detect_communities.py --dataset cora --method louvain --seed 42
python pipeline/02_analyze_communities.py --dataset cora
bash scripts/run_detect.sh citeseer louvain
bash scripts/run_all_default.sh greedy_modularity
cd ..
```

### `node_feature_enhance/`

Pipeline: read `communities/*.pt` → structure `TruncatedSVD(x)→32` → text template (cluster + neighbors) → BERT → `TruncatedSVD→32` → concat → **`enhanced_x_64`**. Override BERT with `--bert`.

```bash
cd node_feature_enhance && pip install -r requirements.txt
export SA2GFM_DATA_ROOT=/path/to/data
bash scripts/run_build.sh cora --device cuda
bash scripts/run_all_default.sh --device cuda
# Optional: --output "$SA2GFM_DATA_ROOT/ori/cora.pt"
cd ..
```

### `few_shot_gen/`

Default **`--n-splits`** in `pipeline/01_generate_splits.py` is **20**; `run_all_default.sh` uses **20** for every dataset.

```bash
cd few_shot_gen && pip install -r requirements.txt
export SA2GFM_DATA_ROOT=/path/to/data
bash scripts/run_generate.sh cora 5 --n-splits 20
bash scripts/run_all_default.sh
cd ..
```

### `pretrain/`

Single-graph contrastive training (`JointContrastiveModel`); checkpoints **`save_model/{dataset}.pt`** (fixed filename). Logic aligns with the historical `pretrain_eso/` single-graph path when present in a fuller checkout.

```bash
cd pretrain && pip install -r requirements.txt
export SA2GFM_DATA_ROOT=/path/to/data
bash scripts/run_experts_for_downstream.sh cora --nb_epochs 3000 --gpu 0
bash scripts/run_pretrain.sh citeseer --nb_epochs 3000 --gpu 0
bash scripts/run_experts_for_downstream.sh pubmed --no_swanlab
cd ..
```

### `downstream/`

**Requires:** enhanced `ori/{dataset}.pt`, `communities/`, `few_shot/`, and all `save_model/` experts for that target.

```bash
cd downstream && pip install -r requirements.txt
export SA2GFM_DATA_ROOT=/path/to/data
bash scripts/run_downstream.sh --dataset cora --shot_num 1 --gpu 0
bash scripts/run_downstream.sh --dataset cora --shot_num 1 --epochs 50 --no-swanlab
bash scripts/run_downstream.sh --dataset cora --shot_num 1 --attack_type random --random_attack_type feature --attack_ratio 0.1
cd ..
```

---

## Attacks and `downstream`

### `attack_data_gen/`

**Input:** `$SA2GFM_DATA_ROOT/ori/{dataset}.pt` with `enhanced_x_64`, `edge_index`, `y`.  
**Output:** `attack_data_gen/outputs/` (`attack_post/`, `attacked_data_random/`, …; large files are often gitignored).

Pipelines: `pipeline/01`–`03` Nettack chain; `04` random perturbation; `05`–`06` optional DeepRobust Metattack. See `lib/paths.py`, `lib/data_utils.py`, `lib/gcn_surrogate.py`.

```bash
cd attack_data_gen && pip install -r requirements.txt
export SA2GFM_DATA_ROOT=/path/to/data
bash scripts/run_targeted_nettack.sh cora 1
SKIP_SURROGATE_TRAIN=1 bash scripts/run_targeted_nettack.sh cora 1
bash scripts/run_random_all_ratios.sh cora
cd ..
```

**Mapping to `downstream --attack_type`:**

| `attack_type` | Expected artifact (under `outputs/`) |
|---------------|----------------------------------------|
| `targeted_poisoning` | `attack_post/{ds}_p{p}_final/{ds}_poisoning_final.pt` |
| `targeted_evasion` | `attack_post/{ds}_p{p}_final/{ds}_evasion_final.pt` |
| `random` (feature) | `attacked_data_random/{ds}_feature_p{ratio}.pt` |
| `random` (structure) | `attacked_data_random/{ds}_structure_p{ratio}.pt` |

Please cite **Nettack**, **DeepRobust**, and the versions you install, per their licenses.
