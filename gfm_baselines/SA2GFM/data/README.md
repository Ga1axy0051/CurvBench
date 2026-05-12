# Data directory — SA²GFM

**Structure-Aware Semantic Augmentation for Robust Graph Foundation Models**

This folder is the **optional local data root** for the official **SA²GFM** implementation in the parent repository ([`../README.md`](../README.md)). SA²GFM is a graph foundation model that uses **structure-aware semantic augmentation** (communities → text → language-model features), a **self-supervised information bottleneck** for pretraining, and **mixture-of-experts routing** (with a **null expert**) for robust downstream adaptation.

The **code** (community detection, feature enhancement, pretraining, downstream MoE, attacks) lives under `../community_detection/`, `../node_feature_enhance/`, `../pretrain/`, `../downstream/`, and `../attack_data_gen/`. **This `data/` tree only holds datasets and derived artifacts** consumed by those modules.

---

## Pointing the code at this folder

From the repository root:

```bash
export SA2GFM_DATA_ROOT="$(pwd)/data"
```

If `SA2GFM_DATA_ROOT` is unset, the project may default to this repo’s `data/` when no sibling `test_pretrain/data` layout is detected—see path resolution in the codebase and the root README.

---

## Expected layout under the data root

| Path | Role |
|------|------|
| `ori/{dataset}.pt` | PyTorch Geometric `Data`: at minimum `x`, `edge_index`, `y`. **Pretrain and downstream** expect **`enhanced_x_64`** (or `enhanced_x`) after running `node_feature_enhance`. |
| `communities/{dataset}_communities.pt` | Produced by `community_detection/` |
| `few_shot/{dataset}/{k}shot/split_*.pt` | Produced by `few_shot_gen/` |
| `save_model/{name}.pt` | Expert checkpoints from `pretrain/` |

Attack outputs typically live under `../attack_data_gen/outputs/` when you run the attack pipeline.

---

## What is in *this* checkout right now

```text
data/
└── ori/
    ├── arxiv.pt
    ├── citeseer.pt
    ├── cora.pt
    ├── pubmed.pt
    ├── P-home.pt
    ├── P-tech.pt
    └── wikics.pt
```

These are **canonical graph tensors** for the listed datasets. Before running the full SA²GFM pipeline, generate **communities**, **enhanced features**, **few-shot splits**, and **per-source experts** as described in the [root README](../README.md).

---

## Naming note

Graph files should be named `{dataset}.pt` (e.g. `cora.pt`) so scripts resolve `ori/{dataset}.pt` consistently. Duplicate-download names such as `dataset (2).pt` should be renamed to avoid path mismatches.

---

## More documentation

- Full install, pipeline, expert–target table, and commands: [`../README.md`](../README.md)
