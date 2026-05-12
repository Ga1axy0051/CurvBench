from __future__ import annotations

from pathlib import Path

import torch

from attack_data_gen.lib.data_utils import canonical_graph_path, load_graph, normalize_dataset_name, has_enhanced_features
from attack_data_gen.lib.paths import paths
from configs.pretrain_config import PretrainConfig
from utils.runtime import ensure_dir, load_module_from_path, save_yaml, set_data_root_env


class PretrainRunner:
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_root = (self.project_root / self.config.root).resolve() if not Path(self.config.root).is_absolute() else Path(self.config.root).resolve()
        set_data_root_env(self.data_root)

        self.community_module = load_module_from_path(
            "sa2gfm_detect_communities",
            self.project_root / "community_detection" / "pipeline" / "01_detect_communities.py",
        )
        self.enhance_module = load_module_from_path(
            "sa2gfm_build_enhanced_x",
            self.project_root / "node_feature_enhance" / "pipeline" / "01_build_enhanced_x.py",
        )
        self.pretrain_module = load_module_from_path(
            "sa2gfm_train_single",
            self.project_root / "pretrain" / "pipeline" / "train_single.py",
        )

    @property
    def fold_dir(self) -> Path:
        return ensure_dir((self.project_root / self.config.checkpoint_root / self.config.fold_name).resolve())

    @property
    def experts_dir(self) -> Path:
        return ensure_dir(self.fold_dir / "experts")

    @property
    def manifest_path(self) -> Path:
        return self.fold_dir / self.config.manifest_name

    def _ensure_communities(self, dataset: str) -> Path:
        path = paths.communities_dir / f"{dataset}_communities.pt"
        if path.is_file():
            return path
        return self.community_module.run_detection(
            dataset=dataset,
            method=self.config.community_method,
            seed=self.config.seed,
            resolution=self.config.community_resolution,
        )

    def _ensure_enhanced_graph(self, dataset: str) -> Path:
        graph_path = canonical_graph_path(dataset)
        if graph_path.is_file():
            try:
                graph_data = load_graph(dataset)
                if has_enhanced_features(graph_data):
                    return graph_path
            except Exception:
                pass
        return self.enhance_module.build_enhanced_x(
            dataset=dataset,
            output=str(graph_path),
            svd_dim=self.config.svd_dim,
            text_svd_dim=self.config.text_svd_dim,
            bert=self.config.bert,
            batch_size=self.config.bert_batch_size,
            device="cuda" if torch.cuda.is_available() and self.config.gpu >= 0 else "cpu",
            seed=self.config.seed,
            max_neighbors_in_text=self.config.max_neighbors_in_text,
        )

    def _prepare_dataset(self, dataset: str) -> None:
        self._ensure_communities(dataset)
        self._ensure_enhanced_graph(dataset)

    def _train_expert(self, dataset: str) -> Path:
        out_path = self.experts_dir / f"{dataset}.pt"
        return self.pretrain_module.train_expert(
            dataset=dataset,
            seed=self.config.seed,
            gpu=self.config.gpu,
            hid_units=self.config.pretrain_hid_units,
            out_channels=self.config.pretrain_out_channels,
            num_layers=self.config.pretrain_num_layers,
            dropout=self.config.pretrain_dropout,
            lr=self.config.pretrain_lr,
            l2_coef=self.config.pretrain_l2_coef,
            nb_epochs=self.config.pretrain_nb_epochs,
            patience=self.config.pretrain_patience,
            eval_steps=self.config.pretrain_eval_steps,
            neg_samples=self.config.pretrain_neg_samples,
            kl_weight=self.config.pretrain_kl_weight,
            output=str(out_path),
            no_swanlab=self.config.no_swanlab,
        )

    def run(self) -> Path:
        source_datasets = [normalize_dataset_name(name) for name in (self.config.source_datasets or [])]
        if not source_datasets:
            raise ValueError("Pretrain requires non-empty source_datasets in the config.")

        manifest = {
            "fold_name": self.config.fold_name,
            "data_root": str(self.data_root),
            "source_datasets": source_datasets,
            "target_datasets": [normalize_dataset_name(name) for name in (self.config.target_datasets or [])],
            "experts": {},
            "config": self.config.__dict__,
        }

        for dataset in source_datasets:
            print(f"\n=== Preparing dataset: {dataset} ===")
            if self.config.auto_prepare:
                self._prepare_dataset(dataset)
            print(f"\n=== Training expert: {dataset} ===")
            expert_path = self._train_expert(dataset)
            manifest["experts"][dataset] = str(expert_path)

        save_yaml(manifest, self.manifest_path)

        # Compatibility checkpoint path so adapt CLI can still look GraphGlue-like.
        compat_path = self.fold_dir / "pretrain_final_model.pth"
        torch.save({"manifest_path": str(self.manifest_path), "experts": manifest["experts"]}, compat_path)
        print(f"Saved manifest: {self.manifest_path}")
        print(f"Saved compatibility checkpoint: {compat_path}")
        return compat_path
