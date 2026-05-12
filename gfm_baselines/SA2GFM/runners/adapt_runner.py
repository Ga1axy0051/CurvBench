from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from attack_data_gen.lib.data_utils import canonical_graph_path, load_graph, normalize_dataset_name, has_enhanced_features
from attack_data_gen.lib.paths import paths
from configs.adapt_config import AdaptConfig
from utils.runtime import ensure_dir, load_module_from_path, load_yaml, set_data_root_env


class AdaptRunner:
    def __init__(self, config: AdaptConfig):
        self.config = config
        self.project_root = Path(__file__).resolve().parents[1]
        self.data_root = (self.project_root / self.config.root).resolve() if not Path(self.config.root).is_absolute() else Path(self.config.root).resolve()
        set_data_root_env(self.data_root)

        self.community_module = load_module_from_path(
            "sa2gfm_detect_communities_for_adapt",
            self.project_root / "community_detection" / "pipeline" / "01_detect_communities.py",
        )
        self.enhance_module = load_module_from_path(
            "sa2gfm_build_enhanced_x_for_adapt",
            self.project_root / "node_feature_enhance" / "pipeline" / "01_build_enhanced_x.py",
        )
        self.split_module = load_module_from_path(
            "sa2gfm_generate_splits_for_adapt",
            self.project_root / "few_shot_gen" / "pipeline" / "01_generate_splits.py",
        )
        self.downstream_module = load_module_from_path(
            "sa2gfm_train_downstream_for_adapt",
            self.project_root / "downstream" / "pipeline" / "train_downstream.py",
        )

    def _load_manifest(self) -> dict:
        ckpt = Path(self.config.pretrained_checkpoint)
        if not ckpt.is_absolute():
            ckpt = (self.project_root / ckpt).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt}")

        if ckpt.suffix in {".yaml", ".yml"}:
            return load_yaml(ckpt)

        payload = self.downstream_module._torch_load(str(ckpt))
        manifest_path = payload.get("manifest_path")
        if manifest_path is None:
            raise ValueError(f"{ckpt} is not a compatible unified pretrain checkpoint.")
        return load_yaml(Path(manifest_path))

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
            device="cuda" if self.config.gpu >= 0 else "cpu",
            seed=self.config.seed,
            max_neighbors_in_text=self.config.max_neighbors_in_text,
        )

    def _ensure_split_dir(self, dataset: str, k_shot: int) -> Path:
        split_dir = paths.few_shot_dir / dataset / f"{k_shot}shot"
        expected = split_dir / "split_0.pt"
        if expected.is_file():
            return split_dir
        return self.split_module.generate_splits_for_dataset(
            dataset=dataset,
            k_shot=k_shot,
            n_splits=self.config.num_splits,
            n_way=0,
            num_val=self.config.num_val,
            seed=self.config.seed,
            write_example=False,
        )

    def run(self):
        if self.config.task_type != "node_cls":
            raise ValueError(f"Unified SA2GFM adapter currently supports only task_type=node_cls, got {self.config.task_type}")

        dataset = normalize_dataset_name(self.config.data_name)
        manifest = self._load_manifest()
        source_datasets = [normalize_dataset_name(name) for name in manifest.get("source_datasets", [])]
        expert_map = manifest.get("experts", {})
        expert_paths = [expert_map[name] for name in source_datasets if name in expert_map]
        if not expert_paths:
            raise ValueError("No expert paths were found in the provided pretrain manifest/checkpoint.")

        if self.config.auto_prepare:
            self._ensure_communities(dataset)
            self._ensure_enhanced_graph(dataset)
        self._ensure_split_dir(dataset, self.config.k_shot)

        args = Namespace(
            dataset=dataset,
            seed=self.config.seed,
            shot_num=self.config.k_shot,
            source_datasets=source_datasets,
            expert_paths=expert_paths,
            unify_dim=self.config.unify_dim,
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            gamma=self.config.gamma,
            tau=self.config.tau,
            lambda_var=self.config.lambda_var,
            alpha=self.config.alpha,
            gcn_layers=self.config.gcn_layers,
            dropout=self.config.dropout,
            out_channels=self.config.out_channels,
            hid_units=self.config.hid_units,
            lr=self.config.adapt_lr,
            epochs=self.config.adapt_epochs,
            split_id=self.config.split_id,
            moe_weight=self.config.moe_weight,
            structure_weight=self.config.structure_weight,
            bucket_boundaries=self.config.bucket_boundaries,
            inter_cluster_optimizer=self.config.inter_cluster_optimizer,
            appnp_alpha=self.config.appnp_alpha,
            appnp_k=self.config.appnp_k,
            inter_cluster_threshold=self.config.inter_cluster_threshold,
            inter_cluster_temperature=self.config.inter_cluster_temperature,
            moe_embedding_weight=self.config.moe_embedding_weight,
            multi_embedding_weight=self.config.multi_embedding_weight,
            attack_type=self.config.attack_type,
            attack_ratio=self.config.attack_ratio,
            p=self.config.p,
            random_attack_type=self.config.random_attack_type,
            gpu=self.config.gpu,
            num_splits=self.config.num_splits,
            no_swanlab=self.config.no_swanlab,
            pre_train_model_dir_single=str(paths.save_model_dir),
            pre_train_model_dir_many=str(self.data_root / "save_model_many"),
            communitys_dir=str(paths.communities_dir),
            community_file=str(paths.communities_dir / f"{dataset}_communities.pt"),
            down_data_dir=str(paths.few_shot_dir / dataset / f"{self.config.k_shot}shot"),
            txt_features=None,
            num_nodes=0,
            num_classes=0,
            data_path=str(canonical_graph_path(dataset)),
        )
        return self.downstream_module.run_downstream(args)
