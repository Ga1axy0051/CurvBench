"""
Central path configuration for the publication attack-data pipeline.

Override with environment variable:
  SA2GFM_DATA_ROOT  — directory that contains `ori/`, `communities/`, `few_shot/`, `save_model/`, …
                      (default: sibling `../data` when `SA2GFM` lives under the main project, else `SA2GFM/data`).
"""

from __future__ import annotations

import os
from pathlib import Path


class Paths:
    """Resolved once at import; safe to read from any script."""

    def __init__(self) -> None:
        # SA2GFM/attack_data_gen/lib/paths.py -> attack_data_gen = parents[1]
        self.attack_gen_root: Path = Path(__file__).resolve().parents[1]
        self.sa2gfm_root: Path = self.attack_gen_root.parent

        env_root = os.environ.get("SA2GFM_DATA_ROOT", "").strip()
        if env_root:
            self.data_root = Path(env_root).expanduser().resolve()
        else:
            sibling = self.sa2gfm_root.parent / "data"
            if (sibling / "ori").is_dir():
                self.data_root = sibling.resolve()
            else:
                self.data_root = (self.sa2gfm_root / "data").resolve()

        self.graph_ori_dir = self.data_root / "ori"
        # Downstream expects: {data_root}/communities/{dataset}_communities.pt
        self.communities_dir = self.data_root / "communities"
        # Single-dataset MoE experts (downstream loads from here)
        self.save_model_dir = self.data_root / "save_model"
        # Few-shot splits: few_shot/{dataset}/{k}shot/split_{i}.pt
        self.few_shot_dir = self.data_root / "few_shot"
        self.checkpoints_dir = self.attack_gen_root / "checkpoints"
        self.output_root = self.attack_gen_root / "outputs"
        self.attack_post_dir = self.output_root / "attack_post"
        self.attack_random_dir = self.output_root / "attacked_data_random"
        self.surrogate_deeprobust_dir = self.output_root / "surrogate_deeprobust"
        self.metattack_batch_dir = self.output_root / "metattack_batch"

    def ensure_output_dirs(self) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.attack_post_dir.mkdir(parents=True, exist_ok=True)
        self.attack_random_dir.mkdir(parents=True, exist_ok=True)
        self.surrogate_deeprobust_dir.mkdir(parents=True, exist_ok=True)
        self.metattack_batch_dir.mkdir(parents=True, exist_ok=True)


paths = Paths()
