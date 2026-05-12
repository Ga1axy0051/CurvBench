from __future__ import annotations

import importlib.util
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_path(path_str: str, base_dir: Path) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML config {path}, got {type(data)}")
    return data


def save_yaml(data: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file_obj:
        yaml.safe_dump(data, file_obj, sort_keys=False)
    return path


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(config.__dict__)
    raise TypeError(f"Unsupported config type: {type(config)}")


def print_final_config(config: Any) -> None:
    print("Final Configuration:")
    for key, value in config_to_dict(config).items():
        print(f"  {key}: {value}")


def infer_fold_name(config_load_path: str | None, source_datasets: list[str]) -> str:
    if config_load_path:
        return Path(config_load_path).stem
    if source_datasets:
        return "_".join(source_datasets)
    return "manual_run"


def set_data_root_env(data_root: Path) -> None:
    os.environ["SA2GFM_DATA_ROOT"] = str(data_root)


def load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
