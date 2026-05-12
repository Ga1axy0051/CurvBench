from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "datasets_to_run.md"

PROJECT_NAME_MAP = {
    "hgcn": "hgcn",
    "hnn": "hgcn",
    "hgcn/hnn": "hgcn",
    "qgcn": "QGCN-main",
    "graphmore": "GraphMoRE-main",
}

DATASET_ALIASES = {
    "airport": "airport",
    "carcinogenesis_data": "carcinogenesis_data",
    "citeseer": "citeseer",
    "cora": "cora",
    "f1": "f1_ultimate_hetero_graph",
    "f1_ultimate_hetero_graph": "f1_ultimate_hetero_graph",
    "hepatitis_std_data": "hepatitis_std_data",
    "hockey_data": "hockey_data",
    "pte": "pte",
    "pubmed": "pubmed",
    "toxicology_data": "toxicology_data",
}

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def _normalize_dataset_token(raw: str) -> str | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"^[\-\*\s`]+", "", cleaned)
    cleaned = re.sub(r"[\s`:：]+$", "", cleaned)
    cleaned = cleaned.strip("`").strip()
    if not cleaned:
        return None
    return DATASET_ALIASES.get(cleaned.lower())


def _normalize_project_name(raw: str) -> str | None:
    cleaned = raw.strip()
    cleaned = re.sub(r"^[\-\*\s`]+", "", cleaned)
    cleaned = re.sub(r"[\s`:：]+$", "", cleaned)
    cleaned = cleaned.strip("`").strip().lower()
    return PROJECT_NAME_MAP.get(cleaned)


@lru_cache(maxsize=1)
def load_datasets_to_run() -> dict[str, set[str]]:
    scopes = {
        "global": set(),
        "hgcn": set(),
        "QGCN-main": set(),
        "GraphMoRE-main": set(),
    }
    if not CONFIG_PATH.exists():
        return scopes

    in_project_section = False
    for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "其他" in line and "跑" in line:
            in_project_section = True
            continue

        if not in_project_section:
            dataset = _normalize_dataset_token(line)
            if dataset is not None:
                scopes["global"].add(dataset)
            continue

        if ":" not in line and "：" not in line:
            continue
        parts = re.split(r"[:：]", line, maxsplit=1)
        if len(parts) != 2:
            continue
        project_name = _normalize_project_name(parts[0])
        if project_name is None:
            continue
        for token in TOKEN_PATTERN.findall(parts[1]):
            dataset = _normalize_dataset_token(token)
            if dataset is not None:
                scopes[project_name].add(dataset)

    return scopes


def get_compact_search_scope(project_name: str) -> set[str]:
    scopes = load_datasets_to_run()
    return set(scopes["global"]) | set(scopes.get(project_name, set()))


def is_compact_search_target(project_name: str, dataset: str) -> bool:
    normalized = _normalize_dataset_token(dataset)
    if normalized is None:
        normalized = dataset.strip().lower()
    return normalized in get_compact_search_scope(project_name)
