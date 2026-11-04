"""config/params.yaml okuyucu. Her değer {value, status, source} taşır."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "config" / "params.yaml"
PROMPTS_PATH = ROOT / "config" / "prompts.yaml"

STATUSES = {"measured", "datasheet", "literature", "sim", "estimate", "target"}


def load_raw(path: Path = PARAMS_PATH) -> dict[str, dict[str, dict[str, Any]]]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, dict[str, dict[str, Any]]] = yaml.safe_load(f)
    return data


def load(path: Path = PARAMS_PATH) -> dict[str, dict[str, Any]]:
    """Yalnız değerleri döndürür: cfg["blur"]["sigma_px"] -> 18.0"""
    return {sec: {k: v["value"] for k, v in items.items()} for sec, items in load_raw(path).items()}


def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, list[str]] = yaml.safe_load(f)
    return data
