"""config/params.yaml okuyucu. Her değer {value, status, source} taşır."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "config" / "params.yaml"
PROMPTS_PATH = ROOT / "config" / "prompts.yaml"

STATUSES = {"measured", "datasheet", "literature", "sim", "estimate", "target"}


_RAW_CACHE: dict[Path, tuple[int, dict[str, dict[str, dict[str, Any]]]]] = {}


def load_raw(path: Path = PARAMS_PATH) -> dict[str, dict[str, dict[str, Any]]]:
    """YAML'ı dosya değişmedikçe bir kez ayrıştırır (her karede ~6 ms idi). Çağıran kopyayı değiştirebilir."""
    key = Path(path).resolve()
    mtime = key.stat().st_mtime_ns
    hit = _RAW_CACHE.get(key)
    if hit is None or hit[0] != mtime:
        with open(key, encoding="utf-8") as f:
            data: dict[str, dict[str, dict[str, Any]]] = yaml.safe_load(f)
        _RAW_CACHE[key] = hit = (mtime, data)
    return copy.deepcopy(hit[1])


_PROFILE: str | None = None


def set_profile(name: str | None) -> None:
    """Süreç boyunca kullanılacak hassasiyet profili (params.yaml -> profiles). None = varsayılan."""
    global _PROFILE
    if name is not None and name not in load_raw().get("profiles", {}):
        raise ValueError(f"bilinmeyen profil: {name}")
    _PROFILE = name


def load(path: Path = PARAMS_PATH, profile: str | None = None) -> dict[str, dict[str, Any]]:
    """Yalnız değerleri döndürür: cfg["blur"]["sigma_px"] -> 18.0. Profil seçiliyse üzerine yazar."""
    raw = load_raw(path)
    profiles = raw.pop("profiles", {})
    cfg = {sec: {k: v["value"] for k, v in items.items()} for sec, items in raw.items()}
    name = profile or _PROFILE
    for dotted, v in (profiles.get(name, {}) if name else {}).items():
        sec, key = dotted.split(".", 1)
        if key not in cfg.get(sec, {}):
            raise KeyError(f"profil {name}: {dotted} params.yaml'da yok")
        cfg[sec][key] = v["value"]
    return cfg


def load_prompts(path: Path = PROMPTS_PATH) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        data: dict[str, list[str]] = yaml.safe_load(f)
    return data
