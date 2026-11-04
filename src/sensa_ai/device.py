"""Cihaz seçimi (D-012): cuda -> mps -> cpu."""

from __future__ import annotations

from typing import Any


def pick(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for(device: str) -> Any:
    """2060 (Turing) FP16 destekler; MPS'te FP16 bazı işlemlerde sorunlu, FP32 kalır."""
    import torch

    return torch.float16 if device == "cuda" else torch.float32
