"""Bölgesel yumuşak bulanıklaştırma: yalnız maskenin içi, kenarları yumuşatılmış."""

from __future__ import annotations

import cv2
import numpy as np

from .blurtrack import Box


def box_mask(shape: tuple[int, int], boxes: list[tuple[Box, float]], dilate_px: int, feather_px: int) -> np.ndarray:
    """Normalize kutulardan 0..1 float maske (güç ile ölçekli, kenarı yumuşak)."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    for (x0, y0, x1, y1), strength in boxes:
        a = (max(0, int(x0 * w) - dilate_px), max(0, int(y0 * h) - dilate_px))
        b = (min(w - 1, int(x1 * w) + dilate_px), min(h - 1, int(y1 * h) + dilate_px))
        layer = np.zeros_like(mask)
        cv2.rectangle(layer, a, b, float(np.clip(strength, 0.0, 1.0)), thickness=-1)
        mask = np.maximum(mask, layer)
    if feather_px > 0 and mask.any():
        k = feather_px * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), feather_px / 2)
    return mask


def apply(frame: np.ndarray, mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """mask 0 olan yerde kare aynen kalır; 1 olan yerde tam bulanık."""
    if not mask.any():
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), sigma_px)
    m = mask[..., None]
    return (frame.astype(np.float32) * (1 - m) + blurred.astype(np.float32) * m).astype(frame.dtype)


def apply_full(frame: np.ndarray, sigma_px: float) -> np.ndarray:
    """Güvenli mod: tam kare hafif bulanıklık (D-007)."""
    return cv2.GaussianBlur(frame, (0, 0), sigma_px)
