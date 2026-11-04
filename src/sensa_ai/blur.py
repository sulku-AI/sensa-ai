"""Bölgesel yumuşak bulanıklaştırma: yalnız maskenin içi, kenarları yumuşatılmış."""

from __future__ import annotations

import cv2
import numpy as np

from .blurtrack import Box, Region


def timing(b: dict, sample_fps: float | None = None) -> dict[str, float]:
    """params.yaml blur bölümünden active_regions zamanlaması. hold = bir örnekleme aralığı."""
    from . import config

    sfps = sample_fps or float(config.load()["lookahead"]["sample_fps"])
    hold = float(b.get("hold_factor", 1.0)) / sfps
    return {"ramp_in_s": b["ramp_in_s"], "ramp_out_s": b["ramp_out_s"], "hold_in_s": hold, "hold_out_s": hold}


def box_mask(shape: tuple[int, int], boxes: list[tuple[Box, float]], dilate_px: int, feather_px: int) -> np.ndarray:
    """Normalize kutulardan 0..1 float maske (güç ile ölçekli, kenarı yumuşak)."""
    h, w = shape
    mask: np.ndarray = np.zeros((h, w), np.float32)
    for (x0, y0, x1, y1), strength in boxes:
        a = (max(0, int(x0 * w) - dilate_px), max(0, int(y0 * h) - dilate_px))
        b = (min(w - 1, int(x1 * w) + dilate_px), min(h - 1, int(y1 * h) + dilate_px))
        layer = np.zeros_like(mask)
        cv2.rectangle(layer, a, b, float(np.clip(strength, 0.0, 1.0)), thickness=-1)
        mask = np.maximum(mask, layer).astype(np.float32)
    if feather_px > 0 and mask.any():
        k = feather_px * 2 + 1
        mask = np.maximum(mask, cv2.GaussianBlur(mask, (k, k), feather_px / 2))  # yalnız dışa
    return mask


def region_mask(shape: tuple[int, int], regions: list[Region], dilate_px: int, feather_px: int) -> np.ndarray:
    """Bölgelerden 0..1 maske: poligon varsa şekil, yoksa kutu. Genişletme + yumuşak kenar."""
    h, w = shape
    mask: np.ndarray = np.zeros((h, w), np.float32)
    boxes_only: list[tuple[Box, float]] = []
    for r in regions:
        extra = round(r.pad * max(w, h))
        if not r.poly:
            x0, y0, x1, y1 = r.box
            boxes_only.append(((x0 - r.pad, y0 - r.pad, x1 + r.pad, y1 + r.pad), r.strength))
            continue
        layer: np.ndarray = np.zeros((h, w), np.uint8)
        pts = [
            (np.asarray(p, np.float32).reshape(-1, 2) * (w, h)).round().astype(np.int32) for p in r.poly if len(p) >= 6
        ]
        cv2.fillPoly(layer, pts, 255)
        dl = dilate_px + extra
        if dl > 0:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dl * 2 + 1, dl * 2 + 1))
            layer = cv2.dilate(layer, se)
        mask = np.maximum(mask, layer.astype(np.float32) / 255 * float(np.clip(r.strength, 0.0, 1.0)))
    if boxes_only:
        mask = np.maximum(mask, box_mask(shape, boxes_only, dilate_px, 0))
    if feather_px > 0 and mask.any():
        # yumuşak kenar yalnız DIŞA doğru: genişletilmiş bölgenin içi tam güçte kalır (sızma yok, H-06)
        ks = feather_px * 2 + 1
        mask = np.maximum(mask, cv2.GaussianBlur(mask, (ks, ks), feather_px / 2))
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
