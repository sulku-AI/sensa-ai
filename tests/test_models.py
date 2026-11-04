"""Model sarmalayıcıları: rastgele ağırlıklı küçük yapılandırmalarla, İNDİRMESİZ.

Bu testler gerçek başarımı ölçmez (ağırlıklar rastgele); yalnız arayüzün doğru çağrıldığını,
kutuların normalize olduğunu ve takip maskelerinin boyutunu sınar. torch yoksa atlanır.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sensa_ai import models, tracking  # noqa: E402

from . import _tiny  # noqa: E402

FRAME = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
PROMPTS = ["red balloon", "blood"]


def _check(dets: list) -> None:
    assert dets, "en az bir tespit beklenir (eşik 0)"
    for d in dets:
        assert all(0.0 <= v <= 1.0 for v in d.box) and d.box[0] <= d.box[2] and d.box[1] <= d.box[3]
    assert len(dets) <= 8  # params detect.max_regions


def test_grounding_family(tmp_path: Path) -> None:
    model, proc = _tiny.tiny_grounding(tmp_path)
    det = models.TextDetector(model, proc, "cpu", 0.0, 0.0)
    assert det.family == "grounding"
    _check(det(FRAME, PROMPTS))


def test_owlv2_family() -> None:
    model, proc = _tiny.tiny_owlv2()
    det = models.TextDetector(model, proc, "cpu", 0.0, 0.0)
    assert det.family == "owlv2"
    _check(det(FRAME, PROMPTS))


class _StubGate:
    """SigLIP yerine: logits_per_image döndüren sahte model + işlemci (softmax mantığını sınar)."""

    def __init__(self, logits: list[float]) -> None:
        self.dtype = torch.float32
        self._logits = torch.tensor([logits])

    def __call__(self, **_: object) -> object:
        return type("O", (), {"logits_per_image": self._logits})()


def test_gate_softmax_over_positive_and_negative() -> None:
    def proc(**_: object) -> dict:
        return {"input_ids": torch.zeros(1, 1)}

    gate = models.SiglipGate(_StubGate([2.0, 2.0, 0.0]), proc, "cpu")
    s = gate.score(FRAME, ["blood", "wound"], ["ketchup"])
    e = np.exp([2.0, 2.0, 0.0])
    assert s == pytest.approx((e[0] + e[1]) / e.sum(), rel=1e-5)


def test_gated_detector_skips_when_gate_low() -> None:
    calls = []

    class Det:
        def __call__(self, f: np.ndarray, p: list[str]) -> list:
            calls.append(1)
            return []

    gd = models.GatedDetector(models.SiglipGate(_StubGate([-5.0, 5.0]), lambda **_: {}, "cpu"), Det(), ["x"], 0.3)  # type: ignore[arg-type]
    gd(FRAME, ["blood"])
    assert calls == [] and gd.last_gate_score is not None and gd.last_gate_score < 0.3


@pytest.mark.parametrize("family", ["sam2", "edgetam"])
def test_tracker_mask_shapes(family: str) -> None:
    model, proc = _tiny.tiny_video_tracker(family)
    tr = tracking.Tracker(model, proc, "cpu", family)
    frames = [np.random.default_rng(i).integers(0, 255, (64, 96, 3), dtype=np.uint8) for i in range(3)]
    masks = tr.track(frames, [(0.2, 0.2, 0.5, 0.5)], start_idx=1)
    assert len(masks) == 3 and all(m.shape == (64, 96) and m.dtype == bool for m in masks)
    assert not masks[0].any()  # başlangıç öncesi boş


def test_mask_to_box() -> None:
    m = np.zeros((10, 20), bool)
    m[2:5, 4:10] = True
    assert tracking.mask_to_box(m) == pytest.approx((0.2, 0.2, 0.5, 0.5))
    assert tracking.mask_to_box(np.zeros((4, 4), bool)) is None


def test_resolve_ids() -> None:
    assert models.resolve("detect") == "iSEE-Laboratory/llmdet_tiny"
    assert models.resolve("detect", "owlv2_base").startswith("google/owlv2")
    assert models.resolve("detect", "x/y") == "x/y"
