"""Kırpım doğrulayıcı (Y3): sahte gömme ve sentetik renkli karelerle; model ve gerçek görüntü yok."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sensa_ai import verify
from sensa_ai.detect import Detection


def mean_color(crops: list[np.ndarray]) -> np.ndarray:
    return np.array([c.reshape(-1, 3).mean(0) for c in crops], np.float64)


def test_fit_separates_and_head_roundtrips(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    pos = rng.normal([1.0, 0.0, 0.2], 0.05, (40, 3))
    neg = rng.normal([0.0, 1.0, 0.2], 0.05, (200, 3))  # dengesiz sınıf
    head = verify.fit(np.vstack([pos, neg]), np.array([1] * 40 + [0] * 200))
    assert head.prob(pos).min() > 0.5 > head.prob(neg).max()
    head.save(tmp_path / "v" / "head.npz")
    again = verify.Head.load(tmp_path / "v" / "head.npz")
    assert np.allclose(again.prob(pos), head.prob(pos))


def test_verified_detector_drops_rejected_regions_and_delegates() -> None:
    class Det:
        box_threshold = 0.3

        def __call__(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
            return [Detection("a", (0.0, 0.0, 0.5, 1.0), 0.9), Detection("b", (0.5, 0.0, 1.0, 1.0), 0.9)]

    frame = np.zeros((40, 80, 3), np.uint8)
    frame[:, :40] = (0, 0, 200)  # sol yarı "kan" rengi (sentetik)
    frame[:, 40:] = (0, 200, 0)
    head = verify.fit(np.array([[0, 0, 200.0], [0, 200.0, 0]] * 10), np.array([1, 0] * 10))
    det = Det()
    vd = verify.VerifiedDetector(det, mean_color, head, 0.5, 0.0)
    assert [d.label for d in vd(frame, [])] == ["a"]
    vd.box_threshold = 0.1  # flag_timeline eşiği sarmalı üzerinden ayarlar
    assert det.box_threshold == 0.1 and vd.box_threshold == 0.1
    assert verify.crop(frame, (0.25, 0.25, 0.75, 0.75), 0.5).shape[:2] == (40, 80)


def test_train_verifier_samples_holdout_and_report(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    from tools import train_verifier as tv

    folder = tmp_path / "dex1-1"
    folder.mkdir()
    images, cur_anns, orig_anns = [], [], []
    for i, (stem, color, kept) in enumerate(
        [("dex1-1", (0, 0, 200), True), ("dex1-1", (0, 200, 0), False), ("dex1-2", (0, 0, 200), True),
         ("dex1-2", (0, 200, 0), False)]
    ):  # fmt: skip
        name = f"{stem}_{i:07d}.jpg"
        cv2.imwrite(str(folder / name), np.full((32, 32, 3), color, np.uint8))
        images.append({"id": i, "file_name": name, "width": 32, "height": 32})
        ann = {"id": i, "image_id": i, "bbox": [4, 4, 16, 16], "score": 0.6}
        orig_anns.append(ann)
        if kept:
            cur_anns.append(ann)
    (folder / "_annotations.coco.json").write_text(json.dumps({"images": images, "annotations": cur_anns}))
    (folder / "_annotations.coco.orig.json").write_text(json.dumps({"images": images, "annotations": orig_anns}))

    samples = tv.autolabel_samples(folder)
    assert [(s[2], s[3]) for s in samples] == [(1, "dex1-1"), (0, "dex1-1"), (1, "dex1-2"), (0, "dex1-2")]
    hold = [tv.is_holdout(s, ["dex1-2"]) for s in samples]
    assert hold == [False, False, True, True]
    x = tv.embed_samples(samples, mean_color, 0.0)
    y = np.array([s[2] for s in samples])
    head = verify.fit(x[:2], y[:2])
    row = tv.report(head, x[2:], samples[2:], [0.5])[0]
    assert row["dizi_pozitif_gecis"] == 1.0 and row["dizi_negatif_gecis"] == 0.0 and row["foto_pozitif_n"] == 0
