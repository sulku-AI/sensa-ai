"""Eğitim verisi hazırlığı ve eğitilmiş model değerlendirmesi: torch gerektirmez, sentetik veri."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sensa_ai import models


def test_split_keeps_augmented_copies_together(tmp_path: Path) -> None:
    """Roboflow artırma kopyaları ve aynı videonun kareleri farklı bölümlere düşmez (sızıntı yok)."""
    from tools import train_rfdetr as tr

    items = []
    for split in ("train", "valid", "test"):
        for g in range(30):
            base = f"{split}{g}_jpg"
            n = 3 if split == "train" else 1
            for k in range(n):
                p = tmp_path / "ds" / split / f"{base}.rf.{k:032x}.jpg"
                items.append((p, 10, 10, [], f"ds:{tr.group_key(p)}", split))
    for v in range(20):
        for f in range(5):
            p = tmp_path / "al" / "images" / f"vid{v}_{f * 10:07d}.jpg"
            items.append((p, 10, 10, [], f"al:{tr.group_key(p)}", None))
    out = tr.split_items(items, aug_per_image=2)
    where: dict[str, set[str]] = {}
    for split, rows in out.items():
        for r in rows:
            where.setdefault(r[4], set()).add(split)
    assert all(len(s) == 1 for s in where.values())
    assert len(where) == 90 + 20
    assert len(out["valid"]) >= 30 and len(out["test"]) >= 30
    assert sum(1 for r in out["train"] if r[4].startswith("ds:")) == 30 * 2  # grup başına en çok 2 kopya
    assert sum(1 for r in out["test"] if r[4].startswith("ds:")) == 30


def test_pred_mask_uses_polygon_when_present() -> None:
    from sensa_ai.detect import Detection
    from tools import evaluate

    tri = Detection("blood", (0.0, 0.0, 1.0, 1.0), 1.0, [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
    m = evaluate.pred_mask((100, 100), [tri], 0)
    assert 0.4 < m.mean() < 0.6  # kutu olsaydı 1.0
    box = Detection("x", (0.0, 0.0, 0.5, 0.5), 1.0)
    assert evaluate.pred_mask((100, 100), [box], 0).mean() == pytest.approx(0.25)


def test_seg_detector_returns_polygons() -> None:
    """RF-DETR çıktısı (supervision.Detections biçimi) kutu + maskeden poligona çevrilir."""
    from types import SimpleNamespace

    mask = np.zeros((1, 40, 80), bool)
    mask[0, 10:30, 20:60] = True

    class Fake:
        def predict(self, image: object, threshold: float) -> SimpleNamespace:
            return SimpleNamespace(xyxy=np.array([[20.0, 10.0, 60.0, 30.0]]), confidence=np.array([0.9]), mask=mask)

    det = models.SegDetector(Fake(), "cpu", 0.5, "x.pth")
    (d,) = det(np.zeros((40, 80, 3), np.uint8), ["ignored"])
    assert d.box == pytest.approx((0.25, 0.25, 0.75, 0.75))
    assert d.poly and len(d.poly[0]) >= 6


def test_rfdetr_version_kwargs() -> None:
    from tools import train_rfdetr as tr

    class New:
        model_fields = {"accelerator": 1, "save_dataset_grids": 1, "amp_dtype": 1}

    assert tr.version_kwargs(New, "cuda") == {"accelerator": "gpu", "save_dataset_grids": False}
    assert tr.version_kwargs(None, "mps") == {"device": "mps", "amp": False}


def test_make_detector_routes_checkpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: list[str] = []
    monkeypatch.setattr(models.SegDetector, "from_checkpoint", classmethod(lambda cls, p, device=None: seen.append(p)))
    models.make_detector("rfdetr_seg")
    models.make_detector(f"rfdetr:{tmp_path / 'a.pth'}")
    models.make_detector(str(tmp_path / "b.pth"))
    assert seen == [
        "models/rfdetr_seg_nano/checkpoint_best_total.pth",
        str(tmp_path / "a.pth"),
        str(tmp_path / "b.pth"),
    ]


def test_seg_checkpoint_missing_is_clear() -> None:
    with pytest.raises(FileNotFoundError, match="Drive"):
        models.SegDetector.from_checkpoint("models/yok.pth")


def test_evaluate_end_to_end_with_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Sentetik kareler: sol yarıda 'bölge' bulan sahte dedektör; kaynak dökümü ve güven aralığı yazılır."""
    import json
    import sys

    import cv2

    from sensa_ai.detect import Detection
    from tools import evaluate

    split = tmp_path / "egitim" / "test"
    split.mkdir(parents=True)
    imgs, anns = [], []
    for k in range(6):
        cv2.imwrite(str(split / f"{k}.png"), np.full((40, 40, 3), 128, np.uint8))
        imgs.append({"id": k, "file_name": f"{k}.png", "width": 40, "height": 40, "source": "a" if k < 3 else "b"})
        if k % 3:
            anns.append({"image_id": k, "bbox": [0, 0, 20, 40], "segmentation": [[0, 0, 20, 0, 20, 40, 0, 40]]})
    (split / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns}))

    class Fake:
        box_threshold = text_threshold = 0.3
        device = "cpu"
        name = "sahte.pth"

        def __call__(self, frame: np.ndarray, prompts: list[str]) -> list[Detection]:
            return [Detection("blood", (0.0, 0.0, 0.5, 1.0), 0.9)]

    monkeypatch.setattr(models, "make_detector", lambda *a, **k: Fake())
    monkeypatch.setattr(models, "detector_name", lambda d: d.name)
    res = tmp_path / "sonuc"
    argv = ["ev", "--pos", str(split), "--empty-as-neg", "--no-gate", "--results", str(res)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()
    out = json.loads(next(res.glob("*_eval_sahte_k0.3.json")).read_text())
    assert out["pozitif_goruntu"] == 4 and out["negatif_goruntu"] == 2
    assert out["yakalama_orani"] == 1.0 and out["negatif_yanlis_alarm_orani"] == 1.0
    assert set(out["kaynaklar"]) == {"a", "b"}
    assert out["yakalama_alt_95"] < 1.0
