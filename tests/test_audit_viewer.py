"""Veri denetimi ve kontrollü görüntüleyici: yalnız sentetik şekiller."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import audit_dataset, safe_viewer


def _coco(folder: Path) -> None:
    folder.mkdir(parents=True)
    a = np.zeros((40, 60, 3), np.uint8)
    a[10:30, 10:30] = (0, 0, 200)
    b = np.zeros((40, 60, 3), np.uint8)
    b[:, 30:] = 255
    for name, im in [("a.png", a), ("a2.png", a), ("b.png", b)]:
        cv2.imwrite(str(folder / name), im)
    imgs = [{"id": i, "file_name": n, "width": 60, "height": 40} for i, n in enumerate(["a.png", "a2.png", "b.png"])]
    imgs[1].update(width=600, height=400)  # minik etiket büyük karede: "çok küçük maske"
    anns = [
        {"image_id": 0, "bbox": [10, 10, 20, 20], "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]},
        {"image_id": 1, "bbox": [0, 0, 1, 1], "segmentation": [[0, 0, 1, 0, 1, 1]]},
    ]
    (folder / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns}))


def test_audit_counts(tmp_path: Path) -> None:
    _coco(tmp_path / "ds" / "train")
    out, rows = audit_dataset.audit([tmp_path / "ds"])
    assert out["goruntu"] == 3 and out["bos_etiket"] == 1
    assert out["kopya_fazlasi"] == 1  # a.png ile a2.png aynı
    assert out["cok_kucuk_maske"] == 1
    sil = audit_dataset.silhouette(60, 40, rows[0][3])
    assert set(np.unique(sil)) == {0, 255}  # yalnız şekil, fotoğraf pikseli yok


def test_viewer_hides_paths_and_serves_on_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _coco(tmp_path / "ds")
    lst = tmp_path / "l.json"
    lst.write_text(json.dumps([{"path": str(tmp_path / "ds" / "a.png"), "iou": 0.1, "polys": []}]))
    items = safe_viewer.load_items(lst)
    assert items[0]["w"] == 60 and items[0]["h"] == 40
    monkeypatch.setattr(safe_viewer, "MARKS", tmp_path / "isaret.json")
    srv = safe_viewer.serve(items, port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "a.png" not in page and str(tmp_path) not in page
        r = urllib.request.urlopen(base + "/img/0")
        assert r.headers["Cache-Control"].startswith("no-store")
        req = urllib.request.Request(base + "/mark", data=json.dumps({"i": 0, "mark": "kotu"}).encode(), method="POST")
        urllib.request.urlopen(req).read()
        assert list(json.loads((tmp_path / "isaret.json").read_text()).values()) == ["kotu"]
    finally:
        srv.shutdown()
