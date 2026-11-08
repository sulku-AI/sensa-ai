"""Araçların uçtan uca duman testi: gerçek ağırlıklar yerine rastgele küçük modeller.

Amaç: Emirhan M5'te ilk kez çalıştırdığında arayüz hatası çıkmasın. Başarım ölçülmez.
Pozitif veri yerine sentetik kırmızı kareler kullanılır (kansız).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from sensa_ai import models, tracking  # noqa: E402

from . import _tiny  # noqa: E402
from .conftest import red_square_frame  # noqa: E402


@pytest.fixture
def patched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gd_model, gd_proc = _tiny.tiny_grounding(tmp_path)
    tr_model, tr_proc = _tiny.tiny_video_tracker("edgetam")

    def det_fp(model_id: str | None = None, device: str | None = None) -> models.TextDetector:
        return models.TextDetector(gd_model, gd_proc, "cpu", 0.0, 0.0)

    def gate_fp(model_id: str | None = None, device: str | None = None) -> models.SiglipGate:
        return models.SiglipGate(_tiny.StubGateModel([3.0, 0.0]), _tiny.stub_processor, "cpu")

    def tr_fp(model_id: str | None = None, device: str | None = None) -> tracking.Tracker:
        return tracking.Tracker(tr_model, tr_proc, "cpu", "tiny-edgetam")

    monkeypatch.setattr(models.TextDetector, "from_pretrained", staticmethod(det_fp))
    monkeypatch.setattr(models.SiglipGate, "from_pretrained", staticmethod(gate_fp))
    monkeypatch.setattr(tracking.Tracker, "from_pretrained", staticmethod(tr_fp))

    def no_checkpoint(path: str, device: str | None = None, fp16: bool | None = None) -> models.SegDetector:
        raise FileNotFoundError(path)  # Mac'teki gerçek eğitilmiş model teste karışmasın: küçük modele düş

    monkeypatch.setattr(models.SegDetector, "from_checkpoint", staticmethod(no_checkpoint))
    monkeypatch.setattr(models.dev, "pick", lambda prefer=None: prefer or "cpu")  # MPS/CUDA olan makinede de cpu
    return tmp_path


def _coco_folder(root: Path, n: int = 3) -> Path:
    """Roboflow COCO-seg düzeninde sentetik (kansız) veri: kırmızı kare = 'blood' sınıfı yerine."""
    d = root / "pozitif" / "sentetik" / "train"
    d.mkdir(parents=True)
    coco: dict = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "blood"}]}
    for i in range(n):
        name = f"{i}.jpg"
        cv2.imwrite(str(d / name), red_square_frame(x=20 + 10 * i))
        coco["images"].append({"id": i + 1, "file_name": name, "width": 160, "height": 120})
        x = 20 + 10 * i
        coco["annotations"].append(
            {
                "id": i + 1,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [x, 40, 30, 30],
                "area": 900,
                "iscrowd": 0,
                "segmentation": [[x, 40, x + 30, 40, x + 30, 70, x, 70]],
            }
        )
    (d / "_annotations.coco.json").write_text(json.dumps(coco))
    return d.parent


def test_bench_runs(patched: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import bench

    monkeypatch.setattr(bench, "ROOT", patched)
    monkeypatch.setattr(sys, "argv", ["bench", "--frames", "4", "--res", "96"])
    bench.main()
    res = json.loads(next((patched / "experiments" / "results").glob("*_bench_cpu.json")).read_text())
    assert res["realtime_factor"] > 0 and "takip_ms_kare" in res


def test_evaluate_runs(patched: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools import evaluate

    pos = _coco_folder(patched)
    neg = patched / "negatif"
    neg.mkdir()
    cv2.imwrite(str(neg / "n.jpg"), np.full((120, 160, 3), 128, np.uint8))
    monkeypatch.setattr(evaluate, "ROOT", patched)
    monkeypatch.setattr(evaluate, "RESULTS", patched / "experiments" / "results")
    monkeypatch.setattr(sys, "argv", ["ev", "--pos", str(pos), "--neg", str(neg), "--previews"])
    evaluate.main()
    res = json.loads(next((patched / "experiments" / "results").glob("*_eval_*.json")).read_text())
    assert res["pozitif_goruntu"] == 3 and res["negatif_goruntu"] == 1


def test_autolabel_and_prepare(patched: Path, monkeypatch: pytest.MonkeyPatch, proxy_video: Path) -> None:
    from sensa_ai import safety
    from tools import autolabel, train_rfdetr

    data = patched / "data"
    monkeypatch.setattr(safety, "DATA_ROOT", data)
    monkeypatch.setattr(autolabel, "ROOT", patched)
    vids = data / "pozitif" / "videolar"
    vids.mkdir(parents=True)
    (vids / "v.mp4").write_bytes(proxy_video.read_bytes())
    out = data / "pozitif" / "autolabel"
    monkeypatch.setattr(sys, "argv", ["al", str(vids), "--out", str(out), "--fps", "2", "--min-score", "0"])
    original = safety.write_dataset_frame
    monkeypatch.setattr(safety, "write_dataset_frame", lambda p, f, data_root=data: original(p, f, data_root))
    autolabel.main()
    coco = json.loads((out / "_annotations.coco.json").read_text())
    assert coco["images"] and (out / "v.blurtrack.json").exists()

    monkeypatch.setattr(train_rfdetr, "OUT", data / "pozitif" / "egitim")
    monkeypatch.setattr(sys, "argv", ["tr", "prepare", "--src", str(out)])
    train_rfdetr.main()
    splits = {p.name for p in (data / "pozitif" / "egitim").iterdir()}
    assert splits == {"train", "valid", "test"}


def test_pipeline_cli_with_models(patched: Path, monkeypatch: pytest.MonkeyPatch, proxy_video: Path) -> None:
    from sensa_ai import pipeline

    pipeline.main(["analyze", str(proxy_video), "--prompts", "proxy"])
    assert proxy_video.with_suffix(".blurtrack.json").exists()


def test_autolabel_negative_mode_writes_unlabeled_frames(
    patched: Path, monkeypatch: pytest.MonkeyPatch, proxy_video: Path
) -> None:
    from sensa_ai import safety
    from tools import autolabel

    data = patched / "data"
    monkeypatch.setattr(safety, "DATA_ROOT", data)
    monkeypatch.setattr(autolabel, "ROOT", patched)
    vids = data / "kansiz_videolar"
    vids.mkdir(parents=True)
    (vids / "n.mp4").write_bytes(proxy_video.read_bytes())
    out = data / "pozitif" / "kansiz_zor"  # ham kare yalnız kilitli pozitif altına
    argv = ["al", str(vids), "--out", str(out), "--fps", "2", "--min-score", "0", "--negative", "--keep-empty", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    original = safety.write_dataset_frame
    monkeypatch.setattr(safety, "write_dataset_frame", lambda p, f, data_root=data: original(p, f, data_root))
    autolabel.main()
    coco = json.loads((out / "_annotations.coco.json").read_text())
    assert coco["images"] and not coco["annotations"]  # kansız video: etiket yok
    assert all("time_s" in im for im in coco["images"])
    res = json.loads(next((patched / "experiments" / "results").glob("*_autolabel.json")).read_text())
    assert res["mod"] == "negatif" and "n.mp4" in res["dakika_basina_isaretli"]


def test_flag_timeline_cli_compares_detectors(
    patched: Path, monkeypatch: pytest.MonkeyPatch, proxy_video: Path
) -> None:
    from tools import flag_timeline

    res = patched / "sonuc"
    argv = ["ft", str(proxy_video), "--detector", "grounding_dino_tiny", "--fps", "2", "--min-score", "0"]
    argv += ["--judge", "siglip", "--judge", "grounding_dino_tiny"]
    monkeypatch.setattr(sys, "argv", argv + ["--ranges", "0:01-0:03", "--results", str(res)])
    flag_timeline.main()
    rows = json.loads(next(res.glob("*_isaret_vekil_e0_hakem.json")).read_text())
    assert "disi_dakikada_yanlis_alarm" in next(iter(rows.values()))
    assert "hakem:siglip" in rows and "aday_kan_dakikalari" in rows["ortak"]
    assert "belirsiz_dakikalar" in rows["ortak"]
