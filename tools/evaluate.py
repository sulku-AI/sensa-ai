"""H-01 / H-02 değerlendirmesi: pozitif (COCO-seg) ve negatif klasörler üzerinde dedektör.

    python -m tools.evaluate --pos data/pozitif/blood_seg_joost --neg data/negatif/coco \
        --detector llmdet_tiny --limit 200

Çıktı yalnız sayı: experiments/results/<tarih>_eval_<dedektör>.md (+ .json).
Hiçbir pozitif görüntü gösterilmez ya da kaydedilmez. --previews yalnız NEGATİF görüntüler için
safe_preview küçük resmi yazar.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import cv2

from sensa_ai import config, metrics, models, safety

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"


def iter_coco(folder: Path):  # type: ignore[no-untyped-def]
    for ann_path in sorted(folder.rglob("_annotations.coco.json")):
        d = json.loads(ann_path.read_text())
        by_img: dict[int, list[list[float]]] = {}
        for a in d["annotations"]:
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                by_img.setdefault(a["image_id"], []).extend(seg)
            elif a.get("bbox"):
                x, y, w, h = a["bbox"]
                by_img.setdefault(a["image_id"], []).append([x, y, x + w, y, x + w, y + h, x, y + h])
        for img in d["images"]:
            yield ann_path.parent / img["file_name"], by_img.get(img["id"], [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, action="append", default=[])
    ap.add_argument("--neg", type=Path, action="append", default=[])
    ap.add_argument("--detector", default=None, help="config/models.yaml detect anahtarı ya da HF kimliği")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--previews", action="store_true", help="yalnız negatifler için güvenli küçük resim")
    args = ap.parse_args()

    prompts = config.load_prompts()
    dilate = int(config.load()["blur"]["dilate_px"])
    det = models.TextDetector.from_pretrained(models.resolve("detect", args.detector), device=args.device)
    gate = None if args.no_gate else models.SiglipGate.from_pretrained(device=args.device)
    gd = models.GatedDetector(gate, det, prompts["negative"], config.load()["gate"]["threshold"])
    score, t0, n = metrics.Score(), time.perf_counter(), 0

    for folder in args.pos:
        for i, (path, polys) in enumerate(iter_coco(folder)):
            if args.limit and i >= args.limit:
                break
            img = cv2.imread(str(path))
            if img is None or not polys:
                continue
            boxes = [d.box for d in gd(img, prompts["positive"])]
            score.add_positive(
                metrics.polygons_to_mask(img.shape[:2], polys), metrics.boxes_to_mask(img.shape[:2], boxes, dilate)
            )
            n += 1
    for folder in args.neg:
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in files[: args.limit or None]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            boxes = [d.box for d in gd(img, prompts["positive"])]
            score.add_negative(metrics.boxes_to_mask(img.shape[:2], boxes, dilate))
            if args.previews and boxes:
                safety.write_preview(ROOT / "previews" / "negatif" / f"{path.stem}.png", img)
            n += 1

    out = score.summary()
    out["saniye_goruntu_basi"] = (time.perf_counter() - t0) / max(n, 1)
    out["dedektor"] = det.model.name_or_path
    out["kapi"] = "yok" if gate is None else gate.model.name_or_path
    out["cihaz"] = det.device
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today()}_eval_{Path(str(out['dedektor'])).name}"
    (RESULTS / f"{stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lines = [f"# Değerlendirme {dt.date.today()}", "", "| ölçüt | değer |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in out.items()]
    (RESULTS / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
