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
import numpy as np

from sensa_ai import config, metrics, models, safety
from sensa_ai.detect import Detection

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
            yield ann_path.parent / img["file_name"], by_img.get(img["id"], []), img.get("source", folder.name)


def _fmt(v: object) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def pred_mask(shape: tuple[int, int], dets: list[Detection], dilate: int) -> np.ndarray:
    """Poligonu olan tespit (segmentasyon modeli) şekliyle, olmayan kutuyla maskeye girer."""
    h, w = shape
    m = metrics.boxes_to_mask(shape, [d.box for d in dets if not d.poly], dilate)
    polys = [[v * (w if i % 2 == 0 else h) for i, v in enumerate(p)] for d in dets if d.poly for p in d.poly]
    if polys:
        pm = metrics.polygons_to_mask(shape, polys)
        if dilate > 0:
            pm = cv2.dilate(pm.astype(np.uint8), np.ones((2 * dilate + 1,) * 2, np.uint8)).astype(bool)
        m |= pm
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=Path, action="append", default=[])
    ap.add_argument("--neg", type=Path, action="append", default=[])
    ap.add_argument(
        "--detector",
        default=None,
        help="models.yaml anahtarı, HF kimliği ya da rfdetr:<.pth>",
    )
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--box-threshold", type=float, default=None, help="detect.box_threshold yerine (deneme)")
    ap.add_argument("--gate-threshold", type=float, default=None, help="gate.threshold yerine (deneme)")
    ap.add_argument(
        "--empty-as-neg", action="store_true", help="--pos içindeki etiketsiz görüntüleri negatif say (egitim/test)"
    )
    ap.add_argument("--results", type=Path, default=None, help="sonuç klasörü (Colab: Drive)")
    ap.add_argument("--previews", action="store_true", help="yalnız negatifler için güvenli küçük resim")
    args = ap.parse_args()

    prompts = config.load_prompts()
    dilate = int(config.load()["blur"]["dilate_px"])
    det = models.make_detector(args.detector, device=args.device)
    if args.box_threshold is not None:
        det.box_threshold = det.text_threshold = args.box_threshold
    gate = None if args.no_gate else models.SiglipGate.from_pretrained(device=args.device)
    gthr = args.gate_threshold if args.gate_threshold is not None else config.load()["gate"]["threshold"]
    gd = models.GatedDetector(gate, det, prompts["negative"], gthr)
    score, t0, n = metrics.Score(), time.perf_counter(), 0
    neg_seen = neg_gate_pass = pos_seen = pos_gate_pass = 0
    by_src: dict[str, metrics.Score] = {}

    for folder in args.pos:
        for i, (path, polys, src) in enumerate(iter_coco(folder)):
            if args.limit and i >= args.limit:
                break
            if not polys and not args.empty_as_neg:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            dets = gd(img, prompts["positive"])
            if not polys:  # etiketsiz görüntü (train_rfdetr negatifleri): kansız sayılır
                neg_seen += 1
                neg_gate_pass += gate is None or (gd.last_gate_score or 0.0) >= gd.threshold
                m = pred_mask(img.shape[:2], dets, dilate)
                score.add_negative(m)
                by_src.setdefault(src, metrics.Score()).add_negative(m)
                n += 1
                continue
            pos_seen += 1
            pos_gate_pass += gate is None or (gd.last_gate_score or 0.0) >= gd.threshold
            gt, m = metrics.polygons_to_mask(img.shape[:2], polys), pred_mask(img.shape[:2], dets, dilate)
            score.add_positive(gt, m)
            by_src.setdefault(src, metrics.Score()).add_positive(gt, m)
            n += 1
    for folder in args.neg:
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in files[: args.limit or None]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            dets = gd(img, prompts["positive"])
            neg_seen += 1
            neg_gate_pass += gate is None or (gd.last_gate_score or 0.0) >= gd.threshold
            m = pred_mask(img.shape[:2], dets, dilate)
            score.add_negative(m)
            by_src.setdefault(folder.name, metrics.Score()).add_negative(m)
            if args.previews and dets:
                safety.write_preview(ROOT / "previews" / "negatif" / f"{path.stem}.png", img)
            n += 1
            if n % 25 == 0:
                print(f"  {n} görüntü, {time.perf_counter() - t0:.0f} sn", flush=True)

    out = score.summary()
    if pos_seen:
        out["pozitif_kapi_gecis_orani"] = pos_gate_pass / pos_seen  # düşükse kaçırmanın sebebi kapı
    out["kutu_esigi"] = det.box_threshold
    out["kapi_esigi"] = gd.threshold if gate is not None else "yok"
    if neg_seen:
        out["negatif_kapi_gecis_orani"] = neg_gate_pass / neg_seen  # tools.bench --gate-pass için (H-03)
    out["saniye_goruntu_basi"] = (time.perf_counter() - t0) / max(n, 1)
    out["dedektor"] = models.detector_name(det)
    out["kapi"] = "yok" if gate is None else gate.model.name_or_path
    out["cihaz"] = det.device
    if len(by_src) > 1:
        out["kaynaklar"] = {k: v.summary() for k, v in sorted(by_src.items())}
    res_dir = args.results or RESULTS
    res_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today()}_eval_{Path(str(out['dedektor'])).stem}_k{det.box_threshold:g}"
    (res_dir / f"{stem}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    lines = [f"# Değerlendirme {dt.date.today()}", "", "| ölçüt | değer |", "|---|---|"]
    lines += [f"| {k} | {_fmt(v)} |" for k, v in out.items() if k != "kaynaklar"]
    if len(by_src) > 1:
        cols = ["pozitif_goruntu", "yakalama_orani", "yakalama_alt_95", "negatif_goruntu", "negatif_yanlis_alarm_orani"]
        lines += ["", "Kaynağa göre:", "", "| kaynak | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for name, sc in sorted(by_src.items()):
            sm = sc.summary()
            lines.append(f"| {name} | " + " | ".join(_fmt(sm[c]) for c in cols) + " |")
    (res_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
