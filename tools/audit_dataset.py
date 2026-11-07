"""Veri seti denetimi, görüntüye bakmadan (D-001): sayılar, kopyalar, şüpheli etiketler, silüet.

    python -m tools.audit_dataset data/pozitif/blood_seg_joost data/pozitif/blood_pure_blur
    python -m tools.audit_dataset data/pozitif/egitim/valid --detector rfdetr_seg --worst 30 --silhouettes

Çıktı: experiments/results/<tarih>_denetim.md (yalnız sayı ve nötr dosya adları).
  * etiket: boş görüntü, kutu yalnız (poligonsuz), bozuk poligon, çok küçük / kareyi kaplayan maske
  * kopya: aynı görüntünün tekrarları (8x8 fark özeti, piksel okunur ama gösterilmez)
  * `--detector`: modelin etiketle en çok çeliştiği görüntüler (kötü etiket adayları)
  * `--silhouettes`: yalnız etiket maskesinin beyaz şekli, siyah zeminde, `safety.write_preview` ile
    (fotoğrafın kendisi hiç yazılmaz). previews/siluet/ altına.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from sensa_ai import metrics, safety

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
TINY, HUGE = 0.001, 0.8  # maske alanı / görüntü alanı


def dhash(path: Path) -> int | None:
    im = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if im is None:
        return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def records(folder: Path):  # type: ignore[no-untyped-def]
    """(yol, w, h, poligonlar, yalnız_kutu_sayısı, kaynak) — her COCO görüntüsü için."""
    for ann_path in sorted(folder.rglob("_annotations.coco.json")):
        d = json.loads(ann_path.read_text())
        polys: dict[int, list[list[float]]] = {}
        box_only: Counter[int] = Counter()
        for a in d["annotations"]:
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                polys.setdefault(a["image_id"], []).extend(seg)
            else:
                box_only[a["image_id"]] += 1
                x, y, w, h = a.get("bbox", (0, 0, 0, 0))
                polys.setdefault(a["image_id"], []).append([x, y, x + w, y, x + w, y + h, x, y + h])
        for img in d["images"]:
            yield (
                ann_path.parent / img["file_name"],
                int(img["width"]),
                int(img["height"]),
                polys.get(img["id"], []),
                box_only[img["id"]],
                img.get("source", folder.name),
            )


def audit(folders: list[Path], limit: int = 0) -> tuple[dict, list[tuple]]:
    rows = []
    for f in folders:
        for i, r in enumerate(records(f)):
            if limit and i >= limit:
                break
            rows.append(r)
    areas: list[float] = []
    sizes: Counter[str] = Counter()
    stats: Counter[str] = Counter()
    hashes: dict[int, list[Path]] = {}
    for path, w, h, polys, box_only, _src in rows:
        stats["goruntu"] += 1
        sizes[f"{w}x{h}"] += 1
        if not polys:
            stats["bos_etiket"] += 1
        if box_only:
            stats["yalniz_kutu_bolge"] += box_only
        bad = sum(1 for p in polys if len(p) < 6)
        stats["bozuk_poligon"] += bad
        if polys:
            a = float(metrics.polygons_to_mask((h, w), polys).mean())
            areas.append(a)
            stats["cok_kucuk_maske"] += a < TINY
            stats["kareyi_kaplayan_maske"] += a > HUGE
        if not path.exists():
            stats["eksik_dosya"] += 1
            continue
        hv = dhash(path)
        if hv is not None:
            hashes.setdefault(hv, []).append(path)
    dup_groups = [v for v in hashes.values() if len(v) > 1]
    stats["kopya_grubu"] = len(dup_groups)
    stats["kopya_fazlasi"] = sum(len(v) - 1 for v in dup_groups)
    a = np.asarray(areas) if areas else np.zeros(1)
    out: dict = dict(stats)
    out.update(
        {
            "maske_alani_p5": float(np.percentile(a, 5)),
            "maske_alani_medyan": float(np.median(a)),
            "maske_alani_p95": float(np.percentile(a, 95)),
            "en_sik_boyutlar": ", ".join(f"{k} ({v})" for k, v in sizes.most_common(3)),
            "kaynaklar": dict(Counter(r[5] for r in rows)),
        }
    )
    return out, rows


def silhouette(w: int, h: int, polys: list[list[float]]) -> np.ndarray:
    """Yalnız etiket şekli: siyah zemin, beyaz bölge. Fotoğraftan hiçbir piksel yok."""
    m = metrics.polygons_to_mask((h, w), polys).astype(np.uint8) * 255
    return np.dstack([m] * 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="klasör başına en çok görüntü")
    ap.add_argument("--detector", default=None, help="çelişki listesi için (ör. rfdetr_seg)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--silhouettes", action="store_true")
    ap.add_argument("--list", type=Path, default=None, help="çelişki sırasını JSON'a yaz (tools.safe_viewer için)")
    args = ap.parse_args()

    out, rows = audit(args.folders, args.limit)
    lines = [f"# Veri denetimi {dt.date.today()}", "", "Klasörler: " + ", ".join(f.name for f in args.folders), ""]
    lines += ["| ölçüt | değer |", "|---|---|"]
    lines += [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in out.items()]

    worst: list[tuple[float, Path, int, int, list]] = []
    if args.detector:
        from sensa_ai import models
        from tools.evaluate import pred_mask

        det = models.make_detector(args.detector, device=args.device)
        for path, w, h, polys, _b, _s in rows:
            img = cv2.imread(str(path))
            if img is None:
                continue
            gt = metrics.polygons_to_mask((h, w), polys) if polys else np.zeros((h, w), bool)
            pm = pred_mask(img.shape[:2], det(img, []), 0)
            if gt.shape != pm.shape:
                pm = cv2.resize(pm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            union = (gt | pm).sum()
            iou = float((gt & pm).sum() / union) if union else 1.0
            worst.append((iou, path, w, h, polys))
        worst.sort(key=lambda r: r[0])
        lines += ["", f"## Modelle en çok çelişen {args.worst} görüntü (düşük IoU: kötü etiket ya da model hatası)", ""]
        lines += ["| sıra | dosya | IoU | etiketli mi |", "|---|---|---|---|"]
        for k, (iou, path, *_rest, polys) in enumerate(worst[: args.worst], 1):
            lines.append(f"| {k} | {path.parent.name}/{path.name} | {iou:.2f} | {'evet' if polys else 'boş'} |")
        ious = np.asarray([r[0] for r in worst])
        lines += ["", f"IoU medyan {np.median(ious):.2f}; IoU < 0.1 olan {int((ious < 0.1).sum())} / {len(ious)}"]

    if args.list is not None:
        src = worst if worst else [(0.0, p, w, h, pl) for p, w, h, pl, _b, _s in rows]
        items = [{"path": str(r[1]), "iou": r[0], "polys": r[4]} for r in src[: args.worst]]
        args.list.write_text(json.dumps(items))

    if args.silhouettes:
        pick = worst[: args.worst] if worst else [(0.0, p, w, h, pl) for p, w, h, pl, _b, _s in rows[: args.worst]]
        for k, (_iou, path, w, h, polys) in enumerate(pick, 1):
            if polys:
                safety.write_preview(ROOT / "previews" / "siluet" / f"{k:03d}_{path.stem}.png", silhouette(w, h, polys))
        lines += ["", "Silüetler: previews/siluet/ (yalnız etiket şekli, fotoğraf yok)."]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{dt.date.today()}_denetim.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
