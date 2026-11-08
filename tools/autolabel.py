"""Otomatik etiketleme: insan görmeden veri seti üretir (damıtma için, M5 yol haritası).

    python -m tools.autolabel data/pozitif/videolar --out data/pozitif/autolabel --fps 1

Her videodan `--fps` hızında kare alır, kapı + dedektör (+ istenirse takip maskesi) çalıştırır:
  * kareler data/pozitif/autolabel/images/ altına (kilitli, safety.write_dataset_frame)
  * COCO etiketleri data/pozitif/autolabel/_annotations.coco.json (her karede `time_s`)
  * her video için <video>.blurtrack.json (oynatıcı bunu doğrudan kullanabilir)
  * özet sayılar ekrana ve experiments/results/<tarih>_autolabel.json
Bölge bulunmayan karelerin bir kısmı (--keep-empty) negatif örnek olarak tutulur.

Uyarı (FA-04): ham etiketler modelin kendi hatalarını içerir. Eğitime ancak `tools.relabel_events`
ile (gerçek kan dakikaları dışındaki tespitler zor negatife çevrilerek) girer.

Kansız video için `--negative` (zor negatif madenciliği): videoda kan OLMADIĞI biliniyorsa modelin
"kan" dediği her kare yanlış alarmdır. Bu kareler etiketsiz yazılır, eğitimde `--neg` ile verilir. Ham
kareler kimden gelirse gelsin yalnız kilitli data/pozitif/ altına yazılır (safety.write_dataset_frame):

    python -m tools.autolabel data/kansiz_videolar --out data/pozitif/kansiz_zor --negative --min-score 0.2

Hız: örneklenmeyen kareler yalnız atlanır (grab), çözülüp kopyalanmaz. Uzun videoda her ~5 dakikada
bir ilerleme satırı yazar. Dakika başına işaretli saniye sonuca da yazılır (görüntü yok).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from sensa_ai import config, models, safety
from sensa_ai.blurtrack import BlurTrack
from sensa_ai.detect import Detection
from sensa_ai.pipeline import associate

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
REPORT_EVERY_S = 300.0


def coco_annotation(ann_id: int, img_id: int, d: Detection, w: int, h: int) -> dict[str, Any]:
    """Normalize tespit -> piksel COCO etiketi. Maskeden poligon varsa segmentasyon, yoksa yalnız kutu."""
    x0, y0, x1, y1 = d.box
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    a: dict[str, Any] = {
        "id": ann_id,
        "image_id": img_id,
        "category_id": 1,
        "bbox": [x0 * w, y0 * h, bw, bh],
        "area": bw * bh,
        "iscrowd": 0,
        "score": d.score,
    }
    if d.poly:
        a["segmentation"] = [[v * (w if k % 2 == 0 else h) for k, v in enumerate(q)] for q in d.poly]
    return a


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pozitif" / "autolabel")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-empty", type=float, default=0.1, help="bölgesiz karelerden tutulacak oran")
    ap.add_argument("--min-score", type=float, default=0.35, help="etikete girecek en düşük skor")
    ap.add_argument("--negative", action="store_true", help="kansız video: tespitler yanlış alarm, kare etiketsiz")
    ap.add_argument("--profile", default=None, help="hassasiyet profili (params.yaml)")
    args = ap.parse_args()
    config.set_profile(args.profile)
    pos = (safety.DATA_ROOT / "pozitif").resolve()
    if pos not in args.out.resolve().parents:  # model yüklenmeden önce söyle (D-001: ham kare kilitli klasörde)
        ap.error(f"--out yalnız {pos} altında olabilir, ör. data/pozitif/kansiz_zor")

    prompts = config.load_prompts()
    p = config.load()
    # D-016: varsayılan eğitilmiş RF-DETR-Seg (kapı params'tan); dosya yoksa OWLv2
    gd = models.build_default(device=args.device, detector=args.detector)
    rng = random.Random(0)
    coco: dict[str, list[dict[str, Any]]] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "blood"}],
    }
    stats: Counter[str] = Counter()
    per_minute: dict[str, list[int]] = {}  # video -> dakika başına işaretli örnek (≈ saniye, --fps 1'de)

    for video in sorted(v for v in args.videos.rglob("*") if v.suffix.lower() in VIDEO_EXT):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps / args.fps))
        duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
        bt = BlurTrack(video.name, duration)
        minutes = per_minute.setdefault(video.name, [0] * (int(duration // 60) + 1))
        stats["video"] += 1
        t_start, last_report, i = time.perf_counter(), 0.0, 0
        while True:
            if i % step:
                if not cap.grab():  # örneklenmeyen kare: çözme, kopyalama yok
                    break
                i += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t = i / fps
            i += 1
            if t - last_report >= REPORT_EVERY_S:
                el = time.perf_counter() - t_start
                print(f"  {video.name}: {t / 60:.0f}/{duration / 60:.0f} dk ({el:.0f} sn)", flush=True)
                last_report = t
            dets = [d for d in gd(frame, prompts["positive"]) if d.score >= args.min_score]
            associate(bt.tracks, t, dets, p["track"]["max_gap_s"])
            stats["kare"] += 1
            if dets:
                minutes[min(int(t // 60), len(minutes) - 1)] += 1
            if not dets and rng.random() >= args.keep_empty:
                continue
            img_id = len(coco["images"]) + 1
            name = f"{video.stem}_{i - 1:07d}.jpg"
            safety.write_dataset_frame(args.out / "images" / name, frame)
            h, w = frame.shape[:2]
            coco["images"].append(
                {"id": img_id, "file_name": f"images/{name}", "width": w, "height": h, "time_s": round(t, 3)}
            )
            if args.negative:  # kansız video: tespit = yanlış alarm, etiket yazılmaz
                stats["zor_negatif" if dets else "bos_tutulan"] += 1
                continue
            for d in dets:
                coco["annotations"].append(coco_annotation(len(coco["annotations"]) + 1, img_id, d, w, h))
            stats["etiketli_kare" if dets else "bos_tutulan"] += 1
            stats["bolge"] += len(dets)
        cap.release()
        bt.analyzed_until_s = bt.duration_s
        bt.save(args.out / f"{video.stem}.blurtrack.json")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_annotations.coco.json").write_text(json.dumps(coco))
    out: dict[str, Any] = {**stats, "mod": "negatif" if args.negative else "pozitif", "min_skor": args.min_score}
    out["dakika_basina_isaretli"] = per_minute
    res = ROOT / "experiments" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{dt.date.today()}_autolabel.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(dict(stats))
    for name, mins in per_minute.items():
        busy = [f"{k}. dk: {n}" for k, n in enumerate(mins) if n]
        print(f"{name}: işaretli dakikalar ({len(busy)}/{len(mins)}): " + (", ".join(busy) or "yok"))


if __name__ == "__main__":
    main()
