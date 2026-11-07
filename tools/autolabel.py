"""Otomatik etiketleme: insan görmeden veri seti üretir (damıtma için, M5 yol haritası).

    python -m tools.autolabel data/pozitif/videolar --out data/pozitif/autolabel --fps 1

Her videodan `--fps` hızında kare alır, kapı + dedektör (+ istenirse takip maskesi) çalıştırır:
  * kareler data/pozitif/autolabel/images/ altına (kilitli, safety.write_dataset_frame)
  * COCO etiketleri data/pozitif/autolabel/_annotations.coco.json
  * her video için <video>.blurtrack.json (oynatıcı bunu doğrudan kullanabilir)
  * özet sayılar ekrana ve experiments/results/<tarih>_autolabel.json
Bölge bulunmayan karelerin bir kısmı (--keep-empty) negatif örnek olarak tutulur.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
from pathlib import Path

import cv2

from sensa_ai import config, models, safety
from sensa_ai.blurtrack import BlurTrack
from sensa_ai.pipeline import associate

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pozitif" / "autolabel")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--detector", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--keep-empty", type=float, default=0.1, help="bölgesiz karelerden tutulacak oran")
    ap.add_argument("--min-score", type=float, default=0.35, help="etikete girecek en düşük skor")
    args = ap.parse_args()

    prompts = config.load_prompts()
    p = config.load()
    # D-016: varsayılan eğitilmiş RF-DETR-Seg (kapı params'tan); dosya yoksa OWLv2
    gd = models.build_default(device=args.device, detector=args.detector)
    rng = random.Random(0)
    coco: dict[str, list[dict[str, object]]] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "blood"}],
    }
    stats = {"video": 0, "kare": 0, "etiketli_kare": 0, "bolge": 0, "bos_tutulan": 0}

    for video in sorted(v for v in args.videos.rglob("*") if v.suffix.lower() in VIDEO_EXT):
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, round(fps / args.fps))
        bt = BlurTrack(video.name, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps)
        i = 0
        stats["video"] += 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                t = i / fps
                dets = [d for d in gd(frame, prompts["positive"]) if d.score >= args.min_score]
                associate(bt.tracks, t, dets, p["track"]["max_gap_s"])
                stats["kare"] += 1
                if dets or rng.random() < args.keep_empty:
                    img_id = len(coco["images"]) + 1
                    name = f"{video.stem}_{i:07d}.jpg"
                    safety.write_dataset_frame(args.out / "images" / name, frame)
                    h, w = frame.shape[:2]
                    coco["images"].append({"id": img_id, "file_name": f"images/{name}", "width": w, "height": h})
                    for d in dets:
                        x0, y0, x1, y1 = d.box
                        bw, bh = (x1 - x0) * w, (y1 - y0) * h
                        coco["annotations"].append(
                            {
                                "id": len(coco["annotations"]) + 1,
                                "image_id": img_id,
                                "category_id": 1,
                                "bbox": [x0 * w, y0 * h, bw, bh],
                                # maskeden poligon; yoksa yalnız kutu
                                **(
                                    {
                                        "segmentation": [
                                            [v * (w if k % 2 == 0 else h) for k, v in enumerate(q)] for q in d.poly
                                        ]
                                    }
                                    if d.poly
                                    else {}
                                ),
                                "area": bw * bh,
                                "iscrowd": 0,
                                "score": d.score,
                            }
                        )
                    stats["etiketli_kare" if dets else "bos_tutulan"] += 1
                    stats["bolge"] += len(dets)
            i += 1
        cap.release()
        bt.analyzed_until_s = bt.duration_s
        bt.save(args.out / f"{video.stem}.blurtrack.json")

    (args.out / "_annotations.coco.json").write_text(json.dumps(coco))
    res = ROOT / "experiments" / "results"
    res.mkdir(parents=True, exist_ok=True)
    (res / f"{dt.date.today()}_autolabel.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    print(stats)


if __name__ == "__main__":
    main()
