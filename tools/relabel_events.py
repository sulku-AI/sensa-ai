"""Autolabel etiketlerini kabaca bilinen "gerçek kan" zamanlarıyla düzeltir (görüntüye bakmadan).

    python -m tools.relabel_events data/pozitif/autolabel --video data/pozitif/videolar/dex1-1.mp4 \
        --ranges "12:00-14:30,38:10-41:00"

Model kırmızı her şeye "kan" diyorsa autolabel etiketleri de yanlış olur. İzleyen kişi bölümde
gerçekten kan olan dakikaları kabaca yazar; bu araç:
  * aralık DIŞINDAKİ karelerin bütün etiketlerini siler: model orada yanıldı, kare zor negatif olur
  * aralık İÇİNDE skoru `--pos-min-score` altındaki bölgeleri siler, kalan kare pozitif kalır
  * aralık içinde hiç bölgesi kalmayan kareyi atar (kaçırma mı, gerçekten kansız mı bilinmiyor)
`--ranges` yerine `--events data/etiket/<video>.json` (tools.label_events çıktısı) da verilebilir.
Özgün dosya `_annotations.coco.orig.json` olarak saklanır; her video için verilen aralıklar dosyada
tutulur ve her çalıştırmada özgünden yeniden uygulanır (aynı video için tekrar çalıştırmak güvenli).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PAD_S = 2.0  # aralık kenarı payı: kabaca hatırlanan zamanlar için


def parse_ranges(text: str) -> list[tuple[float, float]]:
    """'12:00-14:30, 38-41' -> saniye aralıkları. Tek sayı dakika demektir."""

    def sec(t: str) -> float:
        parts = [float(p) for p in t.strip().split(":")]
        if len(parts) == 1:
            return parts[0] * 60
        return sum(v * 60**k for k, v in enumerate(reversed(parts)))

    out = []
    for chunk in filter(None, (c.strip() for c in text.split(","))):
        a, b = chunk.split("-")
        out.append((sec(a), sec(b)))
    return out


def frame_index(file_name: str) -> int:
    m = re.search(r"_(\d{7})\.\w+$", file_name)
    if not m:
        raise ValueError(f"autolabel kare adı değil: {file_name}")
    return int(m.group(1))


def relabel(coco: dict, stem: str, fps: float, ranges: list[tuple[float, float]], pos_min: float) -> tuple[dict, dict]:
    def inside(t: float) -> bool:
        return any(a - PAD_S <= t <= b + PAD_S for a, b in ranges)

    anns: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    images, keep_anns = [], []
    st = {"negatif_oldu": 0, "pozitif_kaldi": 0, "atildi": 0, "dokunulmadi": 0, "bos_negatif": 0}
    for img in coco["images"]:
        name = Path(img["file_name"]).name
        if not name.startswith(stem + "_"):
            images.append(img)
            keep_anns += anns.get(img["id"], [])
            st["dokunulmadi"] += 1
            continue
        t = frame_index(name) / fps
        own = anns.get(img["id"], [])
        if not inside(t):
            images.append(img)  # etiketsiz: zor negatif
            st["negatif_oldu" if own else "bos_negatif"] += 1
            continue
        good = [a for a in own if a.get("score", 1.0) >= pos_min]
        if good:
            images.append(img)
            keep_anns += good
            st["pozitif_kaldi"] += 1
        else:
            st["atildi"] += 1
    return {**coco, "images": images, "annotations": keep_anns}, st


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="autolabel çıktısı")
    ap.add_argument("--video", type=Path, required=True, help="karelerin geldiği video (yalnız fps okunur)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ranges", help='gerçek kan aralıkları, ör. "12:00-14:30,38:10-41:00"')
    g.add_argument("--events", type=Path, help="tools.label_events JSON çıktısı")
    ap.add_argument("--pos-min-score", type=float, default=0.5)
    args = ap.parse_args()

    if args.events:
        ranges = [(e["start"], e["end"]) for e in json.loads(args.events.read_text())["events"]]
    else:
        ranges = parse_ranges(args.ranges)
    import cv2

    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    ann = args.folder / "_annotations.coco.json"
    orig = args.folder / "_annotations.coco.orig.json"
    cur = json.loads(ann.read_text())
    if "relabel" not in cur:  # autolabel'den taze çıktı: özgün kopya bu
        orig.write_text(ann.read_text())
    history = {**cur.get("relabel", {}), args.video.stem: {"ranges": ranges, "fps": fps, "pos_min": args.pos_min_score}}
    coco = json.loads(orig.read_text())
    for stem, h in history.items():  # her video kendi aralıklarıyla, hep özgünden
        coco, st = relabel(coco, stem, h["fps"], [tuple(r) for r in h["ranges"]], h["pos_min"])
        print(stem, st)
    coco["relabel"] = history
    ann.write_text(json.dumps(coco))


if __name__ == "__main__":
    main()
