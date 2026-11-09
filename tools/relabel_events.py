"""Autolabel etiketlerini kabaca bilinen "gerçek kan" zamanlarıyla düzeltir (görüntüye bakmadan).

    python -m tools.relabel_events data/pozitif/autolabel --video data/pozitif/videolar/dex1-1.mp4 \
        --ranges "12:00-14:30,38:10-41:00"

Model kırmızı her şeye "kan" diyorsa autolabel etiketleri de yanlış olur. İzleyen kişi bölümde
gerçekten kan olan dakikaları kabaca yazar; bu araç:
  * aralık DIŞINDAKİ karelerin bütün etiketlerini siler: model orada yanıldı, kare zor negatif olur
  * aralık İÇİNDE skoru `--pos-min-score` altındaki bölgeleri siler, kalan kare pozitif kalır
  * aralık içinde hiç bölgesi kalmayan kareyi atar (kaçırma mı, gerçekten kansız mı bilinmiyor)
`--ranges` yerine `--events data/etiket/<video>.json` (tools.label_events çıktısı) da verilebilir.
Kansız bir video için `--ranges ""`: bütün tespitler zor negatif olur.

Aralıklar insan bilgisi değil de model önerisiyse (`tools.flag_timeline --judge`), emin olunmayan
dakikalar `--unsure` ile verilir: oradaki kareler ne pozitif ne negatif, eğitimden çıkar. Bir dakikada
en çok `--neg-per-min` zor negatif kalır (params `dizi.neg_per_min`, eşit aralıklı seçilir): art arda
saniyeler neredeyse aynı kare.

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


def relabel(
    coco: dict,
    stem: str,
    fps: float,
    ranges: list[tuple[float, float]],
    pos_min: float,
    unsure: list[tuple[float, float]] | None = None,
    neg_per_min: int = 0,
) -> tuple[dict, dict]:
    """`stem` videosunun karelerini aralıklara göre düzeltir; diğer videoların karelerine dokunmaz."""

    def near(t: float, rr: list[tuple[float, float]]) -> bool:
        return any(a - PAD_S <= t <= b + PAD_S for a, b in rr)

    anns: dict[int, list[dict]] = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    keep: dict[int, list[dict]] = {}  # kalan kare -> kalan etiketleri
    hard: dict[int, list[tuple[float, int]]] = {}  # dakika -> (zaman, kare) zor negatif adayları
    names = ("negatif_oldu", "pozitif_kaldi", "atildi", "belirsiz_atildi", "fazla_negatif_atildi", "dokunulmadi")
    st = dict.fromkeys((*names, "bos_negatif"), 0)
    for img in coco["images"]:
        name = Path(img["file_name"]).name
        own = anns.get(img["id"], [])
        if not name.startswith(stem + "_"):
            keep[img["id"]] = own
            st["dokunulmadi"] += 1
            continue
        t = float(img["time_s"]) if "time_s" in img else frame_index(name) / fps  # yeni autolabel zamanı yazar
        if near(t, ranges):
            good = [a for a in own if a.get("score", 1.0) >= pos_min]
            if good:
                keep[img["id"]] = good
                st["pozitif_kaldi"] += 1
            else:  # kaçırma mı, gerçekten kansız mı bilinmiyor
                st["atildi"] += 1
        elif unsure and near(t, unsure):
            st["belirsiz_atildi"] += 1
        elif own:  # model burada yanıldı: etiketsiz zor negatif
            hard.setdefault(int(t // 60), []).append((t, img["id"]))
        else:
            keep[img["id"]] = []
            st["bos_negatif"] += 1
    for rows in hard.values():
        rows.sort()
        picked = rows
        if 0 < neg_per_min < len(rows):  # eşit aralıklı seçim: sahne çeşitliliği kalsın
            picked = [rows[int(i * len(rows) / neg_per_min)] for i in range(neg_per_min)]
        for _, img_id in picked:
            keep[img_id] = []
        st["negatif_oldu"] += len(picked)
        st["fazla_negatif_atildi"] += len(rows) - len(picked)
    images = [img for img in coco["images"] if img["id"] in keep]
    kept_anns = [a for img in images for a in keep[img["id"]]]
    return {**coco, "images": images, "annotations": kept_anns}, st


def main() -> None:
    from sensa_ai import config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="autolabel çıktısı")
    ap.add_argument("--video", type=Path, required=True, help="karelerin geldiği video (yalnız fps okunur)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ranges", help='gerçek kan aralıkları, ör. "12:00-14:30,38:10-41:00"; kansız video için ""')
    g.add_argument("--events", type=Path, help="tools.label_events JSON çıktısı")
    ap.add_argument("--unsure", default="", help='emin olunmayan aralıklar, eğitimden çıkar: ör. "16-17,28-30"')
    ap.add_argument("--pos-min-score", type=float, default=0.5)
    ap.add_argument(
        "--neg-per-min", type=int, default=None, help="dakikada en çok zor negatif (yoksa params; 0 sınırsız)"
    )
    args = ap.parse_args()
    neg_per_min = int(config.load()["dizi"]["neg_per_min"]) if args.neg_per_min is None else args.neg_per_min

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
    entry = {
        "ranges": ranges,
        "unsure": parse_ranges(args.unsure),
        "fps": fps,
        "pos_min": args.pos_min_score,
        "neg_per_min": neg_per_min,
    }
    history = {**cur.get("relabel", {}), args.video.stem: entry}
    coco = json.loads(orig.read_text())
    for stem, h in history.items():  # her video kendi aralıklarıyla, hep özgünden
        coco, st = relabel(
            coco,
            stem,
            h["fps"],
            [tuple(r) for r in h["ranges"]],
            h["pos_min"],
            [tuple(r) for r in h.get("unsure", [])],
            int(h.get("neg_per_min", 0)),
        )
        print(stem, st)
    coco["relabel"] = history
    ann.write_text(json.dumps(coco))


if __name__ == "__main__":
    main()
