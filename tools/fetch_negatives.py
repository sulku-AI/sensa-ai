"""Negatif set (kansız, Emirhan bakabilir): COCO 2017 val'den kırmızı/kan benzeri sınıflar.

    python -m tools.fetch_negatives --per-class 40
    python -m tools.fetch_negatives --set zor --per-class 30

data/negatif/coco/ altına görüntüler + liste yazar. `--set zor` ayrı bir sınama seti yazar
(data/negatif/coco_zor/): COCO train2017'den (eğitimdeki val2017 negatifleriyle çakışmaz),
insanlı sahneler ve kana benzeyen kırmızı nesneler. Eğitimde KULLANILMAZ; yalnız yanlış alarm
ölçümü içindir (eğitilmiş model hiç görmediği dağılımda sınanır). COCO görüntüleri Flickr kaynaklı,
lisansları görüntü başına COCO annotations içinde (çoğu CC BY / CC BY-SA / CC BY-NC).
Yalnız değerlendirme içindir, yeniden dağıtılmaz.

Not: COCO'da kan yok ama her ihtimale karşı negatif klasörü de yalnız sınıf adıyla seçilir;
istersen önce `--dry-run` ile yalnız sayılara bak.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEG = ROOT / "data" / "negatif" / "coco"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
HARD = [
    "person",
    "fire hydrant",
    "stop sign",
    "sports ball",
    "kite",
    "teddy bear",
    "bowl",
    "donut",
    "apple",
    "pizza",
    "hot dog",
    "cake",
    "wine glass",
    "tie",
    "umbrella",
    "couch",
    "bed",
    "toothbrush",
]
CLASSES = ["traffic light", "wine glass", "pizza", "apple", "orange", "hot dog", "tie", "stop sign", "umbrella", "cake"]


def load_instances(split: str = "val2017") -> dict:
    cache = NEG.parent / f"instances_{split}.json"
    if not cache.exists():
        NEG.parent.mkdir(parents=True, exist_ok=True)
        zpath = NEG.parent / "annotations_trainval2017.zip"
        if not zpath.exists():
            print("COCO annotations indiriliyor (~250 MB, bir kez)...")
            zpath.write_bytes(urllib.request.urlopen(ANN_URL).read())  # noqa: S310 (sabit URL)
        with zipfile.ZipFile(zpath) as z:
            cache.write_bytes(z.read(f"annotations/instances_{split}.json"))
    return json.loads(cache.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--set", choices=["egitim", "zor"], default="egitim")
    args = ap.parse_args()
    hard = args.set == "zor"
    out_dir = NEG.parent / "coco_zor" if hard else NEG
    d = load_instances("train2017" if hard else "val2017")
    wanted = HARD if hard else CLASSES
    cats = {c["id"]: c["name"] for c in d["categories"] if c["name"] in wanted}
    by_cat: dict[str, set[int]] = {n: set() for n in cats.values()}
    for a in d["annotations"]:
        if a["category_id"] in cats:
            by_cat[cats[a["category_id"]]].add(a["image_id"])
    imgs = {i["id"]: i for i in d["images"]}
    rng = random.Random(0)
    chosen: dict[int, str] = {}
    for name, ids in by_cat.items():
        for i in rng.sample(sorted(ids), min(args.per_class, len(ids))):
            chosen.setdefault(i, name)
    print({n: len(v) for n, v in by_cat.items()}, "->", len(chosen), "görüntü")
    if args.dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, name in chosen.items():
        info = imgs[i]
        out = out_dir / info["file_name"]
        if not out.exists():
            out.write_bytes(urllib.request.urlopen(info["coco_url"]).read())  # noqa: S310
        manifest.append({"file": info["file_name"], "class": name, "license": info["license"]})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("->", out_dir)


if __name__ == "__main__":
    main()
