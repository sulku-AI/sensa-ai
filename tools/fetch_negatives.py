"""Negatif set (kansız, Emirhan bakabilir): COCO 2017 val'den kırmızı/kan benzeri sınıflar.

    python -m tools.fetch_negatives --per-class 40

data/negatif/coco/ altına görüntüler + liste yazar. COCO görüntüleri Flickr kaynaklı,
lisansları görüntü başına COCO annotations içinde (çoğu CC BY / CC BY-SA / CC BY-NC).
Yalnız değerlendirme içindir, yeniden dağıtılmaz.

Not: COCO'da kan yok ama her ihtimale karşı negatif klasörü de yalnız sınıf adıyla seçilir;
istersen önce `--dry-run` ile yalnız sayılara bak.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEG = ROOT / "data" / "negatif" / "coco"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
CLASSES = ["traffic light", "wine glass", "pizza", "apple", "orange", "hot dog", "tie", "stop sign", "umbrella", "cake"]


def load_instances() -> dict:
    cache = NEG.parent / "instances_val2017.json"
    if not cache.exists():
        NEG.parent.mkdir(parents=True, exist_ok=True)
        print("COCO annotations indiriliyor (~250 MB, bir kez)...")
        data = urllib.request.urlopen(ANN_URL).read()  # noqa: S310 (sabit URL)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            cache.write_bytes(z.read("annotations/instances_val2017.json"))
    return json.loads(cache.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    d = load_instances()
    cats = {c["id"]: c["name"] for c in d["categories"] if c["name"] in CLASSES}
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
    NEG.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, name in chosen.items():
        info = imgs[i]
        out = NEG / info["file_name"]
        if not out.exists():
            out.write_bytes(urllib.request.urlopen(info["coco_url"]).read())  # noqa: S310
        manifest.append({"file": info["file_name"], "class": name, "license": info["license"]})
    (NEG / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("->", NEG)


if __name__ == "__main__":
    main()
