"""Damıtma eğitimi (M5 yol haritası): RF-DETR-Seg Nano (Apache 2.0, D-011), varsayılan cihaz M5 / MPS.

    pip install "rfdetr[train]"
    python -m tools.train_rfdetr prepare --src data/pozitif/blood_seg_joost data/pozitif/blood_pure_blur \
        data/pozitif/autolabel --neg data/negatif/coco
    python -m tools.train_rfdetr train --epochs 30

`prepare` kaynakları tek "blood" sınıfında birleştirip data/pozitif/egitim/{train,valid,test}
(Roboflow COCO düzeni) yazar; negatif görüntüler etiketsiz eklenir (yanlış alarmı öğretir).
Kutu yalnız etiketler (autolabel) dikdörtgen poligona çevrilir. Görüntüler kopyalanır,
açılmaz; bölünme sabit tohumla. `train` çıktısı models/rfdetr_seg_nano/ (repoya girmez).

M5 ayarları (tahmin, D-012): batch 2 x grad_accum 8, MPS'te AMP kapalı, num_workers 2.
Fansız kasada uzun eğitim ısınır: geceleri prize takılıyken çalıştır.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pozitif" / "egitim"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _box_poly(b: list[float]) -> list[list[float]]:
    x, y, w, h = b
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def collect(src: list[Path], neg: list[Path]) -> list[tuple[Path, int, int, list[dict]]]:
    items: list[tuple[Path, int, int, list[dict]]] = []
    for folder in src:
        for ann_path in folder.rglob("_annotations.coco.json"):
            d = json.loads(ann_path.read_text())
            by_img: dict[int, list[dict]] = {}
            for a in d["annotations"]:
                seg = a.get("segmentation") if isinstance(a.get("segmentation"), list) and a["segmentation"] else None
                by_img.setdefault(a["image_id"], []).append(
                    {"bbox": a["bbox"], "segmentation": seg or _box_poly(a["bbox"]), "area": a.get("area", 0)}
                )
            for img in d["images"]:
                items.append(
                    (ann_path.parent / img["file_name"], img["width"], img["height"], by_img.get(img["id"], []))
                )
    import cv2

    for folder in neg:
        for p in folder.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                im = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)  # yalnız boyut için
                if im is not None:
                    items.append((p, im.shape[1] * 8, im.shape[0] * 8, []))
    return items


def prepare(args: argparse.Namespace) -> None:
    from sensa_ai import safety

    items = collect(args.src, args.neg)
    random.Random(0).shuffle(items)
    n = len(items)
    splits = {
        "train": items[: int(n * 0.8)],
        "valid": items[int(n * 0.8) : int(n * 0.9)],
        "test": items[int(n * 0.9) :],
    }
    if OUT.exists():
        shutil.rmtree(OUT)
    safety.lock_folder(OUT.parent)
    counts = {}
    for split, rows in splits.items():
        d = OUT / split
        d.mkdir(parents=True)
        coco: dict[str, list] = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "blood"}]}
        for k, (path, w, h, anns) in enumerate(rows, 1):
            name = f"{k:06d}{path.suffix.lower()}"
            shutil.copyfile(path, d / name)
            coco["images"].append({"id": k, "file_name": name, "width": w, "height": h})
            for a in anns:
                coco["annotations"].append(
                    {"id": len(coco["annotations"]) + 1, "image_id": k, "category_id": 0, "iscrowd": 0, **a}
                )
        (d / "_annotations.coco.json").write_text(json.dumps(coco))
        counts[split] = {
            "goruntu": len(rows),
            "bolge": len(coco["annotations"]),
            "bos": sum(1 for r in rows if not r[3]),
        }
    print(counts)


def train(args: argparse.Namespace) -> None:
    import torch
    from rfdetr import RFDETRSegNano

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = RFDETRSegNano()
    model.train(
        dataset_dir=str(OUT),
        output_dir=str(ROOT / "models" / "rfdetr_seg_nano"),
        epochs=args.epochs,
        batch_size=args.batch,
        grad_accum_steps=args.accum,
        lr=args.lr,
        device=device,
        amp=device == "cuda",
        num_workers=2,
        early_stopping=True,
        notes={"proje": "sensa-ai", "veri": "docs/MODEL_ARASTIRMA.md §5"},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", type=Path, nargs="+", required=True)
    p.add_argument("--neg", type=Path, nargs="*", default=[])
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--device", default=None)
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    main()
