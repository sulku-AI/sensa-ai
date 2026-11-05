"""Damıtma eğitimi (M5 yol haritası): RF-DETR-Seg Nano (Apache 2.0, D-011), varsayılan cihaz M5 / MPS.

    pip install "rfdetr[train]"
    python -m tools.train_rfdetr prepare --src data/pozitif/blood_seg_joost data/pozitif/blood_pure_blur \
        data/pozitif/autolabel --neg data/negatif/coco
    python -m tools.train_rfdetr train --epochs 30

`prepare` kaynakları tek "blood" sınıfında birleştirip data/pozitif/egitim/{train,valid,test}
(Roboflow COCO düzeni) yazar; negatif görüntüler etiketsiz eklenir (yanlış alarmı öğretir).
Kutu yalnız etiketler (autolabel) dikdörtgen poligona çevrilir. Görüntüler kopyalanır,
açılmaz. `train` çıktısı models/rfdetr_seg_nano/ (repoya girmez).

Sızıntı önlemi: aynı kaynaktan gelen kopyalar (Roboflow artırması `ad_jpg.rf.<hash>`, autolabel
video kareleri `video_0000123`) tek grup sayılır ve hep aynı bölüme düşer. Kaynağın kendi
train/valid/test klasörü varsa ona uyulur; yoksa grup adının özetiyle 80/10/10 bölünür. valid ve
test'te grup başına tek görüntü, train'de en çok `--aug-per-image` kopya kalır (33 bin görüntülük
artırılmış set T4'te makul sürede biter).

M5 ayarları (tahmin, D-012): batch 2 x grad_accum 8, MPS'te AMP kapalı, num_workers 2.
Fansız kasada uzun eğitim ısınır: geceleri prize takılıyken çalıştır.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "pozitif" / "egitim"
IMG_EXT = {".jpg", ".jpeg", ".png"}


def _box_poly(b: list[float]) -> list[list[float]]:
    x, y, w, h = b
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


Item = tuple[Path, int, int, list[dict], str, str | None]  # yol, w, h, etiketler, grup, kaynak bölümü
SPLITS = ("train", "valid", "test")


def group_key(path: Path) -> str:
    """Aynı özgün görüntünün/videonun kopyaları aynı anahtarı alır."""
    stem = path.name.split(".rf.")[0]
    stem = re.sub(r"_\d{7}(\.\w+)?$", "", stem)  # autolabel kare numarası
    return f"{path.parent.parent.name}/{stem}" if path.parent.name in SPLITS else f"{path.parent}/{stem}"


def hash_split(key: str) -> str:
    u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "train" if u < 0.8 else "valid" if u < 0.9 else "test"


def collect(src: list[Path], neg: list[Path]) -> list[Item]:
    items: list[Item] = []
    for folder in src:
        for ann_path in sorted(folder.rglob("_annotations.coco.json")):
            d = json.loads(ann_path.read_text())
            split = ann_path.parent.name if ann_path.parent.name in SPLITS else None
            by_img: dict[int, list[dict]] = {}
            for a in d["annotations"]:
                seg = a.get("segmentation") if isinstance(a.get("segmentation"), list) and a["segmentation"] else None
                by_img.setdefault(a["image_id"], []).append(
                    {"bbox": a["bbox"], "segmentation": seg or _box_poly(a["bbox"]), "area": a.get("area", 0)}
                )
            for img in d["images"]:
                p = ann_path.parent / img["file_name"]
                key = f"{folder.name}:{group_key(p)}"
                items.append((p, img["width"], img["height"], by_img.get(img["id"], []), key, split))
    import cv2

    for folder in neg:
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMG_EXT:
                im = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)  # yalnız boyut için
                if im is not None:
                    items.append((p, im.shape[1] * 8, im.shape[0] * 8, [], f"neg:{p.name}", None))
    return items


def split_items(items: list[Item], aug_per_image: int) -> dict[str, list[Item]]:
    """Grup bütün kalır; kaynak bölümü varsa en saklı olan (test > valid > train) seçilir."""
    rank = {"train": 0, "valid": 1, "test": 2}
    groups: dict[str, list[Item]] = {}
    for it in items:
        groups.setdefault(it[4], []).append(it)
    out: dict[str, list[Item]] = {s: [] for s in SPLITS}
    for key, members in sorted(groups.items()):
        given = [m[5] for m in members if m[5] is not None]
        split = max(given, key=rank.__getitem__) if given else hash_split(key)
        keep = aug_per_image if split == "train" else 1
        out[split].extend(members[: max(1, keep)])
    rng = random.Random(0)
    for rows in out.values():
        rng.shuffle(rows)
    return out


def prepare(args: argparse.Namespace) -> None:
    from sensa_ai import safety

    splits = split_items(collect(args.src, args.neg), args.aug_per_image)
    if OUT.exists():
        shutil.rmtree(OUT)
    safety.lock_folder(OUT.parent)
    counts = {}
    for split, rows in splits.items():
        d = OUT / split
        d.mkdir(parents=True)
        coco: dict[str, list] = {"images": [], "annotations": [], "categories": [{"id": 0, "name": "blood"}]}
        for k, (path, w, h, anns, key, _) in enumerate(rows, 1):
            name = f"{k:06d}{path.suffix.lower()}"
            shutil.copyfile(path, d / name)
            src = key.split(":", 1)[0]  # kaynak adı (evaluate kaynağa göre döker)
            coco["images"].append({"id": k, "file_name": name, "width": w, "height": h, "source": src})
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
    kw: dict = {
        "dataset_dir": str(OUT),
        "output_dir": str(args.out),
        "epochs": args.epochs,
        "batch_size": args.batch,
        "grad_accum_steps": args.accum,
        "lr": args.lr,
        "num_workers": 2,
        "early_stopping": True,
        "notes": {"proje": "sensa-ai", "veri": "docs/MODEL_ARASTIRMA.md §5"},
    }
    if args.resume:
        kw["resume"] = str(args.resume)
    kw.update(version_kwargs(getattr(model, "_train_config_class", None), device))
    model.train(**kw)


def version_kwargs(config_class: object, device: str) -> dict:
    """rfdetr sürüm farkı: yeni sürüm (Lightning) `device`/`amp` yerine `accelerator` alır.

    Güvenlik (D-001): sürüm destekliyorsa veri seti ızgara görüntüleri diske yazılmaz.
    """
    fields = getattr(config_class, "model_fields", None)
    if not fields or "device" in fields:
        return {"device": device, "amp": device == "cuda"}
    out: dict = {"accelerator": {"cuda": "gpu"}.get(device, device)}
    if "save_dataset_grids" in fields:
        out["save_dataset_grids"] = False
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--src", type=Path, nargs="+", required=True)
    p.add_argument("--neg", type=Path, nargs="*", default=[])
    p.add_argument("--aug-per-image", type=int, default=2, help="train'de özgün görüntü başına en çok kopya")
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--batch", type=int, default=2)
    t.add_argument("--accum", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--device", default=None)
    t.add_argument("--out", type=Path, default=ROOT / "models" / "rfdetr_seg_nano", help="Colab: Drive klasörü")
    t.add_argument("--resume", type=Path, default=None, help="yarım kalan eğitim: <out>/checkpoint.pth")
    args = ap.parse_args()
    prepare(args) if args.cmd == "prepare" else train(args)


if __name__ == "__main__":
    main()
