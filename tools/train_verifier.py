"""Kırpım doğrulayıcıyı eğitir (Y3, D-023): "kan" / "kırmızı ama kan değil". Görüntü göstermez, yazmaz.

    python -m tools.train_verifier --autolabel data/pozitif/autolabel/dex1-* --pos data/pozitif/roboflow \\
        --holdout dex1-12 dex2-12

Örnekler (yalnız kutu kırpımları, bellekte):
  * pozitif: `--pos` fotoğraf setlerinin etiket kutuları ve autolabel klasörlerinde relabel sonrası kalan kutular
  * negatif: autolabel klasörlerinde zor negatife dönen karelerin özgün model kutuları
    (`_annotations.coco.orig.json`: model orada yanılmıştı) ve `--neg` klasörlerindeki görüntülerin tamamı
`--holdout` videoları (dosya adı öneki) eğitime girmez; tablo o videolarda ve fotoğrafların ayrılmış
%10'unda eşik başına geçiş oranlarını verir. Dizi pozitifinin geçiş oranına dikkat: baş "fotoğraf = kan,
dizi = kan değil" kestirmesini öğrendiyse o sayı düşük çıkar.
Çıktı: params `verify.head` (ya da `--out`) ve experiments/results/<tarih>_dogrulayici.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from sensa_ai import config, verify

ROOT = Path(__file__).resolve().parents[1]
IMG_EXT = {".jpg", ".jpeg", ".png"}
Sample = tuple[Path, tuple[float, float, float, float] | None, int, str, str]  # yol, xywh piksel, etiket, grup, kaynak


def video_stem(name: str) -> str:
    return re.sub(r"_\d{7}(\.\w+)?$", "", Path(name).name.split(".rf.")[0])


def autolabel_samples(folder: Path, max_per_image: int = 3) -> list[Sample]:
    """Relabel sonrası kalan kutular pozitif; etiketi silinmiş karelerin özgün kutuları zor negatif."""
    cur = json.loads((folder / "_annotations.coco.json").read_text())
    orig_path = folder / "_annotations.coco.orig.json"
    orig = json.loads(orig_path.read_text()) if orig_path.exists() else {"images": [], "annotations": []}

    def boxes(d: dict) -> dict[str, list[dict]]:
        names = {im["id"]: im["file_name"] for im in d["images"]}
        out: dict[str, list[dict]] = {}
        for a in d["annotations"]:
            out.setdefault(names[a["image_id"]], []).append(a)
        return out

    kept, model = boxes(cur), boxes(orig)
    out: list[Sample] = []
    for im in cur["images"]:
        fn = im["file_name"]
        label = 1 if fn in kept else 0
        anns = sorted(kept.get(fn) or model.get(fn, []), key=lambda a: -float(a.get("score", 1.0)))
        for a in anns[:max_per_image]:
            out.append((folder / fn, tuple(a["bbox"]), label, video_stem(fn), "dizi"))
    return out


def photo_samples(folder: Path, max_per_image: int = 3) -> list[Sample]:
    out: list[Sample] = []
    for ann in sorted(folder.rglob("_annotations.coco.json")):
        d = json.loads(ann.read_text())
        names = {im["id"]: im["file_name"] for im in d["images"]}
        seen: dict[int, int] = {}
        for a in d["annotations"]:
            if seen.setdefault(a["image_id"], 0) < max_per_image:
                seen[a["image_id"]] += 1
                fn = names[a["image_id"]]
                out.append((ann.parent / fn, tuple(a["bbox"]), 1, f"{folder.name}/{video_stem(fn)}", "foto"))
    return out


def negative_samples(folder: Path) -> list[Sample]:
    return [
        (p, None, 0, f"{folder.name}/{video_stem(p.name)}", "foto")
        for p in sorted(folder.rglob("*"))
        if p.suffix.lower() in IMG_EXT
    ]


def is_holdout(s: Sample, holdout: list[str]) -> bool:
    """Dizi: adı `holdout` önekiyle başlayan videolar. Fotoğraf: grup özetine göre sabit %10."""
    if s[4] == "dizi":
        return any(s[3] == h or s[3].startswith(h + "_") for h in holdout)
    return int(hashlib.sha1(s[3].encode()).hexdigest()[:8], 16) % 10 == 0


def embed_samples(samples: list[Sample], embed: verify.Embed, margin: float, batch: int = 64) -> np.ndarray:
    import cv2

    out: list[np.ndarray] = []
    crops: list[np.ndarray] = []

    def flush() -> None:
        if crops:
            out.append(np.asarray(embed(crops)))
            crops.clear()

    for i, (path, bbox, *_) in enumerate(samples):
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        if bbox is not None:
            h, w = im.shape[:2]
            x, y, bw, bh = bbox
            im = verify.crop(im, (x / w, y / h, (x + bw) / w, (y + bh) / h), margin)
        crops.append(im)
        if len(crops) == batch:
            flush()
            print(f"  gömme {i + 1}/{len(samples)}", flush=True)
    flush()
    return np.concatenate(out)


def report(head: verify.Head, x: np.ndarray, samples: list[Sample], thresholds: list[float]) -> list[dict]:
    """Eşik başına geçiş oranı: dizi pozitif, foto pozitif (yüksek olmalı), dizi ve foto negatif (düşük olmalı)."""
    p = head.prob(x) if len(samples) else np.zeros(0)
    rows = []
    for th in thresholds:
        row: dict = {"esik": th}
        for src in ("dizi", "foto"):
            for label, name in ((1, "pozitif"), (0, "negatif")):
                idx = [i for i, s in enumerate(samples) if s[4] == src and s[2] == label]
                row[f"{src}_{name}_n"] = len(idx)
                row[f"{src}_{name}_gecis"] = float((p[idx] >= th).mean()) if idx else None
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--autolabel", type=Path, nargs="*", default=[], help="relabel edilmiş autolabel klasörleri")
    ap.add_argument("--pos", type=Path, nargs="*", default=[], help="fotoğraf setleri (COCO etiketli)")
    ap.add_argument("--neg", type=Path, nargs="*", default=[], help="kansız görüntü klasörleri")
    ap.add_argument("--holdout", nargs="*", default=[], help="eğitime girmeyen videolar, ör. dex1-12")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results", type=Path, default=ROOT / "experiments" / "results")
    args = ap.parse_args()
    v = config.load()["verify"]
    samples = [s for f in args.autolabel for s in autolabel_samples(f)]
    samples += [s for f in args.pos for s in photo_samples(f)]
    samples += [s for f in args.neg for s in negative_samples(f)]
    if not any(s[2] for s in samples) or all(s[2] for s in samples):
        raise SystemExit("hem pozitif hem negatif örnek gerekli")

    from sensa_ai import models

    x = embed_samples(samples, models.SiglipEmbedder.from_pretrained(device=args.device), float(v["margin"]))
    hold = np.array([is_holdout(s, args.holdout) for s in samples])
    y = np.array([s[2] for s in samples])
    head = verify.fit(x[~hold], y[~hold])
    out = args.out or (ROOT / v["head"])
    head.save(out)
    rows = report(head, x[hold], [s for s, h in zip(samples, hold, strict=True) if h], [0.2, 0.3, 0.5, 0.7])
    keys = [k for k in rows[0] if k != "esik"]
    lines = [
        f"# Kırpım doğrulayıcı {dt.date.today()}",
        "",
        f"Eğitim {int((~hold).sum())} kırpım ({int(y[~hold].sum())} pozitif), ayrılmış {int(hold.sum())}. "
        f"Ayrılmış videolar: {', '.join(args.holdout) or 'yok'}. Baş: `{out.name}`. Yalnız sayı.",
        "",
        "| eşik | " + " | ".join(keys) + " |",
        "|---" * (len(keys) + 1) + "|",
    ]
    for r in rows:
        vals = ["-" if r[k] is None else f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys]
        lines.append(f"| {r['esik']:g} | " + " | ".join(vals) + " |")
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / f"{dt.date.today()}_dogrulayici.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
