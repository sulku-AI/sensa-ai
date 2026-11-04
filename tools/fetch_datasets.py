"""Pozitif değerlendirme verisini GÖRMEDEN indirir (D-001, D-004).

    export ROBOFLOW_API_KEY=...        # roboflow.com > Settings > API Keys (o sayfada görüntü yok)
    python -m tools.fetch_datasets --list
    python -m tools.fetch_datasets blood_seg_joost blood_seg_backup

Görüntüler doğrudan data/pozitif/<ad>/ altına COCO segmentasyon biçiminde yazılır. Bu betik
hiçbir görüntüyü açmaz, göstermez, küçük resim üretmez; yalnız sayıları yazar. Emirhan bu
klasörü Finder'da açmamalı (Finder küçük resim gösterir): `data/pozitif` kilitlenir (chmod 700)
ve içine .metadata_never_index konur (Spotlight dizinlemesin).

Kaynaklar ve lisanslar: docs/MODEL_ARASTIRMA.md §5. Lisansı belirsiz set listede yok.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POS = ROOT / "data" / "pozitif"

# ad: (workspace, project, lisans)
DATASETS = {
    "blood_seg_joost": ("joost-bos-rfw3p", "blood-segmentation-y0ygo", "CC BY 4.0"),
    "blood_seg_backup": ("backupgeneral", "blood_segmentation_2", "CC BY 4.0"),
}


def lock(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / ".metadata_never_index").touch()
    os.chmod(folder, 0o700)


def count_coco(folder: Path) -> dict[str, int]:
    out = {"images": 0, "annotations": 0}
    for ann in folder.rglob("_annotations.coco.json"):
        d = json.loads(ann.read_text())
        out["images"] += len(d.get("images", []))
        out["annotations"] += len(d.get("annotations", []))
    return out


def fetch(name: str, api_key: str) -> dict[str, int]:
    from roboflow import Roboflow  # pip install roboflow

    ws, proj, lic = DATASETS[name]
    project = Roboflow(api_key=api_key).workspace(ws).project(proj)
    versions = project.versions()
    if not versions:
        raise SystemExit(f"{name}: sürüm yok")
    latest = max(versions, key=lambda v: int(str(v.version).split("/")[-1]))
    dest = POS / name
    lock(POS)
    latest.download("coco-segmentation", location=str(dest), overwrite=True)
    (dest / "LISANS.txt").write_text(f"{ws}/{proj} · {lic} · Roboflow Universe\n", encoding="utf-8")
    return count_coco(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.names:
        for n, (ws, p, lic) in DATASETS.items():
            print(f"{n:20s} {ws}/{p}  [{lic}]")
        return
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY ortam değişkeni yok")
    for n in args.names:
        print(n, fetch(n, key))


if __name__ == "__main__":
    main()
