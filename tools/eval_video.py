"""H-01 / H-02 video düzeyi: ortağın zaman etiketleri ile iz karşılaştırması. Yalnız sayı.

Etiket biçimi (data/etiket/<video adı>.json), görüntü içermez; ortak videoyu izleyip yazar:
    {"video": "film.mp4", "events": [{"start": 41.0, "end": 46.0, "box": [0.1, 0.5, 0.4, 0.9]}]}
`box` isteğe bağlı (kaba bölge, normalize). Negatif videolar için "events": [] yazılır.

    python -m tools.eval_video data/pozitif/videolar --labels data/etiket            # analiz + değerlendirme
    python -m tools.eval_video data/pozitif/videolar --labels data/etiket --tracks   # var olan .blurtrack.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from sensa_ai import blur, config, metrics, pipeline
from sensa_ai.blurtrack import BlurTrack

ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


def load_events(path: Path) -> list[metrics.Event]:
    d = json.loads(path.read_text(encoding="utf-8"))
    return [metrics.Event(e["start"], e["end"], tuple(e["box"]) if e.get("box") else None) for e in d["events"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", type=Path)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--tracks", action="store_true", help="analiz etme, videonun yanındaki .blurtrack.json'u kullan")
    ap.add_argument("--proxy", action="store_true", help="vekil dedektör (vekil videolar için)")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    config.set_profile(args.profile)
    b = config.load()["blur"]

    detector = tracker = None
    rows = []
    for v in sorted(p for p in args.videos.rglob("*") if p.suffix.lower() in VIDEO_EXT):
        lab = args.labels / f"{v.stem}.json"
        if not lab.exists():
            continue
        if args.tracks:
            bt = BlurTrack.load(v.with_suffix(".blurtrack.json"))
        else:
            if detector is None:
                if args.proxy:
                    from sensa_ai.detect import ColorProxyDetector

                    detector = ColorProxyDetector()
                else:
                    from sensa_ai import models
                    from sensa_ai.tracking import Tracker

                    detector = models.build_default(device=args.device)
                    tracker = Tracker.from_pretrained(device=args.device)
            prompts = config.load_prompts()["proxy" if args.proxy else "positive"]
            bt = pipeline.analyze(v, detector, prompts, tracker)
            bt.save(v.with_suffix(".blurtrack.json"))
        s = metrics.event_scores(bt, load_events(lab), **blur.timing(b))
        rows.append({"video": v.name, **s})
        print(v.name, s)

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today()}_eval_video"
    (out / f"{stem}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    if rows:
        keys = list(rows[0])
        lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
        for r in rows:
            lines.append(
                "| " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |"
            )
        (out / f"{stem}.md").write_text(f"# Video değerlendirmesi {dt.date.today()}\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
