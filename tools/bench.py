"""H-03 hız ölçümü (ana hedef M5 / MPS, D-012). Yalnız sentetik, kansız karelerle çalışır.

    python -m tools.bench                       # varsayılan modeller, otomatik cihaz
    python -m tools.bench --detector owlv2_base --res 640 --frames 20

Sonuç: experiments/results/<tarih>_bench_<cihaz>.md (+ .json). Hesap:
    analiz süresi / video saniyesi = sample_fps * (kapı + p_kapı_geçer * dedektör)
    realtime_factor = 1 / bunun değeri  (>= requirements.realtime_factor olmalı)
Takip (SAM 2 / EdgeTAM) ayrıca ölçülür; kare başı maliyeti track.fps ile çarpılır.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import time
from pathlib import Path

import numpy as np

from sensa_ai import config, models

ROOT = Path(__file__).resolve().parents[1]


def synthetic_frames(n: int, res: int) -> list[np.ndarray]:
    import cv2

    rng = np.random.default_rng(0)
    h, w = res, int(res * 16 / 9)
    out = []
    for i in range(n):
        f = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        cv2.circle(f, (w // 4 + i * 5, h // 2), h // 8, (0, 0, 220), -1)  # vekil: kırmızı balon
        out.append(f)
    return out


def timed(fn, frames, warmup: int = 2) -> float:  # type: ignore[no-untyped-def]
    for f in frames[:warmup]:
        fn(f)
    t0 = time.perf_counter()
    for f in frames[warmup:]:
        fn(f)
    return (time.perf_counter() - t0) / max(1, len(frames) - warmup)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--res", type=int, default=720, help="kare yüksekliği (piksel)")
    ap.add_argument("--gate-pass", type=float, default=0.2, help="kapıyı geçen kare oranı varsayımı")
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    p = config.load()
    prompts = config.load_prompts()
    frames = synthetic_frames(args.frames, args.res)
    gate = models.SiglipGate.from_pretrained(device=args.device)
    det = models.TextDetector.from_pretrained(models.resolve("detect", args.detector), device=args.device)
    t_gate = timed(lambda f: gate.score(f, prompts["positive"], prompts["negative"]), frames)
    t_det = timed(lambda f: det(f, prompts["proxy"]), frames)

    res: dict[str, object] = {
        "tarih": str(dt.date.today()),
        "makine": f"{platform.system()} {platform.machine()} {platform.processor()}",
        "cihaz": det.device,
        "kare": f"{frames[0].shape[1]}x{frames[0].shape[0]}",
        "kapi_model": gate.model.name_or_path,
        "kapi_ms": round(t_gate * 1000, 1),
        "dedektor_model": det.model.name_or_path,
        "dedektor_ms": round(t_det * 1000, 1),
    }
    sfps = float(p["lookahead"]["sample_fps"])
    load = sfps * (t_gate + args.gate_pass * t_det)

    if not args.no_track:
        from sensa_ai import tracking

        tr = tracking.Tracker.from_pretrained(device=args.device)
        t_track = tr.bench(frames, (0.2, 0.35, 0.4, 0.65))
        res["takip_model"] = tr.model_id
        res["takip_ms_kare"] = round(t_track * 1000, 1)
        # takip yalnız bölge varken ve track.fps hızında çalışır
        tfps = float(p["track"]["fps"])
        res["takip_fps"] = tfps
        load += args.gate_pass * tfps * t_track

    res["analiz_s_video_s"] = round(load, 3)
    res["realtime_factor"] = round(1 / load, 2) if load else float("inf")
    res["hedef_realtime_factor"] = p["requirements"]["realtime_factor"]
    res["gecti"] = bool(res["realtime_factor"] >= p["requirements"]["realtime_factor"])  # type: ignore[operator]

    out = ROOT / "experiments" / "results"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today()}_bench_{det.device}"
    (out / f"{stem}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    lines = [f"# Hız ölçümü {res['tarih']} ({res['cihaz']})", "", "| ölçüt | değer |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in res.items()]
    (out / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
