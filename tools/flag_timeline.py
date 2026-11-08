"""Bir videoda modelin nerede "kan" dediğini dakika dakika sayar; görüntü göstermez, kare yazmaz.

    python -m tools.flag_timeline data/pozitif/videolar/dex1-1.mp4
    python -m tools.flag_timeline dex1-1.mp4 --detector rfdetr:models/v1.pth --detector rfdetr:models/v2b.pth
    python -m tools.flag_timeline dex1-1.mp4 --ranges "12-14,38:10-41"

Neden: fotoğraf setindeki yanlış alarm oranı diziye taşınmadı (FA-04). Bu araç dizide iki modeli aynı
karelerde karşılaştırır ve izleyen kişinin kabaca bildiği gerçek kan dakikaları (`--ranges`) verilirse
iki sayı çıkarır:
  * aralık İÇİ: işaretli örnek oranı (kapsama vekili) ve en az bir kez yakalanan aralık oranı
  * aralık DIŞI: dakikada yanlış alarm olayı (art arda işaretli örnekler tek olay; hedef
    requirements.false_alarm_per_min) ve işaretli saniye oranı
Sonuç: experiments/results/<tarih>_isaret_<video>.md (+ .json). Yalnız sayı.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import cv2

from sensa_ai import config
from sensa_ai.detect import Detection, Detector

ROOT = Path(__file__).resolve().parents[1]
EVENT_GAP_S = 2.0  # aradaki boşluk bundan kısaysa iki işaret aynı yanlış alarm olayı sayılır


class GateJudge:
    """SigLIP kapısını hakem olarak kullanır: karede "kan" istemleri ketçap, boya, kırmızı kumaş gibi
    benzerlerinden daha olası mı. `Detector` arayüzü: eşik geçerse tek "kare" tespiti döner."""

    def __init__(self, gate: Any, negative: list[str], threshold: float) -> None:
        self.gate, self.negative, self.threshold = gate, negative, threshold

    def __call__(self, frame: Any, prompts: list[str]) -> list[Detection]:
        sc = float(self.gate.score(frame, prompts, self.negative))
        return [Detection("kare", (0.0, 0.0, 1.0, 1.0), sc)] if sc >= self.threshold else []


def sample_flags(
    video: Path,
    detectors: list[Detector],
    fps: float,
    min_score: float,
    prompts: list[str],
    judges: list[Detector] | None = None,
) -> tuple[list[float], list[list[float]], float]:
    """Her örnek anı için her dedektörün en yüksek skoru (tespit yoksa 0). Kareler yalnız bellekte.

    `judges` yalnız ilk dedektörün işaretlediği anlarda çalışır (bağımsız ikinci görüş, yavaş modeller için);
    skorları dedektörlerden sonra gelir, işaretsiz anlarda 0."""
    judges = judges or []
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / vfps
    step = max(1, round(vfps / fps))
    times: list[float] = []
    scores: list[list[float]] = [[] for _ in detectors + judges]
    i = 0
    while True:
        if i % step:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        times.append(i / vfps)
        for k, det in enumerate(detectors):
            best = max((d.score for d in det(frame, prompts)), default=0.0)
            scores[k].append(best if best >= min_score else 0.0)
        first_flagged = bool(detectors) and scores[0][-1] > 0
        for k, jd in enumerate(judges, start=len(detectors)):
            best = max((d.score for d in jd(frame, prompts)), default=0.0) if first_flagged else 0.0
            scores[k].append(best)
        i += 1
    cap.release()
    return times, scores, duration


def summarize(
    times: list[float], flags: list[bool], duration: float, ranges: list[tuple[float, float]] | None, step_s: float
) -> dict[str, Any]:
    minutes = [0] * (int(duration // 60) + 1)
    for t, f in zip(times, flags, strict=True):
        if f:
            minutes[min(int(t // 60), len(minutes) - 1)] += 1
    out: dict[str, Any] = {
        "isaretli_ornek": int(sum(flags)),
        "ornek": len(flags),
        "isaretli_oran": sum(flags) / max(1, len(flags)),
        "dakika": minutes,
    }
    if ranges is None:
        return out

    def inside(t: float) -> bool:
        return any(a <= t <= b for a, b in ranges)

    ins = [f for t, f in zip(times, flags, strict=True) if inside(t)]
    outs = [(t, f) for t, f in zip(times, flags, strict=True) if not inside(t)]
    events, last = 0, -1e9
    for t, f in outs:
        if f:
            if t - last > EVENT_GAP_S + step_s:
                events += 1
            last = t
    out_minutes = max(1e-9, len(outs) * step_s / 60)
    caught = sum(1 for a, b in ranges if any(f for t, f in zip(times, flags, strict=True) if a <= t <= b))
    out.update(
        {
            "ici_isaretli_oran": sum(ins) / max(1, len(ins)),
            "yakalanan_aralik": f"{caught}/{len(ranges)}",
            "disi_dakika": round(out_minutes, 2),
            "disi_yanlis_alarm_olayi": events,
            "disi_dakikada_yanlis_alarm": events / out_minutes,
            "disi_isaretli_oran": sum(f for _, f in outs) / max(1, len(outs)),
        }
    )
    return out


def candidate_ranges(minutes: list[int], min_s: int) -> str:
    """Hakemlerin de kan dediği saniyesi `min_s` ve üstü olan dakikalar, art arda olanlar birleşik:
    "38-40,49-51" (relabel_events / --ranges biçimi, dakika). Kaba bir öneri; insan bilgisi değil."""
    out, start = [], None
    for m, n in enumerate([*minutes, 0]):
        if n >= min_s and start is None:
            start = m
        elif n < min_s and start is not None:
            out.append(f"{start}-{m}")
            start = None
    return ",".join(out)


def main() -> None:
    from tools.relabel_events import parse_ranges

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--detector", action="append", default=None, help="birden çok verilebilir; yoksa varsayılan")
    ap.add_argument("--proxy", action="store_true", help="modelsiz vekil dedektör (kansız deneme)")
    ap.add_argument("--fps", type=float, default=1.0, help="saniyede örnek (1 = her saniye)")
    ap.add_argument("--min-score", type=float, default=None, help="yoksa params detect.seg_threshold")
    ap.add_argument("--ranges", default=None, help='gerçek kan aralıkları, ör. "12-14,38:10-41"')
    ap.add_argument(
        "--judge",
        action="append",
        default=[],
        help="ilk dedektörün işaretlediği anlarda ikinci görüş: siglip ya da bir dedektör adı (ör. owlv2_base)",
    )
    ap.add_argument("--judge-min-s", type=int, default=5, help="aday kan dakikası için en az ortak saniye")
    ap.add_argument("--device", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--results", type=Path, default=ROOT / "experiments" / "results")
    args = ap.parse_args()
    config.set_profile(args.profile)
    p = config.load()
    min_score = float(p["detect"]["seg_threshold"]) if args.min_score is None else args.min_score

    names: list[str]
    dets: list[Detector]
    if args.proxy:
        from sensa_ai.detect import ColorProxyDetector

        names, dets = ["vekil"], [ColorProxyDetector()]
        prompts = config.load_prompts()["proxy"]
    else:
        from sensa_ai import models

        specs = args.detector or [None]
        dets = []
        for spec in specs:
            d = models.make_detector(spec, device=args.device)
            d.box_threshold = min(d.box_threshold, min_score)  # eşik burada uygulanır
            dets.append(d)
        names = [
            models.detector_name(d) if s is None else s.removeprefix("rfdetr:")
            for d, s in zip(dets, specs, strict=True)
        ]
        prompts = config.load_prompts()["positive"]

    judges: list[Detector] = []
    for spec in args.judge:
        from sensa_ai import models

        if spec == "siglip":
            negative = config.load_prompts()["negative"]
            gate = models.SiglipGate.from_pretrained(device=args.device)
            judges.append(GateJudge(gate, negative, float(p["gate"]["threshold"])))
        else:
            judges.append(models.make_detector(spec, device=args.device))
        names.append(f"hakem:{spec}")
    names = [n if names.index(n) == k else f"{n} ({k + 1})" for k, n in enumerate(names)]  # aynı ad iki kez
    ranges = parse_ranges(args.ranges) if args.ranges else None
    times, scores, duration = sample_flags(args.video, dets, args.fps, min_score, prompts, judges)
    rows = {
        n: summarize(times, [s > 0 for s in sc], duration, ranges, 1 / args.fps)
        for n, sc in zip(names, scores, strict=True)
    }
    suggestion = ""
    if judges:  # ilk dedektör ve bütün hakemler birlikte "kan" diyorsa
        agree = [all(sc[j] > 0 for sc in [scores[0], *scores[len(dets) :]]) for j in range(len(times))]
        rows["ortak"] = summarize(times, agree, duration, ranges, 1 / args.fps)
        names.append("ortak")
        suggestion = candidate_ranges(rows["ortak"]["dakika"], args.judge_min_s)
        rows["ortak"]["aday_kan_dakikalari"] = suggestion

    lines = [
        f"# İşaret zaman çizelgesi {dt.date.today()}",
        "",
        f"Video: `{args.video.name}`, {duration / 60:.1f} dk, "
        f"saniyede {args.fps:g} örnek, eşik {min_score:g}."
        + (f" Gerçek kan aralıkları: {args.ranges}" if ranges else ""),
        "",
    ]
    keys = [k for k in next(iter(rows.values())) if k not in ("dakika", "aday_kan_dakikalari")]
    lines += ["| ölçüt | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for k in keys:
        vals = [rows[n][k] for n in names]
        lines.append(f"| {k} | " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals) + " |")
    lines += ["", "Dakika başına işaretli örnek (yalnız işaretli dakikalar):", ""]
    lines += ["| dakika | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for m in range(len(next(iter(rows.values()))["dakika"])):
        vals = [rows[n]["dakika"][m] for n in names]
        if any(vals):
            lines.append(f"| {m} | " + " | ".join(str(v) for v in vals) + " |")
    if judges:
        lines += [
            "",
            f"Hakemlerle ortak aday kan dakikaları (en az {args.judge_min_s} ortak saniye): `{suggestion or 'yok'}`.",
            "Bu bir model önerisi, insan bilgisi değil: relabel_events'e vermeden önce kaba doğrulama gerekir.",
        ]
    args.results.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today()}_isaret_{args.video.stem}"
    (args.results / f"{stem}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    (args.results / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
