"""Ses ipucu demosu (D-019), tamamen kansız: kırmızı kareler + sentetik sesler.

    python -m tools.audio_demo data/vekil/ses_demo.mp4

Beş sahne üretir, her birini sessiz hat (yalnız görüntü) ve sesli hat ile analiz eder, tabloyu yazar.
Görüntü skoru karenin doygunluğundan gelir (vekil dedektör, `scored=True`); "çığlık" yüksek bir ton,
"mutfak" geniş bant cızırtıdır (vekil ses sınıflandırıcısı). Gerçek modeller aynı kodu kullanır.

Emirhan bu videoyu izleyebilir (kırmızı kare, yeşil arka plan). Bulanık hali için:
    python -m tools.before_after data/vekil/ses_demo.mp4 --track data/vekil/ses_demo.blurtrack.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from sensa_ai import audio_cue, blur, config, metrics, pipeline
from sensa_ai.blurtrack import BlurTrack
from sensa_ai.detect import ColorProxyDetector

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Scene:
    name: str
    saturation: int  # 0 = kare yok
    sound: str  # "scream" | "kitchen" | "quiet"
    should_blur: bool  # demo etiketi: bu sahnede "kan" var mı


# Sıra önemli değil; sahneler arasında 3 sn sessizlik (ses penceresi + pad komşu sahneye taşmasın) var.
SCENES = [
    Scene("dizi: zayıf görüntü + çığlık", 168, "scream", True),
    Scene("dizi: orta görüntü, sessiz", 195, "quiet", True),
    Scene("yemek: orta görüntü + cızırtı", 195, "kitchen", False),
    Scene("sessiz: zayıf görüntü, ses yok", 168, "quiet", False),
    Scene("dizi: emin görüntü + mutfak sesi", 250, "kitchen", True),
]
SCENE_S, GAP_S, OBJ = 6.0, 3.0, (1.5, 4.5)  # nesne sahnenin 1.5-4.5 sn'sinde görünür
FPS, W, H, RATE = 25, 640, 360, 16000


def scene_start(i: int) -> float:
    return i * (SCENE_S + GAP_S)


def _sound(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / RATE
    if kind == "scream":
        tone = np.sin(2 * np.pi * 1800 * t) + 0.4 * np.sin(2 * np.pi * 3600 * t)
        return (0.25 * tone * (0.6 + 0.4 * np.sin(2 * np.pi * 6 * t))).astype(np.float32)
    if kind == "kitchen":
        return (0.12 * rng.standard_normal(n)).astype(np.float32)
    return (0.003 * rng.standard_normal(n)).astype(np.float32)  # oda gürültüsü, eşiğin altında


def make(out: Path) -> tuple[Path, list[metrics.Event]]:
    """Video + ses (ffmpeg ile birleştirilir). Dönüş: video yolu ve demo etiketleri."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    total = len(SCENES) * (SCENE_S + GAP_S)
    silent = out.with_name(out.stem + ".goruntu.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter.fourcc(*"mp4v"), FPS, (W, H))
    bg = np.full((H, W, 3), (70, 120, 60), np.uint8)
    bg = cv2.add(bg, rng.integers(0, 30, (H, W, 3), dtype=np.uint8))
    audio = np.zeros(int(total * RATE), np.float32)
    events: list[metrics.Event] = []
    for i, sc in enumerate(SCENES):
        a = scene_start(i)
        s0, s1 = int(a * RATE), int((a + SCENE_S) * RATE)
        audio[s0:s1] = _sound(sc.sound, s1 - s0, rng)
        if sc.should_blur:
            events.append(metrics.Event(a + OBJ[0], a + OBJ[1]))
    for k in range(int(total * FPS)):
        t = k / FPS
        f = bg.copy()
        i = int(t // (SCENE_S + GAP_S))
        local = t - scene_start(i)
        if i < len(SCENES) and OBJ[0] <= local < OBJ[1] and SCENES[i].saturation:
            hsv = np.uint8([[[0, SCENES[i].saturation, 220]]])
            bgr = tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
            x = int(W * (0.25 + 0.3 * (local - OBJ[0]) / (OBJ[1] - OBJ[0])))
            cv2.rectangle(f, (x, int(H * 0.35)), (x + 90, int(H * 0.35) + 90), bgr, -1)
        label = SCENES[i].name if i < len(SCENES) and local < SCENE_S else ""
        cv2.putText(f, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        vw.write(f)
    vw.release()
    wav = out.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("ffmpeg gerekli (brew install ffmpeg)")
    cmd = [ff, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav)]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    subprocess.run(cmd, check=True)  # noqa: S603
    silent.unlink(missing_ok=True)
    wav.unlink(missing_ok=True)
    return out, events


def scene_rows(bt_silent: BlurTrack, bt_audio: BlurTrack) -> list[dict[str, object]]:
    b = blur.timing(config.load()["blur"])

    def blurred(bt: BlurTrack, i: int) -> bool:
        a = scene_start(i)
        return any(bt.active(float(t), **b) for t in np.arange(a + OBJ[0], a + OBJ[1], 0.1))

    rows = []
    for i, sc in enumerate(SCENES):
        s, a = blurred(bt_silent, i), blurred(bt_audio, i)
        rows.append({"sahne": sc.name, "olmalı": sc.should_blur, "sessiz_hat": s, "sesli_hat": a})
    return rows


def run(video: Path, events: list[metrics.Event]) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    a = config.load()["audio"]
    prompts = config.load_prompts()["proxy"]
    cand = pipeline.analyze(video, ColorProxyDetector(scored=True, min_score=a["candidate_threshold"]), prompts)
    silent = audio_cue.without_audio(cand, float(a["keep_threshold"]))
    clf = audio_cue.ToneNoiseProxyClassifier(rate=int(a["rate"]))
    fused, tl, decisions = audio_cue.run(video, cand, clf)
    fused.save(video.with_suffix(".blurtrack.json"))
    silent.save(video.with_suffix(".sessiz.blurtrack.json"))
    audio_cue.save_report(video.with_suffix(".audiocue.json"), tl, decisions, "proxy")
    timing = blur.timing(config.load()["blur"])
    scores = {
        "sessiz_hat": metrics.event_scores(silent, events, **timing),
        "sesli_hat": metrics.event_scores(fused, events, **timing),
    }
    return scene_rows(silent, fused), scores


def _yn(v: object) -> str:
    return "bulanık" if v else "-"


def report(rows: list[dict[str, object]], scores: dict[str, dict[str, float]]) -> str:
    lines = ["| Sahne | Olmalı | Sessiz hat | Sesli hat |", "|---|---|---|---|"]
    for r in rows:
        ok = "✓" if r["sesli_hat"] == r["olmalı"] else "✗"
        lines.append(f"| {r['sahne']} | {_yn(r['olmalı'])} | {_yn(r['sessiz_hat'])} | {_yn(r['sesli_hat'])} {ok} |")
    lines += ["", "| Hat | Olay yakalama | Olay dışı bulanık sn |", "|---|---|---|"]
    for k, s in scores.items():
        lines.append(f"| {k} | {s['olay_yakalama']:.2f} | {s['olay_disi_bulanik_s']:.1f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path, nargs="?", default=ROOT / "data" / "vekil" / "ses_demo.mp4")
    ap.add_argument("--save", action="store_true", help="tabloyu experiments/results altına da yaz")
    args = ap.parse_args()
    video, events = make(args.out)
    rows, scores = run(video, events)
    text = report(rows, scores)
    print(text)
    if args.save:
        out = ROOT / "experiments" / "results" / f"{dt.date.today()}_ses_ipucu_vekil_demo.md"
        out.write_text(f"# Ses ipucu vekil demosu {dt.date.today()}\n\n`python -m tools.audio_demo`\n\n{text}\n")
        print("->", out)


if __name__ == "__main__":
    main()
