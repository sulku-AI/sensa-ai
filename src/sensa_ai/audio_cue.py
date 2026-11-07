"""Ses ipucu (D-019, demo): çığlık / bağırış / silah sesi görüntüdeki bölgenin kan olduğuna güveni artırır,
mutfak sesleri (kızartma, doğrama, tencere) azaltır.

Soru "model ne kadar iyi" değil, "dizi izlerken doğru anda bulanıklaşıyor mu, yemek programında
boşuna bulanıklaşmıyor mu". Bu yüzden ses tek başına bulanıklık AÇMAZ (görüntüde aday bölge
gerekir) ve görüntü çok eminse ses onu KAPATAMAZ (D-001: kaçırmak yanlış alarmdan pahalı).

Akış:
    ses (16 kHz mono) -> pencere pencere sınıflandırıcı -> zaman çizgisi (sıkıntı, mutfak)
    görüntü izleri (aday eşiği daha düşük) + zaman çizgisi -> `fuse` -> tutulan izler + rapor

Sınıflandırıcılar:
    AstClassifier          AudioSet'te eğitilmiş AST (transformers, BSD-3), gerçek kullanım
    ToneNoiseProxyClassifier  model gerektirmeyen vekil: yüksek ton = "çığlık", geniş bant gürültü = "cızırtı".
                              Gerçek ses tanıma DEĞİLDİR; yalnız boru hattı ve kansız demo içindir (D-005 gibi)

Sayılar: params.yaml `audio`. Sınıf adları: prompts.yaml `audio_*`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import config
from .blurtrack import BlurTrack, Track

GROUPS = ("vocal", "violence", "kitchen")  # prompts.yaml: audio_vocal, audio_violence, audio_kitchen


class AudioClassifier(Protocol):
    rate: int

    def __call__(self, wave: np.ndarray) -> dict[str, float]:
        """wave: mono float32, `rate` Hz. Dönüş: grup adı -> 0..1 olasılık (GROUPS)."""
        ...


# ---------------------------------------------------------------- ses okuma


def load_mono(video: Path, rate: int = 16000) -> np.ndarray:
    """Videonun sesini mono float32 olarak okur (ffmpeg, yoksa PyAV). Ses yoksa boş dizi."""
    ff = shutil.which("ffmpeg")
    if ff:
        cmd = [ff, "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]
        r = subprocess.run(cmd, capture_output=True, check=False)  # noqa: S603
        if r.returncode == 0:
            return np.frombuffer(r.stdout, np.float32).copy()
    try:
        import av
    except ImportError as e:
        raise RuntimeError("ses okumak için ffmpeg ya da PyAV gerekli") from e
    with av.open(str(video)) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        rs = av.AudioResampler(format="flt", layout="mono", rate=rate)
        parts = [r.to_ndarray().reshape(-1) for f in c.decode(c.streams.audio[0]) for r in rs.resample(f)]
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


# ---------------------------------------------------------------- zaman çizgisi


@dataclass
class AudioWindow:
    t0: float
    t1: float
    vocal: float
    violence: float
    kitchen: float


@dataclass
class AudioTimeline:
    windows: list[AudioWindow] = field(default_factory=list)

    def _in(self, a: float, b: float) -> list[AudioWindow]:
        return [w for w in self.windows if w.t1 > a and w.t0 < b]

    def distress(self, a: float, b: float, violence_weight: float = 1.0) -> float:
        """[a, b] aralığında en yüksek sıkıntı olasılığı (çığlık/bağırış ya da ağırlıklı şiddet sesi)."""
        ws = self._in(a, b)
        return max((max(w.vocal, violence_weight * w.violence) for w in ws), default=0.0)

    def kitchen(self, a: float, b: float) -> float:
        """[a, b] aralığında ortalama mutfak olasılığı (tek bir tencere sesi değil, sahnenin geneli)."""
        ws = self._in(a, b)
        return float(np.mean([w.kitchen for w in ws])) if ws else 0.0


def analyze_audio(wave: np.ndarray, clf: AudioClassifier, window_s: float, hop_s: float) -> AudioTimeline:
    """Kayan pencere. Son pencere kısa kalırsa sıfırla doldurulur (sınıflandırıcı sabit uzunluk ister)."""
    rate = clf.rate
    n, hop = int(window_s * rate), max(1, int(hop_s * rate))
    tl = AudioTimeline()
    if wave.size == 0:
        return tl
    for s in range(0, max(1, wave.size - n + hop), hop):
        seg = wave[s : s + n]
        if seg.size < n:
            seg = np.pad(seg, (0, n - seg.size))
        p = clf(seg)
        tl.windows.append(AudioWindow(s / rate, (s + n) / rate, p["vocal"], p["violence"], p["kitchen"]))
    return tl


# ---------------------------------------------------------------- birleştirme


@dataclass
class TrackDecision:
    label: str
    start: float
    end: float
    visual: float  # izin en yüksek görüntü skoru
    distress: float  # 0..1, `distress_full_at` ile ölçeklenmiş
    kitchen: float  # 0..1, `kitchen_full_at` ile ölçeklenmiş
    fused: float
    keep: bool
    reason: str


def _scaled(p: float, full_at: float) -> float:
    return float(min(1.0, max(0.0, p / full_at))) if full_at > 0 else float(p > 0)


def decide(track: Track, tl: AudioTimeline, a: dict[str, Any]) -> TrackDecision:
    """Tek iz için karar. Kurallar (sırayla):
    1. görüntü >= sure_threshold            -> tut (ses ne derse desin)
    2. birleşik = görüntü + boost*sıkıntı - ceza*mutfak >= keep_threshold -> tut
    3. değilse bırak
    Sesin nötr olduğu durumda (ikisi de 0) sonuç sessiz hatla aynıdır: görüntü >= keep_threshold.
    """
    v = max(k.score for k in track.keyframes)
    pad = float(a["pad_s"])
    lo, hi = track.start - pad, track.end + pad
    d = _scaled(tl.distress(lo, hi, float(a["violence_weight"])), float(a["distress_full_at"]))
    kt = _scaled(tl.kitchen(lo, hi), float(a["kitchen_full_at"]))
    fused = v + float(a["distress_boost"]) * d - float(a["kitchen_penalty"]) * kt
    keep_t = float(a["keep_threshold"])
    if v >= float(a["sure_threshold"]):
        keep, why = True, "görüntü çok emin"
    elif fused >= keep_t:
        keep = True
        why = (
            "ses destekledi"
            if v < keep_t
            else ("görüntü yeterli" if kt == 0 else "görüntü yeterli, mutfak sesine rağmen")
        )
    else:
        keep = False
        why = "mutfak sesi düşürdü" if v >= keep_t else "görüntü zayıf, ses desteği yok"
    return TrackDecision(track.label, track.start, track.end, v, d, kt, fused, keep, why)


def fuse(
    bt: BlurTrack, tl: AudioTimeline, params: dict[str, Any] | None = None
) -> tuple[BlurTrack, list[TrackDecision]]:
    """Görüntü izlerini sesle süzer. Dönüş: yalnız tutulan izlerden yeni BlurTrack + her iz için karar."""
    a = params or config.load()["audio"]
    decisions = [decide(tr, tl, a) for tr in bt.tracks if tr.keyframes]
    kept = [tr for tr, dc in zip([t for t in bt.tracks if t.keyframes], decisions, strict=True) if dc.keep]
    return BlurTrack(bt.video, bt.duration_s, bt.analyzed_until_s, kept), decisions


def run(video: Path, bt: BlurTrack, clf: AudioClassifier) -> tuple[BlurTrack, AudioTimeline, list[TrackDecision]]:
    """Videonun sesini okur, zaman çizgisini çıkarır, izleri süzer."""
    a = config.load()["audio"]
    tl = analyze_audio(load_mono(video, clf.rate), clf, float(a["window_s"]), float(a["hop_s"]))
    fused, decisions = fuse(bt, tl, a)
    return fused, tl, decisions


def without_audio(bt: BlurTrack, keep_threshold: float) -> BlurTrack:
    """Karşılaştırma için sessiz hat: yalnız görüntü skoru >= eşik olan izler."""
    kept = [tr for tr in bt.tracks if tr.keyframes and max(k.score for k in tr.keyframes) >= keep_threshold]
    return BlurTrack(bt.video, bt.duration_s, bt.analyzed_until_s, kept)


def save_report(path: Path, tl: AudioTimeline, decisions: list[TrackDecision], model: str) -> None:
    d = {"model": model, "windows": [asdict(w) for w in tl.windows], "decisions": [asdict(x) for x in decisions]}
    path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- sınıflandırıcılar


class ToneNoiseProxyClassifier:
    """Model gerektirmeyen vekil (D-005'in ses karşılığı). Gerçek çığlık/mutfak tanıma DEĞİLDİR.

    vocal   : 900-3500 Hz bandında baskın tek ton (spektral tepe / bant ortalaması yüksek), yeterince yüksek sesle
    kitchen : geniş bant gürültü (spektral düzlük yüksek), yeterince yüksek sesle ("cızırtı")
    violence: hep 0
    """

    def __init__(self, rate: int = 16000, min_rms: float = 0.02) -> None:
        self.rate, self.min_rms = rate, min_rms

    def __call__(self, wave: np.ndarray) -> dict[str, float]:
        x = wave.astype(np.float64)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if rms < self.min_rms:
            return {"vocal": 0.0, "violence": 0.0, "kitchen": 0.0}
        spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) + 1e-12
        freqs = np.fft.rfftfreq(x.size, 1 / self.rate)
        band = spec[(freqs >= 900) & (freqs <= 3500)]
        peakiness = float(band.max() / band.mean()) if band.size else 0.0
        audible = spec[(freqs >= 100) & (freqs <= 7000)]
        flatness = float(np.exp(np.mean(np.log(audible))) / np.mean(audible))
        loud = min(1.0, rms / (4 * self.min_rms))
        vocal = loud * min(1.0, max(0.0, (peakiness - 20) / 80))
        kitchen = loud * min(1.0, max(0.0, (flatness - 0.3) / 0.4))
        return {"vocal": vocal, "violence": 0.0, "kitchen": kitchen}


class AstClassifier:
    """Audio Spectrogram Transformer, AudioSet 527 sınıf (sigmoid). Grup olasılığı = gruptaki en yüksek sınıf.

    Sınıf adları prompts.yaml `audio_vocal` / `audio_violence` / `audio_kitchen`; modelde olmayan ad
    yüklemede uyarı olarak yazılır (adlar AudioSet ontolojisinden, bulut oturumunda doğrulanamadı).
    """

    rate = 16000

    def __init__(self, model: Any, extractor: Any, device: str, groups: dict[str, list[int]]) -> None:
        self.model, self.extractor, self.device, self.groups = model, extractor, device, groups

    @staticmethod
    def group_indices(id2label: dict[int, str], names: dict[str, list[str]]) -> tuple[dict[str, list[int]], list[str]]:
        by_name = {v.lower(): int(k) for k, v in id2label.items()}
        out, missing = {}, []
        for g in GROUPS:
            ids = []
            for n in names.get(f"audio_{g}", []):
                if n.lower() in by_name:
                    ids.append(by_name[n.lower()])
                else:
                    missing.append(n)
            out[g] = ids
        return out, missing

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, device: str | None = None) -> AstClassifier:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        from . import device as dev
        from .models import resolve

        d = dev.pick(device)
        mid = model_id or resolve("audio")
        model = ASTForAudioClassification.from_pretrained(mid).to(d).eval()
        return cls.from_model(model, ASTFeatureExtractor.from_pretrained(mid), d)

    @classmethod
    def from_model(cls, model: Any, extractor: Any, device: str) -> AstClassifier:
        groups, missing = cls.group_indices(model.config.id2label, config.load_prompts())
        if missing:
            print(f"[audio] modelde olmayan sınıf adları (yok sayıldı): {missing}")
        return cls(model, extractor, device, groups)

    def __call__(self, wave: np.ndarray) -> dict[str, float]:
        import torch

        inputs = self.extractor(wave, sampling_rate=self.rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = self.model(**inputs).logits[0].float().sigmoid().cpu().numpy()
        return {g: float(probs[ids].max()) if ids else 0.0 for g, ids in self.groups.items()}


def make_classifier(name: str | None = None, device: str | None = None) -> AudioClassifier:
    """ "proxy" -> vekil; diğerleri models.yaml `audio` adı ya da HF kimliği."""
    if name == "proxy":
        return ToneNoiseProxyClassifier(rate=int(config.load()["audio"]["rate"]))
    return AstClassifier.from_pretrained(name, device=device)
