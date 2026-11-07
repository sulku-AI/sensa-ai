"""Ses ipucu (D-019): yalnız sentetik ses (ton, gürültü) ve kırmızı kareler."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from sensa_ai import audio_cue, config
from sensa_ai.audio_cue import AudioTimeline, AudioWindow, ToneNoiseProxyClassifier
from sensa_ai.blurtrack import BlurTrack, Keyframe, Track

RATE = 16000


def _track(t0: float, t1: float, score: float) -> Track:
    return Track("x", [Keyframe(t0, (0.1, 0.1, 0.3, 0.3), score), Keyframe(t1, (0.2, 0.1, 0.4, 0.3), score)])


def _tl(vocal: float = 0.0, kitchen: float = 0.0, violence: float = 0.0) -> AudioTimeline:
    return AudioTimeline([AudioWindow(float(s), s + 3.0, vocal, violence, kitchen) for s in range(0, 20)])


def _a() -> dict:
    return config.load()["audio"]


def test_neutral_audio_equals_visual_only() -> None:
    """Ses nötrse sonuç sessiz hatla aynı: eşik üstü tutulur, altı bırakılır."""
    a = _a()
    bt = BlurTrack("v", 20, 20, [_track(1, 3, 0.2), _track(5, 7, 0.4), _track(9, 11, 0.9)])
    fused, _ = audio_cue.fuse(bt, _tl(), a)
    silent = audio_cue.without_audio(bt, a["keep_threshold"])
    assert [t.start for t in fused.tracks] == [t.start for t in silent.tracks] == [5, 9]


def test_scream_rescues_weak_candidate() -> None:
    dc = audio_cue.decide(_track(1, 3, 0.18), _tl(vocal=0.5), _a())
    assert dc.keep and dc.reason == "ses destekledi"


def test_kitchen_suppresses_middle_but_never_sure() -> None:
    a = _a()
    mid = audio_cue.decide(_track(1, 3, 0.45), _tl(kitchen=0.6), a)
    sure = audio_cue.decide(_track(1, 3, a["sure_threshold"]), _tl(kitchen=1.0), a)
    assert not mid.keep and mid.reason == "mutfak sesi düşürdü"
    assert sure.keep  # D-001: emin görüntüyü ses kapatamaz


def test_audio_alone_never_creates_blur() -> None:
    """Görüntüde aday yoksa çığlık bulanıklık açmaz (iz yok -> iz yok)."""
    fused, decisions = audio_cue.fuse(BlurTrack("v", 20, 20, []), _tl(vocal=1.0), _a())
    assert fused.tracks == [] and decisions == []


def test_pad_reaches_scream_just_before_track() -> None:
    a = _a()
    tl = AudioTimeline([AudioWindow(0, 3, 0.9, 0, 0)])
    near = audio_cue.decide(_track(3 + a["pad_s"] - 0.5, 8, 0.2), tl, a)
    far = audio_cue.decide(_track(3 + a["pad_s"] + 0.5, 8, 0.2), tl, a)
    assert near.keep and not far.keep


def test_violence_weighted() -> None:
    tl = _tl(violence=1.0)
    assert tl.distress(0, 5, violence_weight=0.5) == pytest.approx(0.5)


def _tone(seconds: float, f: float = 1800.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_proxy_classifier_tone_noise_quiet() -> None:
    c = ToneNoiseProxyClassifier(RATE)
    rng = np.random.default_rng(0)
    tone, noise = c(_tone(3)), c((0.12 * rng.standard_normal(3 * RATE)).astype(np.float32))
    quiet = c((0.003 * rng.standard_normal(3 * RATE)).astype(np.float32))
    assert tone["vocal"] > 0.9 and tone["kitchen"] < 0.1
    assert noise["kitchen"] > 0.9 and noise["vocal"] < 0.1
    assert quiet == {"vocal": 0.0, "violence": 0.0, "kitchen": 0.0}


def test_analyze_audio_windows_cover_wave() -> None:
    wave = np.concatenate([np.zeros(4 * RATE, np.float32), _tone(3)])
    tl = audio_cue.analyze_audio(wave, ToneNoiseProxyClassifier(RATE), window_s=3.0, hop_s=1.0)
    assert tl.windows[0].t0 == 0 and tl.windows[-1].t1 >= 7.0
    assert tl.distress(0, 2) == 0.0 and tl.distress(5, 7) > 0.9
    assert audio_cue.analyze_audio(np.zeros(0, np.float32), ToneNoiseProxyClassifier(RATE), 3, 1).windows == []


def test_ast_group_indices_reports_missing() -> None:
    id2label = {0: "Speech", 1: "Screaming", 2: "Frying (food)", 3: "Gunshot, gunfire"}
    names = {
        "audio_vocal": ["Screaming", "Yell"],
        "audio_violence": ["Gunshot, gunfire"],
        "audio_kitchen": ["frying (food)"],
    }
    groups, missing = audio_cue.AstClassifier.group_indices(id2label, names)
    assert groups == {"vocal": [1], "violence": [3], "kitchen": [2]} and missing == ["Yell"]


def test_prompts_have_audio_groups() -> None:
    p = config.load_prompts()
    assert all(p[f"audio_{g}"] for g in audio_cue.GROUPS)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg yok")
def test_demo_end_to_end(tmp_path: Path) -> None:
    """Kansız demo: sesli hat bütün sahnelerde doğru, sessiz hat iki sahnede yanılıyor."""
    from tools import audio_demo

    video, events = audio_demo.make(tmp_path / "ses_demo.mp4")
    assert audio_cue.load_mono(video).size > 30 * RATE
    rows, scores = audio_demo.run(video, events)
    assert all(r["sesli_hat"] == r["olmalı"] for r in rows)
    assert sum(r["sessiz_hat"] != r["olmalı"] for r in rows) == 2
    assert scores["sesli_hat"]["olay_yakalama"] == 1.0
    assert scores["sesli_hat"]["olay_disi_bulanik_s"] < scores["sessiz_hat"]["olay_disi_bulanik_s"]
    assert (tmp_path / "ses_demo.audiocue.json").exists()


def test_ast_wrapper_with_tiny_random_model() -> None:
    """AST sarmalayıcısı, rastgele ağırlıklı küçük yapılandırma (indirmesiz)."""
    pytest.importorskip("torch")
    tf = pytest.importorskip("transformers")
    labels = {0: "Speech", 1: "Screaming", 2: "Frying (food)", 3: "Gunshot, gunfire"}
    cfg = tf.ASTConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_length=128,
        num_mel_bins=128,
        id2label=labels,
        label2id={v: k for k, v in labels.items()},
    )
    model = tf.ASTForAudioClassification(cfg).eval()
    ext = tf.ASTFeatureExtractor(max_length=128)
    clf = audio_cue.AstClassifier.from_model(model, ext, "cpu")
    p = clf(_tone(1.0))
    assert set(p) == set(audio_cue.GROUPS) and all(0.0 <= v <= 1.0 for v in p.values())
