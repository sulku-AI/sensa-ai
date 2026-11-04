"""Model sarmalayıcıları: rastgele ağırlıklı küçük yapılandırmalarla, İNDİRMESİZ.

Bu testler gerçek başarımı ölçmez (ağırlıklar rastgele); yalnız arayüzün doğru çağrıldığını,
kutuların normalize olduğunu ve takip maskelerinin boyutunu sınar. torch yoksa atlanır.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sensa_ai import models, tracking  # noqa: E402

FRAME = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
PROMPTS = ["red balloon", "blood"]


def _check(dets: list) -> None:
    assert dets, "en az bir tespit beklenir (eşik 0)"
    for d in dets:
        assert all(0.0 <= v <= 1.0 for v in d.box) and d.box[0] <= d.box[2] and d.box[1] <= d.box[3]
    assert len(dets) <= 8  # params detect.max_regions


def test_grounding_family(tmp_path: Path) -> None:
    from transformers import (
        BertTokenizerFast,
        GroundingDinoConfig,
        GroundingDinoForObjectDetection,
        GroundingDinoImageProcessor,
        GroundingDinoProcessor,
    )

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "red", "balloon", "blood"]
    (tmp_path / "vocab.txt").write_text("\n".join(words))
    text = {"model_type": "bert", "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2}
    text |= {"intermediate_size": 64, "vocab_size": len(words)}
    cfg = GroundingDinoConfig(
        text_config=text,
        d_model=64,
        encoder_layers=1,
        decoder_layers=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        num_queries=20,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
    )
    torch.manual_seed(0)
    proc = GroundingDinoProcessor(
        image_processor=GroundingDinoImageProcessor(size={"shortest_edge": 200, "longest_edge": 300}),
        tokenizer=BertTokenizerFast(str(tmp_path / "vocab.txt")),
    )
    det = models.TextDetector(GroundingDinoForObjectDetection(cfg).eval(), proc, "cpu", 0.0, 0.0)
    assert det.family == "grounding"
    _check(det(FRAME, PROMPTS))


def test_owlv2_family() -> None:
    from transformers import CLIPTokenizer, Owlv2Config, Owlv2ForObjectDetection, Owlv2ImageProcessor, Owlv2Processor

    vocab = {"blood</w>": 0, "r": 1, "e": 2, "d</w>": 3, "b": 4, "a": 5, "l": 6, "o": 7, "n</w>": 8}
    vocab |= {"<|startoftext|>": 9, "<|endoftext|>": 10}  # OWL: başlangıç kimliği > 0, bitiş en büyük
    layer = {"hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2}
    cfg = Owlv2Config(
        text_config=layer | {"vocab_size": len(vocab)},
        vision_config=layer | {"image_size": 96, "patch_size": 16},
        projection_dim=32,
    )
    torch.manual_seed(0)
    proc = Owlv2Processor(
        image_processor=Owlv2ImageProcessor(size={"height": 96, "width": 96}),
        tokenizer=CLIPTokenizer(vocab=vocab, merges=[]),
    )
    det = models.TextDetector(Owlv2ForObjectDetection(cfg).eval(), proc, "cpu", 0.0, 0.0)
    assert det.family == "owlv2"
    _check(det(FRAME, PROMPTS))


class _StubGate:
    """SigLIP yerine: logits_per_image döndüren sahte model + işlemci (softmax mantığını sınar)."""

    def __init__(self, logits: list[float]) -> None:
        self.dtype = torch.float32
        self._logits = torch.tensor([logits])

    def __call__(self, **_: object) -> object:
        return type("O", (), {"logits_per_image": self._logits})()


def test_gate_softmax_over_positive_and_negative() -> None:
    def proc(**_: object) -> dict:
        return {"input_ids": torch.zeros(1, 1)}

    gate = models.SiglipGate(_StubGate([2.0, 2.0, 0.0]), proc, "cpu")
    s = gate.score(FRAME, ["blood", "wound"], ["ketchup"])
    e = np.exp([2.0, 2.0, 0.0])
    assert s == pytest.approx((e[0] + e[1]) / e.sum(), rel=1e-5)


def test_gated_detector_skips_when_gate_low() -> None:
    calls = []

    class Det:
        def __call__(self, f: np.ndarray, p: list[str]) -> list:
            calls.append(1)
            return []

    gd = models.GatedDetector(models.SiglipGate(_StubGate([-5.0, 5.0]), lambda **_: {}, "cpu"), Det(), ["x"], 0.3)  # type: ignore[arg-type]
    gd(FRAME, ["blood"])
    assert calls == [] and gd.last_gate_score is not None and gd.last_gate_score < 0.3


@pytest.mark.parametrize("family", ["sam2", "edgetam"])
def test_tracker_mask_shapes(family: str) -> None:
    from transformers import (
        EdgeTamVideoConfig,
        EdgeTamVideoModel,
        Sam2ImageProcessor,
        Sam2VideoConfig,
        Sam2VideoModel,
        Sam2VideoProcessor,
        Sam2VideoVideoProcessor,
    )

    torch.manual_seed(0)
    proc = Sam2VideoProcessor(image_processor=Sam2ImageProcessor(), video_processor=Sam2VideoVideoProcessor())
    model = Sam2VideoModel(Sam2VideoConfig()) if family == "sam2" else EdgeTamVideoModel(EdgeTamVideoConfig())
    tr = tracking.Tracker(model.eval(), proc, "cpu", family)
    frames = [np.random.default_rng(i).integers(0, 255, (64, 96, 3), dtype=np.uint8) for i in range(3)]
    masks = tr.track(frames, [(0.2, 0.2, 0.5, 0.5)], start_idx=1)
    assert len(masks) == 3 and all(m.shape == (64, 96) and m.dtype == bool for m in masks)
    assert not masks[0].any()  # başlangıç öncesi boş


def test_mask_to_box() -> None:
    m = np.zeros((10, 20), bool)
    m[2:5, 4:10] = True
    assert tracking.mask_to_box(m) == pytest.approx((0.2, 0.2, 0.5, 0.5))
    assert tracking.mask_to_box(np.zeros((4, 4), bool)) is None


def test_resolve_ids() -> None:
    assert models.resolve("detect") == "iSEE-Laboratory/llmdet_tiny"
    assert models.resolve("detect", "owlv2_base").startswith("google/owlv2")
    assert models.resolve("detect", "x/y") == "x/y"
