"""Rastgele ağırlıklı küçük modeller (indirme yok). Model ve araç testleri ortak kullanır."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def tiny_grounding(tmp_path: Path) -> tuple[Any, Any]:
    import torch
    from transformers import (
        BertTokenizerFast,
        GroundingDinoConfig,
        GroundingDinoForObjectDetection,
        GroundingDinoImageProcessor,
        GroundingDinoProcessor,
    )

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "red", "balloon", "blood", "wound", "gore", "open"]
    words += ["bleeding", "a", "photo", "of"]
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
    return GroundingDinoForObjectDetection(cfg).eval(), proc


def tiny_owlv2() -> tuple[Any, Any]:
    import torch
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
    return Owlv2ForObjectDetection(cfg).eval(), proc


def tiny_video_tracker(family: str = "sam2") -> tuple[Any, Any]:
    import torch
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
    return model.eval(), proc


class StubGateModel:
    """SigLIP yerine: sabit logits_per_image (kapı mantığını sınamak için)."""

    def __init__(self, logits: list[float]) -> None:
        import torch

        self.dtype = torch.float32
        self._logits = torch.tensor([logits])
        self.name_or_path = "stub-gate"

    def __call__(self, **_: object) -> object:
        return type("O", (), {"logits_per_image": self._logits})()


def stub_processor(**_: object) -> dict:
    import torch

    return {"input_ids": torch.zeros(1, 1)}
