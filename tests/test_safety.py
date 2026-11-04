import ast
from pathlib import Path

import numpy as np

from sensa_ai import safety

from .conftest import red_square_frame

PKG = Path(__file__).resolve().parents[1] / "src" / "sensa_ai"


def test_preview_is_small_gray_and_coarse() -> None:
    frame = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    p = safety.safe_preview(frame)
    assert max(p.shape[:2]) <= 160
    assert np.array_equal(p[..., 0], p[..., 1]) and np.array_equal(p[..., 1], p[..., 2])  # gri
    # 16 px bloklar: bir blok içindeki bütün pikseller aynı
    assert len(np.unique(p[:16, :16, 0])) == 1


def test_preview_hides_red() -> None:
    p = safety.safe_preview(red_square_frame())
    b, g, r = (p[..., i].astype(int) for i in range(3))
    assert (r - g).max() == 0 and (r - b).max() == 0


def test_only_safety_module_writes_images() -> None:
    """D-004: paket içinde cv2.imwrite / VideoWriter yalnız safety.py ve pipeline.render'da."""
    allowed = {"safety.py": {"imwrite"}, "pipeline.py": {"VideoWriter"}}
    for f in PKG.glob("*.py"):
        names = {n.attr for n in ast.walk(ast.parse(f.read_text())) if isinstance(n, ast.Attribute)}
        for fn in ("imwrite", "VideoWriter"):
            if fn in names:
                assert fn in allowed.get(f.name, set()), f"{f.name} {fn} kullanıyor"
