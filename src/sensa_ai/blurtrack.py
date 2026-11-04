"""Bulanıklaştırma izi (.blurtrack.json): altyazı gibi videodan ayrı bir dosya (D-003).

Bir iz, zaman içinde bir bölgenin anahtar karelerini tutar. Kutular ve poligonlar
normalize (0..1) koordinattadır, böylece iz videonun çözünürlüğünden bağımsızdır.
Takip maskesi varsa anahtar kare sadeleştirilmiş poligon da taşır (`poly`); bulanıklık
o zaman kutu yerine bölgenin şeklini izler. Poligon yoksa kutu kullanılır.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

FORMAT_VERSION = 1

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 normalize
Poly = list[float]  # x1, y1, x2, y2, ... normalize


@dataclass
class Keyframe:
    t: float
    box: Box
    score: float = 1.0
    poly: list[Poly] | None = None


@dataclass
class Region:
    box: Box
    strength: float
    poly: list[Poly] | None = None


@dataclass
class Track:
    label: str
    keyframes: list[Keyframe] = field(default_factory=list)

    @property
    def start(self) -> float:
        return self.keyframes[0].t

    @property
    def end(self) -> float:
        return self.keyframes[-1].t

    def poly_at(self, t: float, box: Box) -> list[Poly] | None:
        """En yakın anahtar karenin poligonu, kutunun kaymasına göre ötelenir."""
        kf = min(self.keyframes, key=lambda k: abs(k.t - t))
        if not kf.poly:
            return None
        dx, dy = box[0] - kf.box[0], box[1] - kf.box[1]
        return [[v + (dx if i % 2 == 0 else dy) for i, v in enumerate(p)] for p in kf.poly]

    def box_at(self, t: float) -> Box | None:
        """İki anahtar kare arasında doğrusal ara değer; iz dışında None."""
        kf = self.keyframes
        if not kf or t < kf[0].t or t > kf[-1].t:
            return None
        for a, b in zip(kf, kf[1:], strict=False):
            if a.t <= t <= b.t:
                u = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
                return tuple(pa + u * (pb - pa) for pa, pb in zip(a.box, b.box, strict=True))  # type: ignore[return-value]
        return kf[-1].box


@dataclass
class BlurTrack:
    video: str
    duration_s: float
    analyzed_until_s: float = 0.0
    tracks: list[Track] = field(default_factory=list)

    def active_regions(self, t: float, ramp_in_s: float = 0.0, ramp_out_s: float = 0.0) -> list[Region]:
        """t anında bulanıklaştırılacak bölgeler (kutu, güç 0..1, varsa poligon).

        Rampa sayesinde bulanıklık bölge görünmeden ramp_in_s önce başlar ve
        kaybolduktan ramp_out_s sonra biter. İleri bakma bunu mümkün kılar.
        """
        out: list[Region] = []
        for tr in self.tracks:
            if not tr.keyframes:
                continue
            box = tr.box_at(t)
            if box is not None:
                out.append(Region(box, 1.0, tr.poly_at(t, box)))
            elif tr.start - ramp_in_s <= t < tr.start and ramp_in_s > 0:
                k = tr.keyframes[0]
                out.append(Region(k.box, 1.0 - (tr.start - t) / ramp_in_s, k.poly))
            elif tr.end < t <= tr.end + ramp_out_s and ramp_out_s > 0:
                k = tr.keyframes[-1]
                out.append(Region(k.box, 1.0 - (t - tr.end) / ramp_out_s, k.poly))
        return out

    def active(self, t: float, ramp_in_s: float = 0.0, ramp_out_s: float = 0.0) -> list[tuple[Box, float]]:
        """Geriye uyumluluk: yalnız (kutu, güç)."""
        return [(r.box, r.strength) for r in self.active_regions(t, ramp_in_s, ramp_out_s)]

    def to_dict(self) -> dict[str, object]:
        return {"format": "sensa-blurtrack", "version": FORMAT_VERSION, **asdict(self)}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BlurTrack:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != "sensa-blurtrack" or d.get("version") != FORMAT_VERSION:
            raise ValueError("bilinmeyen iz biçimi")
        tracks = [
            Track(
                tr["label"],
                [Keyframe(k["t"], tuple(k["box"]), k.get("score", 1.0), k.get("poly")) for k in tr["keyframes"]],
            )
            for tr in d["tracks"]
        ]
        return cls(d["video"], d["duration_s"], d.get("analyzed_until_s", 0.0), tracks)
