"""ORTAK İÇİN zaman etiketleme aracı (TEST_PLANI §1). Emirhan bu aracı pozitif videolarda KULLANMAZ.

    python -m tools.label_events data/pozitif/videolar/film.mp4 --ortak

Pencerede video oynar. Tuşlar: s = olay başladı, e = olay bitti, boşluk = duraklat,
← / → (j / l) = 5 sn, q = kaydet ve çık. Çıktı: data/etiket/<video adı>.json (yalnız zaman, görüntü yok).
Olay başlarken kaba bölge için fareyle dikdörtgen çizilebilir (isteğe bağlı).

Güvenlik: `--ortak` bayrağı olmadan açılmaz. data/pozitif dışındaki videolar (vekil, negatif) için bayrak gerekmez.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def needs_partner(video: Path, data_root: Path = ROOT / "data") -> bool:
    return (data_root / "pozitif").resolve() in video.resolve().parents


def save(out: Path, video: str, events: list[dict]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    clean = [e for e in events if e.get("end") is not None and e["end"] > e["start"]]
    out.write_text(json.dumps({"video": video, "events": clean}, indent=1), encoding="utf-8")


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--ortak", action="store_true", help="pozitif videoyu izleyecek kişi ortaktır (Emirhan değil)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if needs_partner(args.video) and not args.ortak:
        raise SystemExit("Bu video pozitif klasörde. Yalnız ortak etiketler: --ortak (D-001).")
    out = args.out or ROOT / "data" / "etiket" / f"{args.video.stem}.json"
    events: list[dict] = json.loads(out.read_text())["events"] if out.exists() else []
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    win = "etiket (s/e/bosluk/j/l/q)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    box: list[float] | None = None
    drag: list[int] = []

    def on_mouse(ev: int, x: int, y: int, *_: object) -> None:
        nonlocal box
        if ev == cv2.EVENT_LBUTTONDOWN:
            drag[:] = [x, y]
        elif ev == cv2.EVENT_LBUTTONUP and drag:
            w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            x0, x1 = sorted((drag[0], x))
            y0, y1 = sorted((drag[1], y))
            box = [round(x0 / w, 3), round(y0 / h, 3), round(x1 / w, 3), round(y1 / h, 3)]

    cv2.setMouseCallback(win, on_mouse)
    paused, frame = False, None
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        show = frame.copy()
        open_ev = bool(events) and events[-1].get("end") is None
        cv2.putText(
            show,
            f"{t:7.2f} s  olay: {len(events)}{'  [ACIK]' if open_ev else ''}",
            (14, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.imshow(win, show)
        key = cv2.waitKey(max(1, int(1000 / fps))) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("s"):
            events.append({"start": round(t, 2), "end": None, "box": box})
            box = None
        elif key == ord("e") and open_ev:
            events[-1]["end"] = round(t, 2)
            if box:
                events[-1]["box"] = box
                box = None
        elif key in (81, 2, ord("j"), 83, 3, ord("l")):
            dt = -5 if key in (81, 2, ord("j")) else 5
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t + dt) * 1000)
            frame = None
    cap.release()
    cv2.destroyAllWindows()
    save(out, args.video.name, events)
    print(f"{len(events)} olay -> {out}")


if __name__ == "__main__":
    main()
