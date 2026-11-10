"""Tarayıcı oynatıcısı (M5 eklentisinin ilk adımı): web/player.html'i yerelde sunar.

    python -m tools.web_player                                   # dosyaları sayfada seç
    python -m tools.web_player data/vekil/balon.mp4 --track data/vekil/balon.blurtrack.json
    python -m tools.web_player --profile guclu

Sayfa `params.json`'u buradan alır: değerler config/params.yaml'dan (profil uygulanmış), kopya
sayı yok. Ham video sayfada hiç görünmez; görüntü yalnız tuvalden, bulanıklıktan sonra çizilir ve
iz yüklenmeden tuval siyah kalır. Sunucu yalnız 127.0.0.1'i dinler.

Tarayıcı codec notu: OpenCV'nin `mp4v` çıktısını Chrome çalmaz. Vekil video için
`python -m tools.make_proxy_video data/vekil/balon.webm` (VP8) ya da H.264 mp4 kullan.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from sensa_ai import config

WEB = config.ROOT / "web"
STATIC = {"/": "player.html", "/player.html": "player.html", "/blurtrack.js": "blurtrack.js"}


def params_json(profile: str | None = None) -> dict:
    if profile is not None and profile not in config.load_raw().get("profiles", {}):
        raise ValueError(f"bilinmeyen profil: {profile}")
    cfg = config.load(profile=profile)
    track = {k: cfg["track"][k] for k in ("min_hits", "strong_score") if k in cfg["track"]}
    return {"lookahead": cfg["lookahead"], "blur": cfg["blur"], "track": track, "profile": profile or "orta"}


def make_handler(video: Path | None, track: Path | None, profile: str | None) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:  # sessiz
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            if u.path in STATIC:
                f = WEB / STATIC[u.path]
                ctype = "text/html; charset=utf-8" if f.suffix == ".html" else "text/javascript; charset=utf-8"
                return self._send(f.read_bytes(), ctype)
            if u.path == "/params.json":
                prof = parse_qs(u.query).get("profile", [profile])[0]
                try:
                    body = params_json(prof)
                except ValueError as e:
                    return self._send(str(e).encode(), "text/plain; charset=utf-8", 400)
                return self._send(json.dumps(body).encode(), "application/json")
            if u.path == "/media/config.json":
                info = {"video": "/media/video" if video else None, "track": "/media/track.json" if track else None}
                return self._send(json.dumps(info).encode(), "application/json")
            if u.path == "/media/video" and video:
                ctype = mimetypes.guess_type(video.name)[0] or "application/octet-stream"
                return self._send(video.read_bytes(), ctype)
            if u.path == "/media/track.json" and track:
                return self._send(track.read_bytes(), "application/json")
            self._send(b"yok", "text/plain", 404)

    return H


def serve(
    video: Path | None = None, track: Path | None = None, profile: str | None = None, port: int = 8765
) -> ThreadingHTTPServer:
    """Arka planda sunucu başlatır (testler de kullanır). port=0 boş bir port seçer."""
    if profile is not None:
        config.set_profile(profile)  # bilinmeyen profili erken reddet
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(video, track, profile))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path, nargs="?")
    ap.add_argument("--track", type=Path, help="varsayılan: <video>.blurtrack.json")
    ap.add_argument("--profile", choices=sorted(config.load_raw().get("profiles", {})))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    track = args.track or (args.video.with_suffix(".blurtrack.json") if args.video else None)
    if track is not None and not track.exists():
        ap.error(f"iz yok: {track} (önce: python -m sensa_ai.pipeline analyze {args.video})")
    srv = serve(args.video, track, args.profile, args.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Oynatıcı: {url}  (Ctrl+C ile kapat)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
