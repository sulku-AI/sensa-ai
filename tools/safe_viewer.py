"""Kontrollü görüntüleyici: kullanıcı açıkça istediğinde (D-001 teyidi alındıktan sonra), birkaç şüpheli
etiketi kademeli bulanıklıkla gösterir. Varsayılan hep en güvenli hal: siyah ekran.

    python -m tools.audit_dataset data/pozitif/blood_seg_joost --detector rfdetr_seg --worst 5 --list denetim.json
    python -m tools.safe_viewer denetim.json

Kurallar (sayfada da yazar):
  * Görüntü sen "Göster"e basana kadar yüklenmez; açılınca gri, küçük ve çok bulanık başlar.
  * Netliği kaydırıcıyla sen artırırsın. Esc ya da "Karart" anında siyaha döner.
  * "Yalnız silüet" fotoğrafı hiç göstermez, yalnız etiket şeklini çizer.
  * Görüntü diske yazılmaz; tarayıcıya önbelleğe almama başlığıyla gider. Sunucu yalnız 127.0.0.1.
  * İşaretlerin (iyi / kötü / emin değilim) data/denetim_isaretleri.json'a yazılır (repoya girmez).
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
MARKS = ROOT / "data" / "denetim_isaretleri.json"

PAGE = """<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>güvenli denetim</title>
<style>
 body{margin:0;background:#000;color:#bbb;font:14px system-ui,sans-serif}
 header{padding:10px 14px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#111}
 button{background:#222;color:#ddd;border:1px solid #444;padding:6px 12px;border-radius:4px}
 #dark{background:#600;color:#fff;font-weight:bold}
 #stage{display:flex;justify-content:center;padding:16px}
 canvas{background:#000;max-width:320px;max-height:320px}
 p{padding:0 14px;color:#777}
</style></head><body>
<header>
 <span id="pos"></span>
 <button id="show">Göster</button>
 <button id="dark">Karart (Esc)</button>
 <label>Netlik <input type="range" id="sharp" min="0" max="100" value="0"></label>
 <label><input type="checkbox" id="color"> Renkli</label>
 <label><input type="checkbox" id="sil"> Yalnız silüet</label>
 <label><input type="checkbox" id="outline" checked> Etiket çizgisi</label>
 <button data-m="iyi">Etiket iyi</button><button data-m="kotu">Etiket kötü</button>
 <button data-m="emin_degil">Emin değilim</button>
 <button id="next">Sonraki</button>
</header>
<div id="stage"><canvas id="c" width="320" height="320"></canvas></div>
<p>Ekran siyah başlar. Görüntü gri ve çok bulanık açılır, netliği sen artırırsın. Rahatsız olursan Esc.
 "Yalnız silüet" fotoğrafı hiç göstermez. İstediğin an sekmeyi kapatabilirsin.</p>
<script>
const items = __ITEMS__;
let i = 0, img = null, shown = false;
const c = document.getElementById('c'), x = c.getContext('2d');
const $ = (id) => document.getElementById(id);
function dark() { shown = false; img = null; x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height); }
function pos() { $('pos').textContent = `${i + 1} / ${items.length}  (IoU ${items[i].iou.toFixed(2)})`; }
function draw() {
  if (!shown) return dark();
  const it = items[i], W = it.w, H = it.h, s = Math.min(320 / W, 320 / H);
  c.width = Math.round(W * s); c.height = Math.round(H * s);
  x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height);
  if (!$('sil').checked && img) {
    const blur = 40 * (1 - $('sharp').value / 100);
    x.filter = `blur(${blur}px)` + ($('color').checked ? '' : ' grayscale(100%)');
    x.drawImage(img, 0, 0, c.width, c.height);
    x.filter = 'none';
  }
  if ($('sil').checked || $('outline').checked) {
    x.strokeStyle = '#0f0'; x.fillStyle = '#fff'; x.lineWidth = 2;
    for (const p of it.polys) {
      x.beginPath();
      for (let k = 0; k < p.length; k += 2) x.lineTo(p[k] * s, p[k + 1] * s);
      x.closePath();
      $('sil').checked ? x.fill() : x.stroke();
    }
  }
}
function show() {
  shown = true;
  if ($('sil').checked) return draw();
  const im = new Image();
  im.onload = () => { if (shown) { img = im; draw(); } };
  im.src = `/img/${i}?t=${Date.now()}`;
}
$('show').onclick = show;
$('dark').onclick = dark;
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') dark(); });
for (const id of ['sharp', 'color', 'sil', 'outline']) $(id).oninput = draw;
$('next').onclick = () => {
  dark(); $('sharp').value = 0; $('color').checked = false; i = (i + 1) % items.length; pos();
};
for (const b of document.querySelectorAll('[data-m]'))
  b.onclick = () => fetch('/mark', {method: 'POST', body: JSON.stringify({i, mark: b.dataset.m})});
dark(); pos();
</script></body></html>
"""


def load_items(path: Path) -> list[dict]:
    from PIL import Image

    out = []
    for it in json.loads(path.read_text()):
        p = Path(it["path"])
        try:
            with Image.open(p) as im:  # yalnız başlık okunur (boyut), piksel çözülmez
                w, h = im.size
        except OSError:
            continue
        out.append({"path": str(p), "iou": float(it.get("iou", 0.0)), "polys": it.get("polys", []), "w": w, "h": h})
    return out


def make_handler(items: list[dict]) -> type[BaseHTTPRequestHandler]:
    public = [{k: v for k, v in it.items() if k != "path"} for it in items]  # dosya yolu sayfaya gitmez

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_: object) -> None:
            pass

        def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            if u.path == "/":
                html = PAGE.replace("__ITEMS__", json.dumps(public))
                return self._send(html.encode(), "text/html; charset=utf-8")
            if u.path.startswith("/img/"):
                try:
                    it = items[int(u.path.removeprefix("/img/"))]
                except (ValueError, IndexError):
                    return self._send(b"yok", "text/plain", 404)
                p = Path(it["path"])
                return self._send(p.read_bytes(), mimetypes.guess_type(p.name)[0] or "image/jpeg")
            self._send(b"yok", "text/plain", 404)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/mark":
                return self._send(b"yok", "text/plain", 404)
            n = int(self.headers.get("Content-Length", 0))
            d = json.loads(self.rfile.read(n) or b"{}")
            marks = json.loads(MARKS.read_text()) if MARKS.exists() else {}
            k = int(d["i"])
            if 0 <= k < len(items):
                marks[items[k]["path"]] = d["mark"]
                MARKS.parent.mkdir(parents=True, exist_ok=True)
                MARKS.write_text(json.dumps(marks, indent=1, ensure_ascii=False))
            self._send(b"ok", "text/plain")

    return H


def serve(items: list[dict], port: int = 8766) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(items))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("list", type=Path, help="tools.audit_dataset --list çıktısı")
    ap.add_argument("--max", type=int, default=5, help="en çok bu kadar görüntü (varsayılan 5)")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    items = load_items(args.list)[: args.max]
    if not items:
        ap.error("listede açılabilir görüntü yok")
    srv = serve(items, args.port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"{len(items)} görüntü. Sayfa siyah başlar. {url}  (Ctrl+C ile kapat)")
    webbrowser.open(url)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
        if MARKS.exists():
            c: dict[str, int] = {}
            for v in json.loads(MARKS.read_text()).values():
                c[v] = c.get(v, 0) + 1
            print("işaretler:", c)


if __name__ == "__main__":
    main()
