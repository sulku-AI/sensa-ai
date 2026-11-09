"""Tarayıcı oynatıcısı: JS iz mantığı Python ile aynı sonucu verir; sunucu params.yaml'ı aynen sunar.

Tarayıcı testi yalnız Node + Playwright varsa çalışır (sentetik renkli kareler, kansız).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

from sensa_ai import blur, config
from sensa_ai.blurtrack import BlurTrack, Keyframe, Track
from tools import web_player

WEB = config.ROOT / "web"
NODE = shutil.which("node")


def sample_track() -> BlurTrack:
    poly = [[0.2, 0.2, 0.4, 0.2, 0.4, 0.5, 0.2, 0.5]]
    a = Track("blood", [Keyframe(2.0, (0.2, 0.2, 0.4, 0.5), 0.8, poly), Keyframe(4.0, (0.4, 0.3, 0.6, 0.6), 0.7)])
    b = Track("blood", [Keyframe(6.0, (0.1, 0.1, 0.2, 0.2)), Keyframe(6.0, (0.1, 0.1, 0.2, 0.2))])
    return BlurTrack("x.mp4", 10.0, 8.0, [a, b, Track("blood", [])])


def run_js(code: str) -> object:
    assert NODE
    mod = (WEB / "blurtrack.js").as_uri()
    src = f"import * as B from {json.dumps(mod)};\n{code}"
    out = subprocess.run([NODE, "--input-type=module", "-e", src], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node yok")
def test_js_matches_python() -> None:
    bt = sample_track()
    timing = blur.timing(config.load()["blur"])
    times = [round(x, 3) for x in np.arange(0.0, 8.0, 0.05)]
    js = run_js(
        f"const bt = B.parse({json.dumps(bt.to_dict())});\n"
        f"const ts = {json.dumps(times)}, tm = {json.dumps(timing)};\n"
        "console.log(JSON.stringify(ts.map(t => [B.activeRegions(bt, t, tm), B.mode(bt, t, 5)])));"
    )
    assert isinstance(js, list)
    for t, (regions, mode) in zip(times, js, strict=True):
        py = bt.active_regions(t, **timing)
        assert len(regions) == len(py), t
        for r, p in zip(regions, py, strict=True):
            assert r["box"] == pytest.approx(list(p.box), abs=1e-9), t
            assert r["strength"] == pytest.approx(p.strength, abs=1e-9), t
            assert (r["poly"] is None) == (p.poly is None), t
            if p.poly:
                assert np.allclose(r["poly"], p.poly), t
        from sensa_ai import lookahead

        assert mode == lookahead.mode(t, bt.analyzed_until_s, bt.duration_s, 5).value


@pytest.mark.skipif(NODE is None, reason="node yok")
def test_js_rejects_unknown_format_and_timing_matches() -> None:
    res = run_js(
        "let ok = false; try { B.parse({format: 'x', version: 1}); } catch (e) { ok = true; }\n"
        f"console.log(JSON.stringify([ok, B.timingFrom({json.dumps(web_player.params_json())})]));"
    )
    assert isinstance(res, list)
    assert res[0] is True
    assert res[1] == pytest.approx(blur.timing(config.load()["blur"]))


def get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_serves_params_and_static(tmp_path: Path) -> None:
    srv = web_player.serve(port=0)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, body = get(base + "/params.json")
        assert code == 200
        assert json.loads(body)["blur"] == config.load()["blur"]
        code, body = get(base + "/params.json?profile=guclu")
        assert json.loads(body)["blur"]["sigma_px"] == config.load(profile="guclu")["blur"]["sigma_px"]
        assert get(base + "/params.json?profile=yok")[0] == 400
        assert b"blurtrack.js" in get(base + "/")[1]
        assert get(base + "/blurtrack.js")[0] == 200
        assert json.loads(get(base + "/media/config.json")[1]) == {"video": None, "track": None}
        assert get(base + "/media/video")[0] == 404
        assert get(base + "/../config/params.yaml")[0] == 404  # yalnız beyaz liste
    finally:
        srv.shutdown()


def _playwright_ok() -> bool:
    if NODE is None:
        return False
    r = subprocess.run([NODE, "-e", "require.resolve('playwright')"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _playwright_ok(), reason="node playwright yok")
def test_browser_blurs_only_region(tmp_path: Path) -> None:
    """Gerçek tarayıcı: iz yokken tuval siyah; izle bölge bulanık, dışı keskin."""
    import os

    import cv2

    video = tmp_path / "v.webm"  # keskin dama deseni, 3 sn
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"VP80"), 10, (320, 240))
    yy, xx = np.mgrid[0:240, 0:320]
    checker = (((yy // 8) + (xx // 8)) % 2 * 255).astype(np.uint8)
    frame = np.dstack([checker] * 3)
    for _ in range(30):
        vw.write(frame)
    vw.release()
    left_half = (0.0, 0.0, 0.5, 1.0)
    bt = BlurTrack("v.webm", 3.0, 3.0, [Track("blood", [Keyframe(0.0, left_half), Keyframe(3.0, left_half)])])
    track = tmp_path / "v.blurtrack.json"
    bt.save(track)
    srv = web_player.serve(video, track, port=0)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    script = tmp_path / "run.cjs"
    script.write_text("""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto(process.argv[2]);
  await p.waitForFunction(() => window.sensa && window.sensa.ready(), null, { timeout: 15000 });
  await p.evaluate(() => window.sensa.seek(1.5));
  const s = await p.evaluate(() => {
    const c = document.getElementById('out'), x = c.getContext('2d');
    const std = (sx) => {
      const d = x.getImageData(sx, 60, 100, 120).data;
      let m = 0, q = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) { m += d[i]; q += d[i] * d[i]; n++; }
      m /= n;
      return Math.sqrt(q / n - m * m);
    };
    const op = getComputedStyle(document.getElementById('v')).opacity;
    return { w: c.width, left: std(30), right: std(190), videoVisible: op };
  });
  console.log(JSON.stringify(s));
  await b.close();
})().catch(e => { console.error(e); process.exit(1); });
""")
    pw = subprocess.check_output([NODE, "-p", "require.resolve('playwright')"], text=True).strip()
    env = os.environ | {"NODE_PATH": str(Path(pw).parents[1])}
    try:
        out = subprocess.run([NODE, str(script), url], capture_output=True, text=True, timeout=60, env=env)
    finally:
        srv.shutdown()
    assert out.returncode == 0, out.stderr
    s = json.loads(out.stdout.strip().splitlines()[-1])
    assert s["w"] == 320
    assert s["videoVisible"] == "0"
    assert s["right"] > 100  # bölge dışı: keskin dama
    assert s["left"] < 0.3 * s["right"]  # bölge içi: bulanık
