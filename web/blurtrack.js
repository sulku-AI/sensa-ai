// .blurtrack.json okuyucu ve oynatıcı kuralları (docs/BLURTRACK_FORMAT.md).
// src/sensa_ai/blurtrack.py ile aynı mantık; tests/test_web.py ikisini aynı girdide karşılaştırır.
// Tarayıcı eklentisinin (M5) çekirdeği: DOM'a bağlı değil, Node'da da çalışır.

export const FORMAT = "sensa-blurtrack";
export const VERSION = 1;

export function parse(obj) {
  if (obj.format !== FORMAT || obj.version !== VERSION) throw new Error("bilinmeyen iz biçimi");
  return {
    video: obj.video,
    duration_s: obj.duration_s,
    analyzed_until_s: obj.analyzed_until_s ?? 0,
    tracks: obj.tracks.map((tr) => ({
      label: tr.label,
      keyframes: tr.keyframes.map((k) => ({ t: k.t, box: k.box, score: k.score ?? 1, poly: k.poly ?? null })),
    })),
  };
}

export function boxAt(tr, t) {
  const kf = tr.keyframes;
  if (!kf.length || t < kf[0].t || t > kf[kf.length - 1].t) return null;
  for (let i = 0; i + 1 < kf.length; i++) {
    const a = kf[i], b = kf[i + 1];
    if (a.t <= t && t <= b.t) {
      const u = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
      return a.box.map((pa, j) => pa + u * (b.box[j] - pa));
    }
  }
  return kf[kf.length - 1].box;
}

export function polyAt(tr, t, box) {
  let kf = tr.keyframes[0];
  for (const k of tr.keyframes) if (Math.abs(k.t - t) < Math.abs(kf.t - t)) kf = k;
  if (!kf.poly || !kf.poly.length) return null;
  const dx = box[0] - kf.box[0], dy = box[1] - kf.box[1];
  return kf.poly.map((p) => p.map((v, i) => v + (i % 2 === 0 ? dx : dy)));
}

// Zamansal filtre (Track.confirmed ile aynı): kısa iz ancak çok eminse bulanıklaştırılır.
export function confirmed(tr, minHits = 1, strongScore = 1) {
  return tr.keyframes.length >= minHits || Math.max(...tr.keyframes.map((k) => k.score)) >= strongScore;
}

// timing: {ramp_in_s, ramp_out_s, hold_in_s, hold_out_s, min_hits, strong_score} (sensa_ai.blur.timing ile aynı)
export function activeRegions(bt, t, timing = {}) {
  const ri = timing.ramp_in_s ?? 0, ro = timing.ramp_out_s ?? 0;
  const hi = timing.hold_in_s ?? 0, ho = timing.hold_out_s ?? 0;
  const out = [];
  for (const tr of bt.tracks) {
    const kf = tr.keyframes;
    if (!kf.length || !confirmed(tr, timing.min_hits ?? 1, timing.strong_score ?? 1)) continue;
    const first = kf[0], last = kf[kf.length - 1];
    const start = first.t, end = last.t;
    const a = start - hi, b = end + ho;
    const box = boxAt(tr, t);
    if (box !== null) out.push({ box, strength: 1, poly: polyAt(tr, t, box) });
    else if (a <= t && t < start) out.push({ box: first.box, strength: 1, poly: first.poly });
    else if (end < t && t <= b) out.push({ box: last.box, strength: 1, poly: last.poly });
    else if (ri > 0 && a - ri <= t && t < a) out.push({ box: first.box, strength: 1 - (a - t) / ri, poly: first.poly });
    else if (ro > 0 && b < t && t <= b + ro) out.push({ box: last.box, strength: 1 - (t - b) / ro, poly: last.poly });
  }
  return out;
}

// sensa_ai.lookahead.mode ile aynı: tampon safe_margin_s altına inerse güvenli mod.
export function mode(bt, t, safeMarginS) {
  if (bt.analyzed_until_s >= bt.duration_s) return "normal";
  return bt.analyzed_until_s - t >= safeMarginS ? "normal" : "failsafe";
}

// params: /params.json (tools/web_player). blur.timing ile aynı hesap.
export function timingFrom(params) {
  const b = params.blur, hold = (b.hold_factor ?? 1) / params.lookahead.sample_fps;
  const tr = params.track ?? {};
  return {
    ramp_in_s: b.ramp_in_s, ramp_out_s: b.ramp_out_s, hold_in_s: hold, hold_out_s: hold,
    min_hits: tr.min_hits ?? 1, strong_score: tr.strong_score ?? 1,
  };
}
