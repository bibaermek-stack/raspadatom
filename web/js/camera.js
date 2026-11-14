/* Камера/сурет → спектр құбыры (DOM-сыз, Node-та да жұмыс істейді).
   1) жолақты табу (PCA, элонгация): бұрыш, ұзындық, қалыңдық;  2) бұрышы мен қисықтығына қарамай 1D профиль;
   3) толқын ұзындығына калибровка (CFL бойынша автоматты / екі нүкте);  4) модель торына (380–780 нм, 1024 нүкте) келтіру;
   5) ЖИ болжамын сызық сәйкестігімен тексеру.
   Сурет форматы: {data: Uint8ClampedArray RGBA, width, height}. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./physics.js'));
  else { root.SL = root.SL || {}; root.SL.camera = factory(root.SL.physics); }
})(typeof window !== 'undefined' ? window : globalThis, function (P) {
  'use strict';

  var WIN_LO = 380, WIN_HI = 780, WIN_N = 1024, WIN_DX = (WIN_HI - WIN_LO) / (WIN_N - 1);
  var LIN = new Float32Array(256);
  for (var q = 0; q < 256; q++) { var c = q / 255; LIN[q] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function median(a) { var b = Array.prototype.slice.call(a).sort(function (x, y) { return x - y; }); return b[b.length >> 1]; }
  function percentile(a, p) {
    var b = Array.prototype.slice.call(a).sort(function (x, y) { return x - y; });
    return b[clamp(Math.floor(p * (b.length - 1)), 0, b.length - 1)];
  }

  // ---------- 1. жолақты табу ----------
  // кішірейтілген карталар: сызықтық жарықтық және хроматика (max−min): спектр түсті, ал шам жарқылы ақ
  function smallMaps(img, f) {
    var W = img.width, H = img.height, w = Math.floor(W / f), h = Math.floor(H / f), d = img.data;
    var luma = new Float32Array(w * h), chroma = new Float32Array(w * h), inv = 1 / (f * f);
    for (var y = 0; y < h; y++) for (var x = 0; x < w; x++) {
      var sl = 0, sc = 0;
      for (var dy = 0; dy < f; dy++) {
        var o = ((y * f + dy) * W + x * f) * 4;
        for (var dx = 0; dx < f; dx++, o += 4) {
          var r = LIN[d[o]], g = LIN[d[o + 1]], b = LIN[d[o + 2]];
          sl += r + g + b; sc += Math.max(r, g, b) - Math.min(r, g, b);
        }
      }
      luma[y * w + x] = sl * inv / 3; chroma[y * w + x] = sc * inv;
    }
    return { luma: { w: w, h: h, data: luma, f: f }, chroma: { w: w, h: h, data: chroma, f: f } };
  }
  function downLuma(img, f) { return smallMaps(img, f).luma; }

  // баяу өзгеретін фонды (ауа жарығы, виньетка) алып тастау: 12×9 тор, әр ұяшықтың 20-перцентилі
  function flatten(L) {
    var gx = 12, gy = 9, w = L.w, h = L.h, bg = new Float32Array(gx * gy);
    for (var j = 0; j < gy; j++) for (var i = 0; i < gx; i++) {
      var x0 = Math.floor(i * w / gx), x1 = Math.floor((i + 1) * w / gx), y0 = Math.floor(j * h / gy), y1 = Math.floor((j + 1) * h / gy), v = [];
      for (var y = y0; y < y1; y += 2) for (var x = x0; x < x1; x += 2) v.push(L.data[y * w + x]);
      bg[j * gx + i] = v.length ? percentile(v, 0.2) : 0;
    }
    var out = new Float32Array(w * h);
    for (var y2 = 0; y2 < h; y2++) {
      var fy = clamp((y2 + 0.5) * gy / h - 0.5, 0, gy - 1), j0 = Math.floor(fy), j1 = Math.min(gy - 1, j0 + 1), ty = fy - j0;
      for (var x2 = 0; x2 < w; x2++) {
        var fx = clamp((x2 + 0.5) * gx / w - 0.5, 0, gx - 1), i0 = Math.floor(fx), i1 = Math.min(gx - 1, i0 + 1), tx = fx - i0;
        var b = (bg[j0 * gx + i0] * (1 - tx) + bg[j0 * gx + i1] * tx) * (1 - ty) + (bg[j1 * gx + i0] * (1 - tx) + bg[j1 * gx + i1] * tx) * ty;
        var v2 = L.data[y2 * w + x2] - b;
        out[y2 * w + x2] = v2 > 0 ? v2 : 0;
      }
    }
    return { w: w, h: h, data: out, f: L.f };
  }

  // 8-байланысты компоненттер: [{n, idx:[...]}]
  function components(mask, w, h, minN) {
    var lab = new Int32Array(w * h), comps = [], stack = new Int32Array(w * h), next = 1;
    for (var s = 0; s < w * h; s++) {
      if (!mask[s] || lab[s]) continue;
      var sp = 0, idx = [];
      stack[sp++] = s; lab[s] = next;
      while (sp) {
        var p = stack[--sp], x = p % w, y = (p / w) | 0;
        idx.push(p);
        for (var dy = -1; dy <= 1; dy++) for (var dx = -1; dx <= 1; dx++) {
          var nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          var np = ny * w + nx;
          if (mask[np] && !lab[np]) { lab[np] = next; stack[sp++] = np; }
        }
      }
      if (idx.length >= (minN || 12)) comps.push({ n: idx.length, idx: idx });
      next++;
    }
    return comps;
  }
  function pca(idx, w, wts) {
    var n = 0, mx = 0, my = 0, i;
    for (i = 0; i < idx.length; i++) { var wt = wts ? wts[i] : 1; n += wt; mx += (idx[i] % w) * wt; my += ((idx[i] / w) | 0) * wt; }
    mx /= n; my /= n;
    var cxx = 0, cxy = 0, cyy = 0;
    for (i = 0; i < idx.length; i++) {
      var wt2 = wts ? wts[i] : 1, dx = (idx[i] % w) - mx, dy = ((idx[i] / w) | 0) - my;
      cxx += wt2 * dx * dx; cxy += wt2 * dx * dy; cyy += wt2 * dy * dy;
    }
    cxx /= n; cxy /= n; cyy /= n;
    var tr = cxx + cyy, det = cxx * cyy - cxy * cxy, disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
    var l1 = tr / 2 + disc, l2 = Math.max(tr / 2 - disc, 1e-6);
    return { cx: mx, cy: my, theta: 0.5 * Math.atan2(2 * cxy, cxx - cyy), l1: l1, l2: l2, ratio: Math.sqrt(l1 / l2) };
  }

  // шам жарқылы (нөлінші рет): өте жарық, ықшам дақ. -> [{cx, cy, r}] (кішірейтілген координата)
  function findBlobs(Lf) {
    var w = Lf.w, h = Lf.h, mask = new Uint8Array(w * h), i, out = [];
    for (i = 0; i < mask.length; i++) mask[i] = Lf.data[i] > 0.55 ? 1 : 0;
    components(mask, w, h, 5).forEach(function (c) {
      var g = pca(c.idx, w, null);
      if (g.ratio < 2.2) out.push({ cx: g.cx, cy: g.cy, r: Math.max(2, 2 * Math.sqrt(g.l1)) });
    });
    return out;
  }

  // карта бойынша жолақ: -> roi немесе null
  function detectOn(map, blobs, f, minThr, carve) {
    var w = map.w, h = map.h, d = map.data, i;
    var med = median(d.slice(0, d.length)), dev = new Float32Array(d.length);
    for (i = 0; i < d.length; i++) dev[i] = Math.abs(d[i] - med);
    var floor = 1.4826 * median(dev), thr = Math.max(med + 8 * floor, minThr);
    var mask = new Uint8Array(w * h), cnt = 0;
    for (i = 0; i < d.length; i++) if (d[i] > thr) { mask[i] = 1; cnt++; }
    if (carve) {
      blobs.forEach(function (b) {
        var r = 3.2 * b.r;
        for (var y = Math.max(0, Math.floor(b.cy - r)); y <= Math.min(h - 1, Math.ceil(b.cy + r)); y++)
          for (var x = Math.max(0, Math.floor(b.cx - r)); x <= Math.min(w - 1, Math.ceil(b.cx + r)); x++)
            if ((x - b.cx) * (x - b.cx) + (y - b.cy) * (y - b.cy) <= r * r) mask[y * w + x] = 0;
      });
    }
    var comps = components(mask, w, h, 12), maxScore = 0;
    if (!comps.length) return null;
    comps.forEach(function (c) { c.g = pca(c.idx, w, null); c.score = c.n * Math.min(c.g.ratio, 12); if (c.score > maxScore) maxScore = c.score; });
    comps = comps.filter(function (c) { return c.score >= 0.05 * maxScore; });
    var idx = [];
    comps.forEach(function (c) { for (var q = 0; q < c.idx.length; q++) idx.push(c.idx[q]); });
    var g = pca(idx, w, null), single = false;
    if (g.ratio < 2.2) {
      // бір ғана (не өте жақын) сызық: ось — штрихке перпендикуляр
      var big = comps.reduce(function (a, c) { return !a || c.n > a.n ? c : a; }, null), gl = big.g, th = gl.theta + Math.PI / 2;
      if (th > Math.PI / 2) th -= Math.PI;
      g = { cx: gl.cx, cy: gl.cy, theta: th, l1: Math.pow(Math.max(w * 0.2, 6), 2), l2: gl.l1, ratio: 3 };
      idx = big.idx; single = true;
    } else {
      for (var it = 0; it < 2; it++) {
        var ct = Math.cos(g.theta), st = Math.sin(g.theta), sv = Math.sqrt(g.l2), keep = [];
        for (i = 0; i < idx.length; i++) {
          var dx = (idx[i] % w) - g.cx, dy = ((idx[i] / w) | 0) - g.cy, s = -dx * st + dy * ct;
          if (Math.abs(s) < 2.6 * sv + 1) keep.push(idx[i]);
        }
        if (keep.length < 12) break;
        idx = keep; g = pca(idx, w, null);
      }
    }
    var ct2 = Math.cos(g.theta), st2 = Math.sin(g.theta), ts = [];
    for (i = 0; i < idx.length; i++) ts.push((idx[i] % w - g.cx) * ct2 + ((idx[i] / w) | 0) * st2 - g.cy * st2);
    var t0 = percentile(ts, 0.005), t1 = percentile(ts, 0.995);
    if (single) { t0 = -w * 0.42; t1 = w * 0.42; }
    var margin = Math.max(0.12 * (t1 - t0), 8), lo = t0 - margin, hi = t1 + margin;
    // жарқыл дақ жағына таралмау (ореолымен бірге)
    blobs.forEach(function (b) {
      var tb = (b.cx - g.cx) * ct2 + (b.cy - g.cy) * st2, sb = -(b.cx - g.cx) * st2 + (b.cy - g.cy) * ct2;
      if (Math.abs(sb) > 4 * Math.sqrt(g.l2) + b.r) return;     // осьтен тыс
      if (tb < t0) lo = Math.max(lo, tb + 3.4 * b.r); else if (tb > t1) hi = Math.min(hi, tb - 3.4 * b.r);
    });
    if (hi - lo < 16) { lo = t0; hi = t1; }
    var hv = clamp(1.9 * Math.sqrt(g.l2), 2.5, h / 3);
    return {
      cx: (g.cx + 0.5) * f, cy: (g.cy + 0.5) * f, theta: g.theta, t0: lo * f, t1: hi * f, hv: hv * f,
      info: { f: f, ratio: g.ratio, found: !single, single: single, blobs: blobs.length, thr: thr, floor: floor, components: comps.length }
    };
  }

  // -> roi {cx, cy, theta, t0, t1, hv} (толық ажыратымдылық пикселімен) немесе null; roi.info — диагностика
  // Алдымен хроматика бойынша (түсті жолақ, ақ жарқылсыз), болмаса жарықтық бойынша (жарқыл дақты кесіп тастап).
  function detectBand(img) {
    var f = Math.max(1, Math.ceil(Math.max(img.width, img.height) / 320)), m = smallMaps(img, f);
    var Lf = flatten(m.luma), Cf = flatten(m.chroma), blobs = findBlobs(Lf);
    var r1 = detectOn(Cf, blobs, f, 0.006, true);
    if (r1 && !r1.info.single) { r1.info.map = 'chroma'; return r1; }
    var r2 = detectOn(Lf, blobs, f, 0.012, true);
    if (r2 && !r2.info.single) { r2.info.map = 'luma'; return r2; }
    var r = r1 || r2;
    if (r) r.info.map = r1 ? 'chroma' : 'luma';
    return r;
  }

  // ---------- 2. профиль ----------
  var smp = { r: 0, g: 0, b: 0 };
  function sample(img, x, y) {
    var W = img.width, H = img.height, d = img.data;
    if (x < 0 || y < 0 || x > W - 1 || y > H - 1) return false;
    var x0 = Math.floor(x), y0 = Math.floor(y), x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1), fx = x - x0, fy = y - y0;
    var w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
    var o00 = (y0 * W + x0) * 4, o10 = (y0 * W + x1) * 4, o01 = (y1 * W + x0) * 4, o11 = (y1 * W + x1) * 4;
    smp.r = LIN[d[o00]] * w00 + LIN[d[o10]] * w10 + LIN[d[o01]] * w01 + LIN[d[o11]] * w11;
    smp.g = LIN[d[o00 + 1]] * w00 + LIN[d[o10 + 1]] * w10 + LIN[d[o01 + 1]] * w01 + LIN[d[o11 + 1]] * w11;
    smp.b = LIN[d[o00 + 2]] * w00 + LIN[d[o10 + 2]] * w10 + LIN[d[o01 + 2]] * w01 + LIN[d[o11 + 2]] * w11;
    smp.sat = (d[o00] > 250 || d[o00 + 1] > 250 || d[o00 + 2] > 250) ? 1 : 0;
    return true;
  }
  function boxSmooth(a, r) {
    var n = a.length, out = new Float32Array(n), s = 0, c = 0, i;
    for (i = 0; i < Math.min(n, r); i++) { s += a[i]; c++; }
    for (i = 0; i < n; i++) {
      if (i + r < n) { s += a[i + r]; c++; }
      if (i - r - 1 >= 0) { s -= a[i - r - 1]; c--; }
      out[i] = s / c;
    }
    return out;
  }
  function gauss(a, sigma) {
    if (sigma <= 0.01) return Float32Array.from(a);
    var r = Math.max(1, Math.ceil(3 * sigma)), k = [], s = 0, i, n = a.length, out = new Float32Array(n);
    for (i = -r; i <= r; i++) { var v = Math.exp(-0.5 * i * i / (sigma * sigma)); k.push(v); s += v; }
    for (i = 0; i < n; i++) {
      var acc = 0, wsum = 0;
      for (var j = -r; j <= r; j++) { var q2 = i + j; if (q2 < 0 || q2 >= n) continue; acc += a[q2] * k[j + r]; wsum += k[j + r]; }
      out[i] = acc / wsum;
    }
    return out;
  }

  // roi бойынша профиль. -> {p (сызықтық, фон алынған), t (толық ажыратымдылық пикселі, ось бойымен), hue, sat, bg}
  function extractProfile(img, roi) {
    var M = Math.max(32, Math.round(roi.t1 - roi.t0)), ct = Math.cos(roi.theta), st = Math.sin(roi.theta);
    var hv = Math.max(2.5, roi.hv), wide = Math.ceil(2.2 * hv), core = Math.max(1, Math.round(0.6 * hv));
    var p = new Float32Array(M), bg = new Float32Array(M), R = new Float32Array(M), G = new Float32Array(M), B = new Float32Array(M), sat = new Float32Array(M), sc = new Float32Array(M);
    var j, s, k;
    // 1) әр бағандағы центроид (қисық/бұрыш қатесін өтеу)
    for (j = 0; j < M; j++) {
      var t = roi.t0 + j * (roi.t1 - roi.t0) / (M - 1), sw = 0, ss = 0;
      for (s = -wide; s <= wide; s++) {
        if (!sample(img, roi.cx + t * ct - s * st, roi.cy + t * st + s * ct)) continue;
        var v = (smp.r + smp.g + smp.b) / 3;
        sw += v; ss += v * s;
      }
      sc[j] = sw > 0 ? ss / sw : 0;
    }
    var scs = boxSmooth(sc, Math.max(4, M >> 5));
    for (j = 0; j < M; j++) { if (Math.abs(scs[j]) > hv) scs[j] = Math.sign(scs[j]) * hv; }
    // 2) ядро бойынша орташа, жақтан фон
    for (j = 0; j < M; j++) {
      var t2 = roi.t0 + j * (roi.t1 - roi.t0) / (M - 1), a = 0, an = 0, ar = 0, ag = 0, ab = 0, sa = 0, bgv = [];
      for (s = -wide - Math.ceil(hv); s <= wide + Math.ceil(hv); s++) {
        var os = s - scs[j];
        if (!sample(img, roi.cx + t2 * ct - s * st, roi.cy + t2 * st + s * ct)) continue;
        var vv = (smp.r + smp.g + smp.b) / 3;
        if (Math.abs(s - scs[j]) <= core) { a += vv; ar += smp.r; ag += smp.g; ab += smp.b; sa += smp.sat; an++; }
        else if (Math.abs(os) > 1.6 * hv && Math.abs(os) < 2.8 * hv + 4) bgv.push(vv);
      }
      if (an) { p[j] = a / an; R[j] = ar / an; G[j] = ag / an; B[j] = ab / an; sat[j] = sa / an; }
      bg[j] = bgv.length ? median(bgv) : 0;
    }
    var bgs = boxSmooth(bg, Math.max(6, M >> 4));
    for (k = 0; k < M; k++) { p[k] = Math.max(0, p[k] - bgs[k]); }
    return { p: p, bg: bgs, R: R, G: G, B: B, sat: sat, M: M, t0: roi.t0, t1: roi.t1, step: (roi.t1 - roi.t0) / (M - 1) };
  }

  // бағыт: көк/күлгін жағы қай жақта? -> {flip: көк жағы оң жақта болса true, conf: 0..1}
  function colorDirection(pr) {
    var M = pr.M, sw = 0, sx = 0, sh = 0, i, hs = new Float32Array(M), ws = new Float32Array(M);
    for (i = 0; i < M; i++) {
      var r = pr.R[i], g = pr.G[i], b = pr.B[i], mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn, h;
      if (d < 1e-6 || mx < 1e-4) { ws[i] = 0; continue; }
      if (mx === r) h = 60 * (((g - b) / d + 6) % 6); else if (mx === g) h = 60 * ((b - r) / d + 2); else h = 60 * ((r - g) / d + 4);
      if (h >= 300) h -= 360;
      hs[i] = h; ws[i] = pr.p[i] * (d / mx);
      sw += ws[i];
    }
    if (sw <= 0) return { flip: false, conf: 0 };
    var mxp = 0, mh = 0;
    for (i = 0; i < M; i++) { sx += ws[i] * i; sh += ws[i] * hs[i]; }
    mxp = sx / sw; mh = sh / sw;
    var cov = 0, vx = 0, vh = 0;
    for (i = 0; i < M; i++) { cov += ws[i] * (i - mxp) * (hs[i] - mh); vx += ws[i] * (i - mxp) * (i - mxp); vh += ws[i] * (hs[i] - mh) * (hs[i] - mh); }
    var corr = vx > 0 && vh > 0 ? cov / Math.sqrt(vx * vh) : 0;
    // hue өседі: қызыл (0°) → көк (240°) → күлгін; индекс өссе hue өссе, көк жағы оң жақта -> flip
    return { flip: corr > 0, conf: Math.min(1, Math.abs(corr) * 1.6) };
  }

  // ---------- 3. шыңдар ----------
  function findPeaks(y, minProm, minSep) {
    var n = y.length, out = [], i;
    for (i = 1; i < n - 1; i++) {
      if (!(y[i] > y[i - 1] && y[i] >= y[i + 1])) continue;
      var l = i, lm = y[i], r = i, rm = y[i];
      while (l > 0 && y[l - 1] <= y[i]) { l--; if (y[l] < lm) lm = y[l]; }
      while (r < n - 1 && y[r + 1] <= y[i]) { r++; if (y[r] < rm) rm = y[r]; }
      var prom = y[i] - Math.max(lm, rm);
      if (prom >= minProm) {
        // параболалық нақтылау
        var den = y[i - 1] - 2 * y[i] + y[i + 1], off = den !== 0 ? 0.5 * (y[i - 1] - y[i + 1]) / den : 0;
        out.push({ i: i, x: i + clamp(off, -1, 1), h: y[i], prom: prom });
      }
    }
    out.sort(function (a, b) { return b.prom - a.prom; });
    var kept = [];
    out.forEach(function (p) { if (!kept.some(function (q) { return Math.abs(q.x - p.x) < (minSep || 3); })) kept.push(p); });
    return kept;
  }

  // ---------- 4. калибровка ----------
  // Флуоресцентті шам (CFL) эталоны: Hg + Tb/Eu фосфорлары. Төмен ажыратымдылықта 543.5 (Tb) пен 546.1 (Hg) бірігеді.
  var CFL_REF = [
    { wl: 404.66, w: 0.25, name: 'Hg 404.7' }, { wl: 435.83, w: 1.0, name: 'Hg 435.8' }, { wl: 487.7, w: 0.5, name: 'Tb 487.7' },
    { wl: 544.5, w: 1.0, name: 'Tb/Hg 544.5' }, { wl: 578.0, w: 0.4, name: 'Hg 577/579' }, { wl: 611.6, w: 1.0, name: 'Eu 611.6' }
  ];
  function calib2(t1, w1, t2, w2) {
    if (t1 === t2 || w1 === w2) return null;
    var a = (w2 - w1) / (t2 - t1);
    return { a: a, b: w1 - a * t1, rms: 0, n: 2, kind: 'two-point' };
  }
  // pr: extractProfile нәтижесі; t — профиль индексі. -> {a, b, rms, n, kind, matched:[{x, wl, name}]} немесе null
  function autoCalibrateCFL(prof) {
    var y = gauss(prof.p, 1.2), mx = 0, i;
    for (i = 0; i < y.length; i++) if (y[i] > mx) mx = y[i];
    if (mx <= 0) return null;
    var peaks = findPeaks(y, 0.06 * mx, Math.max(4, y.length / 150)).slice(0, 9), M = y.length, best = null;
    if (peaks.length < 3) return null;
    var tol = 4.5;
    function score(a, b) {
      var s = 0, used = {}, matched = [];
      peaks.forEach(function (pk) {
        var wl = a * pk.x + b, bi = -1, bd = 1e9;
        CFL_REF.forEach(function (r, ri) { var d = Math.abs(wl - r.wl); if (d < bd) { bd = d; bi = ri; } });
        if (bi >= 0 && bd < tol) {
          var r = CFL_REF[bi], gain = (pk.prom / mx) * r.w * (1 - (bd / tol) * (bd / tol));
          if (!used[bi] || used[bi] < gain) { s += gain - (used[bi] || 0); used[bi] = gain; }
          matched.push({ x: pk.x, wl: r.wl, name: r.name, bi: bi, d: bd });
        }
      });
      return { s: s, matched: matched, n: Object.keys(used).length };
    }
    for (var p1 = 0; p1 < peaks.length; p1++) for (var p2 = p1 + 1; p2 < peaks.length; p2++) {
      var xa = peaks[p1].x, xb = peaks[p2].x;
      if (Math.abs(xb - xa) < M * 0.05) continue;
      for (var r1 = 0; r1 < CFL_REF.length; r1++) for (var r2 = 0; r2 < CFL_REF.length; r2++) {
        if (r1 === r2) continue;
        var a = (CFL_REF[r2].wl - CFL_REF[r1].wl) / (xb - xa), b = CFL_REF[r1].wl - a * xa;
        var l0 = b, l1 = a * (M - 1) + b, lo = Math.min(l0, l1), hi = Math.max(l0, l1);
        if (lo < 250 || hi > 1100 || hi - lo < 120 || hi - lo > 700) continue;      // ақылға сыйымды дисперсия
        var sc = score(a, b);
        if (sc.n >= 3 && (!best || sc.s > best.s)) best = { a: a, b: b, s: sc.s, matched: sc.matched, n: sc.n };
      }
    }
    if (!best) return null;
    // нақтылау: сәйкес шыңдар бойынша ең кіші квадраттар (әр эталонға ең жақыны)
    var one = {};
    best.matched.forEach(function (m) { if (!one[m.bi] || m.d < one[m.bi].d) one[m.bi] = m; });
    var pts = Object.keys(one).map(function (k) { return one[k]; });
    var sx = 0, sy = 0, sxx = 0, sxy = 0, n = pts.length;
    pts.forEach(function (m) { sx += m.x; sy += m.wl; sxx += m.x * m.x; sxy += m.x * m.wl; });
    var den = n * sxx - sx * sx;
    if (n < 3 || Math.abs(den) < 1e-9) return null;
    var A = (n * sxy - sx * sy) / den, Bc = (sy - A * sx) / n, ss = 0;
    pts.forEach(function (m) { var e = A * m.x + Bc - m.wl; ss += e * e; });
    if (n === 3 && Math.sqrt(ss / n) > 1.2) return null;      // 3 сәйкестік тек өте дәл болса ғана
    return { a: A, b: Bc, rms: Math.sqrt(ss / n), n: n, kind: 'cfl', matched: pts.map(function (m) { return { x: m.x, wl: m.wl, name: m.name }; }) };
  }

  // ---------- 5. модель кірісі ----------
  // prof.p индексі → толқын ұзындығы: λ = cal.a·i + cal.b.  -> {x: Float32Array(1024), lo, hi, coverage, max}
  function toModelInput(p, cal, opts) {
    opts = opts || {};
    var n = p.length, ys = gauss(p, opts.smooth === undefined ? 0.8 : opts.smooth), x = new Float32Array(WIN_N), cov = 0, i;
    // базалық деңгей: жарамды үлгілердің төмен перцентилі
    var valid = [];
    for (i = 0; i < n; i++) { var wl = cal.a * i + cal.b; if (wl >= WIN_LO && wl <= WIN_HI) valid.push(ys[i]); }
    if (valid.length < 16) return null;
    var base = percentile(valid, 0.05), mx = 0;
    for (i = 0; i < WIN_N; i++) {
      var l = WIN_LO + i * WIN_DX, t = (l - cal.b) / cal.a;
      if (!(t >= 0 && t <= n - 1)) { x[i] = 0; continue; }
      var t0 = Math.floor(t), t1 = Math.min(n - 1, t0 + 1), f = t - t0, v = ys[t0] * (1 - f) + ys[t1] * f - base;
      x[i] = v > 0 ? v : 0; cov++;
      if (x[i] > mx) mx = x[i];
    }
    if (mx <= 1e-6) return null;
    for (i = 0; i < WIN_N; i++) x[i] /= mx;
    var lo = Math.max(WIN_LO, Math.min(cal.b, cal.a * (n - 1) + cal.b)), hi = Math.min(WIN_HI, Math.max(cal.b, cal.a * (n - 1) + cal.b));
    return { x: x, lo: lo, hi: hi, coverage: cov / WIN_N, max: mx + base, base: base };
  }

  // сызық сигналы: жұқа тегістелген профиль минус тегіс фон (σ ≈ 35 нм). Кең «төбелер» (фосфор, континуум) кетеді.
  function lineSignal(p, cal) {
    var sig = Math.max(6, 35 / Math.max(Math.abs(cal.a), 0.05)), fine = gauss(p, 1.0), bgv = gauss(p, sig), r = new Float32Array(p.length), i;
    for (i = 0; i < p.length; i++) { var v = fine[i] - bgv[i]; r[i] = v > 0 ? v : 0; }
    return { res: r, fine: fine };
  }
  // «сызықтылық»: сызық сигналының шудан асқан максимумы профиль максимумына қатысты. Үздіксіз спектрде ≈ 0.
  function lineness(p, cal) {
    var ls = lineSignal(p, cal), i, mx = 1e-9, rm = 0, d = new Float32Array(p.length);
    for (i = 0; i < p.length; i++) { if (p[i] > mx) mx = p[i]; if (ls.res[i] > rm) rm = ls.res[i]; d[i] = Math.abs(p[i] - ls.fine[i]); }
    var noise = 1.4826 * median(d);
    return Math.max(0, rm - 3 * noise) / mx;
  }

  // тар (атомдық сызыққа ұқсас) шыңдар: сызық сигналында FWHM ≤ maxNm. Үздіксіз спектрде (қыздыру шамы, LED) жоқ.
  function narrowPeaks(p, cal, maxNm) {
    var ls = lineSignal(p, cal), y = ls.res, mx = 0, i;
    for (i = 0; i < y.length; i++) if (y[i] > mx) mx = y[i];
    var out = [];
    if (mx <= 0) return out;
    findPeaks(y, 0.12 * mx, 3).forEach(function (pk) {
      var half = pk.h - pk.prom / 2, l = pk.i, r = pk.i;
      while (l > 0 && y[l] > half) l--;
      while (r < y.length - 1 && y[r] > half) r++;
      var w = (r - l) * Math.abs(cal.a);
      if (w <= (maxNm || 15)) out.push({ x: pk.x, wl: cal.a * pk.x + cal.b, h: pk.h, prom: pk.prom, fwhm: w });
    });
    return out;
  }

  // Толық баға: модель + физикалық қақпа + сызық сәйкестігімен растау.
  // -> {inp, res, lin, narrow, continuum, pred, predGated, confirmed, fwhm, ver}
  function assess(prof, cal, model, byName, species, opts) {
    opts = opts || {};
    var inp = toModelInput(prof.p, cal); if (!inp) return null;
    var res = opts.tta ? predictTTA(model, prof.p, cal).mid.res : model.predict(inp.x), names = model.species;
    var lin = lineness(prof.p, cal), narrow = narrowPeaks(prof.p, cal, 15);
    var continuum = lin < 0.2 || narrow.length === 0, pred = [];
    names.forEach(function (n, k) { if (res.probs[k] > 0.5) pred.push(k); });
    pred.sort(function (a, b) { return res.probs[b] - res.probs[a]; });
    var gated = continuum ? [] : pred, fwhm = estimateFwhm(prof.p, cal);
    var sel = gated.slice(0, 3).map(function (k) { return { sp: byName[names[k]], conc: 1 }; });
    var ver = verify(prof, cal, fwhm, sel, species, 8000), confirmed = [];
    ver.perSpecies.forEach(function (ps) { if (ps.expected.length && ps.found / ps.expected.length >= 0.4) confirmed.push(ps.name); });
    return { inp: inp, res: res, lin: lin, narrow: narrow, continuum: continuum, pred: pred, predGated: gated, confirmed: confirmed, fwhm: fwhm, ver: ver };
  }

  // калибровкаға шағын ығысу/масштаб: λ' = scale·(λ−580)+580+shift
  function perturbCal(cal, shift, scale) {
    return { a: scale * cal.a, b: scale * (cal.b - 580) + 580 + shift, kind: cal.kind, provisional: cal.provisional };
  }
  // калибровка қателігіне төзімділік: ығысулар торы бойынша ықтималдықтардың орташасы (test-time augmentation)
  function predictTTA(model, p, cal, grid) {
    grid = grid || { shifts: [-3, 0, 3], scales: [0.99, 1, 1.01] };
    var acc = null, n = 0, mid = null;
    grid.shifts.forEach(function (sh) {
      grid.scales.forEach(function (sc) {
        var inp = toModelInput(p, perturbCal(cal, sh, sc));
        if (!inp) return;
        var r = model.predict(inp.x);
        if (!acc) acc = new Float32Array(r.probs.length);
        for (var i = 0; i < acc.length; i++) acc[i] += r.probs[i];
        n++;
        if (sh === 0 && sc === 1) mid = { inp: inp, res: r };
      });
    });
    if (!acc) return null;
    for (var i = 0; i < acc.length; i++) acc[i] /= n;
    return { probs: acc, n: n, mid: mid };
  }

  // профильдің ажыратымдылығы: ең күшті сызықтардың FWHM төменгі квартилі (нм), кең фон алынғаннан кейін
  function estimateFwhm(p, cal) {
    var y = lineSignal(p, cal).res, mx = 0, i;
    for (i = 0; i < y.length; i++) if (y[i] > mx) mx = y[i];
    var peaks = findPeaks(y, 0.12 * mx, 4).slice(0, 6), w = [];
    peaks.forEach(function (pk) {
      var half = pk.h - pk.prom / 2, l = pk.i, r = pk.i;
      while (l > 0 && y[l] > half) l--;
      while (r < y.length - 1 && y[r] > half) r++;
      if (r > l) w.push((r - l) * Math.abs(cal.a));
    });
    w.sort(function (x, y2) { return x - y2; });
    return w.length ? clamp(w[Math.floor((w.length - 1) * 0.35)], 1.2, 25) : 4;
  }

  // ---------- 6. ЖИ болжамын тексеру ----------
  var PHOSPHOR = [{ wl: 487.7, name: 'Tb³⁺ (фосфор)' }, { wl: 543.5, name: 'Tb³⁺ (фосфор)' }, { wl: 593, name: 'Eu³⁺ (фосфор)' }, { wl: 611.6, name: 'Eu³⁺ (фосфор)' },
                  { wl: 650, name: 'Eu³⁺ (фосфор)' }, { wl: 690, name: 'Eu³⁺ (фосфор)' }];

  function speciesExpected(sp, T, lo, hi) {
    var st = P.boltzmannStrengths(sp, T), m = 0, i, out = [];
    for (i = 0; i < st.length; i++) if (sp.wl[i] >= lo && sp.wl[i] <= hi && st[i] > m) m = st[i];
    if (m <= 0) return out;
    for (i = 0; i < st.length; i++) if (sp.wl[i] >= lo && sp.wl[i] <= hi) out.push({ wl: sp.wl[i], st: st[i] / m, i: i });
    return out;
  }

  // species: [{sp, conc}] — болжанған түрлер. -> {perSpecies:[{name, expected, found}], unexplained:[{wl, h, cand}], simulated: Float32Array(1024)}
  function verify(prof, cal, fwhmNm, species, allSpecies, T) {
    T = T || 8000;
    var y = lineSignal(prof.p, cal).res, mx = 0, i;
    for (i = 0; i < y.length; i++) if (y[i] > mx) mx = y[i];
    var peaks = [];
    findPeaks(y, 0.12 * Math.max(mx, 1e-9), 4).forEach(function (pk) {
      var half = pk.h - pk.prom / 2, l = pk.i, r = pk.i;
      while (l > 0 && y[l] > half) l--;
      while (r < y.length - 1 && y[r] > half) r++;
      if ((r - l) * Math.abs(cal.a) <= 20) peaks.push({ wl: cal.a * pk.x + cal.b, h: pk.h / mx, prom: pk.prom / mx });      // кең «төбелер» сызық емес
    });
    var lo = Math.max(WIN_LO, Math.min(cal.b, cal.a * (prof.p.length - 1) + cal.b)), hi = Math.min(WIN_HI, Math.max(cal.b, cal.a * (prof.p.length - 1) + cal.b));
    var tol = Math.max(2.5, 0.9 * fwhmNm), explained = new Array(peaks.length).fill(false), per = [];
    var delta = new Float64Array(WIN_N);
    species.forEach(function (s) {
      var ex = speciesExpected(s.sp, T, lo, hi).filter(function (e) { return e.st >= 0.2; }).sort(function (a, b) { return b.st - a.st; }).slice(0, 8), found = 0;
      ex.forEach(function (e) {
        var hit = false;
        peaks.forEach(function (pk, k) { if (Math.abs(pk.wl - e.wl) <= tol) { hit = true; explained[k] = true; } });
        e.found = hit; if (hit) found++;
      });
      per.push({ name: s.sp.name, sp: s.sp, expected: ex, found: found });
      // модельденген қисық
      speciesExpected(s.sp, T, lo - 20, hi + 20).forEach(function (e) {
        var pos = (e.wl - WIN_LO) / WIN_DX, i0 = Math.floor(pos), fr = pos - i0;
        if (i0 >= 0 && i0 < WIN_N) delta[i0] += e.st * (1 - fr);
        if (i0 + 1 >= 0 && i0 + 1 < WIN_N) delta[i0 + 1] += e.st * fr;
      });
    });
    var ker = P.pseudoVoigtKernel(fwhmNm / WIN_DX, 0.3), sim = new Float32Array(WIN_N), k2, half = ker.half;
    for (i = 0; i < WIN_N; i++) {
      if (delta[i] === 0) continue;
      for (k2 = Math.max(0, i - half); k2 <= Math.min(WIN_N - 1, i + half); k2++) sim[k2] += delta[i] * ker.k[half + (k2 - i)];
    }
    var smx = 0; for (i = 0; i < WIN_N; i++) if (sim[i] > smx) smx = sim[i];
    if (smx > 0) for (i = 0; i < WIN_N; i++) sim[i] /= smx;
    var unexplained = [];
    peaks.forEach(function (pk, k) {
      if (explained[k] || pk.prom < 0.15) return;
      var cand = [];
      allSpecies.forEach(function (sp) {
        var j = P.nearestLine(sp, pk.wl);
        if (Math.abs(sp.wl[j] - pk.wl) <= tol) cand.push({ name: sp.name, wl: sp.wl[j], rel: Math.log10(sp.rel[j] + 1) });
      });
      cand.sort(function (a, b) { return b.rel - a.rel; });
      var ph = PHOSPHOR.filter(function (q) { return Math.abs(q.wl - pk.wl) <= tol + 1.5; })[0];
      unexplained.push({ wl: pk.wl, h: pk.h, cand: cand.slice(0, 3), phosphor: ph ? ph.name : null });
    });
    return { perSpecies: per, unexplained: unexplained, simulated: sim, peaks: peaks, lo: lo, hi: hi };
  }

  return {
    WIN_LO: WIN_LO, WIN_HI: WIN_HI, WIN_N: WIN_N, WIN_DX: WIN_DX, CFL_REF: CFL_REF, PHOSPHOR: PHOSPHOR,
    detectBand: detectBand, extractProfile: extractProfile, colorDirection: colorDirection,
    findPeaks: findPeaks, gauss: gauss, calib2: calib2, autoCalibrateCFL: autoCalibrateCFL,
    toModelInput: toModelInput, lineness: lineness, narrowPeaks: narrowPeaks, assess: assess, perturbCal: perturbCal, predictTTA: predictTTA, estimateFwhm: estimateFwhm, verify: verify, downLuma: downLuma
  };
});
