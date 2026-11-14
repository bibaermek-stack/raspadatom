/* Виртуал камера: тор арқылы түсірілген спектрдің модельденген суреті (RGBA).
   Мақсат: камера/сурет құбырын (camera.js) камерасыз сынау және мысал суреттер көрсету. Бұл НЕГІЗІ — модель, нақты фото емес.
   Құрамында: тор бұрышы, қисықтық, виньетка, нөлінші ретті жарқыл, ауа жарығы, шу, RGB сезімталдығы, IR-фильтр, қанығу. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./physics.js'));
  else { root.SL = root.SL || {}; root.SL.camsim = factory(root.SL.physics); }
})(typeof window !== 'undefined' ? window : globalThis, function (P) {
  'use strict';

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function G(l, c, s) { var z = (l - c) / s; return Math.exp(-0.5 * z * z); }
  function sig(x) { return 1 / (1 + Math.exp(-x)); }
  function srgb(v) { v = clamp(v, 0, 1); return v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055; }

  // ---------- көз (сахна спектрі) ----------
  // NIST сызықтарынан: sel=[{sp, conc}], терезеде (380–780) ең күшті сызық = conc
  function nistLines(sel, T, rng, jitter) {
    var lines = [], gj = P.gaussSource(rng);
    sel.forEach(function (s) {
      var st = P.boltzmannStrengths(s.sp, T), m = 0, i;
      for (i = 0; i < st.length; i++) if (s.sp.wl[i] >= 380 && s.sp.wl[i] <= 780 && st[i] > m) m = st[i];
      if (m <= 0) return;
      for (i = 0; i < st.length; i++) {
        var w = s.sp.wl[i]; if (w < 370 || w > 800) continue;
        lines.push({ wl: w, a: st[i] / m * s.conc * Math.exp(gj() * (jitter || 0)) });
      }
    });
    return lines;
  }
  function byName(species) { var m = {}; species.forEach(function (s) { m[s.name] = s; }); return m; }

  // дайын көздер. build(ctx) -> {lines, cont(λ), truth:[...]}
  var SOURCES = [
    { id: 'cfl', title: 'Флуоресцентті шам (CFL)', truth: ['Hg I'], build: function (c) {
      var L = [], rel = { 404.66: 0.12, 435.83: 0.5, 546.07: 0.9, 576.96: 0.13, 579.07: 0.15 };
      Object.keys(rel).forEach(function (k) { L.push({ wl: +k, a: rel[k] }); });
      [[487.7, 0.3], [543.5, 1.0], [585.0, 0.15], [593.0, 0.25], [611.6, 0.95], [620.0, 0.12], [650.0, 0.14], [690.0, 0.2], [700.0, 0.12]].forEach(function (q) { L.push({ wl: q[0], a: q[1] }); });
      return { lines: L, cont: function (l) { return 0.45 * G(l, 450, 22) + 0.04; }, truth: ['Hg I'] };
    } },
    { id: 'ne', title: 'Неон белгісі', truth: ['Ne I'], build: function (c) { return { lines: nistLines([{ sp: c.by['Ne I'], conc: 1 }], 6500, c.rng, 0.15), cont: function () { return 0.01; }, truth: ['Ne I'] }; } },
    { id: 'na', title: 'Натрий көше шамы', truth: ['Na I'], build: function (c) { return { lines: nistLines([{ sp: c.by['Na I'], conc: 1 }], 6000, c.rng, 0.1), cont: function (l) { return 0.04 * G(l, 600, 80); }, truth: ['Na I'] }; } },
    { id: 'h', title: 'Сутегі түтігі', truth: ['H I'], build: function (c) { return { lines: nistLines([{ sp: c.by['H I'], conc: 1 }], 9000, c.rng, 0.1), cont: function () { return 0.015; }, truth: ['H I'] }; } },
    { id: 'he', title: 'Гелий түтігі', truth: ['He I'], build: function (c) { return { lines: nistLines([{ sp: c.by['He I'], conc: 1 }], 10000, c.rng, 0.15), cont: function () { return 0.01; }, truth: ['He I'] }; } },
    { id: 'hg', title: 'Сынап шамы', truth: ['Hg I'], build: function (c) { return { lines: nistLines([{ sp: c.by['Hg I'], conc: 1 }], 8000, c.rng, 0.1), cont: function () { return 0.01; }, truth: ['Hg I'] }; } },
    { id: 'kr', title: 'Криптон түтігі', truth: ['Kr I'], build: function (c) { return { lines: nistLines([{ sp: c.by['Kr I'], conc: 1 }], 7000, c.rng, 0.2), cont: function () { return 0.012; }, truth: ['Kr I'] }; } },
    { id: 'mix', title: 'Қоспа: Ne + He', truth: ['Ne I', 'He I'], build: function (c) { return { lines: nistLines([{ sp: c.by['Ne I'], conc: 0.8 }, { sp: c.by['He I'], conc: 1 }], 8000, c.rng, 0.15), cont: function () { return 0.01; }, truth: ['Ne I', 'He I'] }; } },
    { id: 'bulb', title: 'Қыздыру шамы (үздіксіз)', truth: [], build: function () { return { lines: [], cont: function (l) { return P.planck(l, 2800) / P.planck(700, 2800) * 0.8; }, truth: [] }; } },
    { id: 'led', title: 'Ақ LED (үздіксіз)', truth: [], build: function () { return { lines: [], cont: function (l) { return 0.9 * G(l, 450, 9) + 0.55 * G(l, 570, 55) + 0.02; }, truth: [] }; } }
  ];

  // ---------- камера ----------
  function chans(l, cutR) {
    var cut = sig((l - 398) / 7) * (1 - sig((l - cutR) / 14));
    return [(G(l, 603, 46) + 0.22 * G(l, 700, 55) + 0.06 * G(l, 450, 30)) * cut, (G(l, 541, 44) + 0.04 * G(l, 620, 40)) * cut, (G(l, 456, 31) + 0.05 * G(l, 540, 40)) * cut];
  }

  // o: {W,H,fwhm, ...}; әр параметр кездейсоқ таңдалады, өзгертуге болады. -> {img, truth, geo, params}
  function render(source, species, o, rng) {
    o = o || {};
    var W = o.W || 640, H = o.H || 480, by = byName(species), U = function (a, b) { return a + (b - a) * rng(); };
    var scene = source.build({ by: by, rng: rng });
    var g = P.gaussSource(rng);
    var fwhm = o.fwhm || U(2.5, 8), sg = fwhm / 2.3548;
    var theta = (o.theta !== undefined ? o.theta : U(-9, 9)) * Math.PI / 180, ct = Math.cos(theta), st = Math.sin(theta);
    var Lb = o.Lb || U(0.62, 0.86) * W, ucx = U(-0.04, 0.04), ucy = U(-0.12, 0.12);
    var cx = o.fixCx !== undefined ? o.fixCx : W / 2 + ucx * W, cy = o.fixCy !== undefined ? o.fixCy : H / 2 + ucy * H;
    var sigv = o.sigv || U(3.5, 12), curv = o.curv !== undefined ? o.curv : U(-6e-5, 6e-5);
    var lamLo = o.lamLo || U(372, 410), lamHi = o.lamHi || U(700, 790), flip = o.flip !== undefined ? o.flip : rng() < 0.5;
    var cutR = o.cutR || U(675, 715), peakLevel = o.peak || Math.exp(U(Math.log(0.35), Math.log(1.8)));
    var blobR = o.blobR || U(6, 17), blobOff = o.blobOff || U(0.1, 0.3) * Lb;
    // сахна спектрі + аспаптың кеңеюі (тор бойынша 0.25 нм)
    var l0 = Math.min(lamLo, lamHi) - 6, step = 0.25, n = Math.ceil((Math.max(lamLo, lamHi) + 6 - l0) / step) + 1, S = new Float32Array(n), i, k;
    scene.lines.forEach(function (ln) {
      var c = Math.round((ln.wl - l0) / step), r = Math.ceil(4 * sg / step);
      for (k = c - r; k <= c + r; k++) if (k >= 0 && k < n) S[k] += ln.a * G(l0 + k * step, ln.wl, sg);
    });
    var cs = new Float32Array(n);
    for (i = 0; i < n; i++) cs[i] = scene.cont(l0 + i * step);
    var C = []; for (i = 0; i < n; i++) C.push(chans(l0 + i * step, cutR));
    var img = new Float32Array(W * H * 3), peak = 1e-9, x, y;
    function lamAt(tau) { var f = tau / Lb; return flip ? lamHi - f * (lamHi - lamLo) : lamLo + f * (lamHi - lamLo); }
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) {
      var dx = x - cx, dy = y - cy, t = dx * ct + dy * st, s0 = -dx * st + dy * ct, tau = t + Lb / 2, o3 = (y * W + x) * 3;
      var s = s0 - curv * t * t;
      if (tau >= 0 && tau <= Lb && Math.abs(s) < 5 * sigv) {
        var lam = lamAt(tau), k2 = (lam - l0) / step, i0 = Math.floor(k2), fr = k2 - i0;
        var val = (S[i0] + cs[i0]) * (1 - fr) + (S[i0 + 1] + cs[i0 + 1]) * fr, vert = Math.exp(-0.5 * (s / sigv) * (s / sigv));
        var c0 = C[i0], c1 = C[i0 + 1];
        img[o3] = val * vert * (c0[0] * (1 - fr) + c1[0] * fr); img[o3 + 1] = val * vert * (c0[1] * (1 - fr) + c1[1] * fr); img[o3 + 2] = val * vert * (c0[2] * (1 - fr) + c1[2] * fr);
        var lum = (img[o3] + img[o3 + 1] + img[o3 + 2]) / 3; if (lum > peak) peak = lum;
      }
    }
    var gain = peakLevel / peak;
    // нөлінші ретті жарқыл (шам өзі), ауа жарығы, виньетка, шу
    var bx = cx - Lb / 2 * ct - blobOff * ct, by2 = cy - Lb / 2 * st - blobOff * st, amb = U(0.004, 0.03), gradA = U(-0.4, 0.4), gradDir = U(0, 6.28);
    var read = o.read !== undefined ? o.read : U(0.002, 0.01), out = new Uint8ClampedArray(W * H * 4);
    for (y = 0; y < H; y++) for (x = 0; x < W; x++) {
      var o3b = (y * W + x) * 3, d2 = ((x - bx) * (x - bx) + (y - by2) * (y - by2)), br = Math.exp(-d2 / (2 * blobR * blobR)) * 4 + 0.35 * Math.exp(-Math.sqrt(d2) / (blobR * 5));
      var rx = (x - W / 2) / (W / 2), ry = (y - H / 2) / (H / 2), vig = 1 - 0.28 * (rx * rx + ry * ry);
      var grad = 1 + gradA * 0.5 * (Math.cos(gradDir) * rx + Math.sin(gradDir) * ry);
      var base = amb * grad;
      for (var ch = 0; ch < 3; ch++) {
        var v = (img[o3b + ch] * gain + br * (ch === 2 ? 0.9 : 1) * 0.5) * vig + base;
        v += Math.sqrt(Math.max(v, 0)) * 0.012 * g() + read * g();
        out[(y * W + x) * 4 + ch] = Math.round(srgb(v) * 255);
      }
      out[(y * W + x) * 4 + 3] = 255;
    }
    var geo = { cx: cx, cy: cy, theta: theta, Lb: Lb, lamLo: lamLo, lamHi: lamHi, flip: flip, sigv: sigv, curv: curv, fwhm: fwhm, cutR: cutR, blobR: blobR, blobOff: blobOff };
    // (x,y) нүктесіне сәйкес шынайы толқын ұзындығы
    geo.lambdaAt = function (px, py) {
      var ddx = px - cx, ddy = py - cy, t = ddx * ct + ddy * st;
      return lamAt(t + Lb / 2);
    };
    return { img: { data: out, width: W, height: H }, truth: scene.truth, geo: geo, title: source.title };
  }

  // таңдау бойынша көз: sel=[{sp, conc}]
  function selectionSource(sel, T, jitter, rng, cont) {
    return { title: sel.map(function (s) { return s.sp.name; }).join(' + '), truth: sel.map(function (s) { return s.sp.name; }),
             build: function () { return { lines: nistLines(sel, T, rng, jitter), cont: cont || function () { return 0.01; }, truth: sel.map(function (s) { return s.sp.name; }) }; } };
  }

  return { SOURCES: SOURCES, render: render, nistLines: nistLines, chans: chans, selectionSource: selectionSource };
});
