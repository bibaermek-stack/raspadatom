/* Спектр зертханасы: интерфейс. Физика — physics.js, ЖИ — ai.js, 3D — atomscene.js / benchscene.js. */
(function () {
  'use strict';
  var SL = window.SL, D = window.SL_DATA, M = window.SL_MODEL;
  var P = SL.physics, AI = SL.ai, E = SL.engine3d, LV = SL.levels;

  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function html(el, s) { el.innerHTML = s; }
  function maxOf(a) { var m = -Infinity; for (var i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var SUP = { '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻' };
  function sci(a) {
    var e = Math.floor(Math.log10(a)), m = a / Math.pow(10, e);
    return m.toFixed(2) + '×10' + String(e).replace(/./g, function (c) { return SUP[c]; });
  }
  function roman(s) { return s.charge === 0 ? 'I' : s.charge === 1 ? 'II' : 'III'; }
  function sname(s) { return s.sym + ' ' + roman(s); }

  // ================= деректер =================
  var species = D.species, byName = {};
  species.forEach(function (s, k) {
    s.k = k; byName[s.name] = s;
    var st = P.boltzmannStrengths(s, 8000), best = -1, bi = -1;
    for (var i = 0; i < st.length; i++) if (s.wl[i] >= 400 && s.wl[i] <= 700 && st[i] > best) { best = st[i]; bi = i; }
    s.color = bi >= 0 ? P.wlColor(s.wl[bi]) : [0.66, 0.52, 1];
  });
  var model = new AI.Model(M);
  var MARK = ['#f0b64a', '#4fd1ff', '#ff7ab6'];   // таңдалған түрлердің маркер түстері

  var PRESETS = [
    { name: 'Сутегі түтігі', sel: [['H I', 1]], T: 9000 },
    { name: 'Натрий шамы', sel: [['Na I', 1]], T: 6000 },
    { name: 'Неон белгісі', sel: [['Ne I', 1]], T: 8000 },
    { name: 'Сынап шамы', sel: [['Hg I', 1]], T: 9000, noise: 0.0003 },
    { name: 'Гелий', sel: [['He I', 1]], T: 10000 },
    { name: 'Натрий + сынап', sel: [['Na I', 0.8], ['Hg I', 1]], T: 8000 },
    { name: 'Күн: Фраунгофер сызықтары', sel: [['Ca II', 1], ['H I', 1], ['Na I', 0.8]], T: 5800, mode: 'absorption', tau: 3.5 }
  ];

  var state = {
    mode: 'emission', range: 'full', scale: 'lin', scene: 'atom',
    sel: [{ name: 'H I', conc: 1 }],
    T: 9000, fwhm: 1.2, eta: 0.5, noise: 0.01, shift: 0, tau: 3, exposure: 1.4, seed: 11,
    tr: null, mystery: null, ai: null, hover: null
  };
  var cache = { spec: new Float32Array(P.N), clean: null, strengths: [], obs: null };
  var refLines = [];

  // ================= есептеу =================
  function params() { return state.mystery ? state.mystery.par : state; }
  function truthSel() {
    var src = state.mystery ? state.mystery.truth : state.sel;
    return src.map(function (s) { return { sp: byName[s.name], conc: s.conc }; });
  }
  function refSel() { return state.sel.map(function (s) { return { sp: byName[s.name], conc: s.conc }; }); }
  function noiseOpts(par) { return { read: par.noise, shot: par.noise * 0.7, baseline: 0.0005 + par.noise * 0.5 }; }
  function normStrengths(sels, T) {
    var fn = state.mode === 'emission' ? P.boltzmannStrengths : P.absorptionStrengths;
    return sels.map(function (s) {
      var st = fn(s.sp, T), m = maxOf(st);
      for (var i = 0; i < st.length; i++) st[i] = st[i] / m * s.conc;
      return st;
    });
  }

  function compute() {
    var par = params(), truth = truthSel(), N = P.N, i;
    if (!truth.length) { cache.spec = new Float32Array(N); cache.strengths = []; cache.obs = new Float64Array(N); return; }
    if (state.mode === 'emission') {
      var r = P.emissionClean(truth, par.T, par.fwhm, par.eta, par.shift);
      cache.clean = r.clean; cache.strengths = r.strengths; cache.obs = r.clean;
      cache.spec = P.addNoise(r.clean, noiseOpts(par), par.seed);
    } else {
      var a = P.absorptionTransmission(truth, par.T, par.fwhm, par.eta, par.shift, par.tau);
      var cont = P.continuum(5800), obs = new Float64Array(N);
      for (i = 0; i < N; i++) obs[i] = cont[i] * a.transmission[i];
      cache.clean = obs; cache.obs = obs; cache.strengths = a.strengths; cache.cont = cont; cache.tr = a.transmission;
      cache.spec = P.addNoise(obs, noiseOpts(par), par.seed);
    }
  }

  function view() { return state.range === 'vis' ? [380, 760] : [200, 900]; }
  function maxIn(arr, lo, hi) {
    var a = Math.max(0, Math.floor(P.wlToIndex(lo))), b = Math.min(P.N - 1, Math.ceil(P.wlToIndex(hi))), m = 1e-9;
    for (var i = a; i <= b; i++) if (arr[i] > m) m = arr[i];
    return m;
  }

  function buildRefLines() {
    var sels = refSel(), v = view();
    var T = state.mystery ? (state.ai ? state.ai.tempK : 8000) : state.T;
    T = Math.max(P.T_MIN, Math.min(P.T_MAX, T));
    var st = normStrengths(sels, T);
    refLines = P.topLines(sels, st, 100000, v[0], v[1]);
    var top = refLines.length ? refLines[0].st : 1;
    refLines.forEach(function (l) { l.st /= top; });     // көріністегі ең күштіге қатысты
    refLines.forEach(function (l) { l.mark = MARK[l.k % MARK.length]; });
  }

  // ================= түс жолағы =================
  function fit(cv) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return null;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h, dpr: dpr };
  }
  function specAt(wl, arr) {
    var a = arr || cache.spec, x = P.wlToIndex(wl), i = Math.floor(x); if (i < 0 || i >= P.N - 1) return 0;
    var f = x - i; return a[i] * (1 - f) + a[i + 1] * f;
  }

  function drawStrip() {
    var cv = $('#strip'), f = fit(cv); if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h, dpr = f.dpr, v = view(), px = Math.round(W * dpr), py = Math.round(H * dpr);
    var img = ctx.createImageData(px, py), d = img.data, em = state.mode === 'emission';
    var B = new Float32Array(px), C = new Float32Array(px * 3), k = state.exposure * 3.2;
    var gain = Math.min(50, 1 / maxIn(cache.obs, 380, 760));   // көз көрінетін аймақтың ең жарық жеріне бейімделеді
    for (var x = 0; x < px; x++) {
      var wl = v[0] + (x + 0.5) / px * (v[1] - v[0]), s = Math.min(1, specAt(wl, cache.obs) * gain), b, col = P.wlColor(wl);   // жолақ — шусыз «көз» көрінісі
      if (em) b = 1 - Math.exp(-k * Math.sqrt(Math.max(s, 0)));   // көздің логарифмдік сезімталдығына жақын
      else { var c = s; b = Math.pow(Math.max(c, 0), 0.6) * Math.min(1, 0.55 + state.exposure * 0.3); }
      if (!P.isVisible(wl)) b *= em ? 0.75 : 0.5;
      B[x] = b; C[x * 3] = col[0]; C[x * 3 + 1] = col[1]; C[x * 3 + 2] = col[2];
    }
    // жарқырау (bloom): өлшемі ~ 5 пиксель
    var G = new Float32Array(px), r = Math.max(2, Math.round(5 * dpr));
    for (x = 0; x < px; x++) { var a = 0, n = 0; for (var j = -r; j <= r; j++) { var q = x + j; if (q >= 0 && q < px) { a += B[q]; n++; } } G[x] = a / n; }
    for (var y = 0; y < py; y++) {
      var vy = y / (py - 1), vig = 0.62 + 0.38 * (1 - Math.pow(2 * vy - 1, 2));
      for (x = 0; x < px; x++) {
        var br = em ? Math.min(1.2, B[x] + (em ? 0.55 : 0) * G[x]) : B[x];
        br *= vig;
        var o = (y * px + x) * 4;
        d[o] = Math.min(255, C[x * 3] * br * 255 * 1.05); d[o + 1] = Math.min(255, C[x * 3 + 1] * br * 255 * 1.05);
        d[o + 2] = Math.min(255, C[x * 3 + 2] * br * 255 * 1.05); d[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    ctx.font = '600 10px Inter, system-ui, sans-serif'; ctx.textBaseline = 'top';
    function X(wl) { return (wl - v[0]) / (v[1] - v[0]) * W; }
    if (v[0] < 380) { ctx.fillStyle = 'rgba(200,190,255,0.8)'; ctx.textAlign = 'left'; ctx.fillText('УК', 6, 5); }
    if (v[1] > 760) { ctx.fillStyle = 'rgba(255,150,130,0.8)'; ctx.textAlign = 'right'; ctx.fillText('ИҚ', W - 6, 5); }
    if (state.hover != null) { ctx.fillStyle = 'rgba(255,255,255,0.55)'; var hx = X(state.hover); ctx.fillRect(hx - 0.5, 0, 1, H); }
    if (state.tr) {
      var wl0 = state.tr.sp.wl[state.tr.i];
      if (wl0 >= v[0] && wl0 <= v[1]) { ctx.fillStyle = css('--accent'); var sx = X(wl0); ctx.fillRect(sx - 4, H - 3, 8, 3); }
    }
  }

  // ================= график =================
  var ML = 40, MR = 10, MT = 12, MB = 26;
  function drawGraph() {
    var cv = $('#graph'), f = fit(cv); if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h, v = view(), pw = W - ML - MR, ph = H - MT - MB, bot = MT + ph;
    var th = { bg: css('--graph-bg'), grid: css('--graph-grid'), axis: css('--graph-axis'), curve: css('--graph-curve'), accent: css('--accent'), fg: css('--fg') };
    var dark = th.bg.charAt(1) < '5' || th.bg === '#070b10';
    ctx.fillStyle = th.bg; ctx.fillRect(0, 0, W, H);
    function X(wl) { return ML + (wl - v[0]) / (v[1] - v[0]) * pw; }
    var log = state.scale === 'log', gmax = maxIn(cache.obs, v[0], v[1]);   // шусыз сигналдың ең күшті шыңы = 1
    function Y(val) {
      val = val / gmax;
      if (log) { var u = (Math.log10(Math.max(val, 1e-3)) + 3) / 3; return bot - u * ph; }
      return bot - Math.min(val, 1.05) / 1.05 * ph;
    }
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace'; ctx.lineWidth = 1;
    // тор
    var xs = state.range === 'vis' ? 50 : 100;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var wl = Math.ceil(v[0] / xs) * xs; wl <= v[1]; wl += xs) {
      ctx.strokeStyle = th.grid; ctx.beginPath(); ctx.moveTo(X(wl), MT); ctx.lineTo(X(wl), bot); ctx.stroke();
      ctx.fillStyle = th.axis; ctx.fillText(String(wl), X(wl), bot + 8);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    var yt = log ? [0.001, 0.01, 0.1, 1] : [0, 0.25, 0.5, 0.75, 1];
    yt.forEach(function (t) {
      ctx.strokeStyle = th.grid; ctx.beginPath(); ctx.moveTo(ML, Y(t)); ctx.lineTo(W - MR, Y(t)); ctx.stroke();
      ctx.fillStyle = th.axis; ctx.fillText(log ? (t >= 1 ? '1' : String(t)) : t.toFixed(2).replace(/0$/, ''), ML - 6, Y(t));
    });
    ctx.save(); ctx.translate(10, MT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillStyle = th.axis;
    ctx.fillText(state.range === 'vis' ? 'I (көрінетін max = 1)' : 'I, қатысты', 0, 0); ctx.restore();
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillStyle = th.axis; ctx.fillText('λ, нм', W - MR, H - 1);
    // көрінетін аймақтың кемпірқосақ жолағы
    var g = ctx.createLinearGradient(X(380), 0, X(760), 0);
    for (var s = 0; s <= 12; s++) { var w2 = 380 + (380 * s) / 12; g.addColorStop(s / 12, P.rgbCss(P.wlColor(w2), 1)); }
    ctx.fillStyle = g; ctx.fillRect(Math.max(ML, X(380)), bot, Math.min(X(760), W - MR) - Math.max(ML, X(380)), 3);

    // иілік астындағы түсті бояу
    var i0 = Math.max(0, Math.floor(P.wlToIndex(v[0]))), i1 = Math.min(P.N - 2, Math.ceil(P.wlToIndex(v[1])));
    var fillA = dark ? 0.26 : 0.38;
    for (var i = i0; i <= i1; i++) {
      var wi = P.indexToWl(i), x0 = X(wi), x1 = X(wi + P.DX), yv = Y(cache.spec[i]);
      ctx.fillStyle = P.rgbCss(P.wlColor(wi), fillA);
      ctx.fillRect(x0, yv, Math.max(1, x1 - x0 + 0.5), bot - yv);
    }
    ctx.beginPath();
    for (i = i0; i <= i1; i++) { var xx = X(P.indexToWl(i)), yy = Y(cache.spec[i]); if (i === i0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); }
    ctx.strokeStyle = th.curve; ctx.lineWidth = 1.2; ctx.stroke();

    // анықтамалық сызықтар: төменгі штрихкод + ең күштілерінің атаулары
    refLines.forEach(function (l) {
      var x = X(l.wl); if (x < ML || x > W - MR) return;
      var hgt = 4 + 16 * Math.sqrt(l.st);
      ctx.strokeStyle = l.mark; ctx.globalAlpha = 0.9; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, bot); ctx.lineTo(x, bot - hgt); ctx.stroke(); ctx.globalAlpha = 1;
    });
    ctx.font = '600 11px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    var placed = [];
    refLines.slice(0, 40).forEach(function (l) {
      if (placed.length >= 9) return;
      var x = X(l.wl); if (x < ML + 20 || x > W - MR - 20) return;
      if (placed.some(function (p) { return Math.abs(p - x) < 58; })) return;
      placed.push(x);
      var yv = Y(specAt(l.wl)), ly = Math.max(MT + 12, yv - 7);
      ctx.strokeStyle = l.mark; ctx.globalAlpha = 0.35; ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(x, ly); ctx.lineTo(x, bot); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
      ctx.fillStyle = th.fg; ctx.fillText(sname(l.sp) + ' ' + l.wl.toFixed(1), x, ly);
    });
    // таңдалған сызық
    if (state.tr) {
      var tw = state.tr.sp.wl[state.tr.i], tx = X(tw);
      if (tx >= ML && tx <= W - MR) {
        ctx.strokeStyle = th.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(tx, MT); ctx.lineTo(tx, bot); ctx.stroke();
        ctx.fillStyle = th.accent; ctx.beginPath(); ctx.moveTo(tx, MT + 1); ctx.lineTo(tx - 5, MT - 5); ctx.lineTo(tx + 5, MT - 5); ctx.closePath(); ctx.fill();
      }
    }
    if (state.hover != null) {
      var hx = X(state.hover);
      if (hx >= ML && hx <= W - MR) {
        ctx.strokeStyle = th.fg; ctx.globalAlpha = 0.5; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(hx, MT); ctx.lineTo(hx, bot); ctx.stroke(); ctx.globalAlpha = 1;
      }
    }
  }

  // ================= сызықтар кестесі =================
  function renderTable() {
    var tb = $('#linesTable tbody'), rows = refLines.slice(0, 14);
    if (!rows.length) {
      html(tb, '<tr><td colspan="5" class="hint" style="white-space:normal;padding:14px 8px">' +
        (state.mystery ? 'Гипотеза ретінде түр таңдаңыз: оның NIST сызықтары кестеде және графикте (төменгі штрихтар) шығады.' : 'Түр таңдаңыз.') + '</td></tr>');
      return;
    }
    var out = '';
    rows.forEach(function (l, idx) {
      var sp = l.sp, lo = P.parseLevel(sp.levels[sp.lo[l.i]]), up = P.parseLevel(sp.levels[sp.up[l.i]]);
      var sel = state.tr && state.tr.sp === sp && state.tr.i === l.i;
      var tr = state.mode === 'emission' ? up.label + ' → ' + lo.label : lo.label + ' → ' + up.label;
      out += '<tr data-k="' + l.k + '" data-i="' + l.i + '"' + (sel ? ' aria-selected="true"' : '') + '>' +
        '<td><span class="sw" style="background:' + l.mark + '"></span>' + sname(sp) + '</td>' +
        '<td class="num"><span class="sw" style="border-radius:2px;background:' + P.rgbCss(P.wlColor(l.wl), 1) + '"></span>' + l.wl.toFixed(3) + '</td>' +
        '<td>' + tr + '</td><td class="num">' + (sp.ek[l.i] - sp.ei[l.i]).toFixed(3) + '</td>' +
        '<td><span class="ibar" style="width:' + Math.max(2, Math.round(l.st * 54)) + 'px"></span></td></tr>';
    });
    html(tb, out);
  }

  // ================= ауысу =================
  var atom, bench;
  function selectTransition(sp, i) {
    state.tr = { sp: sp, i: i };
    renderTransition();
    atom.setTransition(sp, i, state.mode);
    drawLevels(); drawGraph(); drawStrip();
    $$('#linesTable tbody tr').forEach(function (tr) {
      var on = byName[sp.name] && +tr.dataset.k === sp.k && +tr.dataset.i === i;
      if (on) tr.setAttribute('aria-selected', 'true'); else tr.removeAttribute('aria-selected');
    });
  }
  function defaultTransition() {
    var cand = refLines.filter(function (l) { return P.isVisible(l.wl); });
    var l = cand[0] || refLines[0];
    if (l) selectTransition(l.sp, l.i); else { state.tr = null; atom.clear(); renderTransition(); drawLevels(); drawGraph(); drawStrip(); }
  }
  function ensureTransition() {
    if (state.tr && state.sel.some(function (s) { return s.name === state.tr.sp.name; })) {
      var v = view(), w = state.tr.sp.wl[state.tr.i];
      if (w >= v[0] && w <= v[1]) return;
    }
    defaultTransition();
  }

  function renderTransition() {
    var box = $('#transInfo'), t = state.tr;
    if (!t) { html(box, '<div class="trans"><span class="hint">Атом ауысуын көру үшін түр таңдаңыз.</span></div>'); $('#sceneCaption').textContent = ''; return; }
    var sp = t.sp, i = t.i, lo = P.parseLevel(sp.levels[sp.lo[i]]), up = P.parseLevel(sp.levels[sp.up[i]]);
    var dE = sp.ek[i] - sp.ei[i], aki = sp.aki[i], hn = sp.name === 'H I' ? P.hydrogenName(lo.n, up.n) : '';
    var a = state.mode === 'emission' ? up : lo, b = state.mode === 'emission' ? lo : up;
    var tau = aki ? 1e9 / aki : null;
    html(box,
      '<div class="trans"><span class="lv">' + a.label + (a.J ? ' <small class="hint">J=' + a.J + '</small>' : '') + '</span><span class="arrow">→</span>' +
      '<span class="lv">' + b.label + (b.J ? ' <small class="hint">J=' + b.J + '</small>' : '') + '</span>' +
      '<span class="nm">' + sname(sp) + ' · ' + sp.kk + (hn ? ' · ' + hn : '') + '</span></div>' +
      '<div class="stats">' +
      stat('λ (ауада)', '<span class="sw" style="border-radius:2px;background:' + P.rgbCss(P.wlColor(sp.wl[i]), 1) + '"></span>' + sp.wl[i].toFixed(3), 'нм' + (sp.wl[i] < 380 ? ' · УК' : sp.wl[i] > 760 ? ' · ИҚ' : '')) + stat('ΔE = hν', dE.toFixed(4), 'эВ') + stat('ν', P.freqTHz(dE).toFixed(1), 'ТГц') +
      stat('A<sub>ki</sub>', aki ? sci(aki) : '—', aki ? 'с⁻¹' : '') + stat('τ ≈ 1/A', tau ? (tau < 1000 ? tau.toFixed(tau < 10 ? 2 : 1) : (tau / 1000).toFixed(1) + ' мк') : '—', tau && tau < 1000 ? 'нс' : (tau ? 'с' : '')) +
      '</div>');
  }
  function stat(k, v, u) { return '<div class="stat"><div class="k">' + k + '</div><div class="v">' + v + (u ? '<small>' + u + '</small>' : '') + '</div></div>'; }

  function drawLevels() {
    var cv = $('#levels');
    if (!state.tr) { var f = fit(cv); if (f) { f.ctx.fillStyle = css('--graph-bg'); f.ctx.fillRect(0, 0, f.w, f.h); } return; }
    LV.draw(cv, state.tr.sp, state.tr.i, state.mode, { grid: css('--graph-grid'), axis: css('--graph-axis'), level: css('--muted'), accent: css('--accent'), fg: css('--fg') });
    $('#levHint').textContent = sname(state.tr.sp) + ': ' + state.tr.sp.levels.length + ' деңгей · ' + state.tr.sp.wl.length + ' сызық';
  }

  // ================= 3D сахналарын жаңарту =================
  function updateBench() {
    var truth = truthSel(), rays = [], em = state.mode === 'emission';
    if (!truth.length) { bench.setData({ mode: state.mode, rays: [] }); return; }
    if (em) {
      P.topLines(truth, cache.strengths, 90, 380, 900).forEach(function (l) { rays.push({ wl: l.wl, I: l.st }); });
      bench.setData({ mode: 'emission', rays: rays, tubeColor: P.mixedColor(truth, cache.strengths) });
    } else {
      for (var wl = 380; wl <= 900; wl += 4) rays.push({ wl: wl, I: Math.min(1, cache.obs[Math.round(P.wlToIndex(wl))] * 1.05) });
      bench.setData({ mode: 'absorption', rays: rays });
    }
  }

  // ================= ЖИ =================
  var aiTimer = 0;
  function runAI() {
    clearTimeout(aiTimer);
    if (state.mode !== 'emission') { state.ai = null; renderAI(); return; }
    aiTimer = setTimeout(function () {
      var t0 = performance.now();
      state.ai = model.predict(cache.spec);
      state.ai.ms = performance.now() - t0;
      renderAI();
      if (state.mystery) { buildRefLines(); renderTable(); drawGraph(); }
    }, 90);
  }
  function renderAI() {
    var box = $('#aiBody'), res = state.ai, my = state.mystery;
    if (state.mode !== 'emission') {
      html(box, '<p class="ai-off">Модель сәулелену спектрлерінде оқытылған, сондықтан жұтылу режимінде ЖИ талдауы өшірулі. «Сәулелену» режиміне өтіңіз.</p>');
      return;
    }
    if (!res) { html(box, '<p class="ai-off">Талдау…</p>'); return; }
    var names = model.species, truthSet = null;
    if (!my) truthSet = state.sel.map(function (s) { return s.name; });
    else if (my.revealed) truthSet = my.truth.map(function (s) { return s.name; });
    var pred = []; names.forEach(function (n, k) { if (res.probs[k] > 0.5) pred.push(k); });
    var order = names.map(function (n, k) { return k; }).sort(function (a, b) { return res.probs[b] - res.probs[a]; });
    var show = order.slice(0, 8);
    if (truthSet) truthSet.forEach(function (n) { var k = names.indexOf(n); if (show.indexOf(k) < 0) show.push(k); });
    var bars = '';
    show.forEach(function (k) {
      var n = names[k], p = res.probs[k], isP = p > 0.5, isT = truthSet ? truthSet.indexOf(n) >= 0 : false, mk = '', cls = isP ? ' pred' : '';
      if (truthSet) { if (isT && isP) { mk = '<span class="mk ok" title="дұрыс">✓</span>'; cls = ' hit'; } else if (isT) mk = '<span class="mk no" title="өткізіп алды">✗</span>'; else if (isP) mk = '<span class="mk no" title="артық">!</span>'; else mk = '<span class="mk"></span>'; }
      else mk = '<span class="mk"></span>';
      bars += '<div class="bar' + cls + '"><span class="num">' + sname(byName[n]) + '</span><span class="track"><span class="fill" style="display:block;width:' + (p * 100).toFixed(1) + '%"></span></span><span class="pc">' + Math.round(p * 100) + '%</span>' + mk + '</div>';
    });
    var verdict = '';
    var predNames = pred.map(function (k) { return sname(byName[names[k]]); });
    if (truthSet) {
      var hit = truthSet.filter(function (n) { return res.probs[names.indexOf(n)] > 0.5; }), missN = truthSet.filter(function (n) { return res.probs[names.indexOf(n)] <= 0.5; });
      var extra = pred.map(function (k) { return names[k]; }).filter(function (n) { return truthSet.indexOf(n) < 0; });
      verdict = 'ЖИ анықтады: <b>' + (predNames.join(', ') || 'ешнәрсе') + '</b>. ';
      if (!missN.length && !extra.length) verdict += '<span class="good">Құрам толық дұрыс.</span>';
      else verdict += (hit.length ? '<span class="good">Дұрыс: ' + hit.map(function (n) { return sname(byName[n]); }).join(', ') + '.</span> ' : '') +
        (missN.length ? '<span class="bad">Өткізіп алды: ' + missN.map(function (n) { return sname(byName[n]); }).join(', ') + '.</span> ' : '') +
        (extra.length ? '<span class="bad">Артық: ' + extra.map(function (n) { return sname(byName[n]); }).join(', ') + '.</span>' : '');
    } else {
      verdict = 'ЖИ-дің пікірі: <b>' + (predNames.join(', ') || 'сенімді түр жоқ') + '</b>. Жауап жасырын: өз болжамыңызды салыстырыңыз.';
    }
    var par = params(), tTrue = (!my || my.revealed) ? par.T : null;
    var f1 = M.metrics && M.metrics.f1 ? ' Валидациядағы F1 = ' + M.metrics.f1.toFixed(2) + '.' : '';
    html(box,
      '<div class="ai-grid"><div><div class="bars">' + bars + '</div><p class="verdict" style="margin:14px 0 0">' + verdict + '</p></div>' +
      '<div><div class="tbox">' +
      '<div class="stat"><div class="k">T, ЖИ бағасы</div><div class="v">' + Math.round(res.tempK / 10) * 10 + '<small>K</small></div></div>' +
      '<div class="stat"><div class="k">T, шынайы</div><div class="v">' + (tTrue != null ? Math.round(tTrue) + '<small>K</small>' : '?') + '</div></div></div>' +
      '<p class="note">Температура — Больцман бөлінуіне негізделген бағалау' + (M.metrics && M.metrics.T_MAE_K ? '; валидациядағы орташа қателік ≈ ' + Math.round(M.metrics.T_MAE_K / 10) * 10 + ' K' : '') + '.' + f1 + ' Модель синтетикалық спектрлерде оқытылған, нақты құралмен тексерілмеген. Есептеу уақыты: ' + res.ms.toFixed(0) + ' мс.</p></div></div>');
  }

  // ================= басқару элементтері =================
  function buildSpeciesGrid() {
    var order = species.slice().sort(function (a, b) { return (a.z * 10 + a.charge) - (b.z * 10 + b.charge); });
    var out = '';
    order.forEach(function (s) {
      out += '<button type="button" class="sp" data-n="' + s.name + '" aria-pressed="false" title="' + s.kk + ' ' + roman(s) + ' · ' + s.wl.length + ' сызық">' +
        '<span class="dot" style="background:' + P.rgbCss(s.color, 1) + '"></span><span class="sym">' + s.sym + '</span><span class="ion">' + roman(s) + '</span></button>';
    });
    html($('#speciesGrid'), out);
  }
  function buildPresets() {
    var out = '';
    PRESETS.forEach(function (p, i) { out += '<button type="button" class="chip" data-i="' + i + '">' + p.name + '</button>'; });
    html($('#presets'), out);
  }

  function fmtT(v) { return Math.round(v) + ' K'; }
  function syncControls() {
    var my = !!state.mystery, par = state;
    $$('#speciesGrid .sp').forEach(function (b) { b.setAttribute('aria-pressed', state.sel.some(function (s) { return s.name === b.dataset.n; }) ? 'true' : 'false'); });
    $('#selCount').textContent = my ? '(гипотеза: ' + state.sel.length + ')' : '(' + state.sel.length + ' / 3)';
    var rows = '';
    state.sel.forEach(function (s, k) {
      var sp = byName[s.name];
      rows += '<div class="sel-row"><span class="nm"><span class="sw" style="background:' + MARK[k % 3] + '"></span>' + sname(sp) + '<small>' + sp.kk + '</small></span>' +
        '<span></span><button class="x" type="button" data-rm="' + k + '" aria-label="Алып тастау">×</button>' +
        (my ? '' : '<input type="range" min="0.1" max="1" step="0.05" value="' + s.conc + '" data-conc="' + k + '" aria-label="Концентрация ' + sname(sp) + '" style="grid-column:1/-1;width:100%;accent-color:var(--accent)">') + '</div>';
    });
    html($('#selList'), rows);
    function set(id, v, out) { $('#in' + id).value = v; $('#out' + id).textContent = out; }
    set('T', par.T, fmtT(par.T)); set('Fwhm', par.fwhm, par.fwhm.toFixed(2) + ' нм'); set('Noise', Math.sqrt(par.noise / 0.06) * 100, par.noise.toFixed(4));
    set('Tau', par.tau, par.tau.toFixed(1)); set('Eta', par.eta, par.eta.toFixed(2)); set('Exp', par.exposure, par.exposure.toFixed(1) + '×');
    set('Shift', par.shift, par.shift.toFixed(2) + ' нм');
    ['T', 'Fwhm', 'Noise', 'Tau', 'Eta', 'Shift'].forEach(function (id) { $('#in' + id).disabled = my; });
    $('#ctlTau').hidden = state.mode !== 'absorption';
    $('#btnNoise').disabled = my;
    $$('#modeSeg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.mode === state.mode ? 'true' : 'false'); b.disabled = my && b.dataset.mode === 'absorption'; });
    $$('#rangeSeg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.range === state.range ? 'true' : 'false'); });
    $$('#scaleSeg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.scale === state.scale ? 'true' : 'false'); });
    $$('#sceneSeg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.scene === state.scene ? 'true' : 'false'); });
    $('#btnMystery').textContent = my ? 'Жаңа қоспа' : 'Белгісіз қоспа';
  }

  function renderBanner() {
    var b = $('#mysteryBanner'), my = state.mystery;
    if (!my) { b.hidden = true; return; }
    b.hidden = false;
    var msg = my.msg || 'Белгісіз газ қоспасы: спектрге қарап құрамын табыңыз. Гипотеза ретінде түр таңдаңыз (оның NIST сызықтары графикте белгіленеді) немесе ЖИ-ден сұраңыз.';
    html(b, '<span class="badge-mystery">Белгісіз қоспа</span><span class="grow">' + msg + '</span>' +
      (my.revealed ? '' : '<button class="btn small primary" id="bCheck" type="button">Тексеру</button><button class="btn small" id="bReveal" type="button">Жауабын көрсету</button>') +
      '<button class="btn small ghost" id="bExit" type="button">Шығу</button>');
  }
  function checkGuess() {
    var my = state.mystery, truth = my.truth.map(function (s) { return s.name; }), g = state.sel.map(function (s) { return s.name; });
    var hit = g.filter(function (n) { return truth.indexOf(n) >= 0; }).length, extra = g.length - hit, miss = truth.length - hit;
    if (!miss && !extra) { my.revealed = true; my.msg = '<b>Дұрыс!</b> Құрам: ' + truth.map(function (n) { return sname(byName[n]); }).join(' + ') + '. T = ' + Math.round(my.par.T) + ' K.'; }
    else my.msg = 'Табылғаны: <b>' + hit + ' / ' + truth.length + '</b>' + (extra ? ', артық түр: <b>' + extra + '</b>' : '') + (miss ? ', жетіспейтіні: <b>' + miss + '</b>' : '') + '. Әлі де байқап көріңіз.';
    renderBanner(); renderAI();
  }
  function startMystery() {
    var rnd = Math.random, k = 1 + Math.floor(rnd() * 3), idx = [];
    while (idx.length < k) { var j = Math.floor(rnd() * species.length); if (idx.indexOf(j) < 0) idx.push(j); }
    var truth = idx.map(function (j) { return { name: species[j].name, conc: Math.round(Math.pow(10, -0.9 * rnd()) * 20) / 20 }; });
    state.mode = 'emission';
    state.mystery = {
      truth: truth, revealed: false, msg: '',
      par: { T: Math.round((5000 + rnd() * 9000) / 100) * 100, fwhm: 0.8 + rnd() * 1.6, eta: rnd(), noise: 0.005 + rnd() * 0.02, shift: (rnd() - 0.5) * 0.4, tau: 3, exposure: state.exposure, seed: 1 + Math.floor(rnd() * 1e6) }
    };
    state.sel = []; state.tr = null;
    refreshAll();
  }
  function exitMystery() {
    var my = state.mystery; if (!my) return;
    state.sel = my.truth.map(function (s) { return { name: s.name, conc: s.conc }; });
    state.T = my.par.T; state.fwhm = my.par.fwhm; state.eta = my.par.eta; state.noise = my.par.noise; state.shift = my.par.shift; state.seed = my.par.seed;
    state.mystery = null; state.tr = null;
    refreshAll();
  }

  function applyPreset(p) {
    state.mystery = null;
    state.sel = p.sel.map(function (s) { return { name: s[0], conc: s[1] }; });
    state.T = p.T; state.mode = p.mode || 'emission'; if (p.tau) state.tau = p.tau; state.noise = p.noise === undefined ? 0.01 : p.noise;
    state.tr = null; state.seed += 1;
    refreshAll();
  }

  // ================= жаңарту тізбегі =================
  var rafPending = false;
  function redrawSoon() { if (rafPending) return; rafPending = true; requestAnimationFrame(function () { rafPending = false; drawStrip(); drawGraph(); drawLevels(); }); }

  function refreshSpectrum(keepTr) {
    compute(); buildRefLines();
    if (!keepTr) ensureTransition();
    renderTable(); drawStrip(); drawGraph(); updateBench(); runAI();
  }
  function refreshAll() {
    syncControls(); renderBanner();
    compute(); buildRefLines();
    if (state.tr && !state.sel.some(function (s) { return s.name === state.tr.sp.name; })) state.tr = null;
    if (!state.tr) defaultTransition(); else { renderTransition(); atom.setTransition(state.tr.sp, state.tr.i, state.mode); drawLevels(); }
    atom.setMode(state.mode);
    renderTable(); drawStrip(); drawGraph(); updateBench(); runAI();
  }

  function toggleSpecies(name) {
    var k = state.sel.findIndex(function (s) { return s.name === name; });
    if (k >= 0) { if (state.sel.length === 1 && !state.mystery) return; state.sel.splice(k, 1); }
    else { state.sel.push({ name: name, conc: 1 }); if (state.sel.length > 3) state.sel.shift(); }
    state.tr = state.tr && state.sel.some(function (s) { return s.name === state.tr.sp.name; }) ? state.tr : null;
    if (state.mystery) { syncControls(); buildRefLines(); if (!state.tr) defaultTransition(); renderTable(); drawGraph(); return; }
    refreshAll();
  }

  function bind() {
    $('#speciesGrid').addEventListener('click', function (e) { var b = e.target.closest('.sp'); if (b) toggleSpecies(b.dataset.n); });
    $('#presets').addEventListener('click', function (e) { var b = e.target.closest('.chip'); if (b) applyPreset(PRESETS[+b.dataset.i]); });
    $('#selList').addEventListener('click', function (e) {
      var x = e.target.closest('[data-rm]'); if (!x) return;
      if (state.sel.length === 1 && !state.mystery) return;
      state.sel.splice(+x.dataset.rm, 1); state.tr = null; refreshAll();
    });
    $('#selList').addEventListener('input', function (e) {
      var k = e.target.dataset.conc; if (k == null) return;
      state.sel[+k].conc = +e.target.value; refreshSpectrum(true);
    });
    function slider(id, key, fmt) {
      $('#in' + id).addEventListener('input', function (e) {
        state[key] = key === 'noise' ? 0.06 * Math.pow(+e.target.value / 100, 2) : +e.target.value;
        $('#out' + id).textContent = fmt(state[key]);
        if (key === 'exposure') { drawStrip(); return; }
        refreshSpectrum(key !== 'T' && key !== 'tau');
        if (key === 'T' && state.tr) { /* қатынастар өзгереді, кестеге әсер етеді */ }
      });
    }
    slider('T', 'T', fmtT); slider('Fwhm', 'fwhm', function (v) { return v.toFixed(2) + ' нм'; });
    slider('Noise', 'noise', function (v) { return v.toFixed(4); }); slider('Tau', 'tau', function (v) { return v.toFixed(1); });
    slider('Eta', 'eta', function (v) { return v.toFixed(2); }); slider('Exp', 'exposure', function (v) { return v.toFixed(1) + '×'; });
    slider('Shift', 'shift', function (v) { return v.toFixed(2) + ' нм'; });
    $('#btnNoise').addEventListener('click', function () { state.seed += 1; refreshSpectrum(true); });
    $('#btnMystery').addEventListener('click', startMystery);
    $('#mysteryBanner').addEventListener('click', function (e) {
      var id = e.target.id; if (id === 'bCheck') checkGuess(); else if (id === 'bReveal') { state.mystery.revealed = true; state.mystery.msg = 'Жауабы: <b>' + state.mystery.truth.map(function (s) { return sname(byName[s.name]) + ' (' + Math.round(s.conc * 100) + '%)'; }).join(' + ') + '</b>, T = ' + Math.round(state.mystery.par.T) + ' K.'; renderBanner(); renderAI(); }
      else if (id === 'bExit') exitMystery();
    });
    $$('#modeSeg button').forEach(function (b) { b.addEventListener('click', function () { if (state.mode === b.dataset.mode) return; state.mode = b.dataset.mode; refreshAll(); }); });
    $$('#rangeSeg button').forEach(function (b) { b.addEventListener('click', function () { state.range = b.dataset.range; syncControls(); refreshSpectrum(false); }); });
    $$('#scaleSeg button').forEach(function (b) { b.addEventListener('click', function () { state.scale = b.dataset.scale; syncControls(); drawGraph(); }); });
    $$('#sceneSeg button').forEach(function (b) { b.addEventListener('click', function () { setScene(b.dataset.scene); }); });
    $('#linesTable tbody').addEventListener('click', function (e) {
      var tr = e.target.closest('tr'); if (!tr || tr.dataset.k == null) return;
      selectTransition(species[+tr.dataset.k], +tr.dataset.i);
    });

    // спектр бойынша меңзер / басу
    function wlFrom(e, cv, left, right) {
      var r = cv.getBoundingClientRect(), x = e.clientX - r.left, v = view();
      var f = (x - left) / (r.width - left - right);
      return f < 0 || f > 1 ? null : v[0] + f * (v[1] - v[0]);
    }
    function candidatesAt(wl) {
      var v = view(), pw = Math.max(200, $('#graph').clientWidth - ML - MR);
      var tol = Math.max(0.9, params().fwhm * 0.8, 7 * (v[1] - v[0]) / pw), out = [];   // ~7 пиксель
      var pool = state.sel.length ? state.sel.map(function (s) { return byName[s.name]; }) : species;
      pool.forEach(function (sp) {
        var i = P.nearestLine(sp, wl);
        if (Math.abs(sp.wl[i] - wl) <= tol) out.push({ sp: sp, i: i, d: Math.abs(sp.wl[i] - wl), w: Math.log10(sp.rel[i] + 1) / Math.log10(sp.rel[P.nearestLine(sp, sp.wl[i])] + 2) });
      });
      return out.sort(function (a, b) { return a.d - b.d; }).slice(0, 5);
    }
    function onMove(cv, left, right) {
      return function (e) {
        var wl = wlFrom(e, cv, left, right);
        state.hover = wl;
        var ro = $('#readout');
        if (wl == null) { state.hover = null; ro.innerHTML = ''; redrawSoon(); return; }
        var c = candidatesAt(wl), s = specAt(wl);
        var h = '<span>λ = <b class="num">' + wl.toFixed(2) + ' нм</b></span><span>I = <b class="num">' + s.toFixed(3) + '</b></span><span>hν = <b class="num">' + (P.HC / wl).toFixed(3) + ' эВ</b></span>';
        if (c.length) h += '<span>Жақын: ' + c.map(function (q) { return '<b>' + sname(q.sp) + ' ' + q.sp.wl[q.i].toFixed(2) + '</b>'; }).join(', ') + '</span>';
        ro.innerHTML = h; redrawSoon();
      };
    }
    function onClick(cv, left, right) {
      return function (e) {
        var wl = wlFrom(e, cv, left, right); if (wl == null) return;
        var c = candidatesAt(wl); if (!c.length) return;
        if (!state.sel.some(function (s) { return s.name === c[0].sp.name; }) && state.mystery) { /* гипотезаға қоспаймыз: тек атомды көрсетеміз */ }
        selectTransition(c[0].sp, c[0].i);
      };
    }
    var strip = $('#strip'), graph = $('#graph');
    strip.addEventListener('pointermove', onMove(strip, 0, 0)); graph.addEventListener('pointermove', onMove(graph, ML, MR));
    [strip, graph].forEach(function (cv) { cv.addEventListener('pointerleave', function () { state.hover = null; $('#readout').innerHTML = ''; redrawSoon(); }); });
    strip.addEventListener('click', onClick(strip, 0, 0)); graph.addEventListener('click', onClick(graph, ML, MR));

    // 3D басқаруы
    $('#btnPlay').addEventListener('click', function () {
      var on = !(atom.playing); atom.playing = on; bench.playing = on; this.textContent = on ? 'Тоқтату' : 'Ойнату';
    });
    $('#btnReplay').addEventListener('click', function () { atom.replay(); bench.time = 0; });
    $('#inSpeed').addEventListener('input', function (e) { atom.speed = bench.speed = +e.target.value; });
    $('#inRotate').addEventListener('change', function (e) { atom.vp.autoRotate = e.target.checked; });
    window.addEventListener('resize', redrawSoon);
    if (typeof ResizeObserver !== 'undefined') { var ro2 = new ResizeObserver(redrawSoon); ro2.observe($('#graph')); ro2.observe($('#levels')); }
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    if (mq.addEventListener) mq.addEventListener('change', redrawSoon);
    new MutationObserver(redrawSoon).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  var CAP_BENCH = 'Снелл заңы: көк сәуле күштірек бұрылады. Дисперсия үлкейтілген; УК әйнекте жұтылады.';
  function setScene(name) {
    state.scene = name;
    $('#atomCanvas').hidden = name !== 'atom'; $('#benchCanvas').hidden = name !== 'bench';
    $$('#sceneSeg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.scene === name ? 'true' : 'false'); });
    $('#transInfo').hidden = name !== 'atom';
    $('#sceneCaption').textContent = name === 'bench' ? CAP_BENCH : (atom.phaseText || '');
    if (name === 'atom' && state.tr && atom.tr && atom.onPhase) atom.onPhase(atom._caption(atom.phaseName || 'ground'));
    $('#inRotate').disabled = name === 'bench';
  }

  // ================= іске қосу =================
  function init() {
    $('#nSpecies').textContent = species.length;
    $('#footer').innerHTML = 'Деректер: NIST Atomic Spectra Database (толқын ұзындығы ауада, деңгей энергиялары мен салыстырмалы интенсивтілік). ' +
      'Сәулелену интенсивтілігі I ∝ Rel·exp(−E<sub>k</sub>/kT) (T<sub>ref</sub> = 10 000 K), профиль — pseudo-Voigt; жұтылу күші S ∝ Rel·λ³·exp(E<sub>k</sub>/kT<sub>ref</sub>)·exp(−E<sub>i</sub>/kT) NIST-тегі Rel шамасынан бағаланған. ' +
      'Атомдағы орбита радиусы схемалық (r ∝ n<sub>eff</sub>); сутегі үшін Бор моделі. Модель: ' + M.layers + ' қабатты Transformer, ' + (M.tensors.reduce(function (a, t) { return a + t.shape.reduce(function (x, y) { return x * y; }, 1); }, 0)).toLocaleString('ru-RU') + ' параметр.';
    var rm = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    atom = new SL.atomscene.AtomScene($('#atomCanvas'));
    bench = new SL.benchscene.BenchScene($('#benchCanvas'));
    if (rm) { atom.vp.autoRotate = false; $('#inRotate').checked = false; }
    atom.onPhase = function (text) { atom.phaseText = text; if (state.scene === 'atom') $('#sceneCaption').textContent = text; };
    buildSpeciesGrid(); buildPresets(); bind();
    $('#aiMeta').textContent = '1D-CNN + Transformer · синтетикалық NIST спектрлерінде оқытылған';
    refreshAll();
    setScene('atom');
    window.__SL = { state: state, atom: atom, bench: bench, cache: cache, model: model, startMystery: startMystery, applyPreset: applyPreset, PRESETS: PRESETS, refreshAll: refreshAll };
    var last = performance.now();
    function loop(now) {
      var dt = Math.min(0.05, (now - last) / 1000); last = now;
      if (!document.hidden) { (state.scene === 'atom' ? atom : bench).frame(dt); }
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { redrawSoon(); });
  init();
})();
