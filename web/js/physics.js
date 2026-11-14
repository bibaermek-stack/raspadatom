/* Спектр физикасы: Python-дағы spectra_ai/synth.py генераторының JS көшірмесі (модель сол спектрлерде оқытылған)
   + жұтылу спектрі, түс, деңгей белгілері, Бор радиустары.  Браузерде де, Node-та да жұмыс істейді. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.SL = root.SL || {}; root.SL.physics = factory(); }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  var LO = 200, HI = 900, N = 2048, DX = (HI - LO) / (N - 1);
  var KB = 8.617333262e-5;        // эВ/К
  var T_REF = 10000, T_MIN = 4000, T_MAX = 15000;
  var HC = 1239.84198;            // эВ·нм
  var EV_TO_THZ = 241.7989242;    // 1 эВ = 241.799 ТГц
  var RY = 13.605693;             // эВ

  // ---------- кездейсоқ сандар (seed бойынша қайталанатын шу) ----------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gaussSource(rng) {
    var spare = null;
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var u = 0, v = 0;
      while (u === 0) u = rng();
      v = rng();
      var m = Math.sqrt(-2 * Math.log(u));
      spare = m * Math.sin(2 * Math.PI * v);
      return m * Math.cos(2 * Math.PI * v);
    };
  }

  // ---------- сызық профилі мен интенсивтілік ----------
  function boltzmannStrengths(sp, T) {
    var f = 1 / (KB * T) - 1 / (KB * T_REF), out = new Float64Array(sp.wl.length);
    for (var i = 0; i < out.length; i++) out[i] = sp.rel[i] * Math.exp(-sp.ek[i] * f);
    return out;
  }
  // Жұтылу: S ∝ Rel·λ³·exp(Ek/kT_ref)·exp(−Ei/kT)  (g_k·A_ki ∝ Rel·λ·e^{Ek/kT_ref}, f_ik ∝ g_k A_ki λ², төменгі деңгей Больцман)
  function absorptionStrengths(sp, T) {
    var out = new Float64Array(sp.wl.length);
    for (var i = 0; i < out.length; i++) {
      var w = sp.wl[i] / 500;
      out[i] = sp.rel[i] * w * w * w * Math.exp(sp.ek[i] / (KB * T_REF)) * Math.exp(-sp.ei[i] / (KB * T));
    }
    return out;
  }
  function maxOf(a) { var m = -Infinity; for (var i = 0; i < a.length; i++) if (a[i] > m) m = a[i]; return m; }

  function pseudoVoigtKernel(fwhmPts, eta) {
    var half = Math.trunc(Math.min(Math.max(4 * fwhmPts, 6), 200));
    var k = new Float64Array(2 * half + 1), sigma = fwhmPts / 2.3548, gamma = fwhmPts / 2, mx = 0;
    for (var j = -half; j <= half; j++) {
      var g = Math.exp(-0.5 * (j / sigma) * (j / sigma)), l = 1 / (1 + (j / gamma) * (j / gamma));
      var v = (1 - eta) * g + eta * l;
      k[j + half] = v; if (v > mx) mx = v;
    }
    for (var i = 0; i < k.length; i++) k[i] /= mx;
    return { k: k, half: half };
  }
  function addLines(out, wl, st, shift) {
    for (var i = 0; i < wl.length; i++) {
      var pos = (wl[i] + shift - LO) / DX, i0 = Math.floor(pos), fr = pos - i0;
      if (i0 >= 0 && i0 < N) out[i0] += st[i] * (1 - fr);
      if (i0 + 1 >= 0 && i0 + 1 < N) out[i0 + 1] += st[i] * fr;
    }
  }
  function convolveSame(delta, ker) {
    var out = new Float64Array(N), k = ker.k, half = ker.half;
    for (var j = 0; j < N; j++) {
      var d = delta[j];
      if (d === 0) continue;
      var a = Math.max(0, j - half), b = Math.min(N - 1, j + half);
      for (var n = a; n <= b; n++) out[n] += d * k[half + (n - j)];
    }
    return out;
  }

  // sel: [{sp, conc}]  ->  { clean: Float64Array (max=1), strengths: [Float64Array], gmax }
  function emissionClean(sel, T, fwhmNm, eta, shiftNm) {
    var delta = new Float64Array(N), strengths = [], gmax = 0;
    sel.forEach(function (s) {
      var st = boltzmannStrengths(s.sp, T), m = maxOf(st);
      for (var i = 0; i < st.length; i++) st[i] = st[i] / m * s.conc;
      strengths.push(st); gmax = Math.max(gmax, maxOf(st));
      addLines(delta, s.sp.wl, st, shiftNm);
    });
    var sig = convolveSame(delta, pseudoVoigtKernel(fwhmNm / DX, eta)), mx = maxOf(sig) + 1e-12;
    for (var i = 0; i < N; i++) sig[i] /= mx;
    return { clean: sig, strengths: strengths, gmax: gmax };
  }

  // o: {read, shot, baseline}; айырмасы Python-дан: баптауға болатын шу деңгейі, baseline — a0..a2 σ-сы
  function addNoise(clean, o, seed) {
    var rng = mulberry32(seed), gauss = gaussSource(rng), out = new Float32Array(N);
    var a0 = gauss() * o.baseline, a1 = gauss() * o.baseline, a2 = gauss() * o.baseline, mx = 0;
    for (var i = 0; i < N; i++) {
      var x = -1 + 2 * i / (N - 1), s = clean[i] > 0 ? clean[i] : 0;
      var v = s + a0 + a1 * x + a2 * x * x + gauss() * Math.sqrt(s) * o.shot + gauss() * o.read;
      if (v < 0) v = 0;
      out[i] = v; if (v > mx) mx = v;
    }
    for (var j = 0; j < N; j++) out[j] /= (mx + 1e-8);
    return out;
  }

  // ---------- жұтылу ----------
  function planck(wlNm, T) {
    var l = wlNm * 1e-9, c2 = 1.438777e-2;
    return 1 / (Math.pow(l, 5) * (Math.exp(c2 / (l * T)) - 1));
  }
  function continuum(T) {
    var c = new Float64Array(N), mx = 0;
    for (var i = 0; i < N; i++) { c[i] = planck(LO + i * DX, T); if (c[i] > mx) mx = c[i]; }
    for (var j = 0; j < N; j++) c[j] /= mx;
    return c;
  }
  // -> { transmission (0..1), tau, strengths }
  function absorptionTransmission(sel, T, fwhmNm, eta, shiftNm, tauMax) {
    var delta = new Float64Array(N), strengths = [];
    sel.forEach(function (s) {
      var st = absorptionStrengths(s.sp, T), m = maxOf(st);
      for (var i = 0; i < st.length; i++) st[i] = st[i] / m * s.conc;
      strengths.push(st);
      addLines(delta, s.sp.wl, st, shiftNm);
    });
    var tau = convolveSame(delta, pseudoVoigtKernel(fwhmNm / DX, eta)), tr = new Float64Array(N);
    for (var i = 0; i < N; i++) { tau[i] *= tauMax; tr[i] = Math.exp(-tau[i]); }
    return { transmission: tr, tau: tau, strengths: strengths };
  }

  // ---------- түс ----------
  function bruton(w) {
    var r = 0, g = 0, b = 0, f;
    if (w >= 380 && w < 440) { r = -(w - 440) / 60; b = 1; }
    else if (w < 490) { g = (w - 440) / 50; b = 1; }
    else if (w < 510) { g = 1; b = -(w - 510) / 20; }
    else if (w < 580) { r = (w - 510) / 70; g = 1; }
    else if (w < 645) { r = 1; g = -(w - 645) / 65; }
    else if (w <= 780) { r = 1; }
    if (w >= 380 && w < 420) f = 0.3 + 0.7 * (w - 380) / 40;
    else if (w < 701) f = 1;
    else if (w <= 780) f = 0.3 + 0.7 * (780 - w) / 80;
    else f = 0;
    return [Math.pow(r * f, 0.8), Math.pow(g * f, 0.8), Math.pow(b * f, 0.8)];
  }
  // көрінетін аймақ — нақты түс; УК — ақшыл күлгін, ИҚ — күңгірт қызыл (көзге көрінбейді, жалған түс)
  function wlColor(w) {
    if (w < 380) return [0.66, 0.52, 1.0];
    if (w > 760) return [0.62, 0.12, 0.10];
    return bruton(w);
  }
  function isVisible(w) { return w >= 380 && w <= 760; }
  function rgbCss(c, a) {
    return 'rgba(' + Math.round(c[0] * 255) + ',' + Math.round(c[1] * 255) + ',' + Math.round(c[2] * 255) + ',' + (a === undefined ? 1 : a) + ')';
  }
  // разрядтық түтіктің көзге көрінетін жалпы түсі (көрінетін сызықтардың қосындысы)
  function mixedColor(sel, strengths) {
    var r = 0, g = 0, b = 0;
    sel.forEach(function (s, k) {
      for (var i = 0; i < s.sp.wl.length; i++) {
        var w = s.sp.wl[i]; if (w < 400 || w > 700) continue;
        var c = bruton(w), v = strengths[k][i];
        r += c[0] * v; g += c[1] * v; b += c[2] * v;
      }
    });
    var m = Math.max(r, g, b, 1e-9);
    return [Math.pow(r / m, 0.7), Math.pow(g / m, 0.7), Math.pow(b / m, 0.7)];
  }

  // ---------- сызықтар ----------
  // Қоспаның ең күшті сызықтары: [{k, i, sp, wl, st}] (st — ең күштіге қатысты 0..1)
  function topLines(sel, strengths, n, lo, hi) {
    var all = [], gmax = 0;
    strengths.forEach(function (st) { gmax = Math.max(gmax, maxOf(st)); });
    sel.forEach(function (s, k) {
      for (var i = 0; i < s.sp.wl.length; i++) {
        var w = s.sp.wl[i];
        if (w < lo || w > hi) continue;
        all.push({ k: k, i: i, sp: s.sp, wl: w, st: strengths[k][i] / gmax });
      }
    });
    all.sort(function (a, b) { return b.st - a.st; });
    return all.slice(0, n);
  }
  // wl-ге ең жақын сызық (sp.wl өсу ретімен сұрыпталған)
  function nearestLine(sp, wl) {
    var a = sp.wl, lo = 0, hi = a.length - 1;
    while (hi - lo > 1) { var m = (lo + hi) >> 1; if (a[m] < wl) lo = m; else hi = m; }
    return Math.abs(a[lo] - wl) <= Math.abs(a[hi] - wl) ? lo : hi;
  }

  // ---------- деңгей белгілері ----------
  var SUP = { 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  function sup(s) { return String(s).replace(/\d/g, function (d) { return SUP[d]; }); }
  function fmtOrb(tok) {
    var m = /^(\d+)([spdfghik])(\d*)$/.exec(tok);
    return m ? m[1] + m[2] + (m[3] && m[3] !== '1' ? sup(m[3]) : '') : tok;
  }
  function isClosed(tok) {
    var m = /^(\d+)([spdfghik])(\d*)$/.exec(tok);
    if (!m || !m[3]) return false;
    var cap = { s: 2, p: 6, d: 10, f: 14, g: 18, h: 22, i: 26, k: 30 }[m[2]];
    return +m[3] >= cap;
  }
  function fmtTerm(t) {
    var m = /^(?:([a-z]) )?(\d+)([A-Z])(\*?)$/.exec(t);
    if (m) return (m[1] ? m[1] : '') + sup(m[2]) + m[3] + (m[4] ? '°' : '');
    m = /^(\d+)\[([\d/]+)\](\*?)$/.exec(t);
    if (m) return sup(m[1]) + '[' + m[2] + ']' + (m[3] ? '°' : '');
    return t;
  }
  // "2p6.3s|2S|1/2" -> {conf, short, term, J, n, l, label}
  function parseLevel(s) {
    var p = String(s).split('|');
    if (p.length < 3) {
      var n = parseInt(p[0], 10);
      return { conf: p[0], short: 'n=' + p[0], term: '', J: '', n: isNaN(n) ? 0 : n, l: '', label: 'n=' + p[0] };
    }
    var toks = p[0].split('.').filter(function (t) { return t.charAt(0) !== '('; });
    var last = toks[toks.length - 1] || '', prev = toks[toks.length - 2];
    var short = fmtOrb(last);
    if (prev && !isClosed(prev) && !/^\d+[spdfghik]$/.test(prev)) short = fmtOrb(prev) + short;
    var m = /^(\d+)([spdfghik])/.exec(last);
    var term = fmtTerm(p[1].trim());
    return { conf: p[0], short: short, term: term, J: p[2].trim(), n: m ? +m[1] : 0, l: m ? m[2] : '',
             label: short + ' ' + term };
  }

  // ---------- атом модельдері ----------
  // Орбита радиусына сәйкес тиімді кванттық сан: E_ион − E = Ry·(Z_eff)²/n_eff²
  function nEff(E, ion, charge) {
    var z = charge + 1;
    return Math.max(0.8, z * Math.sqrt(RY / Math.max(ion - E, 0.04)));
  }
  function orbitRadius(neff) { return 0.34 * Math.pow(neff, 0.85); }

  var H_SERIES = ['', 'Лайман', 'Бальмер', 'Пашен', 'Брэкет', 'Пфунд', 'Хэмфри'];
  var GREEK = ['', 'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ'];
  function hydrogenName(nLo, nUp) {
    var s = H_SERIES[nLo]; if (!s) return '';
    var d = nUp - nLo, g = GREEK[d] || ('n=' + nUp);
    return s + ' ' + g + (nLo === 2 ? ' (H' + g + ')' : '');
  }

  function freqTHz(dE) { return dE * EV_TO_THZ; }

  return {
    LO: LO, HI: HI, N: N, DX: DX, KB: KB, T_REF: T_REF, T_MIN: T_MIN, T_MAX: T_MAX, HC: HC, RY: RY,
    mulberry32: mulberry32, gaussSource: gaussSource,
    boltzmannStrengths: boltzmannStrengths, absorptionStrengths: absorptionStrengths,
    pseudoVoigtKernel: pseudoVoigtKernel, addLines: addLines, convolveSame: convolveSame,
    emissionClean: emissionClean, addNoise: addNoise,
    planck: planck, continuum: continuum, absorptionTransmission: absorptionTransmission,
    bruton: bruton, wlColor: wlColor, isVisible: isVisible, rgbCss: rgbCss, mixedColor: mixedColor,
    topLines: topLines, nearestLine: nearestLine,
    parseLevel: parseLevel, nEff: nEff, orbitRadius: orbitRadius, hydrogenName: hydrogenName, freqTHz: freqTHz,
    wlToIndex: function (w) { return (w - LO) / DX; }, indexToWl: function (i) { return LO + i * DX; }
  };
});
