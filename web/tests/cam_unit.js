/* camera.js бірлік тесттері: калибровка, шыңдар, модель кірісі, жолақты табу. node web/tests/cam_unit.js */
'use strict';
const assert = require('assert');
const P = require('../js/physics.js');
const C = require('../js/camera.js');

const G = (x, c, s) => Math.exp(-0.5 * ((x - c) / s) ** 2);
let n = 0;
const ok = (name, fn) => { fn(); n++; };

ok('calib2: екі нүкте', () => {
  const c = C.calib2(100, 435.8, 700, 611.6);
  assert(Math.abs(c.a - 175.8 / 600) < 1e-9 && Math.abs(c.a * 100 + c.b - 435.8) < 1e-9);
  assert.strictEqual(C.calib2(5, 400, 5, 500), null);
});

ok('findPeaks: орны дәлдігі', () => {
  const y = new Float32Array(500);
  [[120, 1], [250, 0.6], [400, 0.3]].forEach(([c, a]) => { for (let i = 0; i < 500; i++) y[i] += a * G(i, c + 0.3, 3); });
  const pk = C.findPeaks(y, 0.1, 4);
  assert.strictEqual(pk.length, 3);
  [120.3, 250.3, 400.3].forEach((c, i) => assert(Math.abs(pk.find(p => Math.abs(p.x - c) < 2).x - c) < 0.25));
});

// CFL профилі: берілген сызықтық сәйкестік пен шу
function cflProfile(a, b, M, fwhm, seed) {
  const rng = P.mulberry32(seed), gs = P.gaussSource(rng), p = new Float32Array(M), s = fwhm / 2.3548;
  const lines = [[404.66, 0.12], [435.83, 0.5], [487.7, 0.3], [544.5, 1.0], [578.0, 0.18], [611.6, 0.95], [650, 0.14], [690, 0.2]];
  for (let i = 0; i < M; i++) {
    const l = a * i + b;
    let v = 0.04 + 0.4 * G(l, 450, 22);
    lines.forEach(([c, h]) => { v += h * G(l, c, s); });
    p[i] = Math.max(0, v + 0.01 * gs());
  }
  return { p };
}
ok('autoCalibrateCFL: оң және теріс бағыт', () => {
  [[0.55, 385], [-0.5, 760], [0.8, 360], [-0.35, 700]].forEach(([a, b], k) => {
    const M = Math.round(Math.abs(380 / a)) + 40, prof = cflProfile(a, b, M, 5, 10 + k);
    const c = C.autoCalibrateCFL(prof);
    assert(c, `калибровка табылмады a=${a}`);
    assert(Math.abs(c.a - a) / Math.abs(a) < 0.015 && Math.abs((c.a * (M / 2) + c.b) - (a * (M / 2) + b)) < 2.5, `a=${a}: ${c.a.toFixed(3)}, ${c.b.toFixed(1)}`);
  });
});
ok('autoCalibrateCFL: шыңсыз профиль -> null', () => {
  const p = new Float32Array(600).fill(0.1);
  assert.strictEqual(C.autoCalibrateCFL({ p }), null);
});

ok('toModelInput: шың дұрыс торға түседі, max=1', () => {
  const M = 800, a = 0.5, b = 390, p = new Float32Array(M);
  for (let i = 0; i < M; i++) p[i] = 0.05 + G(a * i + b, 589, 3);
  const inp = C.toModelInput(p, { a, b }, { smooth: 0 });
  assert(inp && inp.x.length === 1024);
  let im = 0; inp.x.forEach((v, i) => { if (v > inp.x[im]) im = i; });
  assert(Math.abs(C.WIN_LO + im * C.WIN_DX - 589) < 1.0, 'шың орны');
  assert(Math.abs(inp.x[im] - 1) < 1e-6 && Math.min(...inp.x) >= 0);
  assert(inp.coverage > 0.9);
  // қамту: профиль 380–780-нің жартысын ғана қамтиса
  const half = C.toModelInput(p.slice(0, 400), { a, b });
  assert(half.coverage < 0.55 && half.coverage > 0.35);
});

// сурет: айналған түсті жолақ + қаныққан ақ жарқыл
function bandImage(W, H, thetaDeg, withBlob) {
  const d = new Uint8ClampedArray(W * H * 4), th = thetaDeg * Math.PI / 180, ct = Math.cos(th), st = Math.sin(th);
  const cx = W * 0.55, cy = H * 0.5, L = W * 0.55, sv = 6, rng = P.mulberry32(3), gs = P.gaussSource(rng);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const dx = x - cx, dy = y - cy, t = dx * ct + dy * st, s = -dx * st + dy * ct, tau = t / L + 0.5;
    let r = 0.02, g = 0.02, b = 0.02;
    if (tau >= 0 && tau <= 1) {
      const v = Math.exp(-0.5 * (s / sv) ** 2) * (0.35 + 0.65 * Math.sin(tau * 14) ** 2);
      r = 0.02 + v * (0.2 + 0.8 * tau); g = 0.02 + v * (1 - Math.abs(tau - 0.5) * 1.6); b = 0.02 + v * (1 - tau);
    }
    if (withBlob) { const bx = cx - L / 2 * ct - 50 * ct, by = cy - L / 2 * st - 50 * st, d2 = (x - bx) ** 2 + (y - by) ** 2; const br = 3 * Math.exp(-d2 / (2 * 8 * 8)) + 0.25 * Math.exp(-Math.sqrt(d2) / 40); r += br; g += br; b += br; }
    const o = (y * W + x) * 4, q = v => Math.round(255 * Math.min(1, Math.max(0, v <= 0.0031308 ? 12.92 * v : 1.055 * Math.pow(v, 1 / 2.4) - 0.055)));
    d[o] = q(r + 0.004 * gs()); d[o + 1] = q(g + 0.004 * gs()); d[o + 2] = q(b + 0.004 * gs()); d[o + 3] = 255;
  }
  return { data: d, width: W, height: H };
}
ok('detectBand: бұрыш және жарқылды кесу', () => {
  [-14, -3, 0, 6, 17].forEach(a => {
    const img = bandImage(480, 360, a, true), roi = C.detectBand(img);
    assert(roi, `ROI жоқ ${a}°`);
    assert(Math.abs(roi.theta * 180 / Math.PI - a) < 1.5, `${a}° -> ${(roi.theta * 180 / Math.PI).toFixed(2)}`);
    // жолақтың басынан (жарқыл жағы) 50 пиксель бұрын ROI басталмауы керек
    assert(roi.t1 - roi.t0 > 150 && roi.t1 - roi.t0 < 420, 'ROI ұзындығы ' + (roi.t1 - roi.t0));
  });
  assert.strictEqual(C.detectBand({ data: new Uint8ClampedArray(100 * 100 * 4).fill(10), width: 100, height: 100 }), null);
});

ok('extractProfile: толқындылық орны сақталады', () => {
  const img = bandImage(480, 360, 8, false), roi = C.detectBand(img), pr = C.extractProfile(img, roi);
  // sin(14τ)² -> 14·τ/π ≈ 4.46 период: тербеліс бар
  const y = C.gauss(pr.p, 1.5), pk = C.findPeaks(y, 0.1 * Math.max(...y), 8);
  assert(pk.length >= 3, 'шыңдар саны ' + pk.length);
});

ok('colorDirection: көк оң жақта болса flip', () => {
  const M = 200, mk = (rev) => {
    const o = { M, p: new Float32Array(M).fill(1), R: new Float32Array(M), G: new Float32Array(M), B: new Float32Array(M) };
    for (let i = 0; i < M; i++) { const t = rev ? 1 - i / (M - 1) : i / (M - 1); o.R[i] = t; o.B[i] = 1 - t; o.G[i] = 0.5 * (1 - Math.abs(2 * t - 1)); }   // t=0: көк, t=1: қызыл
    return o;
  };
  assert.strictEqual(C.colorDirection(mk(false)).flip, false);
  assert.strictEqual(C.colorDirection(mk(true)).flip, true);
});

console.log(JSON.stringify({ ok: true, tests: n }));
