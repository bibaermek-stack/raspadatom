/* Камера құбырының толық сынағы (виртуал фотолар): сурет → жолақ → профиль → калибровка → камера моделі.
   node web/tests/cam_bench.js [N]  ->  JSON. Нақты камера емес: симулятор + модель ортақ болжамдарды бөліседі (README-де айтылған). */
'use strict';
const D = require('../data/lines.js');
const M = require('../data/model_cam.js');
const P = require('../js/physics.js');
const AI = require('../js/ai.js');
const CS = require('../js/camsim.js');
const C = require('../js/camera.js');

const N = +process.argv[2] || 150, TTA = process.argv[3] === 'tta';
const rng = P.mulberry32(777), U = (a, b) => a + (b - a) * rng();
const model = new AI.Model(M), names = model.species;
const by = Object.fromEntries(D.species.map(s => [s.name, s]));
const atomic = names.filter(n => by[n]);          // «CFL» сияқты көз кластары атомдық түр емес
const kinds = {}, kindsGate = {}, kindsConf = {};
let roiFail = 0, calFail = 0, cflN = 0, cflOk = 0, cflRms = [];

function add(kind, pred, truth, store) {
  store = store || kinds;
  for (const k of [kind, 'all']) {
    const s = store[k] = store[k] || { tp: 0, fp: 0, fn: 0, exact: 0, n: 0 };
    let same = true;
    names.forEach((nm, j) => {
      const p = pred.has(nm), t = truth.has(nm);
      if (p && t) s.tp++; else if (p) s.fp++; else if (t) s.fn++;
      if (p !== t) same = false;
    });
    if (same) s.exact++; s.n++;
  }
}

for (let n = 0; n < N; n++) {
  const u = rng();
  let src, kind;
  if (u < 0.1) { const cid = rng() < 0.5 ? 'bulb' : 'led'; src = CS.SOURCES.find(s => s.id === cid); kind = 'continuum'; }
  else if (u < 0.22) { src = CS.SOURCES.find(s => s.id === 'cfl'); kind = 'cfl'; }
  else {
    const k = 1 + Math.floor(rng() * 3), ch = [];
    while (ch.length < k) { const j = Math.floor(rng() * atomic.length); if (!ch.includes(j)) ch.push(j); }
    const sel = ch.map(j => ({ sp: by[atomic[j]], conc: Math.pow(10, U(-0.9, 0)) }));
    src = CS.selectionSource(sel, U(5000, 12000), U(0.1, 0.4), rng, () => 0.01 + 0.2 * rng() * 0.1);
    kind = 'nist' + k;
  }
  const r = CS.render(src, D.species, {}, rng), g = r.geo;
  const roi = C.detectBand(r.img);
  if (!roi) { roiFail++; for (const st of [kinds, kindsGate, kindsConf]) add(kind, new Set(), new Set(r.truth), st); continue; }
  const pr = C.extractProfile(r.img, roi), ct = Math.cos(roi.theta), st = Math.sin(roi.theta), Mi = pr.M;
  let cal = null;
  if (kind === 'cfl') {
    cflN++;
    cal = C.autoCalibrateCFL(pr);
    if (cal) { cflOk++; cflRms.push(cal.rms); }
  }
  if (!cal) {      // қолмен екі нүкте: 20% және 80%, ±1 нм қателікпен (пайдаланушы 2 белгілі сызықты шамамен көрсетеді)
    const j1 = Math.round(0.2 * (Mi - 1)), j2 = Math.round(0.8 * (Mi - 1));
    const lam = j => { const t = pr.t0 + j * pr.step; return g.lambdaAt(roi.cx + t * ct, roi.cy + t * st); };
    cal = C.calib2(j1, lam(j1) + 1.0 * (rng() - 0.5) * 2, j2, lam(j2) + 1.0 * (rng() - 0.5) * 2);
    if (kind === 'cfl') calFail++;
  }
  const a = C.assess(pr, cal, model, by, D.species, { tta: TTA }), truth = new Set(r.truth.filter(t => names.includes(t)));
  if (!a) { for (const st of [kinds, kindsGate, kindsConf]) add(kind, new Set(), truth, st); continue; }
  add(kind, new Set(a.pred.map(k => names[k])), truth, kinds);                 // шикі модель
  add(kind, new Set(a.predGated.map(k => names[k])), truth, kindsGate);        // + «үздіксіз спектр» қақпасы
  add(kind, new Set(a.confirmed), truth, kindsConf);                           // + сызық сәйкестігімен расталған
}
const fmt = s => { const p = s.tp / Math.max(s.tp + s.fp, 1), rc = s.tp / Math.max(s.tp + s.fn, 1); return { n: s.n, precision: +p.toFixed(3), recall: +rc.toFixed(3), f1: +(2 * p * rc / Math.max(p + rc, 1e-9)).toFixed(3), exact: +(s.exact / s.n).toFixed(3) }; };
const res = { n: N, roi_fail: roiFail, cfl_auto_cal_ok: cflN ? cflOk / cflN : null, cfl_cal_rms_nm: cflRms.length ? +(cflRms.reduce((a, b) => a + b, 0) / cflRms.length).toFixed(2) : null, by_kind: {} };
res.gate = {}; res.confirmed = {};
Object.keys(kinds).sort().forEach(k => { res.by_kind[k] = fmt(kinds[k]); res.gate[k] = fmt(kindsGate[k]); res.confirmed[k] = fmt(kindsConf[k]); });
console.log(JSON.stringify(res));
