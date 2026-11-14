/* Нақты фотоны камера құбырынан өткізу (диагностика). Суретті алдын ала шикі RGBA-ға айналдырады tools/eval_real.py.
   node web/tests/real_eval.js <base> [roi_json]   (<base>.rgba + <base>.json {w,h})
   roi_json — қолмен ROI: {"cx":..,"cy":..,"theta":..,"hv":..,"t0":..,"t1":..} (радианмен). Жоқ болса, жолақ автоматты табылады. */
'use strict';
const fs = require('fs');
const D = require('../data/lines.js'), M = require('../data/model_cam.js'), AI = require('../js/ai.js'), C = require('../js/camera.js');
const by = Object.fromEntries(D.species.map(s => [s.name, s])), model = new AI.Model(M);
const base = process.argv[2], meta = JSON.parse(fs.readFileSync(base + '.json'));
const img = { data: new Uint8ClampedArray(fs.readFileSync(base + '.rgba')), width: meta.w, height: meta.h };
const manual = process.argv[3] ? JSON.parse(process.argv[3]) : null;
const roi = manual || C.detectBand(img);
const r2 = (v, k = 2) => +v.toFixed(k);
const out = { image: base.split('/').pop(), roi: roi ? { manual: !!manual, theta_deg: r2(roi.theta * 180 / Math.PI, 1), length_px: r2(roi.t1 - roi.t0, 0), single: !!(roi.info && roi.info.single) } : null };
if (!roi) { console.log(JSON.stringify(out)); process.exit(0); }
const pr = C.extractProfile(img, roi), cal = C.autoCalibrateCFL(pr), hue = C.calibrateByHue(pr);
out.hue_calibration = hue ? { a: r2(hue.a, 3), b: r2(hue.b, 1), rms_nm: r2(hue.rms, 1), n: hue.n } : null;
out.cfl_calibration = cal ? { a: r2(cal.a, 3), b: r2(cal.b, 1), c: cal.c ? +cal.c.toExponential(2) : 0, rms_nm: r2(cal.rms), matched: cal.matched.map(m => `${m.name}@${m.x.toFixed(0)}`) } : null;
const use = cal || hue;
if (use) {
  const a = C.assess(pr, use, model, by, D.species);
  if (a) {
    out.calibration_used = cal ? 'cfl' : 'hue (шамамен)';
    out.lineness = r2(a.lin); out.narrow_peaks = a.narrow.length; out.continuum = a.continuum;
    out.top = Array.from(a.res.probs).map((p, k) => [p, model.species[k]]).sort((u, v) => v[0] - u[0]).slice(0, 5).map(([p, n]) => `${n} ${(p * 100).toFixed(0)}%`);
    out.predicted = a.predGated.map(k => model.species[k]);
    out.confirmed = a.confirmed;
  }
}
console.log(JSON.stringify(out));
