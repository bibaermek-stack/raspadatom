/* JS-құбырдың (генератор + модель) сапасы: оқыту таралымындағы кездейсоқ қоспаларда.
   node web/tests/bench.js [N]  ->  JSON {f1, precision, recall, exact_match, T_MAE_K, n}
   Нәтиже Python-дағы валидация метрикасына (model.js ішіндегі `metrics`) жақын болуы керек. */
'use strict';
const D = require('../data/lines.js');
const M = require('../data/model.js');
const P = require('../js/physics.js');
const AI = require('../js/ai.js');

const N = +process.argv[2] || 300;
const rng = P.mulberry32(20241114), gauss = P.gaussSource(rng);
const U = (a, b) => a + (b - a) * rng();
const model = new AI.Model(M);
const names = model.species, byName = Object.fromEntries(D.species.map(s => [s.name, s]));
let tp = 0, fp = 0, fn = 0, exact = 0, tErr = 0;

for (let n = 0; n < N; n++) {
  const k = 1 + Math.floor(rng() * 3), chosen = [];
  while (chosen.length < k) { const j = Math.floor(rng() * names.length); if (!chosen.includes(j)) chosen.push(j); }
  const sel = chosen.map(j => ({ sp: byName[names[j]], conc: Math.pow(10, U(-1, 0)) }));
  const T = U(P.T_MIN, P.T_MAX);
  const r = P.emissionClean(sel, T, U(0.6, 3.0), rng(), gauss() * 0.3);
  const spec = P.addNoise(r.clean, { read: U(0.002, 0.03), shot: U(0, 0.02), baseline: 0.02 }, 1 + Math.floor(rng() * 1e9));
  const out = model.predict(spec);
  let same = true;
  names.forEach((nm, j) => {
    const pred = out.probs[j] > 0.5, truth = chosen.includes(j);
    if (pred && truth) tp++; else if (pred) fp++; else if (truth) fn++;
    if (pred !== truth) same = false;
  });
  if (same) exact++;
  tErr += Math.abs(out.tempK - T);
}
const p = tp / Math.max(tp + fp, 1), rc = tp / Math.max(tp + fn, 1);
console.log(JSON.stringify({ n: N, precision: p, recall: rc, f1: 2 * p * rc / Math.max(p + rc, 1e-9), exact_match: exact / N, T_MAE_K: tErr / N }));
