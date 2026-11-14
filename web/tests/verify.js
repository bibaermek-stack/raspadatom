/* JS ↔ Python сәйкестік тексерісі.  node web/tests/verify.js vectors.json
   1) физика: таза спектр (Больцман + pseudo-Voigt) Python-мен сәйкес келуі керек;
   2) ЖИ: JS forward өтуі PyTorch логиттерімен сәйкес келуі керек. */
'use strict';
const path = require('path');
const fs = require('fs');
const D = require('../data/lines.js');
const M = require('../data/model.js');
const P = require('../js/physics.js');
const AI = require('../js/ai.js');

const vec = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const byName = Object.fromEntries(D.species.map(s => [s.name, s]));
let worstPhys = 0, worstLogit = 0, worstT = 0;

for (const c of vec.physics) {
  const sel = c.species.map((n, i) => ({ sp: byName[n], conc: c.conc[i] }));
  const r = P.emissionClean(sel, c.T, c.fwhm, c.eta, c.shift);
  let d = 0;
  for (let i = 0; i < P.N; i++) d = Math.max(d, Math.abs(r.clean[i] - c.clean[i]));
  worstPhys = Math.max(worstPhys, d);
}
const model = new AI.Model(M);
if (JSON.stringify(model.species) !== JSON.stringify(D.species.map(s => s.name))) throw new Error('түрлер реті сәйкес емес');
for (const c of vec.ai) {
  const out = model.predict(Float32Array.from(c.spec));
  for (let i = 0; i < c.logits.length; i++) worstLogit = Math.max(worstLogit, Math.abs(out.logits[i] - c.logits[i]));
  worstT = Math.max(worstT, Math.abs(out.tNorm - c.t));
}
const res = { physics_cases: vec.physics.length, ai_cases: vec.ai.length,
              max_abs_diff_clean: worstPhys, max_abs_diff_logit: worstLogit, max_abs_diff_tnorm: worstT };
console.log(JSON.stringify(res));
if (worstPhys > 1e-9 || worstLogit > 2e-3 || worstT > 2e-3) { console.error('СӘЙКЕС КЕЛМЕЙДІ'); process.exit(1); }
