'use strict';

const assert = require('node:assert/strict');
const { test } = require('node:test');
const D = require('../data/lines.js');
const M = require('../data/model_cam.js');
const AI = require('../js/ai.js');
const C = require('../js/camera.js');
const S = require('../js/spectrum-confidence.js');

const names = ['Na I', 'Hg I'];
const calibration = { a: 0.5, b: 380 };
const model = new AI.Model(M);
const byName = Object.fromEntries(D.species.map(species => [species.name, species]));

function assessment(overrides = {}) {
  return {
    cal: calibration,
    inp: { coverage: 0.9 },
    continuum: false,
    narrow: [{ wl: 589 }],
    pred: [0, 1],
    predGated: [0, 1],
    confirmed: ['Na I'],
    res: { probs: [0.91, 0.72] },
    ver: { unexplained: [{ wl: 650 }] },
    ...overrides
  };
}

test('Line evidence separates supported and uncertain model candidates', () => {
  assert.deepEqual(S.summarize(assessment(), names), {
    status: 'partial',
    reasons: ['unverified-candidates', 'unexplained-peaks'],
    supported: [{ index: 0, name: 'Na I', score: 0.91 }],
    uncertain: [{ index: 1, name: 'Hg I', score: 0.72 }],
    unexplainedCount: 1
  });
});

test('Unreliable wavelength calibration keeps every candidate uncertain', () => {
  for (const cal of [{ ...calibration, provisional: true }, { ...calibration, approx: true }, null]) {
    const summary = S.summarize(assessment({ cal }), names);
    assert.equal(summary.status, 'unknown');
    assert.deepEqual(summary.supported, []);
    assert.deepEqual(summary.uncertain.map(candidate => candidate.name), names);
    assert.ok(summary.reasons.includes('unreliable-calibration'));
  }
});

test('Limited coverage and absent narrow peaks block high model scores', () => {
  const lowCoverage = S.summarize(assessment({ inp: { coverage: 0.49 } }), names);
  assert.equal(lowCoverage.status, 'unknown');
  assert.ok(lowCoverage.reasons.includes('limited-coverage'));

  const continuum = S.summarize(assessment({ continuum: true, predGated: [] }), names);
  assert.equal(continuum.status, 'unknown');
  assert.deepEqual(continuum.reasons, ['no-narrow-lines']);
  assert.equal(continuum.unexplainedCount, 0);
});

test('No threshold-crossing candidates keeps the spectrum unknown', () => {
  const summary = S.summarize(assessment({ pred: [], predGated: [], confirmed: [] }), names);
  assert.equal(summary.status, 'unknown');
  assert.ok(summary.reasons.includes('no-model-candidates'));
  assert.deepEqual(summary.uncertain, []);
});

test('Irregular wavelength sampling preserves a known sodium line', () => {
  const wavelengths = Array.from({ length: 392 }, (_, i) => 380 + i + (i % 2) / 4);
  for (const background of [0.02, -2]) {
    const intensity = wavelengths.map(wl => background + Math.exp(-0.5 * ((wl - 589.3) / 1.5) ** 2));
    const { assessment, summary } = S.assessMeasurement({ wavelengths, intensity }, model, byName, D.species, C);
    assert.equal(summary.status, 'supported');
    assert.deepEqual(summary.supported.map(candidate => candidate.name), ['Na I']);
    assert.ok(Math.abs(assessment.narrow[0].wl - 589.3) < 0.5);
  }
});

test('Imports with gaps, no visible wavelengths, or no signal skip inference', () => {
  const cases = [
    { wavelengths: [380, 390, 400], intensity: [0, 1, 0], reason: 'sparse-sampling' },
    { wavelengths: [800, 801], intensity: [0, 1], reason: 'outside-model-range' },
    { wavelengths: [380, 381], intensity: [0, 0], reason: 'no-signal' }
  ];
  for (const { wavelengths, intensity, reason } of cases) {
    const result = S.assessMeasurement({ wavelengths, intensity }, model, byName, D.species, C);
    assert.equal(result.assessment, null);
    assert.equal(result.summary.status, 'unknown');
    assert.deepEqual(result.summary.reasons, [reason]);
  }
});
