'use strict';

const assert = require('node:assert/strict');
const C = require('../js/camera.js');
const calibration = require('../js/calibration.js');
const CS = require('../js/camsim.js');
const P = require('../js/physics.js');
const { species } = require('../data/lines.js');

const img = { width: 640, height: 480 };
const roi = {
  cx: 320, cy: 235, theta: 0.2, hv: 8, t0: -100, t1: 100,
  curve: { center: 0, scale: 100, coefficients: [0, -0.2, 3] }
};
const prof = { M: 200 };
const cal = C.calib2(20, 650, 170, 425);
cal.matched = [{ x: 20, wl: 650, name: '1-нүкте' }, { x: 170, wl: 425, name: '2-нүкте' }];

const d = calibration.diagnostics({
  a: 1, b: 400, c: 0.01, kind: 'cfl', rms: 100,
  matched: [{ x: 20, wl: 425, name: 'B' }, { x: 10, wl: 410, name: 'A' }]
});
assert.deepEqual(d.lines, [
  { name: 'A', x: 10, referenceNm: 410, fittedNm: 411, errorNm: 1 },
  { name: 'B', x: 20, referenceNm: 425, fittedNm: 424, errorNm: -1 }
]);
assert.equal(d.rms, 1);
assert.equal(d.maxError, 1);
assert.equal(calibration.diagnostics({ a: 1, b: 400, kind: 'hue', rms: 17 }).rms, 17);

const values = new Map();
const storage = {
  getItem: key => values.get(key),
  setItem: (key, value) => values.set(key, value)
};
const profile = calibration.makeProfile(' Қондырғы A ', cal, img, roi, prof);
assert.equal(profile.name, 'Қондырғы A');
assert.deepEqual(calibration.listProfiles(storage), []);
calibration.saveProfile(storage, profile);
assert.deepEqual(calibration.listProfiles(storage), [profile]);
assert.throws(() => calibration.saveProfile(storage, profile), /Бұл атаумен/);
const restored = calibration.restoreProfile(calibration.listProfiles(storage)[0], img);
assert.deepEqual(restored.roi, roi);
assert.equal(restored.length, prof.M);
assert.equal(C.lam(restored.cal, 170), 425);
assert.equal(calibration.axisMatches(profile.axis, img, roi, prof), true);
assert.equal(calibration.axisMatches(profile.axis, img, { ...roi, t0: -99 }, prof), false);
assert.equal(calibration.axisMatches(profile.axis, img, { ...roi, curve: null }, prof), false);
assert.equal(calibration.axisMatches(profile.axis, img, roi, { M: 201 }), false);
assert.throws(() => calibration.restoreProfile(profile, { ...img, width: 1280 }), /640 × 480/);
assert.throws(() => calibration.makeProfile('X', { ...cal, approx: true }, img, roi, prof), /CFL/);
assert.throws(() => calibration.makeProfile('X', { ...cal, c: 0.01 }, img, roi, prof), /CFL/);
assert.throws(() => calibration.makeProfile('X', cal, img, { ...roi, t1: 1000000 }, prof), /спектр жолағын/);
assert.throws(() => calibration.saveProfile({ ...storage, setItem: () => { throw new Error(); } }, { ...profile, name: 'B' }), /жазу мүмкін/);
assert.deepEqual(calibration.listProfiles(storage), [profile]);
assert.deepEqual(calibration.deleteProfile(storage, profile.name), []);
assert.deepEqual(calibration.listProfiles(storage), []);

const cfl = CS.SOURCES.find(source => source.id === 'cfl');
const sample = CS.render(cfl, species, {}, P.mulberry32(7919)).img;
const autoRoi = C.detectBand(sample);
const autoProf = C.extractProfile(sample, autoRoi);
const autoCal = C.autoCalibrateCFL(autoProf);
const autoProfile = calibration.makeProfile('CFL', autoCal, sample, autoRoi, autoProf);
const autoDiagnostics = calibration.diagnostics(autoProfile.cal);
assert.equal(autoDiagnostics.lines.length, autoCal.n);
assert.ok(Math.abs(autoDiagnostics.rms - autoCal.rms) < 1e-9);
assert.ok(autoDiagnostics.maxError >= autoDiagnostics.rms);

console.log(JSON.stringify({ ok: true }));
