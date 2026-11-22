/* Калибровка қалдықтары және камера қондырғысының сақталатын профильдері. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./camera.js'));
  else root.SL.calibration = factory(root.SL.camera);
})(typeof window !== 'undefined' ? window : globalThis, function (camera) {
  'use strict';

  var STORAGE_KEY = 'raspadatom.calibration-profiles.v1';
  var ROI_KEYS = ['cx', 'cy', 'theta', 'hv', 't0', 't1'];

  function diagnostics(cal) {
    if (!cal || !Number.isFinite(cal.a) || !Number.isFinite(cal.b)) return null;
    var matches = Array.isArray(cal.matched) ? cal.matched : [];
    var lines = matches.filter(function (m) {
      return m && Number.isFinite(m.x) && Number.isFinite(m.wl);
    }).map(function (m) {
      var fitted = camera.lam(cal, m.x);
      return { name: m.name || '', x: m.x, referenceNm: m.wl, fittedNm: fitted, errorNm: fitted - m.wl };
    }).filter(function (row) { return Number.isFinite(row.errorNm); });
    lines.sort(function (a, b) { return a.referenceNm - b.referenceNm; });

    var rms = Number.isFinite(cal.rms) && cal.rms >= 0 ? cal.rms : null;
    var maxError = null;
    if (lines.length) {
      var residuals = lines.map(function (row) { return Math.abs(row.errorNm); });
      rms = Math.hypot.apply(null, residuals) / Math.sqrt(lines.length);
      maxError = Math.max.apply(null, residuals);
    }
    return { kind: cal.kind, rms: rms, maxError: maxError, lines: lines };
  }

  function copyRoi(roi) {
    var copied = {};
    ROI_KEYS.forEach(function (key) { copied[key] = roi[key]; });
    if (roi.curve) {
      copied.curve = { center: roi.curve.center, scale: roi.curve.scale, coefficients: roi.curve.coefficients.slice() };
    }
    return copied;
  }

  function validCurve(curve, limit) {
    return curve == null || (
      Number.isFinite(curve.center) && Math.abs(curve.center) <= limit &&
      Number.isFinite(curve.scale) && curve.scale > 0 && curve.scale <= limit &&
      Array.isArray(curve.coefficients) && curve.coefficients.length === 3 &&
      curve.coefficients.every(function (c) { return Number.isFinite(c) && Math.abs(c) <= limit; })
    );
  }

  function captureAxis(img, roi, prof) {
    if (!img || !roi || !prof || !Number.isSafeInteger(img.width) || !Number.isSafeInteger(img.height) ||
        img.width < 1 || img.height < 1 || img.width > 16384 || img.height > 16384) return null;
    var limit = 2 * Math.hypot(img.width, img.height);
    if (!ROI_KEYS.every(function (key) { return Number.isFinite(roi[key]) && Math.abs(roi[key]) <= limit; }) ||
        roi.hv <= 0 || roi.t1 <= roi.t0 || roi.t1 - roi.t0 > limit ||
        prof.M !== Math.max(32, Math.round(roi.t1 - roi.t0)) || !validCurve(roi.curve, limit)) return null;
    return { width: img.width, height: img.height, length: prof.M, roi: copyRoi(roi) };
  }

  function axisMatches(axis, img, roi, prof) {
    if (!axis || !img || !roi || !prof || axis.width !== img.width || axis.height !== img.height || axis.length !== prof.M) return false;
    var previous = axis.roi;
    function equal(a, b) { return Math.abs(a - b) < 1e-6; }
    if (!previous || !ROI_KEYS.every(function (key) { return equal(previous[key], roi[key]); })) return false;
    var a = previous.curve, b = roi.curve;
    if (!a || !b) return !a && !b;
    return equal(a.center, b.center) && equal(a.scale, b.scale) &&
      a.coefficients.every(function (c, i) { return equal(c, b.coefficients[i]); });
  }

  function copyCal(cal, length) {
    if (!cal || cal.provisional || cal.approx || (cal.kind !== 'cfl' && cal.kind !== 'two-point') ||
        !Number.isFinite(cal.a) || !Number.isFinite(cal.b) || (cal.c != null && !Number.isFinite(cal.c))) return null;
    var c = cal.c || 0;
    var end = length - 1;
    if (!(cal.a * (cal.a + 2 * c * end) > 0) || !Number.isFinite(camera.lam(cal, end))) return null;
    var d = diagnostics(cal);
    var count = d.lines.length;
    if ((cal.kind === 'two-point' && count !== 2) || (cal.kind === 'cfl' && (count < 3 || count > camera.CFL_REF.length)) ||
        !d.lines.every(function (row) { return row.x >= 0 && row.x <= end && row.referenceNm > 0 && typeof row.name === 'string'; }) ||
        new Set(d.lines.map(function (row) { return row.x; })).size !== count ||
        new Set(d.lines.map(function (row) { return row.referenceNm; })).size !== count) return null;
    var saved = {
      a: cal.a, b: cal.b, kind: cal.kind, rms: d.rms, n: count,
      matched: d.lines.map(function (row) { return { x: row.x, wl: row.referenceNm, name: row.name }; })
    };
    if (c) saved.c = c;
    return saved;
  }

  function makeProfile(name, cal, img, roi, prof) {
    name = typeof name === 'string' ? name.trim() : '';
    if (!name) throw new Error('Калибровка профилінің атауын енгізіңіз.');
    var axis = captureAxis(img, roi, prof);
    if (!axis) throw new Error('Профильді сақтау үшін алдымен суреттегі спектр жолағын таңдаңыз.');
    var savedCal = copyCal(cal, axis.length);
    if (!savedCal) throw new Error('Профиль үшін CFL немесе екі нүкте бойынша калибровка жасаңыз.');
    return { name: name, cal: savedCal, axis: axis };
  }

  function validatedProfile(entry) {
    if (!entry || !entry.axis) throw new Error('Калибровка профилінің деректері жарамсыз.');
    return makeProfile(entry.name, entry.cal, entry.axis, entry.axis.roi, { M: entry.axis.length });
  }

  function listProfiles(storage) {
    try {
      var raw = storage.getItem(STORAGE_KEY);
      if (!raw) return [];
      var entries = JSON.parse(raw);
      return entries.map(validatedProfile);
    } catch (error) {
      throw new Error('Калибровка профильдерін браузер қоймасынан оқу мүмкін болмады.');
    }
  }

  function writeProfiles(storage, profiles) {
    try {
      storage.setItem(STORAGE_KEY, JSON.stringify(profiles));
    } catch (error) {
      throw new Error('Калибровка профильдерін браузер қоймасына жазу мүмкін болмады.');
    }
    return profiles;
  }

  function saveProfile(storage, profile) {
    var validated = validatedProfile(profile);
    var profiles = listProfiles(storage);
    if (profiles.some(function (p) { return p.name === validated.name; })) throw new Error('Бұл атаумен калибровка профилі бар.');
    profiles.push(validated);
    return writeProfiles(storage, profiles);
  }

  function deleteProfile(storage, name) {
    var retained = listProfiles(storage).filter(function (p) { return p.name !== name; });
    return writeProfiles(storage, retained);
  }

  function restoreProfile(profile, img) {
    var stored = validatedProfile(profile);
    if (!img || img.width !== stored.axis.width || img.height !== stored.axis.height) {
      throw new Error('Сурет өлшемі профильге сәйкес емес (' + stored.axis.width + ' × ' + stored.axis.height + ' px қажет).');
    }
    return { cal: stored.cal, roi: stored.axis.roi, length: stored.axis.length };
  }

  return {
    diagnostics: diagnostics, captureAxis: captureAxis, axisMatches: axisMatches, makeProfile: makeProfile,
    listProfiles: listProfiles, saveProfile: saveProfile, deleteProfile: deleteProfile, restoreProfile: restoreProfile
  };
});
