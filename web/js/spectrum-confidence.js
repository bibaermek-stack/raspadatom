(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.SL = root.SL || {}; root.SL.spectrumConfidence = factory(); }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  function unknown(reason) {
    return { status: 'unknown', reasons: [reason], supported: [], uncertain: [], unexplainedCount: 0 };
  }

  function summarize(assessment, speciesNames, calibration = assessment && assessment.cal) {
    if (!assessment) return unknown('no-signal');

    const reasons = [];
    const reliableAxis = calibration && !calibration.provisional && !calibration.approx
      && Number.isFinite(calibration.a) && calibration.a !== 0 && Number.isFinite(calibration.b);
    const enoughCoverage = assessment.inp.coverage >= 0.5;
    const hasLines = !assessment.continuum && assessment.narrow.length > 0;
    if (!reliableAxis) reasons.push('unreliable-calibration');
    if (!enoughCoverage) reasons.push('limited-coverage');
    if (!hasLines) reasons.push('no-narrow-lines');
    if (!assessment.pred.length) reasons.push('no-model-candidates');

    const supported = [];
    const uncertain = [];
    const canVerify = reliableAxis && enoughCoverage && hasLines;
    assessment.pred.forEach(index => {
      const candidate = { index, name: speciesNames[index], score: assessment.res.probs[index] };
      if (canVerify && assessment.predGated.includes(index) && assessment.confirmed.includes(candidate.name)) {
        supported.push(candidate);
      } else {
        uncertain.push(candidate);
      }
    });

    if (canVerify && uncertain.length) reasons.push('unverified-candidates');
    const unexplainedCount = hasLines ? assessment.ver.unexplained.length : 0;
    if (unexplainedCount) reasons.push('unexplained-peaks');
    const status = supported.length === 0 ? 'unknown' : reasons.length ? 'partial' : 'supported';
    return { status, reasons, supported, uncertain, unexplainedCount };
  }

  // Interpolate the measured part of the camera model's window onto an even axis.
  // Gaps wider than 5 nm can hide narrow lines, so they preclude model interpretation.
  const MAX_GAP_NM = 5;

  function assessMeasurement(snapshot, model, byName, species, camera) {
    const fail = reason => ({ assessment: null, summary: unknown(reason) });
    const { wavelengths, intensity } = snapshot;
    const lo = Math.max(camera.WIN_LO, wavelengths[0]);
    const hi = Math.min(camera.WIN_HI, wavelengths[wavelengths.length - 1]);
    if (!(hi > lo)) return fail('outside-model-range');

    let scale = 0;
    let first = 0;
    while (wavelengths[first + 1] <= lo) first++;
    for (let i = first; wavelengths[i] < hi; i++) {
      if (wavelengths[i + 1] - wavelengths[i] > MAX_GAP_NM) return fail('sparse-sampling');
      scale = Math.max(scale, Math.abs(intensity[i]), Math.abs(intensity[i + 1]));
    }
    if (!(scale > 0)) return fail('no-signal');

    const size = Math.ceil((hi - lo) / camera.WIN_DX) + 1;
    const cal = { a: (hi - lo) / (size - 1), b: lo, kind: 'csv' };
    const p = new Float32Array(size);
    let left = first;
    let minimum = 0;
    for (let i = 0; i < size; i++) {
      const wavelength = i === size - 1 ? hi : lo + i * cal.a;
      while (wavelengths[left + 1] < wavelength) left++;
      const fraction = (wavelength - wavelengths[left]) / (wavelengths[left + 1] - wavelengths[left]);
      p[i] = (1 - fraction) * (intensity[left] / scale) + fraction * (intensity[left + 1] / scale);
      minimum = Math.min(minimum, p[i]);
    }
    // The camera model expects nonnegative detector intensities.
    if (minimum < 0) for (let i = 0; i < size; i++) p[i] -= minimum;

    const result = camera.assess({ p }, cal, model, byName, species);
    const assessment = result && { ...result, cal };
    return { assessment, summary: summarize(assessment, model.species) };
  }

  return { summarize, assessMeasurement };
});
