'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const M = require('../js/measurements.js');

const snapshot = {
  name: 'Неон',
  source: 'camera',
  wavelengths: [588.2, 600, 620],
  intensity: [1, -0.5, 0.1],
  peaks: [{ wavelength: 588.2, intensity: 1 }, { wavelength: 608.65, intensity: 0.55 }],
  createdAt: '2026-11-23T05:00:00.000Z'
};

test('CSV preserves sample values and both on-grid and fitted peaks', () => {
  const csv = M.toCSV(snapshot);
  assert.match(csv, /^wavelength_nm,intensity,peak\r\n588\.2,1,1\r\n/);
  assert.match(csv, /608\.65,0\.55,peak_only\r\n$/);
  const imported = M.fromCSV(csv, 'Неон көшірмесі');
  assert.deepEqual(imported.wavelengths, snapshot.wavelengths);
  assert.deepEqual(imported.intensity, snapshot.intensity);
  assert.deepEqual(imported.peaks, snapshot.peaks);
  assert.equal(imported.name, 'Неон көшірмесі');
  assert.equal(imported.source, 'csv');
});

test('external instrument files accept headerless, reordered, quoted and decimal comma data', () => {
  const imported = M.fromCSV('\uFEFF"Intensity (a.u.)";"Wavelength [nm]";"Detector"\r\n-2,5;"601,25";"a;""b"""\r\n4e-1;589,5;"a\nb"');
  assert.deepEqual(imported.wavelengths, [589.5, 601.25]);
  assert.deepEqual(imported.intensity, [0.4, -2.5]);
  assert.deepEqual(M.fromCSV('404\t.5\n500\t-3').intensity, [0.5, -3]);
  assert.deepEqual(M.fromCSV('400,1\n501,2').wavelengths, [400, 501]);
});

test('invalid imports fail before plotting or persistence', () => {
  assert.throws(() => M.fromCSV('wavelength_um,intensity\n0.4,2\n0.5,3'), /wavelength_nm/);
  assert.throws(() => M.fromCSV('400,1\n400,2'), /қайталанады/);
  assert.throws(() => M.fromCSV('400,1\n,\n500,0'), /жарамсыз/);
  assert.throws(() => M.fromCSV('400,1\n500,NaN'), /жарамсыз/);
  assert.throws(() => M.fromCSV('400,1\n500,"0'), /жабылмаған/);
  assert.throws(() => M.fromCSV('400,1\n500,"0"x'), /дұрыс қолданылмаған/);
  assert.throws(() => M.fromCSV('400,1\n500,0\n' + 'а'.repeat(M.MAX_CSV_BYTES / 2)), /8 МБ/);
});

test('camera and lab typed arrays stay aligned when calibrated wavelengths run backwards', () => {
  const data = M.create({
    wavelengths: Float64Array.of(610, 598, 580),
    intensity: Float32Array.of(0.5, 1, 0.25)
  });
  assert.deepEqual(data.wavelengths, [580, 598, 610]);
  assert.deepEqual(data.intensity, [0.25, 1, 0.5]);
});

function memoryStorage() {
  const items = new Map();
  return {
    get length() { return items.size; },
    key: i => [...items.keys()][i] ?? null,
    getItem: key => items.get(key) ?? null,
    setItem: (key, value) => items.set(key, value),
    removeItem: key => items.delete(key)
  };
}

test('history persists independent measurements, sorted newest first, and removes only the selected one', () => {
  const storage = memoryStorage();
  storage.setItem('unrelated', 'preserved');
  const first = M.save(snapshot, storage);
  const latest = M.save({ ...snapshot, name: 'Жаңа өлшеу', createdAt: '2026-11-24T05:00:00.000Z' }, storage);
  assert.deepEqual(M.list(storage), [latest, first]);
  M.remove(first.id, storage);
  assert.deepEqual(M.list(storage), [latest]);
  assert.equal(storage.getItem('unrelated'), 'preserved');
});

test('invalid storage and quota errors are surfaced and preserve existing measurements', () => {
  const storage = memoryStorage();
  const first = M.save(snapshot, storage);
  storage.setItem = () => { throw new Error('quota'); };
  assert.throws(() => M.save(snapshot, storage), /сақтау мүмкін болмады/);
  assert.deepEqual(M.list(storage), [first]);
  const corrupt = memoryStorage();
  corrupt.setItem('raspadatom.measurement.v1.broken', '{}');
  assert.throws(() => M.list(corrupt), /оқу мүмкін болмады/);
});
