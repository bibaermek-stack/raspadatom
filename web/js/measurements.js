/* Спектр өлшеулерін CSV арқылы тасымалдау және браузерде сақтау. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(root);
  else root.SL_MEASUREMENTS = factory(root);
})(typeof window !== 'undefined' ? window : globalThis, function (root) {
  'use strict';

  var MAX_CSV_BYTES = 8 * 1024 * 1024;
  var MAX_POINTS = 50000;
  var RECORD_PREFIX = 'raspadatom.measurement.v1.';
  var WAVELENGTH_HEADERS = ['wavelengthnm', 'wavelength', 'lambda', 'lambdanm', 'λ', 'λnm', 'nm', 'толқынұзындығы', 'толқынұзындығынм'];
  var INTENSITY_HEADERS = ['intensity', 'relativeintensity', 'normalizedintensity', 'intensitycounts', 'intensityau', 'signal', 'counts', 'absorbance', 'amplitude', 'қарқындылық'];
  var DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

  function numericArray(values) {
    return Array.isArray(values) || ArrayBuffer.isView(values) && typeof values.length === 'number';
  }

  function create({ name, wavelengths, intensity, peaks = [], source = 'lab', createdAt } = {}) {
    if (!numericArray(wavelengths) || !numericArray(intensity) ||
        wavelengths.length !== intensity.length || wavelengths.length < 2 || wavelengths.length > MAX_POINTS) {
      throw new Error('Спектрде толқын ұзындығы мен қарқындылығы бар 2–50 000 нүкте болуы керек.');
    }
    var samples = Array.from(wavelengths, function (wavelength, i) {
      if (!Number.isFinite(wavelength) || wavelength <= 0 || !Number.isFinite(intensity[i])) {
        throw new Error('Толқын ұзындығы оң сан, ал қарқындылық ақырлы сан болуы керек.');
      }
      return { wavelength: wavelength, intensity: intensity[i] };
    }).sort(function (a, b) { return a.wavelength - b.wavelength; });
    if (samples.some(function (sample, i) { return i > 0 && sample.wavelength === samples[i - 1].wavelength; })) {
      throw new Error('Үлгі нүктелерінде бірдей толқын ұзындығы қайталанады.');
    }
    if (!Array.isArray(peaks) || peaks.length > MAX_POINTS) throw new Error('Спектр шыңдарының тізімі жарамсыз.');
    var minWavelength = samples[0].wavelength, maxWavelength = samples[samples.length - 1].wavelength;
    var checkedPeaks = Array.from(peaks, function (peak) {
      if (!peak || !Number.isFinite(peak.wavelength) || !Number.isFinite(peak.intensity) ||
          peak.wavelength < minWavelength || peak.wavelength > maxWavelength) {
        throw new Error('Шыңның мәндері ақырлы сан болып, толқын ұзындығы спектр аралығында жатуы керек.');
      }
      return { wavelength: peak.wavelength, intensity: peak.intensity };
    }).sort(function (a, b) { return a.wavelength - b.wavelength || b.intensity - a.intensity; });
    var time = createdAt === undefined ? Date.now() : Date.parse(createdAt);
    if (!Number.isFinite(time)) throw new Error('Өлшеу күні жарамсыз.');
    return {
      name: typeof name === 'string' && name.trim() ? name.trim() : 'Спектр',
      wavelengths: samples.map(function (sample) { return sample.wavelength; }),
      intensity: samples.map(function (sample) { return sample.intensity; }),
      peaks: checkedPeaks,
      source: typeof source === 'string' && source ? source : 'lab',
      createdAt: new Date(time).toISOString()
    };
  }

  function toCSV(snapshot) {
    var data = create(snapshot);
    var markedSamples = new Set(), offGrid = [];
    var sampleIndex = new Map(data.wavelengths.map(function (wavelength, i) { return [wavelength, i]; }));
    data.peaks.forEach(function (peak) {
      var i = sampleIndex.get(peak.wavelength);
      if (i !== undefined && data.intensity[i] === peak.intensity && !markedSamples.has(i)) {
        markedSamples.add(i);
      } else {
        offGrid.push([peak.wavelength, peak.intensity, 'peak_only'].join(','));
      }
    });
    var rows = data.wavelengths.map(function (wavelength, i) {
      return [wavelength, data.intensity[i], markedSamples.has(i) ? 1 : 0].join(',');
    });
    return ['wavelength_nm,intensity,peak'].concat(rows, offGrid).join('\r\n') + '\r\n';
  }

  function delimiterFor(text) {
    var quoted = false;
    for (var i = 0; i < text.length; i++) {
      var char = text[i];
      if (char === '"') quoted = !quoted;
      if (quoted) continue;
      if (char === '\n' || char === '\r') break;
      if (char === ';' || char === '\t') return char;
    }
    return ',';
  }

  function csvRows(text, delimiter) {
    var rows = [], row = [], field = '', inQuotes = false, closedQuote = false;
    for (var i = 0; i <= text.length; i++) {
      var char = text[i];
      if (inQuotes) {
        if (char === undefined) throw new Error('CSV бағанындағы тырнақша жабылмаған.');
        if (char !== '"') {
          field += char;
        } else if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
          closedQuote = true;
        }
        continue;
      }
      var endRow = char === undefined || char === '\r' || char === '\n';
      if (endRow || char === delimiter) {
        row.push(field.trim());
        field = '';
        closedQuote = false;
        if (endRow) {
          if (row.length > 1 || row[0]) rows.push(row);
          if (rows.length > 2 * MAX_POINTS + 1) throw new Error('CSV файлында нүктелер тым көп.');
          row = [];
          if (char === '\r' && text[i + 1] === '\n') i++;
        }
      } else if (char === '"' && !closedQuote && !field.trim()) {
        field = '';
        inQuotes = true;
      } else if (char === '"' || closedQuote && char.trim()) {
        throw new Error('CSV бағанындағы тырнақша дұрыс қолданылмаған.');
      } else if (!closedQuote) {
        field += char;
      }
    }
    return rows;
  }

  function csvNumber(field, delimiter) {
    var decimal = delimiter === ',' ? field : field.replace(',', '.');
    return DECIMAL.test(decimal) ? Number(decimal) : NaN;
  }

  function fromCSV(text, name = 'Импортталған спектр') {
    if (typeof text !== 'string' || !text.trim()) throw new Error('CSV файлы бос.');
    if (text.length > MAX_CSV_BYTES || new TextEncoder().encode(text).length > MAX_CSV_BYTES) {
      throw new Error('CSV файлы 8 МБ-тан аспауы керек.');
    }
    var contents = text.trim(), delimiter = delimiterFor(contents), rows = csvRows(contents, delimiter);
    var header = rows[0];
    var bare = header && header.length === 2 && header.every(function (cell) {
      return Number.isFinite(csvNumber(cell, delimiter));
    });
    var headings = bare || !header ? [] : header.map(function (cell) {
      return cell.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, '');
    });
    var wavelengthCol = bare ? 0 : headings.findIndex(function (cell) { return WAVELENGTH_HEADERS.includes(cell); });
    var intensityCol = bare ? 1 : headings.findIndex(function (cell) { return INTENSITY_HEADERS.includes(cell); });
    var peakCol = headings.indexOf('peak');
    if (wavelengthCol === -1 || intensityCol === -1) {
      throw new Error('CSV үшін wavelength_nm (нм) және intensity бағандары қажет.');
    }
    var wavelengths = [], intensity = [], peaks = [];
    for (var i = bare ? 0 : 1; i < rows.length; i++) {
      var row = rows[i], wavelength = csvNumber(row[wavelengthCol] || '', delimiter);
      var value = csvNumber(row[intensityCol] || '', delimiter);
      var flag = peakCol < 0 ? '' : (row[peakCol] || '').toLowerCase();
      if (row.length !== header.length || !Number.isFinite(wavelength) || wavelength <= 0 || !Number.isFinite(value) ||
          !['', '0', '1', 'peak_only'].includes(flag)) {
        throw new Error('CSV файлындағы ' + (i + 1) + '-жолда жарамсыз мән бар.');
      }
      if (flag !== 'peak_only') {
        wavelengths.push(wavelength);
        intensity.push(value);
      }
      if (flag === '1' || flag === 'peak_only') peaks.push({ wavelength: wavelength, intensity: value });
    }
    return create({ name: name, wavelengths: wavelengths, intensity: intensity, peaks: peaks, source: 'csv' });
  }

  function list(storage = root.localStorage) {
    var records = [];
    for (var i = 0; i < storage.length; i++) {
      var key = storage.key(i);
      if (!key || !key.startsWith(RECORD_PREFIX)) continue;
      var raw = storage.getItem(key);
      if (raw === null) continue;
      try {
        records.push({ ...create(JSON.parse(raw)), id: key.slice(RECORD_PREFIX.length) });
      } catch (error) {
        throw new Error('Сақталған өлшеу дерегін оқу мүмкін болмады.', { cause: error });
      }
    }
    return records.sort(function (a, b) { return b.createdAt.localeCompare(a.createdAt); });
  }

  function save(snapshot, storage = root.localStorage) {
    var record = create(snapshot);
    var id = root.crypto && root.crypto.randomUUID ? root.crypto.randomUUID() :
      Date.now().toString(36) + '-' + Math.random().toString(36).slice(2);
    try {
      storage.setItem(RECORD_PREFIX + id, JSON.stringify(record));
    } catch (error) {
      throw new Error('Өлшеуді браузерде сақтау мүмкін болмады.', { cause: error });
    }
    return { ...record, id: id };
  }

  function remove(id, storage = root.localStorage) {
    storage.removeItem(RECORD_PREFIX + id);
  }

  return { create: create, fromCSV: fromCSV, toCSV: toCSV, list: list, save: save, remove: remove, MAX_CSV_BYTES: MAX_CSV_BYTES };
});
