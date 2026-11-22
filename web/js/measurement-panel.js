/* Өлшемдерді CSV арқылы тасымалдау, сақтау және салыстыру. */
(function () {
  'use strict';

  const { SL, SL_MEASUREMENTS: measurements, SL_DATA, SL_MODEL_CAM } = window;
  const elements = {};
  for (const name of ['Name', 'File', 'Save', 'Export', 'History', 'Compare', 'Open', 'Remove',
    'Status', 'Graph', 'Summary', 'Analysis']) {
    elements[name] = document.getElementById('measurement' + name);
  }

  const number = new Intl.NumberFormat('kk-KZ', { maximumFractionDigits: 2 });
  const date = new Intl.DateTimeFormat('kk-KZ', { dateStyle: 'short', timeStyle: 'short' });
  const reasons = {
    'outside-model-range': 'Спектр 380–780 нм аралығымен қиылыспайды.',
    'sparse-sampling': 'Өлшенген нүктелердің арасы тым алшақ.',
    'no-signal': 'Талдауға жеткілікті сигнал табылмады.',
    'limited-coverage': 'Өлшенген аралық модель терезесінің аз бөлігін қамтиды.',
    'no-narrow-lines': 'Атомдық сызықтарға ұқсас тар шыңдар жеткіліксіз.',
    'no-model-candidates': 'Модельдің іріктеу шегінен бірде-бір түр өтпеді.',
    'unverified-candidates': 'Кей болжамдарды спектр сызықтары жеткілікті қолдамайды.',
    'unexplained-peaks': 'Түсіндірілмеген шыңдар бар.',
    'unreliable-calibration': 'Толқын ұзындықтарының калибрленуі жеткіліксіз.'
  };
  let active = null;
  let records = [];
  let cameraModel;
  let analysis;
  let importRequest = 0;

  function status(message, error = false) {
    elements.Status.textContent = message;
    elements.Status.classList.toggle('bad', error);
  }

  function attempt(operation) {
    try {
      operation();
    } catch (error) {
      status(error instanceof Error ? error.message : 'Әрекетті орындау мүмкін болмады.', true);
    }
  }

  function findRecord(id) {
    return records.find(record => record.id === id);
  }

  function fillChoices(select, label, description) {
    const previous = select.value;
    select.replaceChildren(new Option(label, ''));
    for (const record of records) select.add(new Option(description(record), record.id));
    if (findRecord(previous)) select.value = previous;
  }

  function refreshHistory() {
    records = measurements.list();
    fillChoices(elements.History, 'Өлшем таңдаңыз', record =>
      record.name + ' · ' + date.format(new Date(record.createdAt)));
    fillChoices(elements.Compare, 'Салыстырмау', record => record.name);
  }

  function updateButtons() {
    elements.Save.disabled = elements.Export.disabled = elements.Name.disabled = !active;
    elements.Open.disabled = elements.Remove.disabled = !findRecord(elements.History.value);
    elements.Compare.disabled = !active || records.length === 0;
  }

  function spectrumSummary(snapshot) {
    const wavelengths = snapshot.wavelengths;
    return snapshot.name + ' · ' + number.format(wavelengths[0]) + '–' +
      number.format(wavelengths[wavelengths.length - 1]) + ' нм · ' + number.format(wavelengths.length) + ' нүкте';
  }

  function setActive(snapshot) {
    active = snapshot;
    importRequest++;
    elements.Name.value = snapshot.name;
    if (snapshot.id && snapshot.id === elements.Compare.value) elements.Compare.value = '';
    analysis = null;
    render();
  }

  function draw() {
    const reference = active && findRecord(elements.Compare.value);
    const summary = active
      ? 'Өлшем: ' + spectrumSummary(active) + (reference ? ' · Үзік сызық: ' + spectrumSummary(reference) : '')
      : 'Зертханадан не камерадан өлшем қосыңыз немесе CSV файлын ашыңыз.';
    elements.Summary.textContent = summary;
    const canvas = elements.Graph;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    const ctx = canvas.getContext('2d');
    const style = getComputedStyle(document.documentElement);
    const color = key => style.getPropertyValue(key).trim();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = color('--graph-bg');
    ctx.fillRect(0, 0, width, height);
    if (!active) return;

    const curves = reference ? [active, reference] : [active];
    const low = Math.min(...curves.map(curve => curve.wavelengths[0]));
    const high = Math.max(...curves.map(curve => curve.wavelengths[curve.wavelengths.length - 1]));
    const signed = curves.some(curve => curve.intensity.some(value => value < 0));
    const left = 42;
    const top = 13;
    const right = width - 14;
    const bottom = height - 40;
    if (right <= left || bottom <= top) return;
    const x = wavelength => left + (wavelength - low) / (high - low) * (right - left);
    const y = value => bottom - (value + (signed ? 1 : 0)) / (signed ? 2 : 1) * (bottom - top);

    ctx.font = '11px system-ui, sans-serif';
    ctx.strokeStyle = color('--graph-grid');
    ctx.fillStyle = color('--graph-axis');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const ticks = Math.max(2, Math.floor((right - left) / 72));
    for (let i = 0; i <= ticks; i++) {
      const wavelength = low + (high - low) * i / ticks;
      const px = x(wavelength);
      ctx.beginPath();
      ctx.moveTo(px, top);
      ctx.lineTo(px, bottom);
      ctx.stroke();
      ctx.fillText(number.format(wavelength), px, bottom + 5);
    }
    ctx.textAlign = 'right';
    ctx.fillText('λ, нм', right, bottom + 22);
    ctx.textBaseline = 'middle';
    for (const value of signed ? [-1, -0.5, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1]) {
      const py = y(value);
      ctx.beginPath();
      ctx.moveTo(left, py);
      ctx.lineTo(right, py);
      ctx.stroke();
      ctx.fillText(number.format(value), left - 5, py);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(left - 2, top - 2, right - left + 4, bottom - top + 4);
    ctx.clip();
    curves.forEach((curve, index) => {
      const scale = curve.intensity.reduce((max, value) => Math.max(max, Math.abs(value)), 0) || 1;
      ctx.strokeStyle = ctx.fillStyle = color(index ? '--focus' : '--graph-curve');
      ctx.lineWidth = 1.6;
      ctx.setLineDash(index ? [6, 4] : []);
      ctx.beginPath();
      curve.wavelengths.forEach((wavelength, i) => {
        const px = x(wavelength);
        const py = y(curve.intensity[i] / scale);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
      for (const peak of curve.peaks) {
        const py = y(Math.max(-1, Math.min(1, peak.intensity / scale)));
        ctx.beginPath();
        ctx.arc(x(peak.wavelength), py, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      }
    });
    ctx.restore();
  }

  function displayAnalysis() {
    elements.Analysis.replaceChildren();
    if (!active || active.source !== 'csv') return;
    elements.Analysis.textContent = 'Талдау 380–780 нм аймағындағы камера моделіне негізделеді. ' +
      'Модель камера спектрін имитациялайтын синтетикалық деректерде оқытылған.';
    if (!SL_MODEL_CAM) return;
    if (!cameraModel) cameraModel = new SL.ai.Model(SL_MODEL_CAM);
    if (!analysis) {
      const species = SL_DATA.species;
      const byName = Object.fromEntries(species.map(entry => [entry.name, entry]));
      analysis = SL.spectrumConfidence.assessMeasurement(active, cameraModel, byName, species, SL.camera).summary;
    }

    const result = document.createElement('ul');
    function add(text) {
      const item = document.createElement('li');
      item.textContent = text;
      result.append(item);
    }
    const displayNames = entries => entries.map(entry => entry.name).join(', ');
    if (analysis.supported.length) add('Сызықтармен қолдау тапқан түрлер: ' + displayNames(analysis.supported) + '.');
    else add('Құрамы белгісіз. Анықтауға жеткілікті дәлел жиналмады.');
    if (analysis.uncertain.length) add('Күмәнді түрлер: ' + displayNames(analysis.uncertain) + '.');
    for (const reason of analysis.reasons) {
      if (reasons[reason]) add(reasons[reason]);
    }
    elements.Analysis.append(result);
  }

  function render() {
    updateButtons();
    draw();
    displayAnalysis();
  }

  function nameSnapshot() {
    return measurements.create({ ...active, name: elements.Name.value });
  }

  elements.Save.addEventListener('click', () => attempt(() => {
    const saved = measurements.save(nameSnapshot());
    refreshHistory();
    elements.History.value = saved.id;
    setActive(saved);
    status('Өлшем осы браузерде сақталды.');
  }));
  elements.Export.addEventListener('click', () => attempt(() => {
    const snapshot = nameSnapshot();
    const url = URL.createObjectURL(new Blob([measurements.toCSV(snapshot)], { type: 'text/csv;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = (snapshot.name.replace(/[\\/:*?"<>|\x00-\x1f]/g, '_').trim() || 'spectrum') + '.csv';
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    status('CSV файлы жүктелуде.');
  }));
  elements.Open.addEventListener('click', () => attempt(() => {
    const snapshot = findRecord(elements.History.value);
    if (snapshot) {
      setActive(snapshot);
      status('Сақталған өлшем ашылды.');
    }
  }));
  elements.Remove.addEventListener('click', () => attempt(() => {
    measurements.remove(elements.History.value);
    refreshHistory();
    render();
    status('Өлшем тарихтан өшірілді.');
  }));
  elements.History.addEventListener('change', updateButtons);
  elements.Compare.addEventListener('change', draw);
  elements.File.addEventListener('change', async () => {
    const file = elements.File.files[0];
    elements.File.value = '';
    if (!file) return;
    const request = ++importRequest;
    let text;
    try {
      if (file.size > measurements.MAX_CSV_BYTES) throw new Error('CSV файлы тым үлкен.');
      text = await file.text();
    } catch (error) {
      if (request === importRequest) status(error.message, true);
      return;
    }
    if (request !== importRequest) return;
    attempt(() => {
      setActive(measurements.fromCSV(text, file.name.replace(/\.(csv|tsv)$/i, '')));
      status('CSV файлы ашылды.');
    });
  });
  window.addEventListener('resize', draw);
  window.addEventListener('storage', () => attempt(() => {
    refreshHistory();
    updateButtons();
    draw();
  }));
  new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  SL.measurementPanel = {
    capture(input) {
      const snapshot = measurements.create(input);
      setActive(snapshot);
      document.querySelector('[data-view="measure"]').click();
      status('Ағымдағы спектр өлшеулерге қосылды.');
      return snapshot;
    }
  };
  attempt(refreshHistory);
  render();
})();
