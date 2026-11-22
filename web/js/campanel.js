/* «Камера» бөлімінің интерфейсі: сурет (файл / фото / тікелей камера / виртуал камера) → жолақ → спектр → калибровка → ЖИ.
   Есептеу — camera.js, виртуал сурет — camsim.js, модель — ai.js (камера домені, SL_MODEL_CAM). */
(function () {
  'use strict';
  var SL = window.SL, D = window.SL_DATA, MC = window.SL_MODEL_CAM;
  function $(s, r) { return (r || document).querySelector(s); }
  function $$(s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); }
  function html(el, s) { el.innerHTML = s; }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }

  var hasCamera = Boolean(MC && SL && SL.camera);
  var views = { lab: ['#viewLab', '#lab'], cam: ['#viewCam', '#camera'], measure: ['#viewMeasure', '#measurements'] };
  $('#viewSeg').addEventListener('click', function (e) {
    var b = e.target.closest('button[data-view]'); if (b) setView(b.dataset.view);
  });
  window.addEventListener('hashchange', function () { setView(viewFromHash()); });
  if (!hasCamera) {
    $('[data-view=cam]').hidden = true;
    setView(viewFromHash());
    return;
  }
  var P = SL.physics, AI = SL.ai, C = SL.camera, CS = SL.camsim, CAL = SL.calibration;
  var species = D.species, byName = {};
  species.forEach(function (s) { byName[s.name] = s; });
  var model = new AI.Model(MC);
  var MARK = ['#f0b64a', '#4fd1ff', '#ff7ab6'];
  var MAXW = 1280;                           // жүктелген суретті осы ендікке дейін кішірейтеміз

  var cs = {
    img: null, roi: null, base: null, lock: false, prof: null, dir: { flip: false, conf: 0 },
    cal: null, calAxis: null, anchors: [{ x: null }, { x: null }], pick: 0, hover: null, zoom: null, selected: null, flipUser: null,
    ai: null, ver: null, fwhm: 4, live: null, frozen: false, label: '', source: '', drawing: false,
    sim: { seed: 1, geo: null, src: null }, lastAI: 0, aiTimer: 0
  };

  // ================= көрініс ауыстыру =================
  function viewFromHash() {
    if (location.hash === views.measure[1]) return 'measure';
    if (hasCamera && location.hash === views.cam[1]) return 'cam';
    return 'lab';
  }
  function setView(v) {
    if (v === 'cam' && !hasCamera) v = 'lab';
    Object.keys(views).forEach(function (key) { $(views[key][0]).hidden = key !== v; });
    $$('#viewSeg button').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.view === v ? 'true' : 'false'); });
    try { if (location.hash !== views[v][1]) history.replaceState(null, '', views[v][1]); } catch (e) { /* ішкі қорғалған орта */ }
    if (hasCamera && v !== 'cam') { stopLive(); setDrawing(false); }
    window.dispatchEvent(new Event('resize'));
    if (v === 'cam') { drawAll(); if (!cs.img) loadSample(0); }
  }

  // ================= көмекші =================
  function fit(cv) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return null;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: h };
  }
  function msg(id, text, cls) { var el = $(id); el.textContent = text; el.className = 'hint' + (cls ? ' ' + cls : ''); }

  function effectiveRoi() {
    if (!cs.roi) return null;
    var r = cs.roi, a = $('#camAngle').value, s0 = Math.min(98, +$('#camStart').value);
    var s1 = Math.max(+$('#camEnd').value, s0 + 2), span = r.t1 - r.t0, sameAngle = a === cs.baseAngle;
    return {
      cx: r.cx, cy: r.cy, theta: sameAngle ? r.theta : +a * Math.PI / 180, hv: +$('#camThick').value,
      t0: r.t0 + span * s0 / 100, t1: s1 === 100 ? r.t1 : r.t0 + span * s1 / 100,
      curve: sameAngle ? r.curve : undefined
    };
  }
  function setRoiControls(r) {
    $('#camAngle').value = (r.theta * 180 / Math.PI).toFixed(1);
    cs.baseAngle = $('#camAngle').value;
    $('#camThick').value = clamp(Math.round(r.hv), 3, 120);
    $('#camStart').value = 0; $('#camEnd').value = 100;
    roiLabels();
  }
  function roiLabels() {
    $('#camAngleO').textContent = (+$('#camAngle').value).toFixed(1) + '°'; $('#camThickO').textContent = $('#camThick').value + ' px';
    $('#camStartO').textContent = $('#camStart').value + '%'; $('#camEndO').textContent = $('#camEnd').value + '%';
  }

  // ================= кескінді жүктеу =================
  function setImage(img, label, keepRoi) {
    cs.img = img; cs.label = label || '';
    if (!keepRoi) cs.flipUser = null;
    resetGraph();
    var cv = $('#camCanvas'); cv.width = img.width; cv.height = img.height;
    if (!keepRoi || !cs.roi || !cs.lock) {
      var roi = C.detectBand(img);
      cs.roi = roi;
      if (roi) setRoiControls(roi);
    }
    reprocess(true);
    calState();
  }
  function toImageData(source, w, h) {
    var k = Math.min(1, MAXW / w), cw = Math.round(w * k), ch = Math.round(h * k), c = document.createElement('canvas');
    c.width = cw; c.height = ch;
    var x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(source, 0, 0, cw, ch);
    var id = x.getImageData(0, 0, cw, ch);
    return { data: id.data, width: cw, height: ch };
  }
  function loadFile(file) {
    if (!file) return;
    stopLive();
    var done = function (src, w, h) {
      clearCalibration(); cs.source = 'file'; cs.lock = false; $('#camLock').checked = false;
      setImage(toImageData(src, w, h), file.name); updateStatus('Файл: ' + file.name);
    };
    if (window.createImageBitmap) {
      createImageBitmap(file, { imageOrientation: 'from-image' }).then(function (bm) { done(bm, bm.width, bm.height); })
        .catch(function () { viaImg(); });
    } else viaImg();
    function viaImg() {
      var url = URL.createObjectURL(file), im = new Image();
      im.onload = function () { done(im, im.naturalWidth, im.naturalHeight); URL.revokeObjectURL(url); };
      im.onerror = function () { updateStatus('Суретті оқи алмадым: JPG/PNG таңдаңыз'); URL.revokeObjectURL(url); };
      im.src = url;
    }
  }

  // ================= виртуал камера =================
  var SAMPLE_ORDER = ['cfl', 'ne', 'na', 'h', 'he', 'hg', 'kr', 'mix', 'bulb', 'led'];
  function sources() { return SAMPLE_ORDER.map(function (id) { return CS.SOURCES.filter(function (s) { return s.id === id; })[0]; }); }
  function buildSamples() {
    var out = '';
    sources().forEach(function (s, i) { out += '<button type="button" class="chip" data-i="' + i + '">' + s.title + '</button>'; });
    html($('#camSamples'), out);
  }
  // бір қондырғы: геометрия бекітіледі (калибровка басқа көздерге де жарамды); әр түсіруде шу мен экспозиция өзгереді
  function loadSample(i, newSetup) {
    stopLive();
    var src = sources()[i], rng = P.mulberry32(cs.sim.seed * 1000 + i * 17 + (cs.sim.shot = (cs.sim.shot || 0) + 1));
    if (newSetup || !cs.sim.geo) {
      cs.sim.seed = (cs.sim.seed || 1) + (newSetup ? 7 : 0);
      var g0 = CS.render(sources()[0], species, {}, P.mulberry32(cs.sim.seed * 7919)).geo;
      cs.sim.geo = { theta: g0.theta * 180 / Math.PI, Lb: g0.Lb, cx: g0.cx, cy: g0.cy, sigv: g0.sigv, lamLo: g0.lamLo, lamHi: g0.lamHi, flip: g0.flip, fwhm: g0.fwhm, curv: g0.curv, cutR: g0.cutR, blobR: g0.blobR, blobOff: g0.blobOff };
    }
    if (newSetup || cs.source !== 'simulation') {
      clearCalibration(); cs.lock = false; cs.roi = null; $('#camLock').checked = false;
    }
    cs.source = 'simulation';
    var o = cs.sim.geo, r = CS.render(src, species, { theta: o.theta, Lb: o.Lb, fixCx: o.cx, fixCy: o.cy, sigv: o.sigv, lamLo: o.lamLo, lamHi: o.lamHi, flip: o.flip, fwhm: o.fwhm, curv: o.curv, cutR: o.cutR, blobR: o.blobR, blobOff: o.blobOff }, rng);
    cs.sim.src = i;
    $$('#camSamples .chip').forEach(function (b) { b.setAttribute('aria-pressed', +b.dataset.i === i ? 'true' : 'false'); });
    setImage(r.img, src.title, true);
    $('#camTruth').textContent = 'Виртуал: ' + src.title + (r.truth.length ? ' (құрамы: ' + r.truth.join(' + ') + ')' : ' (атомдық түр жоқ)');
    updateStatus('Виртуал камера: ' + src.title);
  }

  // ================= тікелей камера =================
  function stopLive() {
    if (!cs.live) return;
    cs.live.stop = true;
    try { cs.live.stream.getTracks().forEach(function (t) { t.stop(); }); } catch (e) { /* ештеңе */ }
    cs.live = null; $('#camLive').textContent = 'Камераны қосу'; $('#camFreeze').hidden = true;
    $('#camDrawRoi').disabled = false;
  }
  function startLive() {
    if (cs.live) { stopLive(); updateStatus('Камера тоқтатылды'); return; }
    var md = navigator.mediaDevices;
    if (!md || !md.getUserMedia) { updateStatus('Бұл браузерде камера API жоқ: суретті жүктеңіз'); return; }
    md.getUserMedia({ video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false }).then(function (stream) {
      var v = $('#camVideo'); v.srcObject = stream; v.setAttribute('playsinline', ''); v.muted = true;
      return v.play().then(function () {
        clearCalibration(); cs.source = 'live'; cs.label = 'Тікелей камера'; cs.roi = null;
        resetGraph(); setDrawing(false); $('#camDrawRoi').disabled = true;
        cs.live = { stream: stream, stop: false, n: 0, last: 0 }; cs.frozen = false; cs.lock = false; $('#camLock').checked = false;
        $('#camLive').textContent = 'Камераны өшіру'; $('#camFreeze').hidden = false; $('#camFreeze').textContent = 'Қатыру';
        updateStatus('Тікелей камера: калибровка үшін алдымен флуоресцентті шамды түсіріңіз');
        requestAnimationFrame(liveLoop);
      });
    }).catch(function (e) {
      updateStatus('Камераға рұқсат жоқ (' + (e && e.name || 'қате') + '). Қорғалған орта (мыс. Artifact) камераны өшіреді: «Фото түсіру» немесе «Сурет таңдау» қолданыңыз, не бетті файл ретінде жергілікті ашыңыз.');
    });
  }
  function liveLoop(now) {
    var L = cs.live; if (!L || L.stop) return;
    if (!cs.frozen && now - L.last > 90) {
      L.last = now;
      var v = $('#camVideo');
      if (v.videoWidth) {
        var img = toImageData(v, v.videoWidth, v.videoHeight);
        cs.img = img; var cv = $('#camCanvas'); if (cv.width !== img.width) { cv.width = img.width; cv.height = img.height; }
        if (!cs.lock && (L.n % 20 === 0 || !cs.roi)) { var roi = C.detectBand(img); if (roi) { cs.roi = roi; setRoiControls(roi); } }
        L.n++;
        reprocess(false, true);
      }
    }
    requestAnimationFrame(liveLoop);
  }

  // ================= өңдеу =================
  function defaultCal(prof) {
    var M = prof.M, flip = cs.flipUser !== null ? cs.flipUser : cs.dir.flip;
    // 400–700 нм деп шамалаймыз: көк → қызыл (flip болса, керісінше)
    return flip ? { a: -300 / (M - 1), b: 700, provisional: true } : { a: 300 / (M - 1), b: 400, provisional: true };
  }
  function currentCal() { return cs.cal || (cs.prof ? defaultCal(cs.prof) : null); }
  function resetGraph() {
    cs.zoom = cs.hover = cs.selected = null;
    $('#camReadout').replaceChildren();
  }

  function reprocess(full, live) {
    if (!cs.img) return;
    var r = effectiveRoi();
    if (!r) {
      cs.prof = cs.ai = cs.ver = null;
      clearCalibration(); drawAll(); renderAI();
      updateStatus('Спектр жолағы табылмады: суретті жарығырақ етіп, не ROI-ны қолмен баптаңыз');
      return;
    }
    var prof = C.extractProfile(cs.img, r);
    if (cs.cal && !CAL.axisMatches(cs.calAxis, cs.img, r, prof)) {
      clearCalibration(); resetGraph();
      msg('#camCalMsg', 'Жолақ немесе кадр өлшемі өзгерді. Калибровканы қайта жасаңыз.');
    }
    cs.dir = C.colorDirection(prof);
    if (cs.flipUser === null) $('#camFlip').checked = cs.dir.flip;      // автоматты бағыт; қолмен өзгертсе — ол басым
    if (live && cs.prof && cs.prof.M === prof.M) for (var i = 0; i < prof.M; i++) prof.p[i] = 0.35 * prof.p[i] + 0.65 * cs.prof.p[i];     // уақыт бойынша тегістеу
    cs.prof = prof;
    if (full) { cs.ver = null; cs.ai = null; }
    updateProfileButtons();
    drawImage(); drawGraph();
    if (full || !cs.aiTimer) scheduleAI(full ? 0 : 280);
  }

  function scheduleAI(delay) {
    clearTimeout(cs.aiTimer);
    cs.aiTimer = setTimeout(function () { cs.aiTimer = 0; analyze(); }, delay);
  }

  function analyze() {
    if (!cs.prof) return;
    var cal = currentCal(), t0 = performance.now(), a = C.assess(cs.prof, cal, model, byName, species);
    if (!a) { cs.ai = null; cs.ver = null; renderAI(); return; }
    a.cal = cal; a.ms = performance.now() - t0;
    cs.fwhm = a.fwhm; cs.ver = a.ver; cs.ai = a;
    renderAI(); drawGraph();
  }

  // ================= сурет + ROI =================
  function drawImage() {
    var cv = $('#camCanvas');
    if (!cs.img) return;
    var ctx = cv.getContext('2d'), img = cs.img;
    if (cv.width !== img.width || cv.height !== img.height) { cv.width = img.width; cv.height = img.height; }
    ctx.putImageData(new ImageData(img.data instanceof Uint8ClampedArray ? img.data : new Uint8ClampedArray(img.data), img.width, img.height), 0, 0);
    if (cs.drag) {
      var lw0 = Math.max(1.5, img.width / 400);
      ctx.lineWidth = lw0; ctx.strokeStyle = '#4fd1ff'; ctx.setLineDash([lw0 * 4, lw0 * 3]);
      ctx.beginPath(); ctx.moveTo(cs.drag.a.x, cs.drag.a.y); ctx.lineTo(cs.drag.b.x, cs.drag.b.y); ctx.stroke(); ctx.setLineDash([]);
    }
    var r = effectiveRoi(); if (!r) return;
    var ct = Math.cos(r.theta), st = Math.sin(r.theta), hv = r.hv;
    function pt(t, s) {
      s += C.bandOffset(r, t);
      return [r.cx + t * ct - s * st, r.cy + t * st + s * ct];
    }
    function trace(from, to, s, move) {
      var n = r.curve ? 24 : 1;
      for (var i = 0; i <= n; i++) {
        var p = pt(from + (to - from) * i / n, s);
        if (i === 0 && move) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
    }
    var c = [pt(r.t0, -hv), pt(r.t1, -hv), pt(r.t1, hv), pt(r.t0, hv)], lw = Math.max(1.5, img.width / 500);
    ctx.lineWidth = lw; ctx.strokeStyle = cs.lock ? '#78cdff' : '#f0b64a';
    ctx.beginPath(); trace(r.t0, r.t1, -hv, true); trace(r.t1, r.t0, hv, false); ctx.closePath(); ctx.stroke();
    ctx.setLineDash([lw * 4, lw * 3]); ctx.beginPath(); trace(r.t0, r.t1, 0, true); ctx.stroke(); ctx.setLineDash([]);
    var cal = currentCal(), fs = Math.max(12, img.width / 55);
    ctx.font = '600 ' + fs + 'px Inter, system-ui, sans-serif'; ctx.fillStyle = ctx.strokeStyle; ctx.textBaseline = 'bottom';
    if (cal && cs.prof) {
      var M = cs.prof.M, l0 = C.lam(cal, 0), l1 = C.lam(cal, M - 1);
      ctx.textAlign = 'left'; ctx.fillText(Math.round(l0) + ' нм' + (cal.provisional ? '?' : ''), c[0][0] + 3, c[0][1] - 3);
      ctx.textAlign = 'right'; ctx.fillText(Math.round(l1) + ' нм' + (cal.provisional ? '?' : ''), c[1][0] - 3, c[1][1] - 3);
    }
  }

  // ================= график =================
  var ML = 40, MR = 10, MT = 12, MB = 26;
  function profilePeaks(p) {
    var mx = p.reduce(function (a, b) { return Math.max(a, b); }, 1e-9);
    return C.findPeaks(p, 0.14 * mx, Math.max(4, p.length / 60));
  }
  function drawGraph() {
    $('#camPrevPeak').disabled = $('#camNextPeak').disabled = !cs.prof;
    $('#camZoomReset').disabled = !cs.zoom;
    $('#camZoomIn').disabled = !cs.prof || !!(cs.zoom && cs.zoom[1] - cs.zoom[0] <= 20);
    var cv = $('#camGraph'), f = fit(cv); if (!f) return;
    var ctx = f.ctx, W = f.w, H = f.h, pw = W - ML - MR, ph = H - MT - MB, bot = MT + ph;
    var th = { bg: css('--graph-bg'), grid: css('--graph-grid'), axis: css('--graph-axis'), curve: css('--graph-curve'), accent: css('--accent'), fg: css('--fg') };
    ctx.fillStyle = th.bg; ctx.fillRect(0, 0, W, H);
    if (!cs.prof) { cs._plot = null; ctx.fillStyle = th.axis; ctx.font = '13px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.fillText('Спектр жоқ', W / 2, H / 2); return; }
    var cal = currentCal(), M = cs.prof.M, p = C.gauss(cs.prof.p, 0.8), mx = 1e-9, i;
    for (i = 0; i < M; i++) if (p[i] > mx) mx = p[i];
    var la = C.lam(cal, 0), lb = C.lam(cal, M - 1), lo = Math.min(la, lb), hi = Math.max(la, lb);
    if (cs.zoom) { lo = cs.zoom[0]; hi = cs.zoom[1]; }
    function X(l) { return ML + (l - lo) / (hi - lo) * pw; }
    function Y(v) { return bot - clamp(v / mx, 0, 1.05) / 1.05 * ph; }
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace'; ctx.lineWidth = 1;
    var span = hi - lo, xs = span > 500 ? 100 : span > 250 ? 50 : span > 80 ? 20 : span > 40 ? 10 : 5;
    $('#camZoomIn').disabled = span <= 20;
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (var w = Math.ceil(lo / xs) * xs; w <= hi; w += xs) { ctx.strokeStyle = th.grid; ctx.beginPath(); ctx.moveTo(X(w), MT); ctx.lineTo(X(w), bot); ctx.stroke(); ctx.fillStyle = th.axis; ctx.fillText(String(w), X(w), bot + 8); }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [0, 0.5, 1].forEach(function (t) { ctx.strokeStyle = th.grid; ctx.beginPath(); ctx.moveTo(ML, Y(t * mx)); ctx.lineTo(W - MR, Y(t * mx)); ctx.stroke(); ctx.fillStyle = th.axis; ctx.fillText(t.toFixed(1), ML - 6, Y(t * mx)); });
    var approximate = cal.provisional || cal.approx;
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillStyle = approximate ? th.accent : th.axis;
    ctx.fillText(approximate ? 'λ, нм (шамамен)' : 'λ, нм', W - MR, H - 1);
    ctx.save(); ctx.beginPath(); ctx.rect(ML, MT, pw, ph); ctx.clip();
    // 380–780 терезесі
    [380, 780].forEach(function (e) { if (e > lo && e < hi) { ctx.setLineDash([4, 4]); ctx.strokeStyle = th.axis; ctx.beginPath(); ctx.moveTo(X(e), MT); ctx.lineTo(X(e), bot); ctx.stroke(); ctx.setLineDash([]); } });
    // профиль: түсті бояу + қисық
    var dark = th.bg !== '#fbfcfe';
    for (i = 0; i < M - 1; i++) {
      var l = C.lam(cal, i), x0 = X(l), x1 = X(C.lam(cal, i + 1)), yv = Y(p[i]);
      ctx.fillStyle = P.rgbCss(P.wlColor(l), dark ? 0.26 : 0.38);
      ctx.fillRect(Math.min(x0, x1), yv, Math.max(1, Math.abs(x1 - x0) + 0.5), bot - yv);
    }
    ctx.beginPath();
    for (i = 0; i < M; i++) { var xx = X(C.lam(cal, i)), yy = Y(p[i]); if (i === 0) ctx.moveTo(xx, yy); else ctx.lineTo(xx, yy); }
    ctx.strokeStyle = th.curve; ctx.lineWidth = 1.3; ctx.stroke();
    // модельденген қисық (ЖИ болжамы бойынша)
    if (cs.ver && cs.ai && cs.ver.simulated && cs.ai.predGated.length) {
      var sim = cs.ver.simulated, x = cs.ai.inp.x, num = 0, den = 1e-9;
      for (i = 0; i < sim.length; i++) { num += x[i] * sim[i]; den += sim[i] * sim[i]; }
      var sc = num / den;
      ctx.beginPath(); var started = false;
      for (i = 0; i < sim.length; i++) {
        var lam = C.WIN_LO + i * C.WIN_DX; if (lam < lo || lam > hi) continue;
        var px = X(lam), py = Y(sim[i] * sc * mx);
        if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
      }
      ctx.setLineDash([5, 3]); ctx.strokeStyle = th.accent; ctx.lineWidth = 1.4; ctx.stroke(); ctx.setLineDash([]);
    }
    // болжанған түрлердің сызықтары: төменгі штрихтар (табылғаны — қою, табылмағаны — жарты)
    if (cs.ver) cs.ver.perSpecies.forEach(function (ps, k) {
      ps.expected.forEach(function (e) {
        var xe = X(e.wl); if (xe < ML || xe > W - MR) return;
        ctx.strokeStyle = MARK[k % 3]; ctx.globalAlpha = e.found ? 1 : 0.35; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(xe, bot); ctx.lineTo(xe, bot - 8 - 14 * e.st); ctx.stroke(); ctx.globalAlpha = 1;
      });
    });
    // шыңдардың атаулары
    var peaks = profilePeaks(p).filter(function (pk) { var l = C.lam(cal, pk.x); return l >= lo && l <= hi; }).slice(0, 6);
    ctx.font = '600 11px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillStyle = th.fg;
    var placed = [];
    peaks.forEach(function (pk) {
      var l = C.lam(cal, pk.x), x = X(l); if (x < ML + 16 || x > W - MR - 16) return;
      if (placed.some(function (q) { return Math.abs(q - x) < 40; })) return; placed.push(x);
      ctx.fillText(l.toFixed(0), x, Math.max(MT + 12, Y(pk.h) - 4));
    });
    // калибровка нүктелері
    cs.anchors.forEach(function (a, k) {
      if (a.x == null) return;
      var xa = X(C.lam(cal, a.x)); ctx.setLineDash([2, 3]); ctx.strokeStyle = MARK[k]; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(xa, MT); ctx.lineTo(xa, bot); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = MARK[k]; ctx.textBaseline = 'top'; ctx.fillText(String(k + 1), xa, MT + 1);
    });
    var focus = cs.hover == null ? cs.selected : cs.hover;
    if (focus != null) { var hx = X(focus); if (hx >= ML && hx <= W - MR) { ctx.strokeStyle = th.fg; ctx.globalAlpha = 0.5; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(hx, MT); ctx.lineTo(hx, bot); ctx.stroke(); ctx.globalAlpha = 1; } }
    ctx.restore();
    cs._plot = { lo: lo, hi: hi, W: W, cal: cal, M: M };
  }
  function drawAll() { drawImage(); drawGraph(); }

  // ================= ЖИ панелі =================
  function sname(n) { var s = byName[n]; return s ? s.sym + ' ' + (s.charge ? 'II' : 'I') : n; }
  function renderAI() {
    var box = $('#camAiBody'), a = cs.ai;
    if (!cs.img) { html(box, '<p class="cam-hint">Сурет таңдаңыз.</p>'); return; }
    if (!cs.prof) { html(box, '<p class="cam-hint">Спектр жолағы табылмады.</p>'); return; }
    if (!a) { html(box, '<p class="cam-hint">Сигнал жеткіліксіз: калибровка аралығы 380–780 нм-мен қиылыспайды не профиль тегіс.</p>'); return; }
    var res = a.res, names = model.species, order = names.map(function (n, k) { return k; }).sort(function (x, y) { return res.probs[y] - res.probs[x]; });
    var show = order.slice(0, 6), bars = '';
    show.forEach(function (k) {
      var p = res.probs[k], isP = p > 0.5;
      bars += '<div class="bar' + (isP ? ' pred' : '') + '" style="grid-template-columns:52px 1fr 44px"><span class="num">' + sname(names[k]) + '</span><span class="track"><span class="fill" style="display:block;width:' + (p * 100).toFixed(1) + '%"></span></span><span class="pc">' + Math.round(p * 100) + '%</span></div>';
    });
    var pills = '', verdict, conf = a.confirmed, pg = a.predGated;
    var evidence = SL.spectrumConfidence.summarize(a, names);
    if (a.cal.provisional) pills += '<span class="pill warn">калибровка жоқ: сенімсіз</span> ';
    else if (a.cal.approx) pills += '<span class="pill warn">калибровка түс бойынша: шамамен ±15–20 нм</span> ';
    else pills += '<span class="pill ok">калибровка: ' + (a.cal.kind === 'cfl' ? 'CFL' : '2 нүкте') + (a.cal.c ? ', квадраттық' : '') + (a.cal.rms ? ', RMS ' + a.cal.rms.toFixed(1) + ' нм' : '') + '</span> ';
    pills += '<span class="pill">ажыратымдылық ≈ ' + cs.fwhm.toFixed(1) + ' нм</span> <span class="pill' + (a.continuum ? ' warn' : '') + '">сызықтылық ' + a.lin.toFixed(2) + '</span>';
    function candidateList(list) { return list.map(function (p) { return '<b>' + sname(p.name) + '</b>'; }).join(', '); }
    if (evidence.status === 'unknown') verdict = '<b>Құрамы белгісіз.</b> Анықтауға жеткілікті дәлел жиналмады.';
    else verdict = (evidence.status === 'partial' ? 'Жартылай анықталды. ' : '') + 'Спектр сызықтарымен қолдау тапқан түрлер: ' + candidateList(evidence.supported) + '.';
    if (evidence.uncertain.length) verdict += ' Күмәнді түрлер: ' + candidateList(evidence.uncertain) + '.';
    if (evidence.reasons.includes('no-narrow-lines')) verdict += ' Айқын тар атомдық сызықтар табылмады.';
    if (evidence.supported.some(function (p) { return p.name === 'CFL'; })) verdict += ' <i>CFL</i> — люминофорлы флуоресцентті (үнемдегіш) шам: Hg сызықтары + Tb³⁺/Eu³⁺ фосфор сызықтары.';
    var warn = '';
    var satN = 0, top = 0, M = cs.prof.M, thr = 0, i;
    for (i = 0; i < M; i++) thr = Math.max(thr, cs.prof.p[i]);
    for (i = 0; i < M; i++) if (cs.prof.p[i] > 0.6 * thr) { top++; if (cs.prof.sat[i] > 0.4) satN++; }
    if (top && satN / top > 0.3) warn += '<li>Сигнал қаныққан: ең жарық сызықтар жалпақталған. Экспозицияны азайтыңыз.</li>';
    var d = [], mdn;
    for (i = 1; i < M; i++) d.push(Math.abs(cs.prof.p[i] - cs.prof.p[i - 1]));
    d.sort(function (x, y) { return x - y; }); mdn = d[d.length >> 1] * 1.4826 / 1.414;
    if (thr / Math.max(mdn, 1e-9) < 12) warn += '<li>Сигнал шумен салыстырғанда әлсіз: көз қараңғы бөлмеде не жарығырақ болсын.</li>';
    if (a.inp.coverage < 0.5) warn += '<li>Калибровка 380–780 нм-нің жартысынан азын қамтиды: нәтиже толық емес.</li>';
    if (cs.roi && cs.roi.info && cs.roi.info.single && !cs.lock) warn += '<li>Бір ғана сызық көрінеді: жолақ бағытын «Бұрыш» жүгірткісімен тексеріңіз.</li>';
    var ver = '', v = cs.ver;
    if (v) {
      v.perSpecies.forEach(function (ps) {
        var n = ps.expected.length, ok = ps.found, fr = n ? ok / n : 0, cls = fr >= 0.6 ? 'ok' : fr >= 0.35 ? 'warn' : 'bad';
        ver += '<li><span class="pill ' + cls + '">' + ok + ' / ' + n + '</span> <b>' + sname(ps.name) + '</b>: болжамның негізгі сызықтарының ' + Math.round(fr * 100) + '% спектрде бар</li>';
      });
      (a.continuum ? [] : v.unexplained).slice(0, 4).forEach(function (u) {
        var what = u.phosphor ? u.phosphor : (u.cand.length ? 'жақын: ' + u.cand.map(function (c) { return sname(c.name) + ' ' + c.wl.toFixed(1); }).join(', ') : 'белгісіз');
        ver += '<li><span class="pill">?</span> түсіндірілмеген шың <b>' + u.wl.toFixed(1) + ' нм</b> (' + what + ')</li>';
      });
      if (pg.length && !conf.length) warn += '<li class="bad">ЖИ болжамы сызықтармен расталмады: калибровканы тексеріңіз.</li>';
    }
    html(box, '<div>' + pills + '</div><div class="bars" style="margin-top:12px;grid-template-columns:1fr">' + bars + '</div><p class="verdict" style="margin:12px 0 0">' + verdict + '</p>' +
      (warn ? '<ul class="warns">' + warn + '</ul>' : '') + (ver ? '<ul class="verify">' + ver + '</ul>' : '') +
      '<p class="note">Бағандар модель ұпайын көрсетеді. Құрамды бағалауда калибровка, өлшенген аралық және спектр сызықтарының сәйкестігі ескеріледі. Есептеу: ' + a.ms.toFixed(0) + ' мс.</p>');
  }

  function updateStatus(t) { $('#camStatus').textContent = t; }
  function calState() {
    var c = cs.cal, el = $('#camCalState');
    el.textContent = c ? 'Калибровка: ' + c.kind + (c.c ? ' (квадраттық)' : '') + ', ' + Math.abs(c.a).toFixed(3) + ' нм/пиксель' : 'Калибровка жоқ';
    var d = CAL.diagnostics(c), rows = d ? d.lines : [];
    $('#camCalDetails').hidden = !rows.length;
    $('#camCalQuality').textContent = rows.length
      ? 'RMS ' + d.rms.toFixed(2) + ' нм · Ең үлкен ауытқу ' + d.maxError.toFixed(2) + ' нм.' +
          (d.kind === 'two-point' ? ' Екі нүкте арқылы есептелген сәйкестікті бөлек эталон сызығымен тексеруге болады.' : '')
      : '';
    $('#camCalLines').replaceChildren(...rows.map(function (match) {
      var tr = document.createElement('tr');
      [match.name, match.referenceNm.toFixed(2), match.fittedNm.toFixed(2), match.errorNm.toFixed(2)].forEach(function (v) {
        tr.insertCell().textContent = v;
      });
      return tr;
    }));
    updateProfileButtons();
  }
  function clearCalibration() {
    cs.cal = cs.calAxis = null;
    cs.anchors = [{ x: null }, { x: null }]; cs.pick = 0;
    msg('#camCalMsg', ''); msg('#camPickMsg', '');
    calState();
  }
  function useCalibration(cal) {
    cs.calAxis = CAL.captureAxis(cs.img, effectiveRoi(), cs.prof);
    cs.cal = cal;
    lockRoi(true); resetGraph(); calState(); reprocess(true);
  }

  function hasMeasuredAxis() {
    return Boolean(cs.prof && cs.cal && cs.calAxis && !cs.cal.approx && !cs.cal.provisional &&
      Math.min(C.lam(cs.cal, 0), C.lam(cs.cal, cs.prof.M - 1)) > 0);
  }
  function updateProfileButtons() {
    var selected = Boolean($('#camProfiles').value);
    var measured = hasMeasuredAxis();
    $('#camApplyProfile').disabled = !selected || !cs.img;
    $('#camDeleteProfile').disabled = !selected;
    $('#camSaveProfile').disabled = $('#camCapture').disabled = !measured;
  }
  function refreshProfiles(selected) {
    var select = $('#camProfiles'), profiles = CAL.listProfiles(window.localStorage);
    select.replaceChildren(new Option('Профиль таңдаңыз', ''), ...profiles.map(function (p) { return new Option(p.name, p.name); }));
    if (selected && profiles.some(function (p) { return p.name === selected; })) select.value = selected;
    updateProfileButtons();
  }
  function changeProfile(operation) {
    try { msg('#camProfileStatus', operation()); }
    catch (e) { msg('#camProfileStatus', e.message || 'Профильмен жұмыс істеу мүмкін болмады.', 'bad'); }
  }
  function applyProfile(name) {
    var profile = CAL.listProfiles(window.localStorage).find(function (p) { return p.name === name; });
    if (!profile) throw new Error('Калибровка профилі табылмады.');
    var restored = CAL.restoreProfile(profile, cs.img);
    var prof = C.extractProfile(cs.img, restored.roi);
    if (prof.M !== restored.length) throw new Error('Спектр осінің ұзындығы профильге сәйкес емес.');
    cs.roi = restored.roi; setRoiControls(cs.roi); cs.prof = prof;
    clearCalibration(); useCalibration(restored.cal);
    return '«' + name + '» профилі қолданылды.';
  }

  // ================= калибровка =================
  function lockRoi(on) { cs.lock = on; $('#camLock').checked = on; drawImage(); }
  function doAutoCal() {
    if (!cs.prof) { msg('#camCalMsg', 'Алдымен сурет таңдаңыз.'); return; }
    var c = C.autoCalibrateCFL(cs.prof);
    if (!c) {
      var h = C.calibrateByHue(cs.prof);        // CFL сызықтары табылмаса: түстен шамамен
      if (h) {
        useCalibration(h);
        msg('#camCalMsg', 'CFL сызықтары табылмады. Түс бойынша шамамен калибровка қолданылды (қателігі ±15–20 нм, RMS ' + h.rms.toFixed(0) + ' нм): нәтижені тек бағдар ретінде қолданыңыз, не CFL түсіріп қайталаңыз.', 'bad');
        return;
      }
      msg('#camCalMsg', 'Флуоресцентті шам сызықтары табылмады. Кадрда флуоресцентті/CFL шамның спектрі көрінетініне көз жеткізіңіз, не қолмен калибровка жасаңыз.', 'bad'); return;
    }
    useCalibration(c);
    msg('#camCalMsg', 'Калибровка дайын: ' + c.matched.map(function (m) { return m.name; }).join(', ') + '; RMS ' + c.rms.toFixed(1) + ' нм. ROI бекітілді: енді басқа көздерді түсіріңіз.');
  }
  function doManualCal() {
    var a = cs.anchors, w1 = parseFloat($('#camWl1').value), w2 = parseFloat($('#camWl2').value);
    if (!cs.prof || a[0].x == null || a[1].x == null || !isFinite(w1) || !isFinite(w2) || w1 < 300 || w2 < 300 || w1 > 900 || w2 > 900) {
      msg('#camPickMsg', 'Екі нүктені графиктен таңдап, 300–900 нм аралығындағы толқын ұзындықтарын енгізіңіз.', 'bad'); return;
    }
    var c = C.calib2(a[0].x, w1, a[1].x, w2);
    if (!c) { msg('#camPickMsg', 'Нүктелер немесе толқын ұзындықтары бірдей.', 'bad'); return; }
    c.matched = [{ x: a[0].x, wl: w1, name: '1-нүкте' }, { x: a[1].x, wl: w2, name: '2-нүкте' }];
    if (Math.min(C.lam(c, 0), C.lam(c, cs.prof.M - 1)) <= 0) {
      msg('#camPickMsg', 'Бұл нүктелер спектр шетінде жарамсыз толқын ұзындығын береді. Эталон сызықтарын тексеріңіз.', 'bad'); return;
    }
    useCalibration(c);
    msg('#camPickMsg', 'Калибровка қолданылды: ' + Math.abs(c.a).toFixed(3) + ' нм/пиксель.');
  }

  // ================= әрекеттерді байлау =================
  function setDrawing(active) {
    cs.drawing = active; cs.drag = null;
    $('#camDrawRoi').setAttribute('aria-pressed', String(active));
    $('.cam-stage').classList.toggle('is-drawing', active);
    if (!$('#viewCam').hidden) drawImage();
  }

  function bind() {
    $('#camSamples').addEventListener('click', function (e) { var b = e.target.closest('.chip'); if (b) loadSample(+b.dataset.i); });
    $('#camReshoot').addEventListener('click', function () { loadSample(cs.sim.src == null ? 0 : cs.sim.src, true); });
    $('#camPickFile').addEventListener('click', function () { $('#camFile').click(); });
    $('#camPickShot').addEventListener('click', function () { $('#camShot').click(); });
    $('#camFile').addEventListener('change', function (e) { loadFile(e.target.files[0]); e.target.value = ''; });
    $('#camShot').addEventListener('change', function (e) { loadFile(e.target.files[0]); e.target.value = ''; });
    $('#camLive').addEventListener('click', startLive);
    $('#camFreeze').addEventListener('click', function () { cs.frozen = !cs.frozen; this.textContent = cs.frozen ? 'Жалғастыру' : 'Қатыру'; });
    ['camAngle', 'camThick', 'camStart', 'camEnd'].forEach(function (id) {
      $('#' + id).addEventListener('input', function () { lockRoi(true); roiLabels(); reprocess(true); });
    });
    $('#camAuto').addEventListener('click', function () {
      if (!cs.img) return;
      setDrawing(false); lockRoi(false); clearCalibration(); resetGraph();
      cs.roi = C.detectBand(cs.img);
      if (cs.roi) { setRoiControls(cs.roi); updateStatus('Спектр жолағы автоматты түрде табылды.'); }
      reprocess(true);
    });
    $('#camLock').addEventListener('change', function () { cs.lock = this.checked; drawImage(); });
    $('#camFlip').addEventListener('change', function () {
      cs.flipUser = this.checked; clearCalibration(); resetGraph(); reprocess(true);
    });
    $('#camAutoCal').addEventListener('click', doAutoCal);
    $('#camClearCal').addEventListener('click', function () { clearCalibration(); resetGraph(); reprocess(true); });
    $('#camPick1').addEventListener('click', function () { cs.pick = 1; msg('#camPickMsg', 'Графиктен 1-сызықты басыңыз.'); });
    $('#camPick2').addEventListener('click', function () { cs.pick = 2; msg('#camPickMsg', 'Графиктен 2-сызықты басыңыз.'); });
    $('#camApplyCal').addEventListener('click', doManualCal);
    $$('[data-laser]').forEach(function (b) { b.addEventListener('click', function () { var i1 = $('#camWl1'), i2 = $('#camWl2'); (i1.value === '' ? i1 : i2).value = b.dataset.laser; }); });
    $('#camProfiles').addEventListener('change', updateProfileButtons);
    $('#camSaveProfile').addEventListener('click', function () { changeProfile(function () {
      var profile = CAL.makeProfile($('#camProfileName').value, cs.cal, cs.img, effectiveRoi(), cs.prof);
      CAL.saveProfile(window.localStorage, profile);
      refreshProfiles(profile.name); $('#camProfileName').value = '';
      return '«' + profile.name + '» профилі сақталды.';
    }); });
    $('#camApplyProfile').addEventListener('click', function () { changeProfile(function () { return applyProfile($('#camProfiles').value); }); });
    $('#camDeleteProfile').addEventListener('click', function () { changeProfile(function () {
      CAL.deleteProfile(window.localStorage, $('#camProfiles').value);
      refreshProfiles(); return 'Профиль өшірілді.';
    }); });
    $('#camCapture').addEventListener('click', function () {
      if (!hasMeasuredAxis()) return;
      var cal = cs.cal, p = cs.prof.p;
      var wavelengths = Array.from(p, function (_, i) { return C.lam(cal, i); });
      var peaks = profilePeaks(C.gauss(p, 0.8)).map(function (pk) { return { wavelength: C.lam(cal, pk.x), intensity: pk.h }; });
      SL.measurementPanel.capture({ name: cs.label || 'Камера спектрі', source: 'camera', wavelengths: wavelengths, intensity: p, peaks: peaks });
    });

    // қолмен ROI: суретте спектр жолағы бойымен сүйреңіз (нақты фотода автоматты табу сәтсіз болғанда)
    var cvImg = $('#camCanvas');
    $('#camDrawRoi').addEventListener('click', function () {
      setDrawing(!cs.drawing);
      if (cs.drawing) updateStatus('Жолақтың бір шетінен екінші шетіне сызық сүйреңіз.');
    });
    function imgPt(e) {       // CSS object-fit: contain ескеріледі (max-height-тан кейін сурет жиектерінде бос орын болады)
      var r = cvImg.getBoundingClientRect(), k = Math.min(r.width / cvImg.width, r.height / cvImg.height);
      return {
        x: clamp((e.clientX - r.left - (r.width - cvImg.width * k) / 2) / k, 0, cvImg.width - 1),
        y: clamp((e.clientY - r.top - (r.height - cvImg.height * k) / 2) / k, 0, cvImg.height - 1)
      };
    }
    cvImg.addEventListener('pointerdown', function (e) {
      if (!cs.img || cs.live || !e.isPrimary || e.button !== 0 || (e.pointerType === 'touch' && !cs.drawing)) return;
      cs.drag = { a: imgPt(e), b: imgPt(e), id: e.pointerId };
      cvImg.setPointerCapture(e.pointerId);
    });
    cvImg.addEventListener('pointermove', function (e) {
      if (cs.drag && cs.drag.id === e.pointerId) { cs.drag.b = imgPt(e); drawImage(); }
    });
    cvImg.addEventListener('pointerup', function (e) {
      var d = cs.drag; if (!d || d.id !== e.pointerId) return;
      cs.drag = null;
      d.b = imgPt(e);
      var dx = d.b.x - d.a.x, dy = d.b.y - d.a.y, len = Math.hypot(dx, dy);
      if (len < 20) { drawImage(); return; }                       // кездейсоқ басу
      if (dx < 0) { dx = -dx; dy = -dy; }                           // бағыт: сол → оң (түс бағыты өзі анықталады)
      var th = Math.atan2(dy, dx), hv = clamp(Math.round(+$('#camThick').value || 12), 3, 120);
      cs.roi = { cx: (d.a.x + d.b.x) / 2, cy: (d.a.y + d.b.y) / 2, theta: th, hv: hv, t0: -len / 2, t1: len / 2, info: { manual: true } };
      setRoiControls(cs.roi); setDrawing(false); lockRoi(true); clearCalibration(); resetGraph(); reprocess(true);
      updateStatus('ROI қолмен берілді: ' + Math.round(th * 180 / Math.PI) + '°, ұзындығы ' + Math.round(len) + ' px');
    });
    ['pointercancel', 'lostpointercapture'].forEach(function (name) {
      cvImg.addEventListener(name, function (e) {
        if (cs.drag && cs.drag.id === e.pointerId) { cs.drag = null; drawImage(); }
      });
    });

    var g = $('#camGraph');
    function lamFrom(e) {
      if (!cs._plot) return null;
      var r = g.getBoundingClientRect(), x = e.clientX - r.left, f = (x - ML) / (r.width - ML - MR);
      return f < 0 || f > 1 ? null : cs._plot.lo + f * (cs._plot.hi - cs._plot.lo);
    }
    function readPeak(l) {
      var ro = $('#camReadout');
      if (l == null || !cs.prof) { ro.replaceChildren(); return; }
      var cal = cs._plot.cal, idx = C.idx(cal, l, cs.prof.M), i0 = clamp(Math.round(idx), 0, cs.prof.M - 1);
      var near = [];
      species.forEach(function (sp) { var j = P.nearestLine(sp, l); if (Math.abs(sp.wl[j] - l) <= Math.max(2.5, cs.fwhm * 0.8)) near.push({ sp: sp, wl: sp.wl[j], rel: Math.log10(sp.rel[j] + 1) }); });
      near.sort(function (a, b) { return b.rel - a.rel; });
      ro.innerHTML = '<span>λ = <b class="num">' + l.toFixed(1) + ' нм</b></span><span>пиксель = <b class="num">' + i0 + '</b></span>' + (near.length ? '<span>Жақын: ' + near.slice(0, 4).map(function (q) { return '<b>' + sname(q.sp.name) + ' ' + q.wl.toFixed(1) + '</b>'; }).join(', ') + '</span>' : '');
    }
    g.addEventListener('pointermove', function (e) {
      if (e.pointerType === 'touch') return;
      cs.hover = lamFrom(e); readPeak(cs.hover == null ? cs.selected : cs.hover); drawGraph();
    });
    g.addEventListener('pointerleave', function () {
      cs.hover = null; readPeak(cs.selected); drawGraph();
    });
    g.addEventListener('click', function (e) {
      if (!cs.prof) return;
      var l = lamFrom(e); if (l == null) return;
      selectPeak(l);
    });
    function selectPeak(l, index) {
      cs.hover = null;
      var cal = cs._plot.cal, best = index;
      if (best === undefined) {
        var center = clamp(Math.round(C.idx(cal, l, cs.prof.M)), 0, cs.prof.M - 1);
        var y = C.gauss(cs.prof.p, 1), last = Math.min(cs.prof.M - 1, center + 10);
        best = center;
        for (var j = Math.max(0, center - 10); j <= last; j++) if (y[j] > y[best]) best = j;
      }
      cs.selected = C.lam(cal, best);
      if (cs.pick) {
        cs.anchors[cs.pick - 1].x = best;
        msg('#camPickMsg', cs.pick + '-нүкте: пиксель ' + best + '. Оның нақты толқын ұзындығын енгізіңіз.');
        cs.pick = 0;
      }
      readPeak(cs.selected); drawGraph();
    }
    function stepPeak(direction) {
      if (!cs._plot) return;
      var plot = cs._plot, peaks = profilePeaks(C.gauss(cs.prof.p, 0.8)).map(function (p) {
        return { wl: C.lam(plot.cal, p.i), i: p.i };
      }).filter(function (p) { return p.wl >= plot.lo && p.wl <= plot.hi; });
      peaks.sort(function (a, b) { return direction * (a.wl - b.wl); });
      var peak = peaks.find(function (p) { return cs.selected == null || direction * (p.wl - cs.selected) > 1e-6; }) || peaks[0];
      if (peak) selectPeak(peak.wl, peak.i);
    }
    $('#camPrevPeak').addEventListener('click', function () { stepPeak(-1); });
    $('#camNextPeak').addEventListener('click', function () { stepPeak(1); });
    $('#camZoomIn').addEventListener('click', function () {
      if (!cs._plot) return;
      var plot = cs._plot, width = Math.max(20, (plot.hi - plot.lo) / 2);
      var center = cs.selected == null ? (plot.lo + plot.hi) / 2 : cs.selected;
      var lo = clamp(center - width / 2, plot.lo, plot.hi - width);
      cs.zoom = [lo, lo + width]; drawGraph();
    });
    $('#camZoomReset').addEventListener('click', function () { cs.zoom = null; drawGraph(); });

    // сүйреп апару және буфер (Ctrl+V)
    var stage = $('.cam-stage');
    ['dragover', 'dragenter'].forEach(function (ev) { stage.addEventListener(ev, function (e) { e.preventDefault(); }); });
    stage.addEventListener('drop', function (e) { e.preventDefault(); var f = e.dataTransfer && e.dataTransfer.files[0]; if (f) loadFile(f); });
    document.addEventListener('paste', function (e) {
      if ($('#viewCam').hidden) return;
      var it = Array.prototype.filter.call((e.clipboardData && e.clipboardData.items) || [], function (x) { return x.type.indexOf('image') === 0; })[0];
      if (it) loadFile(it.getAsFile());
    });
    window.addEventListener('resize', function () { if (!$('#viewCam').hidden) drawGraph(); });
    var mq = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)');
    if (mq && mq.addEventListener) mq.addEventListener('change', drawGraph);
    new MutationObserver(drawGraph).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }

  function init() {
    buildSamples(); bind(); roiLabels(); calState();
    changeProfile(function () { refreshProfiles(); return ''; });
    $('#camAiMeta').textContent = '1D-CNN + Transformer · камера домені (380–780 нм)';
    $('#camFooter').innerHTML = 'Кескіннен спектр: жолақ PCA арқылы табылады, бұрышы мен қисықтығы ескеріліп профиль алынады (sRGB → сызықтық, фон алынады), калибровка λ = a·пиксель + b. ' +
      'Көрінетін аймақтағы (380–780 нм) 1024 нүктелі спектр камера моделіне беріледі; ол ажыратымдылық 1.5–28 нм (кең, қаныққан сызықтарды қоса), калибровка қателігі мен сызықтық емес дисперсия, сезімталдық қисығы, қанығу мен блуминг, фон, бөгде (фосфор) сызықтар және «CFL» көз класы қосылған синтетикалық спектрлерде оқытылған. ' +
      'Виртуал камера — модель, нақты фото емес. Нақты камерада сапа тексерілмеген: нәтижені графиктегі сызық сәйкестігімен салыстырыңыз.';
    setView(viewFromHash());
  }
  init();
})();
