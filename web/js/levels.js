/* Энергия деңгейлерінің диаграммасы (Гротриан типті): бағандар — сыртқы электронның l-і (s, p, d, f...),
   биіктік — NIST деңгей энергиясы (эВ). Түр сызықтары ауысу толқын ұзындығының түсімен сызылады. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./physics.js'));
  else { root.SL = root.SL || {}; root.SL.levels = factory(root.SL.physics); }
})(typeof window !== 'undefined' ? window : globalThis, function (P) {
  'use strict';

  var ORDER = 'spdfghik';

  // деңгей кестесі: [{E, label, col}]
  function collect(sp) {
    var E = {}, info = sp.levels.map(function (s) { return P.parseLevel(s); });
    for (var i = 0; i < sp.wl.length; i++) { E[sp.lo[i]] = sp.ei[i]; E[sp.up[i]] = sp.ek[i]; }
    var cols = {};
    Object.keys(E).forEach(function (k) { var l = info[k].l || 'n'; cols[l] = 1; });
    var colList = Object.keys(cols).sort(function (a, b) {
      var ia = ORDER.indexOf(a), ib = ORDER.indexOf(b); return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    return { E: E, info: info, cols: colList };
  }

  function draw(cv, sp, idx, mode, th) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    var L = collect(sp), padL = 44, padR = 14, padT = 22, padB = 24;
    var pw = w - padL - padR, ph = h - padT - padB, ncol = L.cols.length;
    var emin = Math.min.apply(null, Object.keys(L.E).map(function (k) { return L.E[k]; }));
    var e0 = emin > 2 ? Math.floor(emin - 1) : 0;           // негізгі күй жоқ түрлерде (H I) осьті жоғарырақтан бастаймыз
    var ion = sp.ion, y = function (e) { return padT + ph * (1 - (e - e0) / (ion - e0)); };
    var cw = pw / ncol, cx = function (c) { return padL + cw * (L.cols.indexOf(c) + 0.5); };
    function colOf(k) { return L.info[k].l || 'n'; }

    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    // ось энергии
    var step = ion <= 8 ? 1 : ion <= 16 ? 2 : 5;
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (var e = Math.ceil(e0 / step) * step; e <= ion + 1e-6; e += step) {
      ctx.strokeStyle = th.grid; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(padL, y(e)); ctx.lineTo(w - padR, y(e)); ctx.stroke();
      ctx.fillStyle = th.axis; ctx.fillText(String(e), padL - 6, y(e));
    }
    ctx.save(); ctx.translate(11, padT + ph / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center';
    ctx.fillStyle = th.axis; ctx.fillText('E, эВ', 0, 0); ctx.restore();
    // иондану шегі
    ctx.setLineDash([5, 4]); ctx.strokeStyle = th.axis; ctx.beginPath(); ctx.moveTo(padL, y(ion)); ctx.lineTo(w - padR, y(ion)); ctx.stroke(); ctx.setLineDash([]);
    ctx.textAlign = 'left'; ctx.fillStyle = th.axis; ctx.fillText('иондану ' + ion.toFixed(2) + ' эВ', padL + 4, y(ion) - 8);

    // түр сызықтары
    var order = sp.rel.map(function (r, i) { return i; });
    ctx.lineWidth = 1;
    order.forEach(function (i) {
      var x1 = cx(colOf(sp.lo[i])), x2 = cx(colOf(sp.up[i])), vis = P.isVisible(sp.wl[i]);
      ctx.strokeStyle = P.rgbCss(P.wlColor(sp.wl[i]), vis ? 0.38 : 0.2);
      ctx.beginPath(); ctx.moveTo(x1, y(sp.ei[i])); ctx.lineTo(x2, y(sp.ek[i])); ctx.stroke();
    });
    // деңгей белгілері
    var barW = Math.min(cw * 0.5, 46);
    Object.keys(L.E).forEach(function (k) {
      var x = cx(colOf(k)); ctx.strokeStyle = th.level; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x - barW / 2, y(L.E[k])); ctx.lineTo(x + barW / 2, y(L.E[k])); ctx.stroke();
    });
    // бағандар атаулары
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = th.axis; ctx.font = '600 12px Inter, system-ui, sans-serif';
    L.cols.forEach(function (c) { ctx.fillText(c === 'n' ? 'n' : c, cx(c), h - padB + 6); });

    // таңдалған ауысу
    var lo = sp.lo[idx], up = sp.up[idx];
    var xl = cx(colOf(lo)), xu = cx(colOf(up)), yl = y(sp.ei[idx]), yu = y(sp.ek[idx]);
    var col = P.wlColor(sp.wl[idx]), em = mode === 'emission';
    ctx.strokeStyle = th.accent; ctx.lineWidth = 3;
    [[lo, xl, yl], [up, xu, yu]].forEach(function (a) {
      ctx.beginPath(); ctx.moveTo(a[1] - barW / 2 - 2, a[2]); ctx.lineTo(a[1] + barW / 2 + 2, a[2]); ctx.stroke();
    });
    ctx.strokeStyle = P.rgbCss(col, 1); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(xu, yu); ctx.lineTo(xl, yl); ctx.stroke();
    // көрсеткі ұшы
    var from = em ? [xu, yu] : [xl, yl], to = em ? [xl, yl] : [xu, yu];
    var ang = Math.atan2(to[1] - from[1], to[0] - from[0]);
    ctx.fillStyle = P.rgbCss(col, 1); ctx.beginPath();
    ctx.moveTo(to[0], to[1]);
    ctx.lineTo(to[0] - 10 * Math.cos(ang - 0.4), to[1] - 10 * Math.sin(ang - 0.4));
    ctx.lineTo(to[0] - 10 * Math.cos(ang + 0.4), to[1] - 10 * Math.sin(ang + 0.4));
    ctx.closePath(); ctx.fill();
    // белгілер
    ctx.font = '600 11px Inter, system-ui, sans-serif'; ctx.fillStyle = th.fg; ctx.textBaseline = 'middle';
    function side(x) { return x > w * 0.55 ? 'right' : 'left'; }
    var li = L.info[lo], ui = L.info[up];
    ctx.textAlign = side(xu); ctx.fillText(ui.label + '  ' + sp.ek[idx].toFixed(2) + ' эВ', xu + (side(xu) === 'left' ? barW / 2 + 6 : -barW / 2 - 6), yu - 8);
    var lowY = yl > padT + ph - 26 ? yl - 10 : yl + 10;
    ctx.textAlign = side(xl); ctx.fillText(li.label + '  ' + sp.ei[idx].toFixed(2) + ' эВ', xl + (side(xl) === 'left' ? barW / 2 + 6 : -barW / 2 - 6), lowY);
  }

  return { draw: draw };
});
