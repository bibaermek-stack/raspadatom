/* 3D атом: электронның деңгейден деңгейге секіруі және фотон (толқын пакеті).
   Деңгей энергиялары — NIST (нақты); орбита радиусы — схемалық: r ∝ n_eff^0.85, E_ион − E = Ry·Z²/n_eff².
   Сутегі үшін n_eff = n (Бор моделі). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./physics.js'), require('./engine3d.js'));
  else { root.SL = root.SL || {}; root.SL.atomscene = factory(root.SL.physics, root.SL.engine3d); }
})(typeof window !== 'undefined' ? window : globalThis, function (P, E) {
  'use strict';

  var TILTS = [[0.0, 0.0], [0.55, 0.25], [-0.5, 0.6], [0.9, -0.4], [-0.8, -0.7], [0.3, 1.0], [-0.4, -1.1]];
  var ELEC = [0.55, 0.85, 1.0], PROTON = [1.0, 0.42, 0.35], NEUTRON = [0.6, 0.68, 0.8];

  function norm(v) { var l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }
  function add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
  function mul(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
  function ease(x) { x = E.clamp(x, 0, 1); return x * x * (3 - 2 * x); }
  function rotBasis(tx, tz) {
    // XZ жазықтығындағы орбита: u=(1,0,0), v=(0,0,1); алдымен X осінен tx, сосын Z осінен tz бұрыш
    var u = [1, 0, 0], v = [0, 0, 1];
    function rx(p) { var c = Math.cos(tx), s = Math.sin(tx); return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c]; }
    function rz(p) { var c = Math.cos(tz), s = Math.sin(tz); return [p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2]]; }
    return { u: rz(rx(u)), v: rz(rx(v)) };
  }

  function fmtE(e) { return e < 0.005 ? '0' : e.toFixed(e < 10 ? 3 : 2); }

  function AtomScene(canvas) {
    this.vp = new E.Viewport(canvas, { yaw: 0.55, pitch: 0.5, autoRotate: true });
    this.mode = 'emission';
    this.playing = true;
    this.speed = 1;
    this.time = 0;
    this.phi = 0;
    this.photons = [];
    this.trail = [];
    this.levels = [];
    this.tr = null;
    this.onPhase = null;
    this._phase = '';
    this.theme = { label: '#b9c6d6', dim: '#6f8197' };
    var rnd = P.mulberry32(7);
    this.stars = [];
    for (var i = 0; i < 90; i++) {
      var th = rnd() * 6.2832, ph = Math.acos(2 * rnd() - 1), r = 4.5 + rnd() * 4;
      this.stars.push([r * Math.sin(ph) * Math.cos(th), r * Math.cos(ph), r * Math.sin(ph) * Math.sin(th), 0.25 + rnd() * 0.6]);
    }
    this.nucleons = [];
    for (i = 0; i < 11; i++) {
      var d = norm([rnd() - 0.5, rnd() - 0.5, rnd() - 0.5]);
      this.nucleons.push({ p: mul(d, 0.035 * Math.cbrt(rnd()) * 1.4), proton: i % 2 === 0 });
    }
  }

  // tr: {sp, i} — таңдалған ауысу
  AtomScene.prototype.setTransition = function (sp, i, mode) {
    this.mode = mode || this.mode;
    var wl = sp.wl[i], ei = sp.ei[i], ek = sp.ek[i];
    var lo = P.parseLevel(sp.levels[sp.lo[i]]), up = P.parseLevel(sp.levels[sp.up[i]]);
    var isH = sp.name === 'H I', lv = [];
    function mk(Ev, label, kind, n) {
      var neff = isH ? n : P.nEff(Ev, sp.ion, sp.charge);
      return { E: Ev, neff: neff, r: P.orbitRadius(neff), label: label, kind: kind };
    }
    if (isH) {
      var ns = {};
      for (var q = 1; q <= 5; q++) ns[q] = 1;
      ns[lo.n] = 1; ns[up.n] = 1; if (up.n < 7) ns[up.n + 1] = 1;
      Object.keys(ns).map(Number).sort(function (a, b) { return a - b; }).forEach(function (n) {
        var En = 13.5984 * (1 - 1 / (n * n));
        lv.push(mk(En, 'n=' + n, n === lo.n ? 'lower' : n === up.n ? 'upper' : 'mid', n));
      });
    } else {
      if (ei > 0.02) lv.push(mk(0, 'негізгі күй', 'ground'));
      lv.push(mk(ei, lo.label, 'lower'));
      lv.push(mk(ek, up.label, 'upper'));
    }
    // сақиналар бір-біріне жабысып қалмасын (тек көрнекілік)
    lv.sort(function (a, b) { return a.E - b.E; });
    for (var j = 1; j < lv.length; j++) if (lv[j].r < lv[j - 1].r + 0.11) lv[j].r = lv[j - 1].r + 0.11;
    lv.forEach(function (L, k) {
      L.basis = rotBasis(TILTS[k % TILTS.length][0], TILTS[k % TILTS.length][1]);
      L.omega = 1.5 * Math.pow(0.5 / L.r, 1.5);
    });
    this.levels = lv;
    this.lower = lv.filter(function (L) { return L.kind === 'lower'; })[0];
    this.upper = lv.filter(function (L) { return L.kind === 'upper'; })[0];
    var rmax = lv[lv.length - 1].r;
    this.vp.home.zoom = this.vp.zoom = 0.78 / rmax;
    this.tr = {
      sp: sp, i: i, wl: wl, ei: ei, ek: ek, dE: ek - ei, lo: lo, up: up,
      color: P.wlColor(wl), visible: P.isVisible(wl), aki: sp.aki[i]
    };
    this.replay();
  };

  AtomScene.prototype.clear = function () { this.levels = []; this.tr = null; this.photons = []; this.trail = []; };

  AtomScene.prototype.setMode = function (mode) { this.mode = mode; this.replay(); };
  AtomScene.prototype.replay = function () { this.time = 0; this.photons = []; this.trail = []; this._phase = ''; };

  // цикл кестесі
  AtomScene.prototype._phases = function () {
    return this.mode === 'emission'
      ? [['ground', 1.8], ['excite', 1.1], ['excited', 1.9], ['decay', 0.8], ['after', 3.0]]
      : [['incoming', 2.0], ['absorb', 0.9], ['excited', 1.8], ['scatter', 0.8], ['after', 2.4]];
  };

  AtomScene.prototype._caption = function (name) {
    var t = this.tr, L = this.lower, U = this.upper;
    var E1 = L ? 'E₁ = ' + fmtE(L.E) + ' эВ' : '', E2 = U ? 'E₂ = ' + fmtE(U.E) + ' эВ' : '';
    var tau = t.aki ? ' (өмір ≈ ' + (1e9 / t.aki).toFixed(t.aki > 1e8 ? 1 : 0) + ' нс)' : '';
    var map = {
      ground: 'Электрон төменгі деңгейде: ' + E1,
      excite: 'Соқтығыс: электрон энергия алып қозады → ' + E2,
      excited: 'Қозған күй: ' + E2 + tau,
      decay: 'Секіру E₂ → E₁: ΔE = ' + t.dE.toFixed(3) + ' эВ фотонға айналады',
      after: this.mode === 'emission'
        ? 'Фотон ұшып барады: λ = ' + t.wl.toFixed(2) + ' нм' + (t.visible ? '' : ' (көзге көрінбейді)')
        : 'Электрон негізгі деңгейде; фотон кездейсоқ бағытқа шашырады',
      incoming: 'Фотон келеді: hν = ' + t.dE.toFixed(3) + ' эВ (λ = ' + t.wl.toFixed(2) + ' нм)',
      absorb: 'Фотон жұтылды: электрон ' + E1.split(' =')[0] + ' → ' + E2.split(' =')[0] + ' өтті',
      scatter: 'Қозған атом фотонды кез келген бағытқа қайта шығарады'
    };
    return map[name] || '';
  };

  AtomScene.prototype._spawnPhoton = function (from, dir, alpha, travelIn) {
    this.photons.push({ p0: from, dir: dir, t0: this.time, alpha: alpha, travelIn: !!travelIn });
  };

  AtomScene.prototype.update = function (dt) {
    var vp = this.vp;
    vp.tick(dt);
    if (!this.playing || !this.levels.length) return;
    dt *= this.speed;
    var prevT = this.time;
    this.time += dt;
    var ph = this._phases(), total = ph.reduce(function (a, p) { return a + p[1]; }, 0);
    if (this.time >= total) { this.time -= total; prevT = -1; this.photons = []; }
    var tt = this.time, name = '', t0 = 0, dur = 0;
    for (var k = 0; k < ph.length; k++) { if (tt < t0 + ph[k][1]) { name = ph[k][0]; dur = ph[k][1]; break; } t0 += ph[k][1]; }
    this.phaseName = name; this.phaseT = tt - t0; this.phaseDur = dur;
    if (name !== this._phase) {
      this._phase = name;
      if (this.onPhase) this.onPhase(this._caption(name), name);
      var B = vp.basis(), pos = this._electronPos(tt);
      if (this.mode === 'emission' && name === 'decay') {
        var d = norm(add(add(mul(B.right, 0.9), mul(B.up, 0.28)), mul(B.forward, 0.12)));
        this._spawnPhoton(pos, d, 1, false);
      } else if (this.mode === 'absorption' && name === 'incoming') {
        var dr = norm(add(add(mul(B.right, 1), mul(B.up, 0.12)), mul(B.forward, 0.1)));
        this._spawnPhoton(mul(dr, -3.2), dr, 1, true);
      } else if (this.mode === 'absorption' && name === 'scatter') {
        var rd = norm([Math.random() - 0.5, Math.random() - 0.3, Math.random() - 0.5]);
        this._spawnPhoton(pos, rd, 0.8, false);
      }
    }
    // электрон бұрышы
    var r = this._electronRadius(tt);
    var om = this._electronOmega(tt, r);
    this.phi += om * dt;
    // із
    var p = this._electronPos(tt);
    this.trail.push(p);
    if (this.trail.length > 26) this.trail.shift();
  };

  // әр фазадағы электрон күйі: {from, to, s} (деңгейдер арасындағы өту үлесі)
  AtomScene.prototype._jump = function (tt) {
    var ph = this._phases(), t0 = 0, i, em = this.mode === 'emission';
    for (i = 0; i < ph.length; i++) { if (tt < t0 + ph[i][1]) break; t0 += ph[i][1]; }
    var name = ph[Math.min(i, ph.length - 1)][0], lt = tt - t0, lo = this.lower, up = this.upper;
    if (em) {
      if (name === 'ground') return { a: lo, b: lo, s: 0 };
      if (name === 'excite') return { a: lo, b: up, s: ease((lt - 0.45) / 0.55) };
      if (name === 'excited') return { a: up, b: up, s: 0 };
      if (name === 'decay') return { a: up, b: lo, s: ease(lt / 0.6) };
      return { a: lo, b: lo, s: 0 };
    }
    if (name === 'incoming') return { a: lo, b: lo, s: 0 };
    if (name === 'absorb') return { a: lo, b: up, s: ease((lt - 0.1) / 0.6) };
    if (name === 'excited') return { a: up, b: up, s: 0 };
    if (name === 'scatter') return { a: up, b: lo, s: ease(lt / 0.6) };
    return { a: lo, b: lo, s: 0 };
  };
  AtomScene.prototype._electronRadius = function (tt) {
    var j = this._jump(tt); return j.a.r + (j.b.r - j.a.r) * j.s;
  };
  AtomScene.prototype._electronOmega = function (tt, r) {
    var j = this._jump(tt), w = j.a.omega + (j.b.omega - j.a.omega) * j.s;
    return j.s > 0 && j.s < 1 ? w * 2.6 : w;   // секіру кезінде тез айналады
  };
  AtomScene.prototype._electronPos = function (tt) {
    var j = this._jump(tt), r = j.a.r + (j.b.r - j.a.r) * j.s, s = j.s;
    var u = norm(add(mul(j.a.basis.u, 1 - s), mul(j.b.basis.u, s)));
    var v0 = add(mul(j.a.basis.v, 1 - s), mul(j.b.basis.v, s));
    var v = norm(add(v0, mul(u, -dot(v0, u))));
    return add(mul(u, r * Math.cos(this.phi)), mul(v, r * Math.sin(this.phi)));
  };

  AtomScene.prototype.render = function () {
    var vp = this.vp;
    if (!vp.visible && !this.forceRender) return;
    vp.resize();
    var ctx = vp.ctx, w = vp.w, h = vp.h, tt = this.time;
    // фон
    var g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.7);
    g.addColorStop(0, '#0d1a2b'); g.addColorStop(1, '#03060b');
    ctx.fillStyle = g; ctx.fillRect(0, 0, w, h);
    if (!this.levels.length) {
      ctx.fillStyle = 'rgba(160,180,205,0.85)'; ctx.font = '13px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('Атом ауысуын көру үшін түр таңдаңыз', w / 2, h / 2);
      return;
    }
    vp.begin();
    var self = this, k0 = vp.project(0, 0, 0).k;

    this.stars.forEach(function (s) {
      var p = vp.project(s[0], s[1], s[2]);
      if (p.z <= 0.5) return;
      vp.add(p.z + 50, function (c) { c.fillStyle = 'rgba(190,210,255,' + s[3] * 0.55 + ')'; c.fillRect(p.x, p.y, 1.4, 1.4); });
    });

    // орбиталар
    var hl = { lower: 'rgba(120,205,255,', upper: 'rgba(255,196,92,', ground: 'rgba(160,175,200,', mid: 'rgba(130,150,180,' };
    this.levels.forEach(function (L, li) {
      var pts = [], N = 96;
      for (var i = 0; i <= N; i++) {
        var a = i / N * 6.2832;
        var q = add(mul(L.basis.u, L.r * Math.cos(a)), mul(L.basis.v, L.r * Math.sin(a)));
        pts.push(vp.project(q[0], q[1], q[2]));
      }
      var strong = L.kind === 'lower' || L.kind === 'upper';
      vp.path(pts, hl[L.kind] + (strong ? '1)' : '1)'), strong ? 1.6 : 1, strong ? 0.75 : 0.28, true);
      // белгі
      var la = li === 0 ? 3.9 : 0.9 + li * 0.62, lq = add(mul(L.basis.u, L.r * Math.cos(la)), mul(L.basis.v, L.r * Math.sin(la)));
      var lp = vp.project(lq[0], lq[1], lq[2]);
      var col = L.kind === 'upper' ? '#ffc45c' : L.kind === 'lower' ? '#78cdff' : self.theme.dim;
      var al = li === 0 ? 'right' : 'left', ox = li === 0 ? -6 : 6;
      vp.text(lp, L.label, col, (strong ? '600 ' : '400 ') + '11px Inter, system-ui, sans-serif', al, ox, -7);
      vp.text(lp, fmtE(L.E) + ' эВ', self.theme.dim, '10px "JetBrains Mono", ui-monospace, monospace', al, ox, 5);
    });

    // ядро
    var c0 = vp.project(0, 0, 0);
    vp.glow(c0, k0 * 0.26, [1, 0.62, 0.3], 0.6);
    var rot = tt * 0.7;
    this.nucleons.forEach(function (n) {
      var cs = Math.cos(rot), sn = Math.sin(rot);
      var q = [n.p[0] * cs + n.p[2] * sn, n.p[1], -n.p[0] * sn + n.p[2] * cs];
      var p = vp.project(q[0], q[1], q[2]);
      vp.ball(p, 0.036 * p.k, n.proton ? PROTON : NEUTRON, 1);
    });

    // электрон
    var pos = this._electronPos(tt), pe = vp.project(pos[0], pos[1], pos[2]);
    var tr = this.trail, em = this.mode === 'emission';
    var jumping = this._jump(tt).s > 0 && this._jump(tt).s < 1;
    for (var i = 1; i < tr.length; i++) {
      var a = vp.project(tr[i - 1][0], tr[i - 1][1], tr[i - 1][2]), b = vp.project(tr[i][0], tr[i][1], tr[i][2]);
      vp.seg(a, b, 'rgb(140,215,255)', jumping ? 3 : 2, (i / tr.length) * (jumping ? 0.9 : 0.5), true);
    }
    vp.glow(pe, pe.k * (jumping ? 0.22 : 0.13), ELEC, jumping ? 0.9 : 0.6);
    vp.ball(pe, 0.055 * pe.k, ELEC, 1, -0.4);

    // соқтығыс бөлшегі / жарқыл
    var name = this.phaseName, lt = this.phaseT;
    if (em && name === 'excite' && lt < 0.5) {
      var s = 1 - lt / 0.5, B = vp.basis(), dir = norm(add(mul(B.right, -0.8), mul(B.up, 0.5)));
      var kp = add(mul(dir, 0.35 + 2.4 * s * s), [0, 0, 0]);
      var pk = vp.project(kp[0], kp[1], kp[2]);
      vp.glow(pk, pk.k * 0.18, [1, 0.85, 0.4], 0.9);
      vp.ball(pk, 0.04 * pk.k, [1, 0.92, 0.6], 1);
    }
    var flash = 0;
    if (name === 'excite' && lt > 0.4 && lt < 0.9) flash = 1 - Math.abs(lt - 0.55) / 0.35;
    if (name === 'absorb' && lt < 0.5) flash = 1 - lt / 0.5;
    if (name === 'decay' && lt < 0.5) flash = 1 - lt / 0.5;
    if (flash > 0) vp.glow(pe, pe.k * 0.5 * flash, this.tr.visible ? this.tr.color : [0.8, 0.8, 1], 0.9 * flash);

    // фотондар
    var T = this.tr, speed = 1.0, wlu = 0.1 * T.wl / 500, ncyc = 7;
    this.photons = this.photons.filter(function (ph) { return tt - ph.t0 < 6; });
    this.photons.forEach(function (ph) {
      var age = tt - ph.t0;
      if (ph.travelIn) {
        // кіре беріс: орталыққа жеткенде жойылады
        var dist = -3.2 + 1.5 * age;
        if (dist > -0.2) return;
        self._drawPhoton(vp, add(ph.p0, mul(ph.dir, 1.5 * age)), ph.dir, wlu, ncyc, 1, age);
      } else {
        var al = ph.alpha * E.clamp(1 - (age - 2.8) / 2.2, 0, 1);
        if (al <= 0) return;
        var c = add(ph.p0, mul(ph.dir, speed * age));
        self._drawPhoton(vp, c, ph.dir, wlu, ncyc, al, age);
      }
    });

    vp.flush();

    // ΔE белгісі (2D жабын)
    ctx.font = '600 12px Inter, system-ui, sans-serif'; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(190,205,225,0.85)';
    var ttl = T.sp.name + (T.sp.name === 'H I' && P.hydrogenName(T.lo.n, T.up.n) ? ' · ' + P.hydrogenName(T.lo.n, T.up.n) : '');
    ctx.fillText(ttl, 12, 10);
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace'; ctx.fillStyle = 'rgba(150,170,195,0.9)';
    ctx.fillText('λ ' + T.wl.toFixed(3) + ' нм   ΔE ' + T.dE.toFixed(4) + ' эВ', 12, 28);
  };

  AtomScene.prototype._drawPhoton = function (vp, c, dir, wlu, ncyc, alpha, age) {
    var T = this.tr, L = wlu * ncyc, B = vp.basis();
    var perp = norm(cross(dir, B.up));
    if (Math.hypot(perp[0], perp[1], perp[2]) < 0.2) perp = norm(cross(dir, B.right));
    var col = T.color, pts = [], N = 64, amp = 0.085;
    var vis = T.visible;
    for (var i = 0; i <= N; i++) {
      var s = (i / N - 0.5) * L, env = Math.pow(Math.sin(Math.PI * i / N), 2);
      var ph = 6.2832 * s / wlu - age * 7;
      var q = add(add(c, mul(dir, s)), mul(perp, amp * env * Math.sin(ph)));
      pts.push(vp.project(q[0], q[1], q[2]));
    }
    var css = E.css(col, 1);
    vp.path(pts, css, 2.8, 0.95 * alpha, false, true);
    var pc = vp.project(c[0], c[1], c[2]);
    vp.glow(pc, pc.k * 0.17, col, 0.55 * alpha);
    if (!vis) {
      vp.text(pc, 'көзге көрінбейді', 'rgba(200,190,230,' + (0.8 * alpha) + ')', '10px Inter, system-ui, sans-serif', 'center', 0, -16);
    }
  };

  AtomScene.prototype.frame = function (dt) { this.update(dt); this.render(); };

  return { AtomScene: AtomScene };
});
