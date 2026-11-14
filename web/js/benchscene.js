/* 3D призмалық спектрометр: көз → саңылау → призма → экран.
   Сәулелердің жолы Снелл заңымен есептеледі (әр толқын ұзындығы үшін өз сыну көрсеткіші, Коши: n = A + B/λ²).
   Көрнекілік үшін әйнектің дисперсиясы нақты флинттен ~3 есе үлкен. УК әйнекте жұтылады, сондықтан көрсетілмейді. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory(require('./physics.js'), require('./engine3d.js'));
  else { root.SL = root.SL || {}; root.SL.benchscene = factory(root.SL.physics, root.SL.engine3d); }
})(typeof window !== 'undefined' ? window : globalThis, function (P, E) {
  'use strict';

  var B_COEF = 0.034, A_COEF = 1.73 - B_COEF / (0.55 * 0.55);
  function nGlass(wlNm) { var um = wlNm / 1000; return A_COEF + B_COEF / (um * um); }
  function v2(a, b) { return [a, b]; }
  function dot2(a, b) { return a[0] * b[0] + a[1] * b[1]; }
  function refract(d, nrm, eta) {   // nrm d-ға қарсы бағытталған бірлік нормаль
    var c = -dot2(nrm, d), k = 1 - eta * eta * (1 - c * c);
    if (k < 0) return null;
    var f = eta * c - Math.sqrt(k);
    return [d[0] * eta + nrm[0] * f, d[1] * eta + nrm[1] * f];
  }

  function Geometry() {
    var A = 60 * Math.PI / 180, n0 = nGlass(550), i0 = Math.asin(n0 * Math.sin(A / 2));
    var th1 = Math.PI - i0, th2 = th1 + 2 * Math.PI / 3, th3 = th2 + 2 * Math.PI / 3;   // 1-бет, табан, 2-бет
    this.a = 0.85;
    this.n1 = [Math.cos(th1), Math.sin(th1)];
    this.nb = [Math.cos(th2), Math.sin(th2)];
    this.n2 = [Math.cos(th3), Math.sin(th3)];
    var a = this.a;
    function vtx(p, q) { return [2 * a * (p[0] + q[0]), 2 * a * (p[1] + q[1])]; }
    this.verts = [vtx(this.n1, this.nb), vtx(this.nb, this.n2), vtx(this.n2, this.n1)];
    this.z0 = a * this.n1[1];               // сәуле 1-беттің ортасына түседі
    this.slitX = -3.0; this.srcX = -5.2;
    var c = this.trace(550);
    this.dc = c.d2; this.Pc = c.P2;
    this.D = 5.2;
    this.S = [c.P2[0] + this.D * c.d2[0], c.P2[1] + this.D * c.d2[1]];
    var w = [-c.d2[1], c.d2[0]];
    this.P1c = c.P1;
    // қызыл экранның оң жағында болсын
    this.w = (this._uw(this.trace(400), w) > this._uw(this.trace(700), w)) ? [-w[0], -w[1]] : w;
  }
  Geometry.prototype._uw = function (r, w) { return dot2([r.P3[0] - this.S[0], r.P3[1] - this.S[1]], w); };
  Geometry.prototype.trace = function (wl) {
    var n = nGlass(wl), a = this.a, n1 = this.n1, n2 = this.n2;
    var d0 = [1, 0], z0 = this.z0;
    var xh = (a - z0 * n1[1]) / n1[0], P1 = [xh, z0];
    var d1 = refract(d0, n1, 1 / n);
    if (!d1) return null;
    var t = (a - dot2(n2, P1)) / dot2(n2, d1), P2 = [P1[0] + t * d1[0], P1[1] + t * d1[1]];
    var d2 = refract(d1, [-n2[0], -n2[1]], n);
    if (!d2) return null;
    var res = { P0: [this.slitX, z0], P1: P1, P2: P2, d2: d2, P3: null };
    if (this.S) {
      var dc = this.dc, tt = dot2([this.S[0] - P2[0], this.S[1] - P2[1]], dc) / dot2(d2, dc);
      res.P3 = [P2[0] + tt * d2[0], P2[1] + tt * d2[1]];
    }
    return res;
  };
  Geometry.prototype.screenU = function (wl) {
    var r = this.trace(wl); if (!r) return null;
    return { r: r, u: dot2([r.P3[0] - this.S[0], r.P3[1] - this.S[1]], this.w) };
  };

  function BenchScene(canvas) {
    this.vp = new E.Viewport(canvas, { yaw: 0.62, pitch: 0.55, autoRotate: false, target: [-0.4, 0, -1.6] });
    this.g = new Geometry();
    this.mode = 'emission';
    this.rays = [];
    this.tubeColor = [1, 0.8, 0.4];
    this.time = 0;
    this.playing = true;
    this.speed = 1;
    this._fitKey = '';
    this.hy = 0.5;
    var u400 = this.g.screenU(380).u, u900 = this.g.screenU(900).u;     // экран ені геометрияға ғана тәуелді
    this.uMin = Math.min(u400, u900) - 0.12; this.uMax = Math.max(u400, u900) + 0.12;
  }

  // rays: [{wl, I (0..1)}]
  BenchScene.prototype.setData = function (d) {
    this.mode = d.mode; this.tubeColor = d.tubeColor || this.tubeColor;
    var g = this.g, rays = [];
    d.rays.forEach(function (ry) {
      if (ry.wl < 380 || ry.wl > 900 || ry.I < 0.0015) return;
      var s = g.screenU(ry.wl);
      if (s) rays.push({ wl: ry.wl, I: ry.I, r: s.r, u: s.u, color: P.wlColor(ry.wl), ph: (ry.wl * 7.31) % 1 });
    });
    this.rays = rays;
  };

  // бүкіл қондырғы кадрға сыйатындай масштаб пен орталық
  BenchScene.prototype.fit = function () {
    var vp = this.vp, g = this.g, hy = this.hy + 0.2, pts = [];
    function w3(p, y) { return [p[0], y, p[1]]; }
    var uA = this.uMin === undefined ? -1 : this.uMin, uB = this.uMax === undefined ? 1 : this.uMax;
    [-0.6, 0.6].forEach(function (y) {
      pts.push(w3([g.srcX - 0.3, g.z0], y));
      g.verts.forEach(function (v) { pts.push(w3(v, y)); });
      [uA, uB].forEach(function (u) { pts.push(w3([g.S[0] + g.w[0] * u, g.S[1] + g.w[1] * u], y)); });
    });
    var c = [0, 0, 0];
    pts.forEach(function (p) { c[0] += p[0] / pts.length; c[1] += p[1] / pts.length; c[2] += p[2] / pts.length; });
    vp.target = c; vp.zoom = 1; vp.offX = 0; vp.offY = 0;
    function box() {
      var minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9;
      pts.forEach(function (p) {
        var q = vp.project(p[0], p[1], p[2]);
        minx = Math.min(minx, q.x); maxx = Math.max(maxx, q.x); miny = Math.min(miny, q.y); maxy = Math.max(maxy, q.y);
      });
      return [minx, maxx, miny, maxy];
    }
    var b0 = box(), bw = Math.max(b0[1] - b0[0], 1), bh = Math.max(b0[3] - b0[2], 1);
    var s = Math.min(vp.w * 0.88 / bw, (vp.h - 92) / bh);
    vp.zoom = s;
    var b1 = box();
    vp.offX = vp.w / 2 - (b1[0] + b1[1]) / 2;
    vp.offY = (vp.h - 64) / 2 - (b1[2] + b1[3]) / 2;
    vp.zoom = vp.home.zoom = s;
  };

  BenchScene.prototype.update = function (dt) {
    this.vp.tick(dt);
    if (this.playing) this.time += dt * this.speed;
  };

  BenchScene.prototype.render = function () {
    var vp = this.vp;
    if (!vp.visible && !this.forceRender) return;
    vp.resize();
    var key = vp.w + 'x' + vp.h;
    if (key !== this._fitKey) { this._fitKey = key; this.fit(); }
    var ctx = vp.ctx, w = vp.w, h = vp.h, g = this.g, hy = this.hy, self = this;
    var bg = ctx.createRadialGradient(w / 2, h * 0.55, 0, w / 2, h * 0.55, Math.max(w, h) * 0.75);
    bg.addColorStop(0, '#0e1826'); bg.addColorStop(1, '#03060b');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
    vp.begin();
    function P3(p, y) { return vp.project(p[0], y, p[1]); }

    // оптикалық үстел торы
    for (var gx = -5; gx <= 4; gx++) vp.seg(P3([gx, -4.6], -0.62), P3([gx, 1.6], -0.62), 'rgba(120,150,190,0.16)', 1, 1);
    for (var gz = -4; gz <= 1; gz++) vp.seg(P3([-5, gz], -0.62), P3([4, gz], -0.62), 'rgba(120,150,190,0.16)', 1, 1);

    // көз: разряд түтігі (эмиссия) немесе ақ шам + газ ұяшығы (жұтылу)
    var em = this.mode === 'emission', tc = em ? this.tubeColor : [1, 0.96, 0.85];
    var sp0 = P3([g.srcX, g.z0], -0.42), sp1 = P3([g.srcX, g.z0], 0.42);
    vp.glow(P3([g.srcX, g.z0], 0), sp0.k * 0.9, tc, 0.55);
    vp.add((sp0.z + sp1.z) / 2, function (c) {
      c.lineCap = 'round';
      c.strokeStyle = 'rgba(210,225,240,0.35)'; c.lineWidth = sp0.k * 0.34;
      c.beginPath(); c.moveTo(sp0.x, sp0.y); c.lineTo(sp1.x, sp1.y); c.stroke();
      c.strokeStyle = E.css(tc, 0.95); c.lineWidth = sp0.k * 0.2;
      c.beginPath(); c.moveTo(sp0.x, sp0.y); c.lineTo(sp1.x, sp1.y); c.stroke();
      c.strokeStyle = 'rgba(255,255,255,0.8)'; c.lineWidth = sp0.k * 0.07;
      c.beginPath(); c.moveTo(sp0.x, sp0.y); c.lineTo(sp1.x, sp1.y); c.stroke();
      c.lineCap = 'butt';
    });
    vp.text(P3([g.srcX, g.z0], -0.42), em ? 'Разряд түтігі' : 'Ақ жарық көзі', '#9fb3c8', '600 11px Inter, system-ui, sans-serif', 'center', 0, 16);
    if (!em) {
      var cx = -4.1, cw = 0.5, cz = g.z0, y = 0.34;
      var q = [[cx - cw / 2, cz - 0.2], [cx + cw / 2, cz - 0.2], [cx + cw / 2, cz + 0.2], [cx - cw / 2, cz + 0.2]];
      vp.poly(q.map(function (p) { return P3(p, -y); }), 'rgba(150,200,255,0.10)', 'rgba(170,210,255,0.5)', 1);
      vp.poly(q.map(function (p) { return P3(p, y); }), 'rgba(150,200,255,0.14)', 'rgba(170,210,255,0.6)', 1);
      for (var k = 0; k < 4; k++) vp.seg(P3(q[k], -y), P3(q[k], y), 'rgba(170,210,255,0.45)', 1, 1);
      vp.text(P3([cx, cz], y), 'Газ ұяшығы', '#9fb3c8', '600 11px Inter, system-ui, sans-serif', 'center', 0, -12);
    }

    // саңылау
    [-1, 1].forEach(function (s) {
      var za = g.z0 + s * 0.035, zb = g.z0 + s * 0.4;
      var q = [[g.slitX, za], [g.slitX, zb]];
      vp.poly([P3(q[0], -hy - 0.12), P3(q[1], -hy - 0.12), P3(q[1], hy + 0.12), P3(q[0], hy + 0.12)],
        'rgba(25,34,48,0.95)', 'rgba(120,145,175,0.7)', 1);
    });
    vp.text(P3([g.slitX, g.z0 + 0.1], -hy - 0.12), 'Саңылау', '#9fb3c8', '600 11px Inter, system-ui, sans-serif', 'center', 0, 14);

    // кіріс сәуле (барлық түстің қосындысы)
    var inc = em ? this.tubeColor : [1, 0.97, 0.9];
    this._ribbon(vp, [g.srcX + 0.1, g.z0], g.P1c, inc, 0.16 + (this.rays.length ? 0.1 : 0), hy, 0.8, true);

    // призма
    var V = g.verts, ymin = -hy - 0.18, ymax = hy + 0.18;
    var side = [[V[0], V[1]], [V[1], V[2]], [V[2], V[0]]];
    side.forEach(function (e) {
      vp.poly([P3(e[0], ymin), P3(e[1], ymin), P3(e[1], ymax), P3(e[0], ymax)], 'rgba(130,185,235,0.10)', 'rgba(185,225,255,0.55)', 1);
    });
    vp.poly(V.map(function (p) { return P3(p, ymax); }), 'rgba(150,205,255,0.13)', 'rgba(200,232,255,0.8)', 1.2);
    vp.poly(V.map(function (p) { return P3(p, ymin); }), 'rgba(110,160,210,0.10)', 'rgba(150,190,230,0.45)', 1);
    vp.text(P3([0, 0], ymax), 'Призма', '#9fb3c8', '600 11px Inter, system-ui, sans-serif', 'center', 0, -12);

    // дисперсияланған сәулелер
    var rays = this.rays;
    for (var r = 0; r < rays.length; r++) {
      var ry = rays[r], col = ry.color, vis = P.isVisible(ry.wl), a = Math.pow(ry.I, 0.4) * (vis ? 1 : 0.55);
      this._ribbon(vp, ry.r.P1, ry.r.P2, col, 0.10 * a, hy, 0.7 * a, false);
      this._ribbon(vp, ry.r.P2, ry.r.P3, col, 0.14 * a, hy, 0.95 * a, false);
      // фотон-нүкте (қозғалыс көрінуі үшін)
      var s = (this.time * 0.35 + ry.ph) % 1, pt, p;
      if (s < 0.5) { var f = s / 0.5; p = [ry.r.P0[0] + (ry.r.P1[0] - ry.r.P0[0]) * f, ry.r.P0[1] + (ry.r.P1[1] - ry.r.P0[1]) * f]; }
      else if (s < 0.6) { f = (s - 0.5) / 0.1; p = [ry.r.P1[0] + (ry.r.P2[0] - ry.r.P1[0]) * f, ry.r.P1[1] + (ry.r.P2[1] - ry.r.P1[1]) * f]; }
      else { f = (s - 0.6) / 0.4; p = [ry.r.P2[0] + (ry.r.P3[0] - ry.r.P2[0]) * f, ry.r.P2[1] + (ry.r.P3[1] - ry.r.P2[1]) * f]; }
      pt = P3(p, 0);
      vp.glow(pt, pt.k * 0.07, col, 0.8 * a);
    }

    // экран + мазмұны бір элементте (тереңдік бойынша сұрыптау қатесін болдырмау үшін)
    var uA = this.uMin === undefined ? -1 : this.uMin, uB = this.uMax === undefined ? 1 : this.uMax;
    function SP(u, y) { return vp.project(g.S[0] + g.w[0] * u, y, g.S[1] + g.w[1] * u); }
    var sh = hy + 0.14, quad = [SP(uA, -sh), SP(uB, -sh), SP(uB, sh), SP(uA, sh)], bars = [], glows = [];
    if (em) {
      rays.forEach(function (ry) {
        var vis = P.isVisible(ry.wl), a = Math.pow(ry.I, 0.4) * (vis ? 1 : 0.5), bw = 0.022;
        bars.push({ q: [SP(ry.u - bw, -hy), SP(ry.u + bw, -hy), SP(ry.u + bw, hy), SP(ry.u - bw, hy)], c: E.css(ry.color, 0.25 + 0.7 * a) });
        var pc = SP(ry.u, 0); glows.push({ p: pc, r: pc.k * (0.1 + 0.12 * a), col: ry.color, a: 0.5 * a });
      });
    } else {
      for (var i = 0; i < rays.length - 1; i++) {
        var A = rays[i], Bn = rays[i + 1];
        bars.push({ q: [SP(A.u, -hy), SP(Bn.u + 0.004, -hy), SP(Bn.u + 0.004, hy), SP(A.u, hy)], c: E.css(A.color, 0.1 + 0.9 * Math.pow(Math.min(1, A.I), 0.8)) });
      }
    }
    vp.add(SP((uA + uB) / 2, 0).z, function (c) {
      c.beginPath(); c.moveTo(quad[0].x, quad[0].y);
      for (var k = 1; k < 4; k++) c.lineTo(quad[k].x, quad[k].y);
      c.closePath(); c.fillStyle = 'rgba(6,10,16,0.96)'; c.fill();
      c.globalCompositeOperation = 'lighter';
      bars.forEach(function (b) {
        c.beginPath(); c.moveTo(b.q[0].x, b.q[0].y);
        for (var k = 1; k < 4; k++) c.lineTo(b.q[k].x, b.q[k].y);
        c.closePath(); c.fillStyle = b.c; c.fill();
      });
      glows.forEach(function (gl) {
        var gr = c.createRadialGradient(gl.p.x, gl.p.y, 0, gl.p.x, gl.p.y, gl.r);
        gr.addColorStop(0, E.css(gl.col, gl.a)); gr.addColorStop(1, E.css(gl.col, 0));
        c.fillStyle = gr; c.beginPath(); c.arc(gl.p.x, gl.p.y, gl.r, 0, 6.2832); c.fill();
      });
      c.globalCompositeOperation = 'source-over';
      c.strokeStyle = 'rgba(150,175,205,0.7)'; c.lineWidth = 1.2;
      c.beginPath(); c.moveTo(quad[0].x, quad[0].y);
      for (var j = 1; j < 4; j++) c.lineTo(quad[j].x, quad[j].y);
      c.closePath(); c.stroke();
    });
    vp.text(SP((uA + uB) / 2, sh), 'Экран (детектор)', '#9fb3c8', '600 11px Inter, system-ui, sans-serif', 'center', 0, -12);
    // экрандағы белгілер: ең күшті көрінетін сызықтар
    if (em) {
      var lab = rays.slice().sort(function (a1, b1) { return b1.I - a1.I; }).slice(0, 9).sort(function (a1, b1) { return a1.u - b1.u; });
      var lastX = -1e9;
      lab.forEach(function (ry) {
        var p = SP(ry.u, -sh);
        if (p.x - lastX < 34) return;
        lastX = p.x;
        vp.text(p, ry.wl.toFixed(0), E.css(ry.color, 0.95), '10px "JetBrains Mono", ui-monospace, monospace', 'center', 0, 12);
      });
    }

    vp.flush();
  };

  BenchScene.prototype._ribbon = function (vp, a, b, color, alpha, hy, lineAlpha) {
    function pj(p, y) { return vp.project(p[0], y, p[1]); }
    var q = [pj(a, -hy), pj(b, -hy), pj(b, hy), pj(a, hy)];
    vp.poly(q, E.css(color, alpha), null, 0, 0, true);
    vp.seg(pj(a, 0), pj(b, 0), E.css(color, 1), 1.4, Math.min(1, lineAlpha), true);
  };

  BenchScene.prototype.frame = function (dt) { this.update(dt); this.render(); };

  return { BenchScene: BenchScene, nGlass: nGlass };
});
