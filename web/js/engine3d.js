/* Шағын 3D қозғалтқыш (Canvas 2D): орбиталық камера, перспектива, painter's algorithm, жарқырау (additive).
   Кітапхана қажет емес, file:// арқылы да жұмыс істейді. */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.SL = root.SL || {}; root.SL.engine3d = factory(); }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  function clamp(v, a, b) { return v < a ? a : v > b ? b : v; }
  function css(c, a) {
    return 'rgba(' + Math.round(clamp(c[0], 0, 1) * 255) + ',' + Math.round(clamp(c[1], 0, 1) * 255) + ',' +
      Math.round(clamp(c[2], 0, 1) * 255) + ',' + (a === undefined ? 1 : a) + ')';
  }

  function Viewport(canvas, opts) {
    opts = opts || {};
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.yaw = opts.yaw === undefined ? 0.6 : opts.yaw;
    this.pitch = opts.pitch === undefined ? 0.35 : opts.pitch;
    this.zoom = opts.zoom || 1;
    this.target = opts.target || [0, 0, 0];
    this.offX = 0; this.offY = 0; // кадрды жылжыту (пиксель)
    this.persp = 7;               // перспектива күші (камера қашықтығы)
    this.home = { yaw: this.yaw, pitch: this.pitch, zoom: this.zoom };
    this.autoRotate = !!opts.autoRotate;
    this.dragging = false;
    this.items = [];
    this.w = 0; this.h = 0; this.dpr = 1;
    this.visible = true;
    this._bind();
    this.resize();
  }

  Viewport.prototype._bind = function () {
    var self = this, cv = this.cv, pts = new Map(), lastPinch = 0;
    cv.style.touchAction = 'none';
    cv.addEventListener('pointerdown', function (e) {
      cv.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      self.dragging = true;
      if (pts.size === 2) lastPinch = 0;
    });
    cv.addEventListener('pointermove', function (e) {
      var p = pts.get(e.pointerId);
      if (!p) return;
      var dx = e.clientX - p.x, dy = e.clientY - p.y;
      p.x = e.clientX; p.y = e.clientY;
      if (pts.size === 1) {
        self.yaw += dx * 0.008;
        self.pitch = clamp(self.pitch + dy * 0.008, -1.45, 1.45);
      } else if (pts.size === 2) {
        var a = Array.from(pts.values()), d = Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
        if (lastPinch) self.zoom = clamp(self.zoom * d / lastPinch, 0.35, 5);
        lastPinch = d;
      }
    });
    function up(e) { pts.delete(e.pointerId); if (!pts.size) self.dragging = false; lastPinch = 0; }
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', up);
    cv.addEventListener('wheel', function (e) {
      e.preventDefault();
      self.zoom = clamp(self.zoom * Math.exp(-e.deltaY * 0.0012), 0.35, 5);
    }, { passive: false });
    cv.addEventListener('dblclick', function () { self.reset(); });
    if (typeof ResizeObserver !== 'undefined') new ResizeObserver(function () { self.resize(); }).observe(cv);
    if (typeof IntersectionObserver !== 'undefined')
      new IntersectionObserver(function (en) { self.visible = en[0].isIntersecting; }).observe(cv);
  };

  Viewport.prototype.reset = function () {
    this.yaw = this.home.yaw; this.pitch = this.home.pitch; this.zoom = this.home.zoom;
  };

  Viewport.prototype.resize = function () {
    var r = this.cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    var w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
    if (w === this.w && h === this.h && dpr === this.dpr) return;
    this.w = w; this.h = h; this.dpr = dpr;
    this.cv.width = Math.round(w * dpr); this.cv.height = Math.round(h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  Viewport.prototype.tick = function (dt) {
    if (this.autoRotate && !this.dragging) this.yaw += 0.15 * dt;
  };

  // әлем координатасы -> экран: {x, y, z (тереңдік, үлкен = алыс), k (масштаб)}
  Viewport.prototype.project = function (x, y, z) {
    x -= this.target[0]; y -= this.target[1]; z -= this.target[2];
    var cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    var x1 = x * cy + z * sy, z1 = -x * sy + z * cy;
    var y2 = y * cp + z1 * sp, z2 = -y * sp + z1 * cp;   // pitch > 0: камера жоғарыдан төмен қарайды
    var zc = z2 + this.persp, base = Math.min(this.w, this.h) * 0.5 * this.zoom;
    var k = base * this.persp / zc;
    return { x: this.w / 2 + this.offX + x1 * k, y: this.h / 2 + this.offY - y2 * k, z: zc, k: k };
  };
  // камераның оң/жоғары/алға бағыттары (әлем координатасында)
  Viewport.prototype.basis = function () {
    var cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    return {
      right: [cy, 0, sy],
      up: [-sy * sp, cp, cy * sp],
      forward: [-sy * cp, -sp, cy * cp]   // тереңдік өсетін бағыт (камерадан сахнаға қарай)
    };
  };

  Viewport.prototype.begin = function () { this.items.length = 0; };
  Viewport.prototype.add = function (z, fn) { this.items.push({ z: z, fn: fn }); };
  Viewport.prototype.flush = function () {
    this.items.sort(function (a, b) { return b.z - a.z; });
    for (var i = 0; i < this.items.length; i++) this.items[i].fn(this.ctx);
  };

  // ---- примитивтер (тереңдік тізіміне қосады) ----
  Viewport.prototype.glow = function (p, r, c, a) {
    if (r < 0.5) return;
    this.add(p.z, function (ctx) {
      var g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r);
      g.addColorStop(0, css(c, a)); g.addColorStop(0.4, css(c, a * 0.35)); g.addColorStop(1, css(c, 0));
      ctx.globalCompositeOperation = 'lighter';
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.2832); ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
    });
  };
  Viewport.prototype.ball = function (p, r, c, a, zBias) {
    if (r < 0.4) return;
    this.add(p.z + (zBias || 0), function (ctx) {
      var g = ctx.createRadialGradient(p.x - r * 0.35, p.y - r * 0.35, r * 0.1, p.x, p.y, r);
      g.addColorStop(0, css([c[0] * 0.4 + 0.6, c[1] * 0.4 + 0.6, c[2] * 0.4 + 0.6], a));
      g.addColorStop(0.55, css(c, a));
      g.addColorStop(1, css([c[0] * 0.35, c[1] * 0.35, c[2] * 0.35], a));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.2832); ctx.fill();
    });
  };
  Viewport.prototype.seg = function (a, b, color, w, alpha, additive) {
    this.add((a.z + b.z) / 2, function (ctx) {
      ctx.globalAlpha = alpha === undefined ? 1 : alpha;
      if (additive) ctx.globalCompositeOperation = 'lighter';
      ctx.strokeStyle = color; ctx.lineWidth = w || 1;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over';
    });
  };
  // көпбұрыш (экран жазықтығындағы проекциялар тізімі)
  Viewport.prototype.poly = function (pts, fill, stroke, w, zBias, additive) {
    var z = 0; pts.forEach(function (p) { z += p.z; }); z = z / pts.length + (zBias || 0);
    this.add(z, function (ctx) {
      ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
      for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.closePath();
      if (additive) ctx.globalCompositeOperation = 'lighter';
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      ctx.globalCompositeOperation = 'source-over';
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = w || 1; ctx.stroke(); }
    });
  };
  // 3D ломаная (орбита, толқын): тереңдікке қарай әр сегмент өз алфасымен
  Viewport.prototype.path = function (P, color, w, alpha, depthFade, additive) {
    for (var i = 0; i + 1 < P.length; i++) {
      var a = P[i], b = P[i + 1], z = (a.z + b.z) / 2, al = alpha;
      if (depthFade) al = alpha * clamp(1.25 - (z - this.persp) * 0.12, 0.25, 1.15);
      this.seg(a, b, color, w, al, additive);
    }
  };
  Viewport.prototype.text = function (p, str, color, font, align, dx, dy) {
    this.add(-1e6, function (ctx) {
      ctx.fillStyle = color; ctx.font = font || '12px Inter, system-ui, sans-serif';
      ctx.textAlign = align || 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(str, p.x + (dx || 0), p.y + (dy || 0));
    });
  };

  return { Viewport: Viewport, css: css, clamp: clamp };
});
