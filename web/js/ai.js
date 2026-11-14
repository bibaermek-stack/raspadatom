/* Оқытылған 1D-CNN + Transformer моделінің браузердегі (таза JS) forward өтуі.
   Архитектура spectra_ai/model.py-мен бірдей; BatchNorm конволюцияға біріктірілген (tools/export_web.py). */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else { root.SL = root.SL || {}; root.SL.ai = factory(); }
})(typeof window !== 'undefined' ? window : globalThis, function () {
  'use strict';

  function b64ToF32(b64) {
    var bin = atob(b64), u8 = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
    return new Float32Array(u8.buffer);
  }

  // erf: Numerical Recipes erfc (салыстырмалы қате < 1.2e-7)
  function erf(x) {
    var z = Math.abs(x), t = 1 / (1 + 0.5 * z);
    var r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
      t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
      t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
  }
  function gelu(x) { return 0.5 * x * (1 + erf(x * Math.SQRT1_2)); }

  function sinusoidal(len, dim) {
    var pe = new Float32Array(len * dim);
    for (var p = 0; p < len; p++) {
      for (var i = 0; i < dim; i += 2) {
        var div = Math.exp(i * (-Math.log(10000) / dim));
        pe[p * dim + i] = Math.sin(p * div);
        pe[p * dim + i + 1] = Math.cos(p * div);
      }
    }
    return pe;
  }

  // x: (cin, L) -> conv(k, same) -> GELU -> maxpool4 -> (cout, L/4)
  function convGeluPool(x, cin, L, w, b, cout, K) {
    var pad = K >> 1, Lp = L >> 2, y = new Float32Array(cout * Lp), tmp = new Float32Array(L);
    for (var o = 0; o < cout; o++) {
      tmp.fill(b[o]);
      for (var c = 0; c < cin; c++) {
        var wo = (o * cin + c) * K, xo = c * L;
        for (var k = 0; k < K; k++) {
          var wv = w[wo + k], sh = k - pad, i0 = Math.max(0, -sh), i1 = Math.min(L, L - sh);
          for (var i = i0; i < i1; i++) tmp[i] += wv * x[xo + i + sh];
        }
      }
      for (var p = 0; p < Lp; p++) {
        var m = -Infinity;
        for (var q = 0; q < 4; q++) { var v = gelu(tmp[p * 4 + q]); if (v > m) m = v; }
        y[o * Lp + p] = m;
      }
    }
    return y;
  }

  function layerNorm(x, T, D, w, b, eps) {
    var y = new Float32Array(T * D);
    for (var t = 0; t < T; t++) {
      var mean = 0, v = 0, o = t * D;
      for (var d = 0; d < D; d++) mean += x[o + d];
      mean /= D;
      for (d = 0; d < D; d++) { var c = x[o + d] - mean; v += c * c; }
      var inv = 1 / Math.sqrt(v / D + eps);
      for (d = 0; d < D; d++) y[o + d] = (x[o + d] - mean) * inv * w[d] + b[d];
    }
    return y;
  }

  // (T, din) · W(dout, din)^T + b
  function linear(x, T, din, W, b, dout) {
    var y = new Float32Array(T * dout);
    for (var t = 0; t < T; t++) {
      var xo = t * din;
      for (var o = 0; o < dout; o++) {
        var s = b[o], wo = o * din;
        for (var d = 0; d < din; d++) s += x[xo + d] * W[wo + d];
        y[t * dout + o] = s;
      }
    }
    return y;
  }

  function Model(meta) {
    this.meta = meta;
    var all = b64ToF32(meta.weights), w = this.w = {};
    meta.tensors.forEach(function (t) {
      var n = t.shape.reduce(function (a, c) { return a * c; }, 1);
      w[t.name] = all.subarray(t.offset, t.offset + n);
    });
    this.species = meta.species;
    this.tokens = meta.nPoints >> 4;
    this.pos = sinusoidal(this.tokens, meta.dModel);
  }

  // spec: Float32Array(2048), 0..1  ->  { probs, logits, tempK, tNorm }
  Model.prototype.predict = function (spec) {
    var m = this.meta, w = this.w, L = m.nPoints, D = m.dModel, T = this.tokens, H = m.heads, hd = D / H;
    var x = new Float32Array(2 * L);
    for (var i = 0; i < L; i++) { var v = spec[i]; x[i] = v; x[L + i] = v > 0 ? Math.sqrt(v) : 0; }
    var h1 = convGeluPool(x, 2, L, w['conv1.w'], w['conv1.b'], 32, 7);
    var h2 = convGeluPool(h1, 32, L >> 2, w['conv2.w'], w['conv2.b'], D, 5);   // (D, T)
    var f = new Float32Array(T * D);
    for (var t = 0; t < T; t++) for (var d = 0; d < D; d++) f[t * D + d] = h2[d * T + t] + this.pos[t * D + d];

    for (var l = 0; l < m.layers; l++) {
      var p = 'l' + l + '.';
      var a = layerNorm(f, T, D, w[p + 'ln1.w'], w[p + 'ln1.b'], m.ln_eps);
      var qkv = linear(a, T, D, w[p + 'qkv.w'], w[p + 'qkv.b'], 3 * D);
      var ctx = new Float32Array(T * D), sc = new Float32Array(T), scale = 1 / Math.sqrt(hd);
      for (var hh = 0; hh < H; hh++) {
        for (t = 0; t < T; t++) {
          var mx = -Infinity;
          for (var s = 0; s < T; s++) {
            var dot = 0;
            for (var j = 0; j < hd; j++) dot += qkv[t * 3 * D + hh * hd + j] * qkv[s * 3 * D + D + hh * hd + j];
            dot *= scale; sc[s] = dot; if (dot > mx) mx = dot;
          }
          var sum = 0;
          for (s = 0; s < T; s++) { sc[s] = Math.exp(sc[s] - mx); sum += sc[s]; }
          for (j = 0; j < hd; j++) {
            var acc = 0;
            for (s = 0; s < T; s++) acc += sc[s] * qkv[s * 3 * D + 2 * D + hh * hd + j];
            ctx[t * D + hh * hd + j] = acc / sum;
          }
        }
      }
      var att = linear(ctx, T, D, w[p + 'out.w'], w[p + 'out.b'], D);
      for (i = 0; i < f.length; i++) f[i] += att[i];
      var b2 = layerNorm(f, T, D, w[p + 'ln2.w'], w[p + 'ln2.b'], m.ln_eps);
      var ff = linear(b2, T, D, w[p + 'ff1.w'], w[p + 'ff1.b'], 2 * D);
      for (i = 0; i < ff.length; i++) if (ff[i] < 0) ff[i] = 0;
      var ff2 = linear(ff, T, 2 * D, w[p + 'ff2.w'], w[p + 'ff2.b'], D);
      for (i = 0; i < f.length; i++) f[i] += ff2[i];
    }

    var z = new Float32Array(2 * D);
    for (d = 0; d < D; d++) {
      var sm = 0, mxv = -Infinity;
      for (t = 0; t < T; t++) { var u = f[t * D + d]; sm += u; if (u > mxv) mxv = u; }
      z[d] = sm / T; z[D + d] = mxv;
    }
    var n = this.species.length;
    var logits = linear(z, 1, 2 * D, w['cls.w'], w['cls.b'], n);
    var tn = linear(z, 1, 2 * D, w['temp.w'], w['temp.b'], 1)[0];
    var probs = new Float32Array(n);
    for (i = 0; i < n; i++) probs[i] = 1 / (1 + Math.exp(-logits[i]));
    return { probs: probs, logits: logits, tNorm: tn, tempK: m.tRange[0] + tn * (m.tRange[1] - m.tRange[0]) };
  };

  return { Model: Model, b64ToF32: b64ToF32, erf: erf, gelu: gelu };
});
