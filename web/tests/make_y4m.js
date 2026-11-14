/* Тест көмекшісі: виртуал камера кадрларынан Y4M (Chromium жалған веб-камерасы үшін). node make_y4m.js out.y4m [көз id] */
'use strict';
const fs = require('fs');
const D = require('../data/lines.js'), P = require('../js/physics.js'), CS = require('../js/camsim.js');
const out = process.argv[2], id = process.argv[3] || 'ne', W = 640, H = 480, frames = 12;
const src = CS.SOURCES.find(s => s.id === id), chunks = [Buffer.from(`YUV4MPEG2 W${W} H${H} F10:1 Ip A1:1 C420jpeg\n`)];
const r0 = CS.render(src, D.species, {}, P.mulberry32(5));
const geo = r0.geo;
for (let f = 0; f < frames; f++) {
  const r = CS.render(src, D.species, { theta: geo.theta * 180 / Math.PI, Lb: geo.Lb, fixCx: geo.cx, fixCy: geo.cy, sigv: geo.sigv, lamLo: geo.lamLo, lamHi: geo.lamHi, flip: geo.flip, fwhm: geo.fwhm, curv: geo.curv, cutR: geo.cutR, blobR: geo.blobR, blobOff: geo.blobOff }, P.mulberry32(100 + f));
  const d = r.img.data, Y = Buffer.alloc(W * H), U = Buffer.alloc(W * H / 4), V = Buffer.alloc(W * H / 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const o = (y * W + x) * 4, R = d[o], G = d[o + 1], B = d[o + 2];
    Y[y * W + x] = Math.max(0, Math.min(255, Math.round(0.299 * R + 0.587 * G + 0.114 * B)));
    if (y % 2 === 0 && x % 2 === 0) {
      const i = (y / 2) * (W / 2) + x / 2;
      U[i] = Math.max(0, Math.min(255, Math.round(128 - 0.168736 * R - 0.331264 * G + 0.5 * B)));
      V[i] = Math.max(0, Math.min(255, Math.round(128 + 0.5 * R - 0.418688 * G - 0.081312 * B)));
    }
  }
  chunks.push(Buffer.from('FRAME\n'), Y, U, V);
}
fs.writeFileSync(out, Buffer.concat(chunks));
console.log('y4m', out, fs.statSync(out).size);
