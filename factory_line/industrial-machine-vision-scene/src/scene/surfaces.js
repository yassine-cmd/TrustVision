import * as T from 'three';
import { CLASSES, G, clamp } from './mechanics.js';
import { REAL_DATASET_TEXTURES } from './dataset_samples.js';

function makeCanvas(w, h = w) {
  if (typeof document === 'undefined') {
    return [{ width: w, height: h, getContext: () => null }, null];
  }
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  return [canvas, canvas.getContext('2d', { willReadFrequently: true })];
}

export function texture(draw, w = 256, h = w) {
  const [canvas, ctx] = makeCanvas(w, h);
  draw(ctx, w, h);
  const tex = new T.CanvasTexture(canvas);
  tex.colorSpace = T.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

const random = (seed) => () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) | 0;
  return (seed >>> 0) / 4294967296;
};

function grain(ctx, w, h, base, seed, brush = 0, strength = 1, cool = 1) {
  const rand = random(seed), img = ctx.createImageData(w, h);
  const columns = Float32Array.from({ length: w }, () => (rand() - .5) * brush);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const cloud = 2.8 * Math.cos(x / 21 + y / 27) + 3.2 * Math.sin(x / 29 - y / 35);
      const v = base + (rand() - .5) * 20 * strength + cloud * strength + columns[x];
      img.data[i] = v - 3 * cool;
      img.data[i + 1] = v;
      img.data[i + 2] = v + 4 * cool;
      img.data[i + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

function sample(code, seed) {
  return texture((ctx, n) => {
    const r = random(seed);
    grain(ctx, n, n, 138, seed, 14, .95);
    for (let i = 0; i < 160; i++) {
      const x = r() * n, y = r() * n;
      ctx.fillStyle = i % 2 ? '#cbd4d830' : '#26313820';
      ctx.fillRect(x, y, .65, 3 + r() * 54);
    }
    if (code === 'Sc' || code === 'In') {
      for (let i = 0; i < (code === 'Sc' ? 20 : 58); i++) {
        const x = code === 'Sc' ? (i < 13 ? 91 : 158) + (r() - .5) * 58 : 32 + r() * 184;
        const y = r() * (code === 'Sc' ? 205 : 235), len = code === 'Sc' ? 26 + r() * 153 : 2 + r() * 26;
        ctx.lineWidth = .45 + r() * (code === 'Sc' ? 1 : 2.8);
        ctx.strokeStyle = `rgba(30,36,39,${.28 + r() * .47})`;
        ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + len * .12, y + len); ctx.stroke();
        if (code === 'Sc') {
          ctx.lineWidth *= .7; ctx.strokeStyle = '#eaf0eeb5';
          ctx.beginPath(); ctx.moveTo(x + 1, y); ctx.lineTo(x + 1 + len * .12, y + len); ctx.stroke();
        }
      }
    } else if (code === 'PS' || code === 'Pa' || code === 'RS') {
      const rough = code !== 'PS';
      for (let i = 0; i < (rough ? 210 : 142); i++) {
        const band = i % 3;
        const x = rough ? 40 + band * 69 + (r() - .5) * 74 : r() * n;
        const y = rough ? (band === 1 ? 154 : 86) + (r() - .5) * 170 : r() * n;
        const a = (rough ? 1.0 : .85) + r() * r() * (rough ? 22 : 6);
        const b = code === 'RS' ? a * (1 + r() * 2.4) : a * (.4 + .8 * r());
        const g = ctx.createRadialGradient(x - a * .15, y - b * .15, 0, x, y, Math.max(a, b));
        if (code === 'Pa') g.addColorStop(0, 'rgba(93,62,44,.80)');
        else g.addColorStop(0, 'rgba(21,30,35,.87)');
        g.addColorStop(.68, rough ? '#5140373d' : '#47545a58');
        g.addColorStop(1, rough ? '#242d3300' : '#d0d4d790');
        ctx.fillStyle = g;
        ctx.beginPath(); ctx.ellipse(x, y, a, b, (r() - .5) * .38, 0, Math.PI * 2); ctx.fill();
      }
    } else if (code === 'Cr') {
      for (let i = 0; i < 61; i++) {
        let x = 12 + r() * 225, y = 18 + r() * 206;
        const sign = r() > .5 ? 1 : -1;
        ctx.lineWidth = .6 + r() * 1.1;
        ctx.strokeStyle = '#1d2a31a8';
        ctx.beginPath(); ctx.moveTo(x, y);
        for (let j = 0; j < 4; j++) { x += (r() - .44) * 13; y += sign * (2 + r() * 10); ctx.lineTo(x, y); }
        ctx.stroke();
      }
    }
  });
}

export function surfaces() {
  const steel = {};
  Object.keys(CLASSES).forEach((code, i) => {
    const realSamples = REAL_DATASET_TEXTURES?.[code];
    if (realSamples && realSamples.length > 0 && typeof document !== 'undefined') {
      steel[code] = realSamples.map((sample) => {
        const [canvas, ctx] = makeCanvas(200, 200);
        const tex = new T.CanvasTexture(canvas);
        tex.colorSpace = T.SRGBColorSpace;
        tex.anisotropy = 4;
        const signal = new Uint8Array(200 * 200);

        if (typeof Image !== 'undefined') {
          const img = new Image();
          img.onload = () => {
            if (ctx) {
              ctx.drawImage(img, 0, 0, 200, 200);
              tex.needsUpdate = true;
              const raw = readSurface(canvas);
              for (let j = 0; j < raw.length; j++) signal[j] = raw[j];
            }
          };
          img.src = sample.data_uri;
        }
        return { map: tex, signal, canvas, sample };
      });
    } else {
      const maps = [0, 1].map((variant) => sample(code, 437 + i * 827 + variant * 1171));
      steel[code] = maps.map((map) => ({ map, signal: readSurface(map.image) }));
    }
  });
  const belt = texture((ctx, w, h) => {
    grain(ctx, w, h, 43, 174, 11, .54);
    ctx.fillStyle = '#99a5a235';
    for (const x of [4, 11, 244, 251]) ctx.fillRect(x, 0, 1, h);
    ctx.fillStyle = '#9da8a052';
    ctx.fillRect(12, 1, w - 24, 1);
    ctx.fillStyle = '#97aba339';
    for (const x of [25, 225]) ctx.fillRect(x, h - 12, 6, 8);
  });
  belt.wrapT = T.RepeatWrapping;
  const floor = texture((ctx, w, h) => {
    grain(ctx, w, h, 105, 294, 0, .78);
    ctx.strokeStyle = '#7a8083';
    ctx.lineWidth = 1.3;
    ctx.strokeRect(1, 1, w - 2, h - 2);
    ctx.strokeStyle = '#b4b9ba49';
    ctx.lineWidth = .75;
    ctx.strokeRect(2.5, 2.5, w - 5, h - 5);
  });
  floor.wrapS = floor.wrapT = T.RepeatWrapping;
  floor.repeat.set(11, 11);
  return { steel, belt, floor };
}

function readSurface(canvas) {
  if (!canvas || typeof canvas.getContext !== 'function') return new Uint8Array(200 * 200);
  const ctx = canvas.getContext('2d');
  if (!ctx) return new Uint8Array(200 * 200);
  const { width: n } = canvas;
  const raw = ctx.getImageData(0, 0, n, n).data;
  const out = new Uint8Array(n * n);
  for (let i = 0; i < out.length; i++) out[i] = Math.round(.299 * raw[4 * i] + .587 * raw[4 * i + 1] + .114 * raw[4 * i + 2]);
  return out;
}

const N = 224;

export class ScanRaster {
  constructor(steel) {
    this.steel = steel;
    [this.canvas, this.ctx] = makeCanvas(N);
    this.img = this.ctx ? this.ctx.createImageData(N, N) : { data: new Uint8ClampedArray(N * N * 4) };
    this.id = '';
    this.rows = 0;
  }

  sample(cap) {
    if (this.id !== cap.id) {
      this.id = cap.id;
      this.rows = 0;
      if (this.ctx) this.ctx.clearRect(0, 0, N, N);
    }
    const end = Math.min(N, Math.floor(cap.f * N));
    if (end <= this.rows || !this.ctx) return;
    const sets = this.steel[cap.plate.code] || [];
    const entry = sets[(cap.plate.id - 1) % sets.length] || sets[0];
    const dim = entry?.map?.image?.width || 200;
    const raw = entry?.signal;
    const d = this.img.data;
    if (raw && raw.length >= dim * dim) {
      for (let y = this.rows; y < end; y++) {
        const v = Math.min(dim - 1, Math.floor((y + .5) * dim / N));
        for (let x = 0; x < N; x++) {
          const u = Math.min(dim - 1, Math.floor((x + .5) * dim / N));
          const lum = raw[v * dim + u];
          const offAxis = ((x + .5) / N - .5) * G.plateW / G.fov;
          const gray = clamp((lum - 106) * 1.16 + 106 - 84 * offAxis * offAxis, 0, 255);
          const i = (y * N + x) * 4;
          d[i] = d[i + 1] = d[i + 2] = gray;
          d[i + 3] = 255;
        }
      }
      this.ctx.putImageData(this.img, 0, 0, 0, this.rows, N, end - this.rows);
    }
    this.rows = end;
  }

  payload(cap, stage) {
    this.sample(cap);
    const [snapCanvas, snapCtx] = makeCanvas(N, N);
    if (snapCtx && this.canvas) snapCtx.drawImage(this.canvas, 0, 0);
    cap.canvas = snapCanvas;
    if (cap.plate) cap.plate.canvas = snapCanvas;
    const sample = cap.plate.datasetSample;
    return {
      version: 1, captureId: cap.id, plateId: cap.plate.id, stage,
      code: cap.plate.code,
      encoding: 'gray-rgb8-png',
      imageDataUrl: sample?.data_uri || (this.canvas?.toDataURL ? this.canvas.toDataURL('image/png') : ''),
      datasetSample: sample,
      width: N, height: N, domain: 'neu-cls-test',
      encoderCount: cap.encoder, lineCount: cap.lines, beltSpeedMps: cap.speed,
    };
  }
}