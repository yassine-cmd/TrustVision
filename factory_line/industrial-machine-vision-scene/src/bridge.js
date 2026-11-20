import { CLASSES, NEU } from './scene/mechanics.js';

export function demoResult(code, captureId) {
  return { captureId, source: 'scripted', predictedClass: null, confidence: null,
    disposition: code === 'OK' ? 'pass' : 'reject', description: `Demo sample: ${CLASSES[code]}` };
}

export function validateResult(data, captureId) {
  if (!data || data.captureId !== captureId) throw new Error('Capture ID did not match');
  const { predictedClass = null, disposition, confidence = null } = data;
  if (predictedClass !== null && ![...NEU, 'OK'].includes(predictedClass)) throw new Error('Unknown class code');
  if (!['pass', 'reject', 'review'].includes(disposition)) throw new Error('Provide pass, reject, or review');
  if (confidence !== null && (!Number.isFinite(confidence) || confidence < 0 || confidence > 1)) throw new Error('Confidence must be in [0, 1]');
  let heatmap = null;
  const h = data.heatmap;
  if (h && Number.isInteger(h.width) && Number.isInteger(h.height) && h.width > 0 && h.height > 0 && h.width <= 64 && h.height <= 64 &&
      Array.isArray(h.values) && h.values.length === h.width * h.height && h.values.every((v) => Number.isFinite(v) && v >= 0 && v <= 1)) {
    heatmap = { width: h.width, height: h.height, values: h.values, method: String(h.method || 'Attribution').slice(0, 28) };
  }
  return { captureId, predictedClass, disposition, confidence, heatmap,
    predictedClassName: data.predictedClassName || (predictedClass ? CLASSES[predictedClass] : null),
    groundTruthClass: data.groundTruthClass || null,
    isCorrect: typeof data.isCorrect === 'boolean' ? data.isCorrect : null,
    description: predictedClass ? CLASSES[predictedClass] : 'Model decision' };
}

export class ModelBridge {
  constructor() {
    this.endpoint = '';
    this.stage = 'S2';
    this.onInspect = null;
    this.pending = new Set();
  }

  get isExternal() { return Boolean(this.onInspect || this.endpoint); }
  get mode() { return this.onInspect ? 'Automate & Modèle S2 : En ligne' : this.endpoint ? 'Modèle API : Connecté' : 'Simulation locale'; }

  setEndpoint(text) {
    const url = new URL(text.trim());
    if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('Enter an HTTP or HTTPS endpoint');
    this.endpoint = url.href;
    this.onInspect = null;
  }

  demo() { this.endpoint = ''; this.onInspect = null; }

  cancel() {
    for (const ctrl of this.pending) ctrl.abort(new Error('Simulation reset'));
    this.pending.clear();
  }

  async inspect(payload, code) {
    if (typeof this.onInspect !== 'function' && !this.endpoint) return demoResult(code, payload.captureId);
    const hook = this.onInspect, endpoint = this.endpoint, ctrl = new AbortController(), started = performance.now();
    this.pending.add(ctrl);
    const timer = setTimeout(() => ctrl.abort(new Error('Model response timed out')), 12000);
    try {
      const aborted = new Promise((_, reject) => ctrl.signal.addEventListener('abort', () => reject(ctrl.signal.reason), { once: true }));
      const operation = Promise.resolve().then(async () => {
        if (hook) return hook(payload, { signal: ctrl.signal });
        const response = await fetch(endpoint, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
          signal: ctrl.signal, cache: 'no-store', credentials: 'omit', redirect: 'error',
        });
        if (!response.ok) throw new Error(`Model endpoint returned HTTP ${response.status}`);
        return response.json();
      });
      const result = validateResult(await Promise.race([aborted, operation]), payload.captureId);
      return { ...result, source: hook ? 'hook' : 'http', latency: performance.now() - started };
    } catch (error) {
      return { captureId: payload.captureId, source: 'error', predictedClass: null, confidence: null,
        disposition: 'review', description: String(error?.message || 'Model unavailable').slice(0, 160) };
    } finally {
      clearTimeout(timer);
      this.pending.delete(ctrl);
    }
  }
}