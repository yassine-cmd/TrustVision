import { TEST_DATASET_PLATES } from './dataset_samples.js';

export const G = Object.freeze({
  beltY: .82, beltWidth: .84, head: 3.10, drumR: .088, beltSkin: .005,
  encoderR: .040, encoderPpr: 4096, pulseDivider: 3, pixels: 4096, fov: .76,
  plateW: .58, plateL: .68, plateH: .006, eyeZ: -.72, scanZ: 0,
  eyeY: .823, sensorX: .49, lensY: 1.514, lightZ: .32, lightAngle: 26 * Math.PI / 180,
  pickX: -1.46, pickZ: -2.52, supplyY: .48, carryY: 1.195,
  rejectZ: 1.66, paddleRest: -.393, stroke: .965, releaseStroke: .852,
  chuteLip: 1.42, outLip: 3.81, scrapY: .175, outputY: .225,
  scrapStop: 2.17, outputStop: 4.35,
});

export const BAYS = Object.freeze({
  bay1: { id: 'bay1', name: 'Baie 1 : Retouche & Ébavurage', z: 0.95, lane: 'rework', classes: ['Sc', 'Pa'] },
  bay2: { id: 'bay2', name: 'Baie 2 : Alerte Usure Cylindres', z: 1.80, lane: 'tooling', classes: ['PS', 'RS'] },
  bay3: { id: 'bay3', name: 'Baie 3 : Rebut Refusion Four', z: 2.65, lane: 'scrap', classes: ['Cr', 'In'] },
});

export function assignPlateBay(p) {
  if (p.route === 'pass') {
    p.targetBay = null;
    p.divertZ = null;
    p.lane = 'pass';
  } else if (['Sc', 'Pa'].includes(p.code)) {
    p.targetBay = 'bay1';
    p.divertZ = 0.95;
    p.lane = 'rework';
  } else if (['PS', 'RS'].includes(p.code)) {
    p.targetBay = 'bay2';
    p.divertZ = 1.80;
    p.lane = 'tooling';
  } else if (['Cr', 'In'].includes(p.code)) {
    p.targetBay = 'bay3';
    p.divertZ = 2.65;
    p.lane = 'scrap';
  } else {
    p.targetBay = 'bay3';
    p.divertZ = 2.65;
    p.lane = 'scrap';
  }
}

export const CLASSES = Object.freeze({
  OK: 'Surface saine (Conforme)', Cr: 'Faïençage', In: 'Inclusions', Pa: 'Incrustations (Patches)',
  PS: 'Surfaces piquées', RS: 'Calamine incrustée', Sc: 'Rayures de laminage',
});
export const NEU = ['Cr', 'In', 'Pa', 'PS', 'RS', 'Sc'];
export const LINE_PITCH = G.pulseDivider * 2 * Math.PI * G.encoderR / G.encoderPpr;
export const BELT_PERIMETER = 4 * G.head + 2 * Math.PI * G.drumR;
export const STEP = 1 / 120;
export const X_CHAIN = { fixed: -1.59, base: 1.851, radius: .09, travel: 1.59, lead: .11 };
export const Y_CHAIN = { fixed: -.045, base: 0, radius: .048, travel: .76, lead: .065 };
export const zCarrierPosition = (cupY) => G.carryY - cupY;
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const lerp = (a, b, t) => a + (b - a) * t;
export const ease = (t) => t * t * t * (10 + t * (-15 + 6 * t));
export const approach = (now, target, maxDelta) => now + clamp(target - now, -maxDelta, maxDelta);
const mod = (x, n) => ((x % n) + n) % n;

export function beltPose(s) {
  const r = G.drumR, h = G.head, y = G.beltY - r, span = 2 * h, arc = Math.PI * r;
  s = mod(s, BELT_PERIMETER);
  if (s < span) return { y: G.beltY, z: s - h, ny: 1, nz: 0 };
  s -= span;
  if (s < arc) {
    const a = s / r;
    return { y: y + r * Math.cos(a), z: h + r * Math.sin(a), ny: Math.cos(a), nz: Math.sin(a) };
  }
  s -= arc;
  if (s < span) return { y: y - r, z: h - s, ny: -1, nz: 0 };
  const a = (s - span) / r;
  return { y: y - r * Math.cos(a), z: -h - r * Math.sin(a), ny: -Math.cos(a), nz: -Math.sin(a) };
}

export function shaftAngles(distance) {
  return { drum: distance / G.drumR, returnIdler: -distance / .027, encoder: -distance / G.encoderR };
}

export function carrierLength(chain) {
  return chain.travel + 2 * chain.lead + Math.PI * chain.radius;
}

export function carrierPose(chain, moving, s) {
  const { fixed, base, radius } = chain;
  const bend = (carrierLength(chain) + fixed + moving - Math.PI * radius) / 2;
  const lower = bend - fixed;
  if (s <= lower) return { x: fixed + s, y: base, angle: 0 };
  const a = (s - lower) / radius;
  if (a < Math.PI) return { x: bend + radius * Math.sin(a), y: base + radius * (1 - Math.cos(a)), angle: a };
  return { x: bend - radius * (a - Math.PI), y: base + 2 * radius, angle: Math.PI };
}

export const rejectRollerX = [.475, .565, .655, .745, .835, .925, 1.015, 1.105, 1.195, 1.285, 1.375];
export const outRollerZ = [3.168, 3.258, 3.348, 3.438, 3.528, 3.618, 3.708, 3.798];

export function support(kind, q) {
  const [flat, tangent, grade] = kind === 'chute' ? [.74, 1.12, .226] : [G.head, 3.41, .167];
  const t = clamp((q - flat) / (tangent - flat), 0, 1);
  const slope = -grade * t;
  const h = G.beltY - grade * (.5 * (tangent - flat) * t * t + Math.max(0, q - tangent));
  return { h, slope, angle: Math.atan(slope) };
}

const half = G.plateH / 2;
const INFEED_CODES = ['Pa', 'OK', 'In', 'PS', 'OK', 'Cr', 'RS', 'OK', 'Sc', 'PS', 'OK', 'Cr', 'OK', 'RS', 'Pa', 'OK', 'In', 'Sc'];
const GRAVITY = 9.81;
const BRAKE = 2.45;

function plate(id, code, x, y, z, state) {
  const p = { id, code, x, y, z, angle: 0, vx: 0, vy: 0, vz: 0, state, route: null,
    armed: state === 'belt' && z + G.plateL / 2 >= G.eyeZ, capture: null, result: null };
  p.prev = { x, y, z, angle: 0 };
  return p;
}

export class Machine {
  constructor(onCapture = () => {}) {
    this.onCapture = onCapture;
    this.runId = 0;
    this.targetSpeed = .8;
    this.reset();
  }

  reset() {
    this.runId++;
    this.time = this.distance = this.speed = 0;
    this.job = this.scan = this.lastResult = this.lastSample = null;
    this.head = { x: G.pickX, cupY: G.carryY, phase: 'home', t: 0, plate: null };
    this.actuators = {
      bay1: { id: 'bay1', z: 0.95, phase: 'home', t: 0, stroke: 0, prevStroke: 0, plate: null },
      bay2: { id: 'bay2', z: 1.80, phase: 'home', t: 0, stroke: 0, prevStroke: 0, plate: null },
      bay3: { id: 'bay3', z: 2.65, phase: 'home', t: 0, stroke: 0, prevStroke: 0, plate: null },
    };
    this.actuator = this.actuators.bay2;
    this.rollers = [
      ...rejectRollerX.map((q) => ({ q, z: 0.95, kind: 'chute', angle: 0, prev: 0, velocity: 0 })),
      ...rejectRollerX.map((q) => ({ q, z: 1.80, kind: 'chute', angle: 0, prev: 0, velocity: 0 })),
      ...rejectRollerX.map((q) => ({ q, z: 2.65, kind: 'chute', angle: 0, prev: 0, velocity: 0 })),
      ...outRollerZ.map((q) => ({ q, z: 0, kind: 'runout', angle: 0, prev: 0, velocity: 0 })),
    ];
    this.prev = { x: G.pickX, cupY: G.carryY, stroke: 0, distance: 0 };
    this.output = { pass: [], rework: [], tooling: [], scrap: [], reject: [] };
    this.log = [];
    this.eye = null;
    this.note = '';
    const y = G.beltY + half;
    const p1Sample = TEST_DATASET_PLATES?.[0];
    const p2Sample = TEST_DATASET_PLATES?.[1];
    const p3Sample = TEST_DATASET_PLATES?.[2];

    const reference = plate(1, p1Sample ? p1Sample.gt_code : 'OK', 0, y, 2.43, 'belt');
    reference.route = 'pass';
    reference.armed = false;
    reference.datasetSample = p1Sample;

    const staging = [
      plate(2, p2Sample ? p2Sample.gt_code : 'Sc', 0, y, -.355, 'belt'),
      plate(3, p3Sample ? p3Sample.gt_code : 'OK', 0, y, -1.58, 'belt'),
    ];
    staging[0].datasetSample = p2Sample;
    staging[1].datasetSample = p3Sample;

    const infeedSamples = (TEST_DATASET_PLATES && TEST_DATASET_PLATES.length > 3)
      ? TEST_DATASET_PLATES.slice(3)
      : INFEED_CODES.map(code => ({ gt_code: code }));

    this.inventory = infeedSamples.map((s, i) => {
      const p = plate(i + 4, s.gt_code, G.pickX, G.supplyY + half + G.plateH * (infeedSamples.length - 1 - i), G.pickZ, 'stock');
      p.datasetSample = s;
      return p;
    }).reverse();

    this.plates = [reference, ...staging, ...this.inventory.slice().reverse()];
    this.plates.forEach(assignPlateBay);
  }

  queue(code) {
    if (!Object.hasOwn(CLASSES, code)) return false;
    const next = this.inventory.slice().reverse().find((p) => p !== this.head.plate && !p.manual);
    if (!next) return false;
    next.code = code;
    next.manual = true;
    return true;
  }

  get count() { return Math.floor(this.distance / (LINE_PITCH / G.pulseDivider)); }
  get review() { return this.plates.find((p) => p.state === 'belt' && p.route === 'review' && Math.abs(p.z - (p.divertZ ?? G.rejectZ)) < .001 && this.speed === 0); }
  get moving() { return this.plates.some((p) => p.state !== 'settled'); }

  resolve(id, result) {
    const p = this.plates.find((item) => item.capture?.id === id && !item.result);
    if (!p || !['pass', 'reject', 'review'].includes(result.disposition)) return false;
    p.result = result;
    p.route = result.disposition;
    this.lastResult = result;
    assignPlateBay(p);
    this.log.unshift({ plate: p.id, captureId: id, result, time: this.time });
    return true;
  }

  routeToReject() {
    const p = this.review;
    if (!p) return false;
    p.route = 'reject';
    assignPlateBay(p);
    this.note = `Tôle #${p.id} aiguillée vers le rebut par l'opérateur`;
    return true;
  }

  routeToPass() {
    const p = this.review;
    if (!p) return false;
    p.route = 'pass';
    assignPlateBay(p);
    this.note = `Tôle #${p.id} validée conforme par l'opérateur`;
    return true;
  }

  step(dt = STEP) {
    this.time += dt;
    this.prev = { x: this.head.x, cupY: this.head.cupY, stroke: this.actuator.stroke, distance: this.distance };
    this.plates.forEach((p) => {
      if (p.state !== 'stock' && p.state !== 'settled') Object.assign(p.prev, { x: p.x, y: p.y, z: p.z, angle: p.angle });
    });
    this.tickLoader(dt);
    let minRoom = Infinity;
    for (const p of this.plates) {
      if (p.state === 'belt' && p.route !== 'pass' && p.divertZ && p.z <= p.divertZ) {
        const dist = p.divertZ - p.z;
        if (dist < minRoom) minRoom = dist;
      }
    }
    const room = minRoom;
    const anyActuatorActive = Object.values(this.actuators).some(a => a.phase !== 'home');
    const interlocked = ['wait', 'place', 'vent'].includes(this.head.phase) ||
      (this.head.phase === 'raise' && this.head.cupY < G.beltY + .05) || anyActuatorActive;
    const demand = this.moving && !interlocked ? Math.min(this.targetSpeed, Math.sqrt(2 * BRAKE * room)) : 0;
    const vNext = approach(this.speed, demand, dt * (demand < this.speed ? BRAKE : 1.75));
    const d = Math.min(room, (this.speed + vNext) * .5 * dt);
    this.speed = d >= room ? 0 : vNext;
    this.distance += d;
    this.note = interlocked ? (this.actuator.phase !== 'home' ? 'Verrouillage éjecteur' : 'Verrouillage alimentation') : '';
    this.eye = null;

    for (const p of this.plates) {
      if (p.state === 'belt') {
        const before = p.z;
        p.z += d;
        p.vz = this.speed;
        const nose = p.z + G.plateL / 2;
        if (!p.armed && before + G.plateL / 2 <= G.eyeZ && nose >= G.eyeZ) p.armed = true;
        if (nose >= G.eyeZ && p.z - G.plateL / 2 < G.eyeZ) this.eye = p;
        if (p.armed && !p.route && nose >= G.scanZ && !p.capture?.complete) this.capture(p);
        if (p.z >= G.head) {
          p.state = 'runout';
          this.onSupport(p, 'runout', p.z);
        }
      } else if (p.state === 'chute' || p.state === 'runout') {
        this.tickGuide(p, dt);
      } else if (p.state === 'fall' || p.state === 'landing') {
        this.tickCatch(p, dt);
      }
    }

    for (const bayKey of ['bay1', 'bay2', 'bay3']) {
      const bay = this.actuators[bayKey];
      const p = this.plates.find(item => item.state === 'belt' && item.targetBay === bayKey && Math.abs(item.z - bay.z) < 0.001);
      if (p && this.speed === 0 && bay.phase === 'home') {
        if (p.route === 'reject') {
          p.state = 'indexed';
          bay.phase = 'extend';
          bay.t = 0;
          bay.stroke = 0;
          bay.plate = p;
        } else {
          this.note = p.route === 'review' ? 'Arrêt contrôle opérateur' : 'En attente du modèle';
        }
      }
    }
    this.tickPusher(dt);
    this.tickRollers(dt);
  }

  tickRollers(dt) {
    for (const r of this.rollers) {
      const reject = r.kind === 'chute';
      const p = this.plates.find((p) => reject
        ? (p.state === 'pushing' || p.state === 'chute') && Math.abs(r.z - p.z) < 0.35 && Math.abs(r.q - p.x) < G.plateW * .47
        : p.state === 'runout' && Math.abs(r.q - p.z) < G.plateL * .47);
      r.prev = r.angle;
      r.velocity = p ? (reject ? -p.vx : p.vz) / .026 : r.velocity * Math.exp(-4.5 * dt);
      r.angle += r.velocity * dt;
    }
  }

  capture(p) {
    const f = clamp((p.z + G.plateL / 2 - G.scanZ) / G.plateL, 0, 1);
    if (!p.capture) {
      const travel = this.distance - f * G.plateL;
      const pulseSpacing = LINE_PITCH / G.pulseDivider;
      const startPulse = Math.max(0, Math.ceil(travel / pulseSpacing));
      p.capture = { id: `${this.runId}:${p.id}`, plate: p, f: 0, lines: 0, complete: false,
        triggerPulse: startPulse + (G.pulseDivider - startPulse % G.pulseDivider) % G.pulseDivider };
    }
    const cap = p.capture;
    cap.f = f;
    const exposedDistance = this.distance - Math.max(0, p.z - G.plateL / 2 - G.scanZ);
    const exposedPulses = exposedDistance / (LINE_PITCH / G.pulseDivider);
    cap.lines = Math.max(0, 1 + Math.floor((exposedPulses - cap.triggerPulse) / G.pulseDivider));
    this.scan = cap;
    if (f >= 1) {
      cap.complete = true;
      cap.speed = this.speed;
      cap.encoder = Math.floor(exposedPulses);
      this.lastSample = cap;
      this.onCapture(cap);
    }
  }

  tickLoader(dt) {
    const h = this.head;
    const clear = !this.plates.some((p) => p.state === 'belt' && Math.abs(p.z - G.pickZ) < G.plateL + .18);
    const go = (phase) => { h.phase = phase; h.t = 0; };
    if (h.phase === 'home') {
      if (this.inventory.length && clear) { h.plate = this.inventory.at(-1); go('lower'); }
      return;
    }
    if (h.phase === 'wait') {
      if (this.speed < 1e-6 && clear && this.actuator.phase === 'home') go('place');
      return;
    }
    const phases = { lower: .48, seal: .12, lift: .48, traverse: 1.15, place: .49, vent: .10, raise: .39, return: 1.05 };
    h.t = Math.min(1, h.t + dt / phases[h.phase]);
    const f = ease(h.t), done = h.t === 1;
    const contactY = G.supplyY + this.inventory.length * G.plateH;
    if (h.phase === 'lower') {
      h.cupY = lerp(G.carryY, contactY, f);
      if (done) go('seal');
    } else if (h.phase === 'seal') {
      if (done) { h.plate.state = 'gripped'; h.sealY = contactY; this.inventory.pop(); go('lift'); }
    } else if (h.phase === 'lift') {
      h.cupY = lerp(h.sealY, G.carryY, f);
      if (done) go('traverse');
    } else if (h.phase === 'traverse') {
      h.x = lerp(G.pickX, 0, f);
      if (done) go('wait');
    } else if (h.phase === 'place') {
      h.cupY = lerp(G.carryY, G.beltY + G.plateH, f);
      if (done) go('vent');
    } else if (h.phase === 'vent') {
      if (done) { h.plate.state = 'belt'; h.plate = null; go('raise'); }
    } else if (h.phase === 'raise') {
      h.cupY = lerp(G.beltY + G.plateH, G.carryY, f);
      if (done) go('return');
    } else if (h.phase === 'return') {
      h.x = lerp(0, G.pickX, f);
      if (done) go('home');
    }
    if (h.plate?.state === 'gripped') {
      h.plate.x = h.x;
      h.plate.y = h.cupY - half;
    }
  }

  tickPusher(dt) {
    for (const bayKey of ['bay1', 'bay2', 'bay3']) {
      const a = this.actuators[bayKey];
      a.prevStroke = a.stroke;
      if (a.phase === 'home') continue;
      const duration = a.phase === 'extend' ? .62 : a.phase === 'dwell' ? .08 : .55;
      a.t = Math.min(1, a.t + dt / duration);
      if (a.phase === 'extend') {
        const before = a.stroke;
        a.stroke = G.stroke * ease(a.t);
        const p = a.plate;
        const face = G.paddleRest + a.stroke;
        if (p && face >= p.x - G.plateW / 2) {
          p.state = 'pushing';
          p.x = face + G.plateW / 2;
          p.z = a.z;
          p.vx = (a.stroke - before) / dt;
          this.onSupport(p, 'chute', p.x);
          if (a.stroke >= G.releaseStroke) { p.state = 'chute'; a.plate = null; }
        }
        if (a.t === 1) { a.phase = 'dwell'; a.t = 0; }
      } else if (a.phase === 'dwell') {
        if (a.t === 1) { a.phase = 'retract'; a.t = 0; }
      } else {
        a.stroke = G.stroke * (1 - ease(a.t));
        if (a.t === 1) { a.phase = 'home'; a.t = 0; }
      }
    }
    this.actuator = this.actuators.bay2;
  }

  onSupport(p, kind, q) {
    const s = support(kind, q);
    p.y = s.h + half / Math.cos(s.angle);
    p.angle = s.angle;
  }

  tickGuide(p, dt) {
    const reject = p.state === 'chute';
    const q = reject ? 'x' : 'z', v = reject ? 'vx' : 'vz';
    const s = support(p.state, p[q]);
    const acc = -GRAVITY * s.slope / (1 + s.slope * s.slope) - .065;
    p[v] = Math.max(.06, p[v] + acc * dt);
    p[q] += p[v] * dt;
    this.onSupport(p, p.state, p[q]);
    if (p[q] >= (reject ? G.chuteLip : G.outLip)) {
      p.vy = p[v] * support(p.state, p[q]).slope;
      p.lane = reject ? 'reject' : 'pass';
      p.state = 'fall';
    }
  }

  tickCatch(p, dt) {
    const reject = ['bay1', 'bay2', 'bay3', 'rework', 'tooling', 'scrap', 'reject'].includes(p.lane);
    const extent = (reject ? G.plateW : G.plateL) / 2;
    const q = reject ? 'x' : 'z', v = reject ? 'vx' : 'vz';
    const stop = reject ? G.scrapStop : G.outputStop;
    const shelf = (reject ? G.scrapY : G.outputY) + ((this.output[p.lane] || []).length) * G.plateH;
    p[q] += p[v] * dt;
    if (p[q] >= stop) { p[q] = stop; p[v] = 0; }
    if (p.state === 'fall') {
      const lip = reject ? G.chuteLip : G.outLip;
      const tail = p[q] - extent * Math.cos(p.angle);
      if (tail < lip) {
        // Natural fulcrum tipping over the support lip:
        // The bottom surface rests on the last roller lip until the tail completely slides off.
        const s = support(reject ? 'chute' : 'runout', lip);
        const overhang = p[q] - lip;
        const maxOverhang = extent * Math.cos(s.angle);
        const prog = clamp(overhang / maxOverhang, 0, 1);
        const f = prog * prog * (3 - 2 * prog);
        const dropAngle = reject ? -.30 : -.26;
        p.angle = s.angle + (dropAngle - s.angle) * f;
        p.y = s.h + half / Math.cos(p.angle) + overhang * Math.tan(p.angle);
        p.vy = p[v] * Math.tan(p.angle);
      } else {
        p.y += dt * (p.vy - .5 * GRAVITY * dt);
        p.vy -= GRAVITY * dt;
        p.angle = Math.max(-.30, p.angle - .18 * dt);
        const base = shelf + Math.abs(Math.sin(p.angle)) * extent + half * Math.cos(p.angle);
        if (p.y <= base) { p.y = base; p.vy = 0; p.state = 'landing'; }
      }
    } else {
      p.angle = approach(p.angle, 0, dt * 1.65);
      p.vx = approach(p.vx, 0, dt * 4.5);
      p.vz = approach(p.vz, 0, dt * 4.5);
      p.y = shelf + Math.abs(Math.sin(p.angle)) * extent + half * Math.cos(p.angle);
      if (!p.angle && !p[v]) {
        p.state = 'settled';
        if (this.output[p.lane]) this.output[p.lane].push(p);
        if (p.lane !== 'pass' && p.lane !== 'reject') this.output.reject.push(p);
      }
    }
  }
}