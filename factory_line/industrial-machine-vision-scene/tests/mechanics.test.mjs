import assert from 'node:assert/strict';
import test from 'node:test';
import { Machine, STEP, G, BELT_PERIMETER, beltPose, shaftAngles, carrierLength,
  carrierPose, X_CHAIN, Y_CHAIN, zCarrierPosition, support, LINE_PITCH } from '../src/scene/mechanics.js';
import { ModelBridge, demoResult, validateResult } from '../src/bridge.js';

function close(a, b, e = 1e-7) { assert.ok(Math.abs(a - b) < e, `${a} differs from ${b}`); }

test('continuous web wraps tangent to both drums; over-belt encoder counter-rotates', () => {
  const span = 2 * G.head, arc = Math.PI * G.drumR;
  for (const s of [0, span, span + arc, 2 * span + arc, BELT_PERIMETER]) {
    const a = beltPose(s - 1e-7), b = beltPose(s + 1e-7);
    close(Math.hypot(a.y - b.y, a.z - b.z), 2e-7, 1e-7);
    close(Math.hypot(a.ny - b.ny, a.nz - b.nz), 0, .00001);
  }
  assert.ok(beltPose(.9).z > beltPose(.8).z);
  assert.ok(beltPose(span + arc + .9).z < beltPose(span + arc + .8).z);
  const a = shaftAngles(.82);
  close(a.drum * G.drumR, .82);
  close(a.returnIdler * .027, -.82);
  close(a.encoder * G.encoderR, -.82);
});

test('each carrier keeps both anchors, its radius, and fixed length at all positions', () => {
  for (const [chain, home, end] of [[X_CHAIN, G.pickX, 0], [Y_CHAIN, 0, zCarrierPosition(G.supplyY + G.plateH)]]) {
    const n = 600, length = carrierLength(chain);
    for (let j = 0; j <= 16; j++) {
      const move = home + (end - home) * j / 16;
      const fixed = carrierPose(chain, move, 0), last = carrierPose(chain, move, length);
      close(fixed.x, chain.fixed);
      close(last.x, move);
      close(last.y - fixed.y, 2 * chain.radius);
      let len = 0, prev = fixed;
      for (let i = 1; i <= n; i++) {
        const here = carrierPose(chain, move, length * i / n);
        len += Math.hypot(here.x - prev.x, here.y - prev.y);
        prev = here;
      }
      close(len, length, .0002);
    }
  }
});

test('optics datum keeps the bars aimed at the sheet and encoder pitch independent of speed', () => {
  const barY = G.beltY + G.plateH + G.lightZ * Math.tan(G.lightAngle);
  close(barY - G.lightZ * Math.tan(G.lightAngle), G.beltY + G.plateH);
  assert.ok(G.eyeY > G.beltY && G.eyeY < G.beltY + G.plateH);
  const counts = [];
  for (const speed of [.25, .8, 1.2]) {
    const line = new Machine();
    line.targetSpeed = speed;
    for (let i = 0; i < 1500 && !line.lastSample; i++) line.step();
    assert.equal(line.lastSample.plate.id, 2);
    close(line.lastSample.lines, G.plateL / LINE_PITCH, 1);
    counts.push(line.lastSample.lines);
  }
  close(Math.max(...counts), Math.min(...counts), 2);
});

test('a pending decision stops at the index; a review requires explicit routing', () => {
  const m = new Machine();
  for (let i = 0; i < 1200; i++) m.step();
  const held = m.plates.find((p) => p.id === 2);
  assert.equal(m.speed, 0);
  close(held.z, held.divertZ ?? G.rejectZ);
  assert.equal(m.actuator.phase, 'home');
  assert.equal(m.resolve('wrong:2', { disposition: 'pass' }), false);
  assert.equal(m.resolve(held.capture.id, { disposition: 'review' }), true);
  assert.equal(m.review, held);
  assert.equal(m.routeToReject(), true);
  for (let i = 0; i < 550; i++) m.step();
  assert.equal(held.state, 'settled');
  assert.equal(held.lane, 'reject');
});

test('a reviewed plate routed to pass resets lane, bypasses ejector, and rolls down outfeed to pass stack', () => {
  const m = new Machine();
  for (let i = 0; i < 1200; i++) m.step();
  const held = m.plates.find((p) => p.id === 2);
  assert.equal(m.speed, 0);
  assert.equal(m.resolve(held.capture.id, { disposition: 'review' }), true);
  assert.equal(m.review, held);
  assert.equal(m.routeToPass(), true);
  assert.equal(held.route, 'pass');
  assert.equal(held.lane, 'pass');
  assert.equal(held.targetBay, null);
  assert.equal(held.divertZ, null);
  for (let i = 0; i < 800; i++) m.step();
  assert.equal(held.state, 'settled');
  assert.equal(held.lane, 'pass');
  assert.equal(held.x, 0);
});

test('one fixed axis pose drives pickup and one cylinder pose drives contact throughout the batch', () => {
  for (const speed of [.25, .8, 1.2]) {
    const m = new Machine((cap) => m.resolve(cap.id, demoResult(cap.plate.code, cap.id)));
    m.targetSpeed = speed;
    let gripped = 0, pushed = 0, fell = 0;
    for (let ii = 0; ii < 29000 && m.moving; ii++) {
      m.step(STEP);
      for (const p of m.plates) {
        if (p.state === 'gripped') {
          close(p.x, m.head.x); close(p.y + G.plateH / 2, m.head.cupY); gripped++;
        }
        if (p.state === 'pushing') {
          const act = m.actuators ? m.actuators[p.targetBay] : m.actuator;
          close(p.x - G.plateW / 2, G.paddleRest + act.stroke);
          close(p.z, p.divertZ ?? G.rejectZ); close(m.speed, 0); pushed++;
        }
        if (p.state === 'chute') {
          assert.ok(p.x - G.plateW / 2 >= G.paddleRest + m.actuator.stroke - .00001);
          close(p.angle, support('chute', p.x).angle);
        }
        if (p.state === 'fall') fell++;
        if (!['stock', 'settled'].includes(p.state)) {
          assert.ok(Math.hypot(p.x - p.prev.x, p.y - p.prev.y, p.z - p.prev.z) < .031, 'pose jumped between fixed steps');
        }
      }
    }
    assert.ok(gripped && pushed && fell);
    assert.equal(m.log.length, m.plates.length - 1);
    assert.equal(m.output.pass.length + m.output.reject.length, m.plates.length);
    assert.equal(m.inventory.length, 0);
    assert.equal(m.moving, false);
    for (const [name, y] of [['pass', G.outputY], ['reject', G.scrapY]]) {
      m.output[name].forEach((p, i) => close(p.y, y + (i + .5) * G.plateH));
    }
  }
});

test('model hook never sees the synthetic class and never inherits a demo prediction', async () => {
  const bridge = new ModelBridge();
  const payload = { captureId: '1:2', imageDataUrl: 'data:image/png;base64,AA' };
  const defaultResult = await bridge.inspect(payload, 'Sc');
  assert.equal(defaultResult.source, 'scripted');
  assert.equal(defaultResult.confidence, null);
  bridge.onInspect = async (request) => {
    assert.equal(Object.hasOwn(request, 'groundTruth'), false);
    return { captureId: request.captureId, predictedClass: 'In', confidence: .4, disposition: 'pass' };
  };
  const real = await bridge.inspect(payload, 'Sc');
  assert.equal(real.source, 'hook');
  assert.equal(real.predictedClass, 'In');
  assert.equal(real.disposition, 'pass');
  bridge.onInspect = () => ({ captureId: 'unrelated', disposition: 'pass' });
  const failed = await bridge.inspect(payload, 'Sc');
  assert.equal(failed.source, 'error');
  assert.equal(failed.disposition, 'review');
  assert.throws(() => validateResult({ captureId: '1:2', disposition: 'pass', predictedClass: 'not-a-class' }, '1:2'));
});