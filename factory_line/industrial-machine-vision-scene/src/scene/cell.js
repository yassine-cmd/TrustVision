import * as T from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { G, BELT_PERIMETER, beltPose, shaftAngles, X_CHAIN, Y_CHAIN, zCarrierPosition,
  rejectRollerX, outRollerZ, support, lerp } from './mechanics.js';
import { bar, batch, box, dragChain, extrusion, fitting, ink, matte, mesh, pin, tag, wire } from './fabrication.js';
import { texture } from './surfaces.js';

const PI = Math.PI;
const drumY = G.beltY - G.drumR;
const stripWidth = .005;
const CYL_FRONT = -.513;
const CYL_BACK = -1.762;
const PUSH_Y = .853;
const railX = .466;
const gantryX = .566;
const RAIL_Y = .738;

function materials() {
  return {
    alu: matte(0xa0b0b9, .38, .64), rail: matte(0x71818a, .38, .6), nickel: matte(0xb1c2c8, .25, .7),
    graphite: matte(0x28333a, .52, .32), black: matte(0x20282c, .58, .16), rubber: matte(0x22282a, .85, .03),
    base: matte(0x3e4d55, .51, .44), shade: matte(0x192326, .65, .19), yellow: matte(0xe7a23d, .43, .28),
    amber: matte(0xc68833, .50, .24), mint: ink(0x9ef0c4), white: ink(0xf7f4de),
    blue: matte(0x437080, .43, .34), brass: matte(0xb68953, .34, .62), 
    hose: matte(0x356373, .47, .12), line: matte(0x252c2c, .65, .13), 
    shell: matte(0x53606a, .61, .32), cage: matte(0x899194, .56, .35), cabinet: matte(0x71808a, .53, .28),
    sign: matte(0xd3a150, .7, .04), glowGreen: ink(0x83e1af), glowAmber: ink(0xfcc561), glowRed: ink(0xfe745d),
  };
}

function beltGeometry() {
  const a = G.head * 2, b = PI * G.drumR, w = G.beltWidth / 2;
  const cuts = [0, a];
  for (let i = 1; i <= 28; i++) cuts.push(a + b * i / 28);
  cuts.push(2 * a + b);
  for (let i = 1; i <= 28; i++) cuts.push(2 * a + b + b * i / 28);
  const pos = [], uv = [], normals = [], idx = [];
  for (const surface of ['outside', 'inside', 'left', 'right']) {
    const start = pos.length / 3;
    for (const s of cuts) {
      const p = beltPose(s), t = s / BELT_PERIMETER * 17;
      for (const side of [-1, 1]) {
        const edge = surface === 'left' ? -1 : surface === 'right' ? 1 : 0;
        const depth = surface === 'inside' || (edge !== 0 && side === 1) ? G.beltSkin : 0;
        const normal = surface === 'inside' ? -1 : 1;
        pos.push((edge || side) * w, p.y - p.ny * depth, p.z - p.nz * depth);
        normals.push(edge, edge ? 0 : p.ny * normal, edge ? 0 : p.nz * normal);
        uv.push(edge === 0 ? (side + 1) / 2 : (edge < 0 ? .025 : .975) + .003 * side, t);
      }
    }
    const reverse = surface === 'inside' || surface === 'left';
    for (let i = 0; i < cuts.length - 1; i++) {
      const u = start + i * 2;
      if (reverse) idx.push(u, u + 1, u + 2, u + 1, u + 3, u + 2);
      else idx.push(u, u + 2, u + 1, u + 1, u + 2, u + 3);
    }
  }
  const geo = new T.BufferGeometry();
  geo.setAttribute('position', new T.Float32BufferAttribute(pos, 3));
  geo.setAttribute('normal', new T.Float32BufferAttribute(normals, 3));
  geo.setAttribute('uv', new T.Float32BufferAttribute(uv, 2));
  geo.setIndex(idx);
  return geo;
}

function opticalFill(mat) {
  mat.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 partWorld;')
      .replace('#include <worldpos_vertex>', '#include <worldpos_vertex>\npartWorld = (modelMatrix * vec4(transformed, 1.0)).xyz;');
    shader.fragmentShader = shader.fragmentShader.replace('#include <common>', '#include <common>\nvarying vec3 partWorld;')
      .replace('#include <opaque_fragment>', `
        float along = partWorld.z * 22.0;
        float fillBand = exp(-along * along);
        fillBand *= (1.0 - smoothstep(0.33, 0.386, abs(partWorld.x)));
        fillBand *= smoothstep(0.79, 0.82, partWorld.y) * (1.0 - smoothstep(0.84, 0.885, partWorld.y));
        fillBand *= max(0.0, dot(normal, mat3(viewMatrix) * vec3(0.0, 1.0, 0.0)));
        outgoingLight += vec3(0.28, 0.39, 0.35) * fillBand * (0.25 + diffuseColor.rgb);
        #include <opaque_fragment>`);
  };
  return mat;
}

function patternedFloor(g, M, kit) {
  const floorMat = new T.MeshStandardMaterial({ map: kit.floor, color: 0x7f8b92, roughness: .94, metalness: .04 });
  const floor = mesh(g, new T.PlaneGeometry(26.4, 26.4), floorMat, [0, -.006, 0], false);
  floor.rotation.x = -PI / 2;
  const lineMat = matte(0x786447, .86, .03);
  for (const x of [-2.32, 2.70]) box(g, [.024, .002, 8.57], [x, -.003, .59], lineMat, 0, false);
  for (const z of [-3.70, 4.88]) box(g, [5.03, .002, .024], [.19, -.003, z], lineMat, 0, false);
  for (let z = -3.65; z <= 4.8; z += .49) box(g, [.024, .002, .247], [-2.73, -.003, z], lineMat, 0, false);
  const floorDecal = texture((ctx, w, h) => {
    ctx.strokeStyle = '#c3a2708e'; ctx.lineWidth = 2; ctx.strokeRect(7, 7, w - 14, h - 14);
    ctx.fillStyle = '#cbb38ea6'; ctx.font = 'bold 44px monospace'; ctx.fillText('LINE 04', 23, 55);
    ctx.strokeStyle = '#c7a368ae'; ctx.lineWidth = 6; ctx.beginPath(); ctx.moveTo(319, 30); ctx.lineTo(339, 45); ctx.lineTo(319, 60); ctx.stroke();
    ctx.font = 'bold 16px monospace'; ctx.fillText('OPERATOR AISLE', 26, 79);
  }, 368, 96);
  const label = mesh(g, new T.PlaneGeometry(1.52, .41), new T.MeshStandardMaterial({ map: floorDecal, transparent: true, depthWrite: false, roughness: .95 }),
    [-2.56, .001, 1.90], false);
  label.rotation.set(-PI / 2, 0, -PI / 2);

  const hall = matte(0x373e44, .89, .12), column = matte(0x3d4a50, .76, .34);
  for (const z of [-6.3, -2.5, 1.2, 5.0, 8.75]) {
    const x = 4.45;
    box(g, [.25, 4.1, .20], [x, 2.05, z], column, .009, false);
    box(g, [.31, .045, .32], [x, .024, z], hall, .004, false);
    box(g, [.16, .45, .22], [x - .09, .38, z], M.shade, .009, false);
  }
  box(g, [.063, .84, 15.6], [4.61, .43, 1.2], hall, .008, false);
  box(g, [.09, .042, 15.6], [4.52, .88, 1.2], column, 0, false);
  for (const z of [-5.55, -4.62, 4.57, 5.85]) {
    box(g, [.58, .77, .65], [3.76, .40, z], M.shell, .026, false);
    box(g, [.008, .57, .55], [3.462, .43, z], column, .004, false);
    for (let i = 0; i < 5; i++) box(g, [.012, .009, .25], [3.446, .58 - i * .04, z], hall, 0, false);
  }

  const aoTex = texture((ctx, w, h) => {
    const shade = ctx.createRadialGradient(w / 2, h / 2, w * .05, w / 2, h / 2, w / 2);
    shade.addColorStop(0, '#0000008f'); shade.addColorStop(.53, '#0000006c'); shade.addColorStop(1, '#00000000');
    ctx.fillStyle = shade; ctx.fillRect(0, 0, w, h);
  }, 64);
  const contact = new T.MeshBasicMaterial({ map: aoTex, transparent: true, depthWrite: false, polygonOffset: true, polygonOffsetFactor: -1 });
  for (const [x, z, w, l] of [[0, .03, 2.02, 7.2], [G.pickX, G.pickZ, 1.58, 1.7], [1.75, G.rejectZ, 1.8, 1.22], [-1.07, G.rejectZ, 1.98, .90], [0, 4.20, 1.20, 1.35]]) {
    const p = mesh(g, new T.PlaneGeometry(w, l), contact, [x, -.002, z], false);
    p.rotation.x = -PI / 2; p.renderOrder = 1;
  }
}

function supports(g, M, x, z, y = RAIL_Y, tall = false) {
  const w = tall ? .095 : .066;
  extrusion(g, w, y - .058, [x, y / 2 + .027, z], M.rail, 'y');
  box(g, [w + .037, .027, w + .051], [x, .036, z], M.graphite, .005);
  pin(g, .036, .018, [x, .010, z], M.black);
  pin(g, .012, .030, [x, .028, z], M.nickel, 'y', 12);
  for (const zz of [-.033, .033]) pin(g, .008, .008, [x, .051, z + zz], M.nickel, 'y', 6);
}

function buildConveyor(s, d, M, kit) {
  const belt = mesh(d, beltGeometry(), opticalFill(new T.MeshStandardMaterial({ map: kit.belt, bumpMap: kit.belt, bumpScale: .00027,
    roughness: .74, metalness: .055 })), [0, 0, 0], true);
  const flat = G.head * 2;
  box(s, [.81, .034, flat - .19], [0, .794, 0], M.graphite, .004);
  for (const x of [-railX, railX]) {
    extrusion(s, .092, flat + .12, [x, RAIL_Y, 0], M.alu);
    box(s, [.009, .026, flat - .085], [x + Math.sign(x) * .046, RAIL_Y, 0], M.shade);
    for (let i = 0; i < 22; i++) {
      const z = -2.94 + i * .28;
      pin(s, .0086, .009, [x + Math.sign(x) * .051, RAIL_Y + .021, z], M.nickel, 'x', 6);
    }
    const guides = x < 0
      ? [[-2.08, -.786], [-.654, -.525], [-.352, .58], [1.32, 1.43], [2.17, 2.28], [2.94, 3.135]]
      : [[-3.00, -.786], [-.654, .58], [2.94, 3.135]];
    for (const [a, b] of guides) {
      const xx = Math.sign(x) * .389;
      box(s, [.029, .026, b - a], [xx, .835, (a + b) / 2], M.black, .003);
      for (let z = a + .11; z < b - .06; z += .50) {
        box(s, [.090, .017, .039], [Math.sign(x) * .430, .816, z], M.rail, .003);
        pin(s, .007, .007, [Math.sign(x) * .444, .827, z], M.nickel, 'y', 6);
      }
    }
    for (const z of [-2.78, -1.04, .71, 2.76]) {
      supports(s, M, x, z, .695);
      box(s, [.042, .135, .075], [x + Math.sign(x) * .036, .678, z], M.rail, .008);
      for (const y of [.642, .718]) pin(s, .010, .011, [x + Math.sign(x) * .061, y, z], M.nickel, 'x', 6);
    }
    box(s, [.05, .065, 5.67], [x, .258, 0], M.base, .006);
  }
  for (const z of [-2.78, -1.04, .71, 2.76]) {
    extrusion(s, .049, .91, [0, .280, z], M.rail, 'x');
    for (const x of [-railX, railX]) bar(s, [x, .245, z], [x, .674, z + .27], .037, .029, M.base, .003);
  }

  const makeRoller = (axis, r, width, at, primary, cap = M.alu) => {
    const g = new T.Group();
    const core = new T.Group();
    pin(core, r, width, [0, 0, 0], primary, axis, r > .06 ? 40 : 18);
    const key = axis === 'x' ? 0 : 2;
    for (const sign of [-1, 1]) {
      const end = [0, 0, 0]; end[key] = sign * (width / 2 + .0035);
      pin(core, r * .80, .009, end, cap, axis, 18);
      end[key] += sign * .005;
      end[1] = r * .59;
      pin(core, r * .115, .002, end, r > .06 ? M.yellow : M.graphite, axis, 10);
    }
    batch(core); g.add(core); g.position.set(...at); d.add(g);
    return g;
  };
  const drums = [-G.head, G.head].map((z) => {
    for (const x of [-.468, .468]) {
      box(s, [.074, .137, .165], [x, drumY, z], M.graphite, .009);
      pin(s, .042, .064, [x, drumY, z], M.alu, 'x');
      pin(s, .017, .075, [x + Math.sign(x) * .022, drumY, z], M.nickel, 'x');
      for (const dz of [-.057, .057]) pin(s, .009, .009, [x + Math.sign(x) * .041, drumY + .016, z + dz], M.nickel, 'x', 6);
      if (z < 0) {
        box(s, [.042, .040, .027], [x, drumY, z - .147], M.base, .004);
        pin(s, .007, .101, [x, drumY, z - .121], M.nickel, 'z', 10);
        pin(s, .014, .014, [x, drumY, z - .181], M.brass, 'z', 6);
      }
    }
    return makeRoller('x', G.drumR - G.beltSkin, .868, [0, drumY, z], M.rubber, M.alu);
  });
  const idlers = [-2.08, -.76, .64, 2.10].map((z) => {
    const y = drumY - G.drumR - .027;
    for (const x of [-railX, railX]) box(s, [.059, .153, .061], [x, y + .062, z], M.base, .004);
    return makeRoller('x', .027, .89, [0, y, z], M.rail);
  });

  // Discharge shaft, reduction case, and radial cooling fins share the motor datum.
  const motorZ = G.head, motorX = -.742;
  box(s, [.242, .183, .209], [-.644, drumY, motorZ], M.base, .021);
  pin(s, .068, .13, [-.556, drumY, motorZ], M.yellow, 'x');
  pin(s, .115, .279, [motorX - .165, drumY, motorZ], M.blue, 'x', 28);
  for (let i = 0; i < 16; i++) {
    const a = (i / 16) * 2 * PI;
    const m = box(s, [.228, .027, .013], [motorX - .16, drumY + Math.cos(a) * .114, motorZ + Math.sin(a) * .114], M.blue, .003);
    m.rotation.x = a;
  }
  pin(s, .119, .038, [-1.059, drumY, motorZ], M.blue, 'x');
  pin(s, .092, .011, [-1.084, drumY, motorZ], M.graphite, 'x');
  for (let i = 0; i < 9; i++) {
    const y = -.067 + i * .016;
    box(s, [.009, .006, 2 * Math.sqrt(.080 ** 2 - y ** 2)], [-1.091, drumY + y, motorZ], M.blue, .001);
  }
  box(s, [.100, .061, .080], [-.899, drumY + .142, motorZ], M.graphite, .006);
  fitting(s, [-.899, drumY + .142, motorZ - .051], 'z', M.alu, M.black, .014, -1);
  box(s, [.013, .043, 4.9], [-.546, .484, .575], M.base, .005);
  const setBelt = (distance) => {
    kit.belt.offset.y = -(distance * (17 / BELT_PERIMETER)) % 1;
    const a = shaftAngles(distance);
    drums.forEach((m) => { m.rotation.x = a.drum; });
    idlers.forEach((m) => { m.rotation.x = a.returnIdler; });
  };
  return { setBelt, makeRoller, belt };
}

function cabinetAndHmi(g, M) {
  const cx = 1.20, cz = -.82;
  box(g, [.47, .91, .53], [cx, .64, cz], M.cabinet, .017);
  box(g, [.48, .105, .54], [cx, .10, cz], M.base, .009);
  for (const z of [cz - .19, cz + .19]) box(g, [.065, .11, .061], [cx, .175, z], M.graphite, .003);
  box(g, [.028, .82, .462], [cx - .234, .65, cz], M.base, .014);
  box(g, [.012, .070, .024], [cx - .256, .655, cz + .162], M.black, .003);
  for (let i = 0; i < 6; i++) box(g, [.148, .010, .009], [cx + .010, .43 + i * .024, cz + .274], M.shade, .001);
  tag(g, [.292, .099], [cx - .250, .871, cz], 'TRUSTVISION • 04', 'VISION & AUTOMATE', { angleY: -PI / 2 });
  for (const x of [1.05, 1.13, 1.21, 1.30]) fitting(g, [x, .186, cz - .11], 'y', M.alu, M.black, .012);

  // Operator HMI Terminal: square base plate aligned with factory floor grid & floor raceway
  const hmi = new T.Group();
  hmi.position.set(-1.68, 0, -.97); g.add(hmi);
  box(hmi, [.40, .026, .40], [0, .013, 0], M.graphite, .006);
  for (const dx of [-.15, .15]) for (const dz of [-.15, .15]) pin(hmi, .009, .014, [dx, .026, dz], M.nickel, 'y', 6);
  // Heavy-duty cast raceway entry flange on east edge (Z = 0, between corner bolts at Z = ±0.15)
  box(hmi, [.042, .028, .072], [.20, .014, 0], M.base, .003);
  pin(hmi, .012, .024, [.20, .024, 0], M.nickel, 'x', 12);

  // Upright structural pedestal extrusion (terminates safely at Y = 0.88m, well below angled console head)
  extrusion(hmi, .108, .854, [0, .453, 0], M.base, 'y');
  // Cast aluminum pedestal top collar and heavy swivel pivot pin
  box(hmi, [.130, .028, .130], [0, .894, 0], M.graphite, .004);
  pin(hmi, .036, .056, [0, .936, 0], M.base, 'y', 16);

  // Angled console head swivel mount (only the head swivels to face the operator)
  const head = new T.Group(); hmi.add(head);
  head.position.set(0, 1.13, 0); head.rotation.set(-.32, -1.14, 0, 'YXZ');
  // Rear swivel mount bracket connecting pedestal pin into back enclosure (entirely behind housing)
  box(head, [.120, .140, .054], [0, -.110, -.068], M.graphite, .004);
  box(head, [.66, .44, .068], [0, 0, 0], M.graphite, .014);
  box(head, [.28, .24, .035], [0, 0, -.048], M.shade, .006);
  const hmiCanvas = typeof document !== 'undefined'
    ? (document.getElementById('hmiDisplay') || document.getElementById('scanDisplay'))
    : null;
  let hmiTex = null;
  if (hmiCanvas) {
    hmiTex = new T.CanvasTexture(hmiCanvas);
    hmiTex.colorSpace = T.SRGBColorSpace;
    hmiTex.anisotropy = 4;
  }
  const screenMat = hmiTex
    ? new T.MeshStandardMaterial({ map: hmiTex, roughness: .38, emissive: 0x222a28, emissiveMap: hmiTex, emissiveIntensity: .85 })
    : M.shade;
  mesh(head, new T.PlaneGeometry(.60, .375), screenMat, [0, 0, .035], false);
  return { hmiX: -1.68, hmiZ: -.97, hmiTex };
}

function optics(s, d, overlay, M) {
  const top = 1.755, y = G.lensY;
  for (const x of [-gantryX, gantryX]) {
    for (const z of [-.415, .415]) {
      extrusion(s, .078, 1.711, [x, .912, z], M.yellow, 'y');
      box(s, [.161, .035, .167], [x, .043, z], M.base, .006);
      for (const a of [-.052, .052]) for (const b of [-.056, .056]) pin(s, .0097, .008, [x + a, .064, z + b], M.nickel, 'y', 6);
      box(s, [.015, .139, .124], [x + Math.sign(x) * .045, 1.697, z], M.amber, .006);
      for (const z2 of [z - .034, z + .034]) for (const y2 of [1.654, 1.742]) pin(s, .009, .012, [x + Math.sign(x) * .054, y2, z2], M.nickel, 'x', 6);
    }
    extrusion(s, .076, .89, [x, top, 0], M.yellow);
  }
  for (const z of [-.415, .415]) extrusion(s, .080, 1.19, [0, top, z], M.yellow, 'x');
  extrusion(s, .060, 1.11, [0, 1.697, .103], M.alu, 'x');
  for (const x of [-.40, .40]) bar(s, [x, 1.716, -.405], [x, 1.716, .405], .041, .043, M.rail);
  box(s, [.280, .035, .144], [0, 1.693, .102], M.graphite, .006);
  for (const x of [-.112, .112]) pin(s, .012, .013, [x, 1.715, .135], M.nickel, 'y', 6);

  const cam = new T.Group(); cam.position.set(0, y, 0); s.add(cam);
  box(cam, [.103, .105, .075], [0, .099, 0], M.graphite, .007);
  box(cam, [.118, .019, .098], [0, .047, 0], M.base, .004);
  for (const z of [-.026, -.013, 0, .013, .026]) box(cam, [.119, .071, .0045], [0, .102, z], M.shade, .002);
  for (const x of [-.046, .046]) for (const z of [-.039, .039]) pin(cam, .0049, .008, [x, .035, z], M.nickel, 'y', 6);
  pin(cam, .035, .014, [0, .039, 0], M.alu, 'y', 28);
  pin(cam, .031, .045, [0, .015, 0], M.black, 'y', 28);
  pin(cam, .038, .013, [0, .009, 0], M.base, 'y', 36);
  for (let i = 0; i < 32; i++) {
    const a = i * PI / 16;
    const rib = box(cam, [.003, .013, .0028], [Math.cos(a) * .038, .009, Math.sin(a) * .038], M.graphite);
    rib.rotation.y = -a;
  }
  pin(cam, .034, .011, [0, -.013, 0], M.nickel, 'y', 28);
  pin(cam, .029, .006, [0, -.020, 0], M.shade, 'y', 28);
  const glass = new T.MeshPhysicalMaterial({ color: 0x172d2b, roughness: .16, metalness: .38, clearcoat: 1, clearcoatRoughness: .14 });
  pin(cam, .0248, .004, [0, -.0246, 0], glass, 'y', 32);
  const mount = box(cam, [.098, .037, .032], [0, .164, .030], M.rail, .004);
  mount.rotation.y = 0;
  tag(cam, [.094, .039], [0, .104, .039], 'TV-4K', 'CAPTEUR LINÉAIRE', { width: 160, accent: '#a1d9bd' });
  for (const cx of [-.024, .023]) {
    fitting(s, [cx, y + .153, -.020], 'y', M.alu, M.black, .009);
    pin(s, .0075, .022, [cx, y + .166, -.020], M.rubber, 'y', 12);
  }
  const trigger = pin(d, .0043, .002, [-.037, y + .126, .041], ink(0x96e4b6), 'z', 10);
  trigger.castShadow = false;

  // Rigid mechanical standoff brackets connecting cable tray to gantry profile at X = 0.40m
  for (const bz of [-.070, -.360]) {
    box(s, [.042, .095, .036], [.400, 1.774, bz], M.base, .003);
    box(s, [.094, .012, .044], [.416, 1.821, bz], M.rail, .002);
    pin(s, .007, .014, [.400, 1.774, bz], M.nickel, 'y', 6);
  }

  // Solid cantilever umbilical support arm anchored directly to the transverse aluminum gantry beam at Z = +0.103m
  // 1. Clamping base plate bolted into the profile T-slots at X = 0.220m, Z = 0.103m
  box(s, [.056, .032, .070], [.220, 1.743, .103], M.base, .004);
  for (const bz of [.078, .128]) pin(s, .0065, .014, [.220, 1.759, bz], M.nickel, 'y', 6);

  // 2. Continuous horizontal cantilever arm (flush bottom plane Y = 1.731m, top Y = 1.759m)
  // Extending seamlessly from base mount (Z = +0.068m) to front face of corner (Z = -0.094m)
  box(s, [.028, .028, .162], [.220, 1.745, -.013], M.base, .003);

  // 3. Monolithic vertical riser post sharing exact same bottom datum (Y = 1.731m) and front face (Z = -0.094m)
  // Completely eliminates any stepped joint or offset seam; perfectly flush monolithic L-elbow
  box(s, [.028, .131, .028], [.220, 1.7965, -.080], M.base, .003);

  // 4. Structural gusset brace cleanly reinforcing the inside 90° angle without any awkward overhangs
  bar(s, [.220, 1.759, -.020], [.220, 1.835, -.066], .022, .016, M.base, .002);
  pin(s, .005, .008, [.220, 1.763, -.020], M.nickel, 'y', 6);
  pin(s, .005, .008, [.220, 1.835, -.062], M.nickel, 'z', 6);

  // 5. Dual cable split-collar clamp with outer clamping bolts flanking the cables (zero screw obstruction)
  // Clamp shelf base
  box(s, [.038, .010, .092], [.220, 1.867, -.090], M.rail, .003);
  // Lower cushion block
  box(s, [.036, .012, .090], [.220, 1.876, -.090], M.graphite, .002);
  // Upper clamp cap
  box(s, [.036, .010, .090], [.220, 1.893, -.090], M.graphite, .002);
  // Clamping hardware on outer flanges flanking the dual cables at Z = -0.052m and Z = -0.128m
  for (const cz of [-.052, -.128]) {
    pin(s, .0045, .014, [.220, 1.900, cz], M.nickel, 'y', 6);
    pin(s, .0065, .002, [.220, 1.898, cz], M.nickel, 'y', 12);
  }

  box(s, [.10, .021, .50], [.416, 1.832, -.210], M.base, .004);
  for (const x of [.373, .456]) box(s, [.012, .032, .50], [x, 1.846, -.210], M.graphite, .003);
  // Rounded radius drop-out roller at exit mouth of cable tray (Z = -0.460m)
  pin(s, .006, .095, [.416, 1.834, -.460], M.graphite, 'x', 14);
  tag(s, [.626, .111], [-.615, 1.743, 0], 'POSTE OPTIQUE', 'LIGNE 04 / BALAYAGE', { angleY: -PI / 2 });

  const lightY = G.beltY + G.plateH + G.lightZ * Math.tan(G.lightAngle);
  for (const sign of [-1, 1]) {
    const z = sign * G.lightZ;
    const group = new T.Group();
    group.position.set(0, lightY, z);
    group.rotation.x = sign > 0 ? PI - G.lightAngle : G.lightAngle;
    s.add(group);
    box(group, [.875, .049, .044], [0, 0, -.027], M.graphite, .006);
    box(group, [.804, .018, .004], [0, 0, -.0025], M.white, .001, false);
    for (const dy of [-.013, .013]) box(group, [.842, .008, .010], [0, dy, -.001], M.alu, .001);
    for (const x of [-.431, .431]) {
      pin(group, .0088, .016, [x, 0, -.027], M.nickel, 'x', 6);
      box(s, [.083, .024, .071], [Math.sign(x) * .479, lightY, z], M.base, .003);
    }
  }
  extrusion(s, .044, .77, [gantryX, 1.018, 0], M.rail);
  box(s, [.020, .140, .790], [gantryX + .044, 1.101, 0], M.graphite, .006);

  const scanMat = ink(0xb3f5d4);
  const scanGeo = new T.PlaneGeometry(1, stripWidth);
  const marks = Array.from({ length: 3 }, () => {
    const m = mesh(d, scanGeo, scanMat, [0, G.beltY + .0016, G.scanZ], false);
    m.rotation.x = -PI / 2; return m;
  });
  const fov = new T.BufferGeometry().setFromPoints([
    new T.Vector3(0, y - .030, 0), new T.Vector3(-G.fov / 2, G.beltY + .004, 0),
    new T.Vector3(0, y - .030, 0), new T.Vector3(G.fov / 2, G.beltY + .004, 0),
  ]);
  const guide = new T.LineSegments(fov, new T.LineDashedMaterial({ color: 0x9ac1b3, dashSize: .027, gapSize: .019, transparent: true, opacity: .53 }));
  guide.computeLineDistances();
  overlay.add(guide);

  const sy = G.eyeY;
  const eye = new T.Group(); s.add(eye);
  for (const sign of [-1, 1]) {
    const x = sign * G.sensorX;
    box(eye, [.043, .066, .044], [x, sy + .010, G.eyeZ], M.graphite, .004);
    box(eye, [.045, .012, .048], [x, sy + .049, G.eyeZ], M.yellow, .003);
    pin(eye, .010, .007, [x - sign * .023, sy, G.eyeZ], M.nickel, 'x');
    pin(eye, .0059, .009, [x - sign * .026, sy, G.eyeZ], sign < 0 ? M.glowRed : M.glowGreen, 'x');
    fitting(eye, [x, sy - .029, G.eyeZ], 'y', M.alu, M.black);
    box(eye, [.041, .065, .055], [x, .778, G.eyeZ + .028], M.rail, .003);
  }
  const rx = mesh(d, new T.SphereGeometry(.0047, 10, 6), ink(0x8beab3), [G.sensorX, sy + .056, G.eyeZ], false);
  const beam = mesh(d, new T.CylinderGeometry(.00085, .00085, 1, 8).rotateZ(-PI / 2), ink(0xe67a61), [0, sy, G.eyeZ], false);
  const encoder = new T.Group(); d.add(encoder);
  encoder.position.set(-.394, G.beltY + G.encoderR, -.44);
  const wheel = new T.Group(); encoder.add(wheel);
  pin(wheel, G.encoderR, .023, [0, 0, 0], M.rubber, 'x', 28);
  for (const x of [-.012, .012]) {
    pin(wheel, .032, .006, [x, 0, 0], M.alu, 'x', 20);
    for (let i = 0; i < 5; i++) {
      const a = i * 2 * PI / 5;
      pin(wheel, .0041, .001, [x + Math.sign(x) * .004, Math.cos(a) * .021, Math.sin(a) * .021], M.shade, 'x', 10);
    }
  }
  batch(wheel);
  const anchor = [-.493, .907, -.542];
  bar(s, anchor, [-.432, .861, -.44], .026, .024, M.alu, .004);
  pin(s, .026, .034, [-.471, .861, -.44], M.graphite, 'x', 18);
  pin(s, .009, .083, [-.448, .860, -.44], M.nickel, 'x');
  box(s, [.038, .096, .086], [-.519, .855, -.542], M.rail, .004);
  pin(s, .014, .032, anchor, M.nickel, 'x', 14);
  fitting(s, [-.485, .841, -.44], 'y', M.brass, M.black);

  const tower = [.55, top, .415];
  pin(s, .015, .166, [tower[0], tower[1] + .11, tower[2]], M.rail);
  const stack = ['green', 'amber', 'red'].map((color, i) => {
    const off = [0x38654d, 0x825734, 0x6b3a36][i];
    const on = [0x7cf2ab, 0xffc05b, 0xff7659][i];
    const acrylic = new T.MeshStandardMaterial({ color: off, roughness: .36, metalness: .07 });
    pin(d, .030, .053, [tower[0], tower[1] + .214 + i * .058, tower[2]], acrylic, 'y', 20);
    pin(s, .032, .005, [tower[0], tower[1] + .185 + i * .058, tower[2]], M.graphite, 'y', 20);
    return { color, off, on, acrylic };
  });
  pin(s, .032, .013, [tower[0], tower[1] + .364, tower[2]], M.graphite);

  function setOptics(machine, a, distance, drawings, cameraView) {
    const p = machine.plates.find((p) => p.state === 'belt' && Math.abs(lerp(p.prev.z, p.z, a) - G.scanZ) < G.plateL * .5);
    const setMark = (i, start, end, height) => {
      const m = marks[i]; m.visible = drawings && end > start;
      if (m.visible) { m.scale.x = end - start; m.position.x = (start + end) / 2; m.position.y = height + .0014; }
    };
    const gate = G.fov / 2;
    if (p) {
      const l = Math.max(-gate, p.x - G.plateW / 2), r = Math.min(gate, p.x + G.plateW / 2);
      setMark(0, -gate, l, G.beltY); setMark(1, l, r, G.beltY + G.plateH); setMark(2, r, gate, G.beltY);
    } else { setMark(0, -gate, gate, G.beltY); setMark(1, 0, 0, 0); setMark(2, 0, 0, 0); }
    guide.visible = drawings && cameraView === 'optics';
    const start = -G.sensorX + .030, end = machine.eye ? machine.eye.x - G.plateW / 2 : G.sensorX - .030;
    beam.visible = drawings;
    beam.scale.x = end - start;
    beam.position.x = (end + start) / 2;
    rx.material.color.setHex(machine.eye ? 0xb16b35 : 0x8beab3);
    trigger.material.color.setHex(p ? 0xafffe3 : 0x546a5b);
    encoder.rotation.x = shaftAngles(distance).encoder;
    const which = machine.review || machine.lastResult?.disposition === 'reject' ? 'red' : p || machine.note ? 'amber' : 'green';
    for (const module of stack) {
      module.acrylic.color.setHex(module.color === which ? module.on : module.off);
      module.acrylic.emissive.setHex(module.on);
      module.acrylic.emissiveIntensity = module.color === which ? .55 : 0;
    }
  }
  return { setOptics, lightY };
}

function buildLoader(s, d, M, wires) {
  const z = G.pickZ, x = G.pickX;
  const columns = [-2.18, .614];
  for (const column of columns) {
    extrusion(s, .091, 1.848, [column, .956, z], M.rail, 'y');
    box(s, [.202, .034, .213], [column, .024, z], M.graphite, .006);
    for (const q of [-.063, .063]) for (const qz of [-.071, .071]) pin(s, .010, .011, [column + q, .048, z + qz], M.nickel, 'y', 6);
    box(s, [.144, .156, .147], [column, 1.815, z], M.amber, .009);
  }
  box(s, [2.91, .155, .177], [-.79, 1.902, z], M.yellow, .011);
  // Precision linear guide rail bed along carriage operating zone (X = -1.75m to +0.06m)
  box(s, [1.81, .040, .014], [-.845, 1.891, z + .094], M.shade, .003);
  for (const y of [1.856, 1.929]) box(s, [1.81, .010, .016], [-.845, y, z + .090], M.alu, .002);

  // Heavy mechanical end-stop buffer blocks at stroke limits
  for (const endX of [-1.75, .06]) {
    box(s, [.024, .056, .022], [endX, 1.891, z + .096], M.graphite, .003);
    pin(s, .009, .014, [endX + (endX > 0 ? -.010 : .010), 1.891, z + .096], M.rubber, 'x');
  }

  // Station identification plaque mounted on clean front face of crossbeam at X = +0.34m
  // Completely outside carriage travel path AND guide rail zone (zero obstruction, zero rail overlap)
  box(s, [.520, .108, .010], [.34, 1.902, z + .090], M.base, .003);
  tag(s, [.500, .095], [.34, 1.902, z + .097], 'PORTIQUE À VIDE', '01 / ALIMENTATION', { width: 350, accent: '#e9a343' });
  
  // Horizontal energy chain guide tray along bridge beam
  box(s, [2.22, .017, .113], [-.68, X_CHAIN.base - .025, z - .165], M.graphite, .003);
  for (const side of [-1, 1]) box(s, [2.22, .029, .010], [-.68, X_CHAIN.base - .006, z - .165 + side * .055], M.rail, .002);
  box(s, [.103, .051, .085], [X_CHAIN.fixed, X_CHAIN.base - .012, z - .165], M.base, .004);
  pin(s, .059, .16, [.735, 1.90, z], M.blue, 'x', 20);
  pin(s, .061, .024, [.829, 1.90, z], M.graphite, 'x');
  box(s, [.085, .067, .076], [.749, 2.002, z - .009], M.blue, .005);
  fitting(s, [.739, 2.018, z - .051], 'z', M.alu, M.black);
  // Gantry servo power & encoder feed into structural column conduit
  wire(wires.power, [[.739, 2.018, z - .051], [.614, 2.018, z - .051], [.614, 1.880, z - .051]], .0055, M.line, .022);
  box(s, [.042, .042, .036], [.614, 1.870, z - .051], M.base, .003);

  // --- INFEED HYDRAULIC SCISSOR LIFT PLATFORM ---
  // Heavy floor base plate (sitting flat on floor Y = 0 to 0.040m)
  const platform = new T.Group(); s.add(platform); platform.position.set(x, 0, z);
  box(platform, [1.08, .040, .98], [0, .020, 0], M.graphite, .006);
  // Four anchor studs securing base plate to floor
  for (const qx of [-.49, .49]) for (const qz of [-.43, .43]) pin(platform, .009, .018, [qx, .036, qz], M.nickel, 'y', 6);

  // Scissor linkage with heavy steel pivot clevises and roller channels
  for (const sign of [-1, 1]) {
    const zz = sign * .31;
    // Lower Fixed Pivot Bracket (at X = -0.34m, Y = 0.040m to 0.080m)
    box(platform, [.064, .040, .056], [-.34, .060, zz], M.base, .003);
    pin(platform, .015, .068, [-.34, .065, zz], M.nickel, 'z', 16);

    // Lower Sliding Roller Track (at X = +0.22m to +0.42m)
    box(platform, [.24, .016, .056], [.32, .048, zz], M.base, .002);
    pin(platform, .020, .052, [.34, .065, zz], M.graphite, 'z', 16);

    // Upper Sliding Guide Channel (under lift table at X = -0.28m, Y = 0.427m)
    box(platform, [.22, .016, .056], [-.28, .427, zz], M.base, .002);
    pin(platform, .020, .052, [-.28, .410, zz], M.graphite, 'z', 16);

    // Upper Fixed Pivot Clevis (under lift table at X = +0.28m, Y = 0.427m)
    box(platform, [.064, .036, .056], [.28, .417, zz], M.base, .003);
    pin(platform, .015, .068, [.28, .410, zz], M.nickel, 'z', 16);

    // Cross Scissor Arms (solid amber structural steel, zero clipping with base plate!)
    pin(platform, .023, .046, [-.34, .065, zz], M.amber, 'z', 16);
    pin(platform, .023, .046, [.34, .065, zz], M.amber, 'z', 16);
    pin(platform, .023, .046, [-.28, .410, zz], M.amber, 'z', 16);
    pin(platform, .023, .046, [.28, .410, zz], M.amber, 'z', 16);
    bar(platform, [-.318, .077, zz], [.258, .398, zz], .044, .042, M.amber, .004);
    bar(platform, [.318, .077, zz], [-.258, .398, zz], .044, .042, M.amber, .004);
    // Heavy center hinge pin
    pin(platform, .026, .072, [0, .238, zz], M.nickel, 'z', 16);
  }

  // Heavy structural cross torque-tubes linking dual scissor arm sets
  pin(platform, .022, .62, [-.34, .065, 0], M.amber, 'z', 16);
  pin(platform, .026, .62, [0, .238, 0], M.amber, 'z', 16);
  pin(platform, .022, .62, [.28, .410, 0], M.amber, 'z', 16);
  pin(platform, .020, .62, [.34, .065, 0], M.graphite, 'z', 16);

  // Dual heavy clevis lugs welded to the center scissor torque-tube at [0, .238, 0]
  for (const dz of [-.028, .028]) box(platform, [.042, .042, .014], [0, .238, dz], M.graphite, .002);
  pin(platform, .014, .076, [0, .238, 0], M.nickel, 'z', 16);

  // Hydraulic Ram Actuator (anchored between stationary base clevis and center scissor torque-tube)
  const ram = new T.Group(); platform.add(ram);
  const a = new T.Vector3(-.32, .065, 0), b = new T.Vector3(0, .238, 0);
  const dir = b.clone().sub(a).normalize();
  const orientation = new T.Quaternion().setFromUnitVectors(new T.Vector3(0, 1, 0), dir);

  // Welded dual-lug base clevis bracket and pivot pin on platform base plate
  box(platform, [.060, .052, .082], [-.32, .055, 0], M.graphite, .003);
  pin(platform, .014, .090, [-.32, .065, 0], M.nickel, 'z', 16);

  // Cylinder rear trunnion clevis eye on base pin
  box(ram, [.036, .036, .046], a.toArray(), M.base, .003);

  // Hydraulic cylinder barrel
  const barrelPos = a.clone().addScaledVector(dir, .118);
  pin(ram, .036, .200, barrelPos.toArray(), M.base, 'y', 24).quaternion.copy(orientation);

  // Cylinder gland collar / front seal cap
  const glandPos = a.clone().addScaledVector(dir, .218);
  pin(ram, .040, .020, glandPos.toArray(), M.graphite, 'y', 24).quaternion.copy(orientation);

  // Hard chrome hydraulic piston rod
  const rodPos = a.clone().addScaledVector(dir, .282);
  pin(ram, .017, .110, rodPos.toArray(), M.nickel, 'y', 20).quaternion.copy(orientation);

  // Rod eye clevis engaging center scissor cross torque-tube pin
  box(ram, [.034, .034, .046], b.toArray(), M.graphite, .003);

  // Push-in port fitting and high-pressure hydraulic supply line
  const portPos = a.clone().addScaledVector(dir, .050);
  fitting(ram, [portPos.x, portPos.y + .036, portPos.z], 'y', M.brass, M.black, .009);
  wire(platform, [[portPos.x, portPos.y + .036, 0], [-.32, .08, .09], [-.32, .040, .09]], .0055, M.hose, .025);

  // Upper Lift Table Platform Top
  box(platform, [.98, .045, .93], [0, G.supplyY - .0225, 0], M.rail, .008);
  box(platform, [.890, .027, .015], [0, .451, .471], M.yellow, .002);
  for (const zz of [-.389, .389]) for (const xx of [-.347, .347]) {
    box(platform, [.068, .192, .053], [xx, .548, zz], M.graphite, .004);
    box(platform, [.051, .016, .056], [xx, .651, zz], M.yellow, .002);
  }

  // --- 3-STAGE PRECISION TELESCOPIC SHEET PICKER GANTRY ---
  const xAxis = new T.Group(), carriage = new T.Group(), head = new T.Group(), lift = new T.Group();
  d.add(xAxis, carriage); carriage.add(head, lift);
  xAxis.position.z = z - .165;
  const xChain = dragChain(xAxis, X_CHAIN, .075, .037, .042, M.black, M.rail);

  const staticCarriage = new T.Group(), staticHead = new T.Group(), staticLift = new T.Group();
  carriage.add(staticCarriage); head.add(staticHead); lift.add(staticLift);

  // Carriage Base Body
  box(staticCarriage, [.267, .142, .226], [0, 1.842, 0], M.base, .008);
  box(staticCarriage, [.176, .046, .087], [0, 1.984, -.151], M.rail, .004);
  // Rigid Tow Bracket mechanically coupling moving end of X_CHAIN to Carriage
  box(staticCarriage, [.088, .056, .052], [0, X_CHAIN.base + 2 * X_CHAIN.radius, -.165], M.base, .004);
  box(staticCarriage, [.048, .110, .040], [0, 1.945, -.165], M.rail, .003);
  pin(staticCarriage, .014, .045, [0, X_CHAIN.base + 2 * X_CHAIN.radius, -.165], M.alu, 'z', 12);
  fitting(staticCarriage, [.044, X_CHAIN.base + 2 * X_CHAIN.radius, -.165], 'x', M.brass, M.black, .010, 1);
  box(staticCarriage, [.160, .175, .306], [0, 1.691, -.205], M.graphite, .007);

  // Carriage Vertical Mast (Stage 1 Guide)
  box(staticCarriage, [.158, .700, .087], [0, 1.380, -.365], M.rail, .005);
  // Dual Precision Ground Linear Guide Rails on Carriage Mast
  for (const xx of [-.052, .052]) {
    bar(staticCarriage, [xx, 1.02, -.318], [xx, 1.72, -.318], .016, .014, M.nickel);
  }
  for (const xx of [-.085, .085]) for (const yy of [1.80, 1.881]) pin(staticCarriage, .0095, .006, [xx, yy, .118], M.nickel, 'z', 6);

  // Fixed Top Anchor Bracket for Vertical Drag Chain Y_CHAIN on Carriage Mast
  // Mounted high at Y = G.carryY + 0.550m (Y = 1.745m) so the loop NEVER drops below Y = 1.087m (26.7cm above conveyor belt!)
  box(staticCarriage, [.054, .044, .052], [.125, G.carryY + .550, -.369], M.base, .004);
  pin(staticCarriage, .012, .042, [.125, G.carryY + .550, -.369], M.nickel, 'y', 12);
  fitting(staticCarriage, [.125, G.carryY + .574, -.369], 'y', M.brass, M.black, .009);
  // Vertical Guide Sheet Metal Channel backing Y_CHAIN to prevent chain sway
  box(staticCarriage, [.012, .68, .054], [.098, 1.50, -.369], M.base, .002);

  // Cable conduit from X_CHAIN tow bracket down to Y_CHAIN fixed top bracket
  // Routes cleanly around the carriage frame: completely clear of cable tray lips and solid metal blocks!
  wire(staticCarriage, [
    [.044, X_CHAIN.base + 2 * X_CHAIN.radius, -.165],
    [.125, X_CHAIN.base + 2 * X_CHAIN.radius, -.165],
    [.125, X_CHAIN.base + 2 * X_CHAIN.radius, -.369],
    [.125, G.carryY + .574, -.369]
  ], .006, M.line, .025);
  box(staticCarriage, [.024, .036, .030], [.125, 1.90, -.369], M.base, .002);

  // Intermediate Telescopic Slide (Stage 2 Lift)
  box(staticLift, [.122, .820, .064], [0, .670, -.281], M.alu, .005);
  // Runner blocks gripping Carriage linear rails
  for (const xx of [-.052, .052]) for (const yy of [.96, .42]) {
    box(staticLift, [.030, .055, .028], [xx, yy, -.318], M.graphite, .003);
  }
  // Linear rails on front face of Stage 2 for Stage 3 Tooling Head
  for (const xx of [-.038, .038]) {
    bar(staticLift, [xx, .28, -.246], [xx, 1.06, -.246], .014, .012, M.nickel);
  }

  // Tooling Head Mast & Crossbeam (Stage 3)
  box(staticHead, [.084, .700, .065], [0, .430, -.220], M.graphite, .006);
  // Runner blocks on Tooling Head gripping Stage 2 rails
  for (const xx of [-.038, .038]) for (const yy of [.70, .35]) {
    box(staticHead, [.026, .048, .024], [xx, yy, -.246], M.graphite, .003);
  }
  box(staticHead, [.095, .053, .172], [0, .117, -.199], M.rail, .004);
  // Rigid central carrier bridge plate spanning from rear mast mount all the way to front crossbar
  box(staticHead, [.220, .046, .448], [0, .113, .004], M.base, .007);

  // Rigid H-frame longitudinal aluminum chassis spars mechanically tying front and rear suction crossbars
  for (const sideX of [-.120, .120]) {
    box(staticHead, [.038, .035, .440], [sideX, .082, 0], M.alu, .004);
  }
  // Transverse suction cup mounting crossbars
  box(staticHead, [.454, .035, .068], [0, .082, -.194], M.alu, .004);
  box(staticHead, [.454, .035, .068], [0, .082, .194], M.alu, .004);

  // Heavy anodized corner cross-clamp blocks securing longitudinal spars to transverse bars with hex hardware
  for (const sideX of [-.120, .120]) for (const barZ of [-.194, .194]) {
    box(staticHead, [.052, .046, .052], [sideX, .085, barZ], M.graphite, .003);
    pin(staticHead, .0045, .008, [sideX, .110, barZ], M.nickel, 'y', 6);
  }

  // Structural stiffener plates securing central carrier plate to suction crossbars
  for (const barZ of [-.194, .194]) {
    box(staticHead, [.180, .024, .046], [0, .102, barZ], M.base, .004);
  }

  // Central Multi-Port Vacuum Ejector Manifold
  box(staticHead, [.133, .059, .202], [0, .104, 0], M.blue, .007);
  fitting(staticHead, [.070, .115, -.080], 'z', M.brass, M.black, .009);

  // Moving Lower Tow Bracket on Tooling Head solidly gripping bottom of Y_CHAIN at y = 0.550m
  box(staticHead, [.054, .042, .054], [.221, .550, -.355], M.base, .004);
  fitting(staticHead, [.221, .522, -.355], 'y', M.brass, M.black, .010);
  // Pneumatic main air supply line from Y_CHAIN fitting into central vacuum manifold fitting
  wire(staticHead, [[.221, .515, -.355], [.221, .250, -.280], [.150, .140, -.180], [.070, .115, -.080]], .006, M.hose, .022);

  // 4 Industrial Suction Cup Assemblies with Push-In Fittings
  for (const xx of [-.173, .173]) for (const zz of [-.194, .194]) {
    pin(staticHead, .012, .045, [xx, .051, zz], M.nickel, 'y', 12);
    pin(staticHead, .023, .012, [xx, .034, zz], M.black, 'y', 24, .030);
    pin(staticHead, .020, .012, [xx, .022, zz], M.rubber, 'y', 24, .032);
    pin(staticHead, .024, .015, [xx, .008, zz], M.rubber, 'y', 24, .034);
    fitting(staticHead, [xx, .074, zz], 'y', M.brass, M.black, .008);
    wire(staticHead, [[Math.sign(xx) * .057, .112, Math.sign(zz) * .07], [xx, .115, Math.sign(zz) * .105], [xx, .103, zz], [xx, .074, zz]], .0045, M.hose, .024);
  }

  const led = pin(head, .0075, .004, [.072, .123, .054], ink(0x737f6f), 'x');
  led.castShadow = false;

  // Vertical Drag Chain Coordinate System (Mounted at Y = G.carryY + 0.550m)
  const zAxis = new T.Group(); carriage.add(zAxis);
  zAxis.position.set(.125, G.carryY + .550, -.369); zAxis.rotation.z = -PI / 2;
  const yChain = dragChain(zAxis, Y_CHAIN, .041, .024, .028, M.black, M.rail);
  [staticCarriage, staticLift, staticHead].forEach(batch);

  // Industrial Gantry Terminal Junction Box under the fixed end of X_CHAIN
  const fixY = X_CHAIN.base - .055;
  box(s, [.110, .072, .082], [X_CHAIN.fixed, fixY, z - .165], M.base, .005);
  // Top gland fitting coupling terminal box directly into X_CHAIN tray
  fitting(s, [X_CHAIN.fixed, X_CHAIN.base - .020, z - .165], 'y', M.brass, M.black, .011);
  // Outer -X conduit fitting on terminal box
  fitting(s, [X_CHAIN.fixed - .055, fixY, z - .165], 'x', M.brass, M.black, .011, -1);

  // Rigid horizontal conduit pipe running along underside of gantry beam from column X = -2.18 to terminal box
  bar(s, [-2.16, fixY, z - .165], [X_CHAIN.fixed - .055, fixY, z - .165], .016, .016, M.rail);
  for (const x of [-2.02, -1.82]) {
    box(s, [.024, .026, .024], [x, fixY, z - .165], M.base, .002);
    pin(s, .004, .010, [x, fixY + .013, z - .165], M.nickel, 'y', 6);
  }

  // Column Elbow & Vertical Conduit Riser running down column to floor junction box
  box(s, [.034, .034, .034], [-2.16, fixY, z - .165], M.rail, .002);
  bar(s, [-2.16, fixY - .017, z - .165], [-2.16, .20, z - .165], .016, .016, M.rail);
  for (const y of [.45, .90, 1.35, 1.65]) {
    box(s, [.026, .024, .026], [-2.16, y, z - .165], M.base, .002);
    pin(s, .004, .010, [-2.16 + .013, y, z - .165], M.nickel, 'x', 6);
  }

  // Column Junction Box at Base
  box(s, [.148, .108, .106], [-2.16, .137, z - .065], M.base, .006);
  fitting(s, [-2.16, .191, z - .065], 'y', M.brass, M.black, .010);
  wire(wires.signal, [[-2.16, .20, z - .165], [-2.16, .20, z - .065], [-2.16, .191, z - .065]], .007, M.line, .025);
  fitting(s, [-2.192, .105, z - .121], 'z', M.alu, M.black, .010);

  return (machine, a) => {
    const cx = lerp(machine.prev.x, machine.head.x, a), cy = lerp(machine.prev.cupY, machine.head.cupY, a);
    carriage.position.set(cx, 0, z);
    head.position.y = cy;
    lift.position.y = lerp(cy, G.carryY, .50);
    xChain(cx);
    yChain(zCarrierPosition(cy));
    led.material.color.setHex(machine.head.plate?.state === 'gripped' ? 0x94f4bf : 0x59665e);
  };
}

function buildActuator(s, d, M, wires) {
  const y = PUSH_Y;
  const length = CYL_FRONT - CYL_BACK, mid = (CYL_BACK + CYL_FRONT) * .5;
  const rodLength = 1.12;

  const STATIONS = [
    { key: 'bay1', z: 0.95, tag: '01 / RETOUCHE', title: 'ÉJECTEUR 01', accent: '#f2b05f' },
    { key: 'bay2', z: 1.80, tag: '02 / OUTILLAGE', title: 'ÉJECTEUR 02', accent: '#62b0d9' },
    { key: 'bay3', z: 2.65, tag: '03 / REBUT', title: 'ÉJECTEUR 03', accent: '#fe745d' },
  ];

  const mobiles = [];
  const stationLights = [];
  const coils = [];

  STATIONS.forEach(({ key, z, tag: stationTag, title, accent }) => {
    const stationary = new T.Group(); s.add(stationary);
    for (const x of [-1.70, -.600]) {
      for (const dz of [-.206, .206]) supports(stationary, M, x, z + dz, .77);
      box(stationary, [.115, .087, .515], [x, .782, z], M.base, .005);
      box(stationary, [.071, .032, .518], [x, .726, z], M.rail, .004);
    }
    box(stationary, [.48, .086, .021], [-1.096, .771, z + .231], M.base, .004);
    tag(stationary, [.288, .064], [-1.096, .771, z + .242], title, stationTag, { width: 240, accent });

    // Main Cylinder Housing & High-Tensile Tie Rods
    box(stationary, [length - .070, .098, .099], [mid, y, z], M.alu, .008);
    for (const sign of [-1, 1]) for (const dz of [-.039, .039]) {
      pin(stationary, .0064, length, [mid, y + sign * .047, z + dz], M.nickel, 'x', 12);
      for (const x of [CYL_BACK - .013, CYL_FRONT + .013]) pin(stationary, .010, .017, [x, y + sign * .047, z + dz], M.rail, 'x', 6);
    }
    for (const x of [CYL_BACK, CYL_FRONT]) box(stationary, [.038, .133, .134], [x, y, z], M.rail, .006);
    pin(stationary, .028, .027, [CYL_FRONT + .024, y, z], M.graphite, 'x');

    // Dual Linear Guide Shafts & Pillow Blocks
    for (const dz of [-.189, .189]) {
      pin(stationary, .0235, length - .035, [mid - .012, y, z + dz], M.graphite, 'x');
      for (const x of [CYL_BACK + .03, CYL_FRONT + .022]) {
        box(stationary, [.082, .073, .072], [x - .026, y, z + dz], M.rail, .006);
        pin(stationary, .025, .027, [x + .013, y, z + dz], M.alu, 'x');
      }
    }

    // Moving Pusher Paddle Assembly
    const mobile = new T.Group(); d.add(mobile); mobile.position.set(G.paddleRest, y, z);
    pin(mobile, .016, rodLength, [-.039 - rodLength / 2, 0, 0], M.nickel, 'x', 22);
    for (const dz of [-.189, .189]) pin(mobile, .0125, rodLength, [-.039 - rodLength / 2, 0, dz], M.nickel, 'x', 20);
    box(mobile, [.046, .056, .444], [-.038, 0, 0], M.alu, .004);
    box(mobile, [.019, .068, .493], [-.0155, .001, 0], M.yellow, .003);
    box(mobile, [.0065, .043, .456], [-.00325, -.011, 0], M.shade, .002);
    for (const dz of [-.197, -.114, .114, .197]) pin(mobile, .006, .008, [-.057, .006, dz], M.nickel, 'x', 6);
    batch(mobile);
    mobiles.push({ key, mobile });

    // Pneumatic Valve Manifold & Service Unit
    const vx = -1.264, vy = .616, vz = z + .239;
    box(s, [.273, .026, .112], [vx, vy - .039, vz], M.base, .004);
    box(s, [.159, .071, .064], [vx, vy, vz], M.graphite, .006);
    for (const u of [-.049, .049]) fitting(s, [vx + u, vy + .039, vz], 'y', M.brass, M.black);

    // Regulator & Dial Pressure Gauge
    const regulatorX = vx - .20, regulatorY = vy + .007;
    box(s, [.077, .057, .066], [regulatorX, regulatorY, vz], M.rail, .006);
    pin(s, .027, .070, [regulatorX, regulatorY - .056, vz], matte(0x3c5559, .38, .35));
    pin(s, .029, .017, [regulatorX, regulatorY - .094, vz], M.graphite, 'y', 20);
    pin(s, .019, .025, [regulatorX, regulatorY + .037, vz], M.graphite, 'y', 16);
    pin(s, .025, .012, [regulatorX, regulatorY - .003, vz + .044], M.graphite, 'z');
    pin(s, .020, .004, [regulatorX, regulatorY - .003, vz + .051], M.white, 'z');
    bar(s, [regulatorX - .007, regulatorY - .010, vz + .054], [regulatorX + .010, regulatorY + .006, vz + .054], .0027, .002, M.black);
    // Push-in Brass Fitting on air supply input port of regulator
    pin(s, .012, .008, [regulatorX - .0385, regulatorY, vz], M.base, 'x');
    fitting(s, [regulatorX - .040, regulatorY, vz], 'x', M.brass, M.black, .010, -1);
    // Machined brass union hex nipple between regulator and solenoid manifold
    pin(s, .011, .038, [(regulatorX + .0385 + vx - .080) / 2, regulatorY, vz], M.brass, 'x', 6);
    fitting(s, [vx - .080, regulatorY, vz], 'x', M.brass, M.black, .009, 1);

    // Shop Air Supply Drop Standpipe (in clear bay between legs, ZERO collision with leg foot)
    const spX = -1.46, spZ = z + .04;
    box(s, [.110, .031, .100], [spX, .019, spZ], M.base, .005);
    pin(s, .016, .077, [spX, .063, spZ], M.rail, 'y', 16);
    fitting(s, [spX, .102, spZ], 'y', M.brass, M.black, .011);
    wire(wires.air, [[spX, .102, spZ], [spX, .165, spZ], [spX, regulatorY, spZ], [regulatorX - .056, regulatorY, vz]], .0068, M.hose, .035);

    // Cylinder Work Port Hoses & Magnetic Reed Proximity Switches
    for (const [i, x] of [CYL_BACK + .052, CYL_FRONT - .052].entries()) {
      fitting(s, [x, y + .065, z + .014], 'y', M.brass, M.black, .011);
      const portOffset = i ? .049 : -.049;
      const turnX = i ? vx + .08 : vx - .08;
      // Work port supply hose runs strictly above cylinder in open clearance:
      // ZERO collision with table frame beams, ZERO collision with guide pillow blocks
      wire(wires.air, [
        [vx + portOffset, vy + .045, vz],
        [vx + portOffset, .720, vz],
        [vx + portOffset, .720, z + .055],
        [turnX, y + .085, z + .055],
        [turnX, y + .085, z + .014],
        [x, y + .085, z + .014],
        [x, y + .071, z + .014]
      ], .0055, M.hose, .018);
      // Magnetic reed switches clipped into cylinder profile slot
      box(s, [.037, .013, .018], [x + (i ? -.025 : .025), y + .061, z - .031], M.black, .003);
      // Sensor cable routes neatly along profile slot and passes underneath cylinder in open bay
      wire(wires.signal, [
        [x + (i ? -.025 : .025), y + .061, z - .032],
        [x + (i ? -.025 : .025), y + .061, z - .054],
        [vx + .06, y + .061, z - .054],
        [vx + .06, .660, z - .054],
        [vx + .06, .660, vz],
        [vx + .06, vy - .020, vz]
      ], .0028, M.line, .014);
    }

    // Indicator Diagnostic LEDs
    const lights = [-1.628, -.615].map((x) => pin(d, .0052, .004, [x, y + .069, z - .031], ink(0x404840), 'y'));
    stationLights.push({ key, lights });
    const coil = pin(d, .0064, .004, [vx + .060, vy + .036, vz + .014], ink(0x404840), 'y');
    coils.push({ key, coil });

    // Multi-conductor Solenoid Cable running safely around the table leg to raceway duct (ZERO phasing through legs)
    wire(wires.signal, [
      [vx + .078, vy + .018, vz],
      [vx + .078, .461, vz],
      [vx + .078, .461, z + .275],
      [-.520, .461, z + .275],
      [-.520, .461, z + .200],
      [-.540, .461, z + .200]
    ], .0048, M.line, .020);
  });

  // Continuous longitudinal industrial cable raceway along conveyor legs for pneumatic sorting stations
  box(s, [.048, .042, 2.25], [-.540, .440, 1.80], M.graphite, .003);
  for (const bz of [0.95, 1.80, 2.65]) {
    box(s, [.032, .024, .040], [-.505, .440, bz], M.base, .002);
    fitting(s, [-.540, .461, bz + .20], 'y', M.alu, M.black, .009);
  }
  // Enclosed continuous main signal trunk cable inside the raceway
  wire(wires.signal, [[-.540, .440, 2.65 + .20], [-.540, .440, -.440]], .0055, M.line, .020);

  return (machine, a) => {
    STATIONS.forEach(({ key }, idx) => {
      const act = machine.actuators ? machine.actuators[key] : machine.actuator;
      const prevStroke = act.prevStroke !== undefined ? act.prevStroke : machine.prev.stroke;
      const stroke = lerp(prevStroke, act.stroke, a);
      mobiles[idx].mobile.position.x = G.paddleRest + stroke;
      stationLights[idx].lights[0].material.color.setHex(stroke < .002 ? 0x92f0b4 : 0x445047);
      stationLights[idx].lights[1].material.color.setHex(stroke > G.stroke - .002 ? 0xffce7a : 0x51443b);
      coils[idx].coil.material.color.setHex(['extend', 'dwell'].includes(act.phase) ? 0xffc470 : 0x46504b);
    });
  };
}

function buildRollerStations(s, M, makeRoller) {
  const u = .026;
  const define = (kind, positions, zStation) => positions.map((q) => {
    const { h, angle } = support(kind, q);
    const x = kind === 'chute' ? q + Math.sin(angle) * u : 0;
    const z = kind === 'chute' ? zStation : q + Math.sin(angle) * u;
    return { x, y: h - Math.cos(angle) * u, z, q };
  });

  const BAYS = [
    { z: 0.95, name: 'BAIE 01 : RETOUCHE', sub: 'ÉBAVURAGE & POLISSAGE (Sc, Pa)', accent: '#f2b05f', color: matte(0x4a3b2b, .55, .35) },
    { z: 1.80, name: 'BAIE 02 : OUTILLAGE', sub: 'USURE CYLINDRES (PS, RS)', accent: '#62b0d9', color: matte(0x283b48, .55, .35) },
    { z: 2.65, name: 'BAIE 03 : REBUT', sub: 'REFUSION FOUR À ARC (Cr, In)', accent: '#fe745d', color: matte(0x482824, .55, .35) },
  ];

  const allChuteRollers = [];
  BAYS.forEach((bay) => {
    const rollers = define('chute', rejectRollerX, bay.z);
    allChuteRollers.push(...rollers);
    for (const data of rollers) {
      for (const sign of [-1, 1]) {
        pin(s, .015, .036, [data.x, data.y, data.z + .382 * sign], M.base, 'z', 14);
      }
    }
    // Structural guide rails and containment hopper for this bay
    // Plate length is 0.68m (half-length 0.34m). Positioning rail centerline at bay.z +/- 0.375m with
    // 0.030m rail width provides a clear inner channel width of 0.720m (20mm clearance each side).
    for (const sign of [-1, 1]) {
      const rz = bay.z + sign * .375;
      // Parallel chute guide rail bars running strictly from conveyor side frame (X = 0.440m) down the chute
      for (const [a, b] of [[.440, .74], [.74, 1.12], [1.12, 1.433]]) {
        bar(s, [a, support('chute', a).h + .008, rz], [b + .008, support('chute', b).h + .008, rz], .030, .065, M.rail, .004);
      }
      supports(s, M, .965, rz, .757);
      // Containment hopper sidewalls placed at bay.z +/- 0.418m (59mm clearance to sheet edge)
      box(s, [1.44, .284, .038], [1.775, .299, bay.z + sign * .418], bay.color, .008);
    }
    box(s, [1.38, .042, .800], [1.806, G.scrapY - .021, bay.z], M.graphite, .009);
    box(s, [.055, .295, .820], [G.scrapStop + G.plateW / 2 + .025, .305, bay.z], bay.color, .009);
    box(s, [.043, .093, .800], [1.092, .196, bay.z], bay.color, .007);
    box(s, [.056, .025, .820], [G.scrapStop + G.plateW / 2 + .025, .453, bay.z], M.yellow, .004);
    box(s, [1.45, .023, .044], [1.770, .449, bay.z + .418], M.yellow, .004);
    for (const xx of [1.18, 1.46, 2.08, 2.36]) box(s, [.039, .247, .029], [xx, .303, bay.z + .436], M.rail, .005);
    for (const x of [1.24, 2.31]) for (const z of [bay.z - .345, bay.z + .345]) box(s, [.151, .129, .124], [x, .075, z], M.graphite, .007);
  });

  // Factory divider capping posts between sorting chute mouths along conveyor edge (X = 0.440m)
  // Strictly flush with conveyor side, leaving full 150mm clearance to plates on the belt (never intruding into the belt zone)
  const entranceH = support('chute', .440).h;
  for (const divZ of [1.375, 2.225]) {
    box(s, [.042, .072, .095], [.440, entranceH + .010, divZ], M.base, .004);
    box(s, [.044, .014, .097], [.440, entranceH + .048, divZ], M.yellow, .002);
  }
  // Upstream and downstream boundary end-posts
  box(s, [.042, .072, .044], [.440, entranceH + .010, 0.555], M.base, .004);
  box(s, [.044, .014, .046], [.440, entranceH + .048, 0.555], M.yellow, .002);
  box(s, [.042, .072, .044], [.440, entranceH + .010, 3.045], M.base, .004);
  box(s, [.044, .014, .046], [.440, entranceH + .048, 3.045], M.yellow, .002);

  const outRollers = define('runout', outRollerZ, 0);
  for (const data of outRollers) {
    for (const sign of [-1, 1]) {
      pin(s, .015, .04, [data.x + .424 * sign, data.y, data.z], M.base, 'x', 14);
    }
  }

  const rollerVisual = [
    ...allChuteRollers.map((r) => ({ ...r, kind: 'chute', mesh: makeRoller('z', u, .740, [r.x, r.y, r.z], M.alu, M.rail) })),
    ...outRollers.map((r) => ({ ...r, kind: 'runout', mesh: makeRoller('x', u, .827, [r.x, r.y, r.z], M.alu, M.rail) }))
  ];

  for (const sign of [-1, 1]) {
    const ox = sign * .438;
    for (const [a, b] of [[3.135, 3.41], [3.41, 3.810]]) bar(s, [ox, support('runout', a).h + .008, a], [ox, support('runout', b).h + .008, b + .005], .042, .068, M.rail, .004);
    bar(s, [sign * .470, support('runout', 3.135).h + .008, 2.94], [ox, support('runout', 3.135).h + .008, 3.135], .042, .068, M.rail, .004);
    box(s, [.042, .068, .016], [ox, support('runout', 3.810).h + .008, 3.810], M.base, .003);
    box(s, [.044, .016, .018], [ox, support('runout', 3.810).h + .042, 3.810], M.yellow, .002);
    supports(s, M, ox, 3.67, .681);
    box(s, [.040, .195, 1.075], [sign * .475, .284, 4.23], M.base, .008);
  }

  box(s, [.941, .047, 1.110], [0, G.outputY - .0235, 4.19], M.graphite, .010);
  box(s, [.963, .203, .044], [0, .292, G.outputStop + G.plateL / 2 + .022], M.base, .006);
  box(s, [.965, .023, .046], [0, .396, G.outputStop + G.plateL / 2 + .022], M.yellow, .004);
  tag(s, [.363, .104], [0, .278, G.outputStop + G.plateL / 2 + .046], 'CONFORME', '04 / QUALITÉ SUPÉRIEURE', { width: 256, accent: '#91cdb1' });
  for (const x of [-.354, .354]) for (const z of [3.76, 4.61]) box(s, [.116, .191, .116], [x, .10, z], M.base, .006);

  return (machine, alpha) => {
    rollerVisual.forEach((row, i) => {
      const rollerData = machine.rollers[i % machine.rollers.length];
      if (rollerData) {
        const { prev, angle } = rollerData;
        row.mesh.rotation[row.kind === 'chute' ? 'z' : 'x'] = lerp(prev, angle, alpha);
      }
    });
  };
}

function guards(s, M) {
  const x = 2.67;
  // Fencing upright posts
  for (const z of [.45, 1.35, 2.25, 3.15]) {
    extrusion(s, .054, 1.22, [x, .648, z], M.yellow, 'y');
    box(s, [.123, .028, .118], [x, .019, z], M.rail, .004);
    for (const off of [-.037, .037]) pin(s, .0085, .009, [x + off, .039, z], M.nickel, 'y', 6);
  }
  // Horizontal frame bars and wire safety mesh
  for (const cc of [.90, 1.80, 2.70]) {
    for (const y of [.169, 1.19]) box(s, [.038, .038, .86], [x, y, cc], M.base, .005);
    for (let i = 0; i < 10; i++) pin(s, .0035, .99, [x, .68, cc - .39 + i * .088], M.cage, 'y', 6);
    for (let i = 0; i < 11; i++) pin(s, .0035, .84, [x + .003, .198 + i * .096, cc], M.cage, 'z', 6);
  }

  // Metal Class Identification Badges mounted directly ON the safety net wire mesh at eye level (Y = 0.90m)
  // Perfectly visible across the line, completely unobstructed by hoppers!
  const FENCE_LABELS = [
    { z: 0.95, name: 'BAIE 01 : RETOUCHE', sub: 'ÉBAVURAGE & POLISSAGE (Sc, Pa)', accent: '#f2b05f' },
    { z: 1.80, name: 'BAIE 02 : OUTILLAGE', sub: 'USURE CYLINDRES (PS, RS)', accent: '#62b0d9' },
    { z: 2.65, name: 'BAIE 03 : REBUT', sub: 'REFUSION FOUR À ARC (Cr, In)', accent: '#fe745d' },
  ];

  FENCE_LABELS.forEach(({ z, name, sub, accent }) => {
    // Mounting backplate attached to mesh
    box(s, [.018, .172, .480], [x - .024, .90, z], M.graphite, .004);
    for (const dz of [-.21, .21]) for (const dy of [-.065, .065]) {
      pin(s, .007, .014, [x - .030, .90 + dy, z + dz], M.nickel, 'x', 6);
    }
    // High-contrast, sharp engraved tag facing the operator aisle
    tag(s, [.46, .155], [x - .034, .90, z], name, sub, { angleY: -PI / 2, width: 290, accent });
  });

  // Top rail general safety warning tag
  tag(s, [.34, .12], [x - .030, 1.16, 3.15], 'ACCÈS DANGER', 'MOUVEMENTS AUTOMATIQUES', { angleY: -PI / 2, width: 240, accent: '#e7a23d' });
}

function staticRouting(s, wires, M, lightY) {
  const c = G.lensY, eye = G.eyeZ, h = G.eyeY;
  for (const [i, x] of [-.024, .023].entries()) {
    const z = -.077 - .026 * i;
    const trayX = .406 + .016 * i;
    const gx = .590 + .022 * i;
    const colZ = -.470 - .012 * i;
    wire(wires.signal, [
      [x, c + .161, -.02],
      [x, 1.755, -.02],
      [x + .08, 1.885, z],
      [.220, 1.885, z],
      [.380, 1.885, z],
      [trayX, 1.850, z],
      [trayX, 1.850, -.465],
      [trayX + .010, 1.850, colZ],
      [gx, 1.850, colZ],
      [gx, 1.718, colZ],
      [gx, .267, colZ],
      [.682, .208, -.475],
      [1.05 + .078 * i, .208, -.925],
      [1.05 + .078 * i, .184, -.93]
    ], .0059, M.line, .045);
  }
  for (const sign of [-1, 1]) {
    const z = sign * G.lightZ;
    fitting(s, [.443, lightY + .010, z], 'x', M.alu, M.black);
    wire(wires.power, [[.447, lightY + .010, z], [gantryX + .018, lightY + .03, z], [.632, 1.075, z], [.677, 1.075, -.457], [.699, .231, -.457], [1.211, .217, -.929], [1.211, .185, -.93]], .0045, M.line, .032);
  }
  for (const sign of [-1, 1]) {
    const x = sign * G.sensorX;
    wire(wires.signal, [[x, h - .041, eye], [x + sign * .026, .719, eye], [x + sign * .032, .56, eye], [x + sign * .026, .498, eye],
      [.535, .498, -.688], [.652, .499, -.676], [.664, .233, -.681], [1.30, .225, -.929], [1.30, .178, -.93]].filter((point, i) => sign < 0 || i !== 4), .0041, M.line, .031);
  }
  wire(wires.signal, [[-.485, .829, -.44], [-.537, .777, -.44], [-.541, .578, -.44], [-.542, .498, -.628], [.510, .498, -.628], [.598, .52, -.648]], .0047, M.line, .042);
  box(s, [1.108, .046, .064], [0, .501, -.659], M.graphite, .003);
  box(s, [.091, .085, .126], [.593, .508, -.677], M.base, .006);
  box(s, [.084, .071, .083], [-.536, .479, -.652], M.graphite, .005);

  // Power Cable Trunk routing along conveyor leg
  wire(wires.power, [[-.899, .874, 3.033], [-.899, .874, 2.92], [-.872, .811, 2.86], [-.698, .758, 2.83], [-.593, .461, 2.761],
    [-.530, .461, -.938], [-.530, .200, -.970]], .007, M.line, .042);
  fitting(s, [-.530, .195, -.970], 'y', M.brass, M.black, .011);

  // Electrical Junction Box bolted solidly to conveyor leg profile at X = -0.466m, Z = -1.04m
  box(s, [.088, .120, .110], [-.530, .135, -.970], M.graphite, .006);
  // Heavy mounting bracket connecting junction box to conveyor leg
  box(s, [.042, .090, .036], [-.475, .135, -.990], M.base, .003);
  // Bottom brass conduit fitting on junction box
  fitting(s, [-.530, .075, -.970], 'y', M.brass, M.black, .011);
  // Cast 90-degree floor elbow transition box directly under fitting
  box(s, [.054, .056, .054], [-.530, .038, -.970], M.rail, .003);

  // Low-profile heavy-duty extruded aluminum floor cable raceway duct
  // Spans perfectly straight from HMI base flange (X = -1.48m) to Conveyor Elbow (X = -0.53m) at Z = -0.970m
  box(s, [.95, .018, .068], [-1.005, .009, -.970], M.rail, .003);
  // Floor saddle clamps anchoring raceway to concrete floor
  for (const cx of [-1.30, -1.005, -.71]) {
    box(s, [.024, .020, .082], [cx, .010, -.970], M.base, .002);
    for (const cz of [-.034, .034]) pin(s, .004, .008, [cx, .020, -.970 + cz], M.nickel, 'y', 6);
  }

  // Enclosed signal line running inside floor duct
  wire(wires.signal, [[-1.48, .018, -.970], [-.530, .018, -.970], [-.530, .075, -.970]], .0045, M.line, .018);

  for (const y of [.414, .66, .94, 1.207, 1.470, 1.682]) {
    box(s, [.119, .013, .035], [.615, y, -.478], M.rail, .002);
    box(s, [.026, .013, .025], [.695, y, -.459], M.black, .002);
  }
  for (const z of [-.75, -.43, .03, .69, 1.10, 1.81, 2.44]) box(s, [.021, .025, .020], [-.579, .463, z], M.rail, .002);
}

function buildFactoryWalls(g, M) {
  const wallMat = matte(0x242d32, .72, .18);
  const kickMat = matte(0x323c42, .65, .28);
  const stripeMat = matte(0xd99834, .48, .24);
  const columnMat = matte(0x3a474d, .55, .35);

  // Note: All building meshes have shadow = false so they never cast harsh triangular floor shadows!
  // Only the actual machinery casts sharp, realistic contact shadows.

  // 1. Back Wall (-Z at Z = -6.40m, spanning X from -5.50m to +4.62m, width = 10.12m)
  box(g, [10.2, .40, .26], [-.44, .20, -6.36], kickMat, .008, false);
  box(g, [10.12, 4.20, .14], [-.44, 2.50, -6.42], wallMat, .006, false);
  for (const x of [-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5]) {
    box(g, [.024, 4.20, .016], [x, 2.50, -6.34], M.shade, .002, false);
  }
  box(g, [10.12, .065, .012], [-.44, 1.25, -6.34], stripeMat, .002, false);

  // Structural Building Columns along Back Wall
  for (const x of [-4.5, -2.25, 0.0, 2.25]) {
    box(g, [.24, 4.60, .024], [x, 2.30, -6.21], columnMat, .004, false);
    box(g, [.020, 4.60, .18], [x, 2.30, -6.31], columnMat, .004, false);
    box(g, [.34, .040, .32], [x, .020, -6.23], M.graphite, .006, false);
    for (const dx of [-.12, .12]) for (const dz of [-.10, .10]) {
      pin(g, .012, .024, [x + dx, .046, -6.23 + dz], M.nickel, 'y', 6);
    }
  }

  // 2. Side Wall (+X at X = +4.62m, spanning Z from -6.40m to +9.50m, length = 15.9m)
  box(g, [.26, .40, 16.0], [4.58, .20, 1.55], kickMat, .008, false);
  box(g, [.14, 4.20, 15.9], [4.64, 2.50, 1.55], wallMat, .006, false);
  for (let z = -5.5; z <= 9.0; z += 1.8) {
    box(g, [.016, 4.20, .024], [4.56, 2.50, z], M.shade, .002, false);
  }
  box(g, [.012, .065, 15.9], [4.56, 1.25, 1.55], stripeMat, .002, false);

  // 3. Overhead Crane Runway Girders along Both Walls at Y = 4.25m
  box(g, [10.12, .24, .18], [-.44, 4.25, -6.18], columnMat, .006, false);
  box(g, [.18, .24, 15.9], [4.38, 4.25, 1.55], columnMat, .006, false);
  bar(g, [-5.5, 4.39, -6.18], [4.38, 4.39, -6.18], .032, .036, M.rail);
  bar(g, [4.38, 4.39, -6.18], [4.38, 4.39, 9.5], .036, .032, M.rail);

  // 4. Continuous Galvanized Cable Tray at Y = 2.80m
  box(g, [9.0, .06, .12], [-.60, 2.80, -6.30], M.rail, .004, false);
  for (const dy of [-.018, .018]) {
    wire(g, [[-4.8, 2.80 + dy, -6.29], [4.1, 2.80 + dy, -6.29]], .008, M.line);
  }

  // 5. Heavy 480V Substation Disconnect Cabinet
  const subX = -1.10, subY = 1.65, subZ = -6.28;
  box(g, [.68, .92, .24], [subX, subY, subZ], M.cabinet, .014, false);
  box(g, [.06, .16, .05], [subX - .35, subY + .02, subZ + .12], M.black, .004, false);
  tag(g, [.30, .10], [subX, subY + .22, subZ + .13], '400V TRIPHASÉ', 'ARMOIRE FORCE 04B', { width: 220, accent: '#fe745d' });

  // Two conduits exactly bridging from cabinet top (Y = 2.11m) to cable tray underside (Y = 2.77m)
  for (const cx of [subX - .14, subX + .14]) {
    pin(g, .020, .66, [cx, 2.44, subZ], M.alu, 'y', 16);
    fitting(g, [cx, 2.12, subZ], 'y', M.brass, M.black, .016);
    fitting(g, [cx, 2.76, subZ], 'y', M.brass, M.black, .016);
  }

  // 6. Overhead Industrial Wall Floodlights (Mounted on structural columns, wired into cable tray)
  for (const lx of [-4.5, -2.25, 2.25]) {
    bar(g, [lx, 3.65, -6.34], [lx, 3.55, -6.08], .026, .026, M.base);
    box(g, [.36, .08, .18], [lx, 3.55, -6.08], M.graphite, .005, false);
    box(g, [.32, .01, .14], [lx, 3.51, -6.08], ink(0xfffaed), .002, false);
    pin(g, .011, .82, [lx, 3.22, -6.33], M.alu, 'y');
    fitting(g, [lx, 2.83, -6.33], 'y', M.brass, M.black, .011);
    fitting(g, [lx, 3.63, -6.33], 'y', M.brass, M.black, .011);
  }

  // 7. Facility Identification Signage
  // Cleanly centered in the open wall bay between column X = 0.0 and column X = 2.25 (at X = 1.125m, Y = 3.45m)
  // Perfectly unobstructed by columns, crane girders, cable tray, or floodlights!
  tag(g, [2.00, .46], [1.125, 3.45, -6.34], 'TRUSTVISION INDUSTRIAL SYSTEMS', 'POSTE DE CONTRÔLE QUALITÉ & TRI LIGNE 04', { width: 920, accent: '#f2b05f' });
}

export function createCell(scene, kit) {
  const staticAssembly = new T.Group(), environment = new T.Group(), live = new T.Group(), overlay = new T.Group();
  scene.add(environment, staticAssembly, live, overlay);
  const M = materials();
  const route = { signal: new T.Group(), power: new T.Group(), air: new T.Group() };
  scene.add(...Object.values(route));
  patternedFloor(environment, M, kit);
  buildFactoryWalls(environment, M);
  environment.traverse((child) => {
    if (child.isMesh) child.castShadow = false;
  });
  const c = buildConveyor(staticAssembly, live, M, kit);
  const { hmiTex } = cabinetAndHmi(staticAssembly, M);
  const o = optics(staticAssembly, live, overlay, M);
  const loader = buildLoader(staticAssembly, live, M, route);
  const actuator = buildActuator(staticAssembly, live, M, route);
  const roll = buildRollerStations(staticAssembly, M, c.makeRoller);
  guards(staticAssembly, M);
  staticRouting(staticAssembly, route, M, o.lightY);
  batch(staticAssembly);
  batch(environment);
  Object.values(route).forEach(batch);

  const highlight = { signal: ink(0x75e0b3), power: ink(0xf2b66c), air: ink(0x79b8df) };
  const original = new Map(Object.values(route).flatMap((group) => group.children.map((m) => [m, m.material])));
  function showRoutes(enabled) {
    for (const [key, g] of Object.entries(route)) {
      g.children.forEach((m) => { m.material = enabled ? highlight[key] : original.get(m); });
    }
  }

  const blanks = new Map();
  const plateGeo = new RoundedBoxGeometry(G.plateW, G.plateH, G.plateL, 1, .0011);
  const uv = plateGeo.attributes.uv, pos = plateGeo.attributes.position;
  const upper = plateGeo.groups[2];
  for (let i = upper.start; i < upper.start + upper.count; i++) uv.setXY(i, .5 + pos.getX(i) / G.plateW, .5 + pos.getZ(i) / G.plateL);
  plateGeo.clearGroups();
  plateGeo.addGroup(0, upper.start, 0);
  plateGeo.addGroup(upper.start, upper.count, 1);
  plateGeo.addGroup(upper.start + upper.count, pos.count - upper.start - upper.count, 0);
  const plateMats = Object.fromEntries(Object.entries(kit.steel).map(([code, sets]) => [code, sets.map((set) => {
    const mat = opticalFill(new T.MeshStandardMaterial({ map: set.map, bumpMap: set.map,
      bumpScale: code === 'OK' ? .00018 : .00042, roughness: code === 'OK' ? .43 : .53, metalness: .48 }));
    return [M.rail, mat];
  })]));

  function rebuildBlanks(machine) {
    for (const part of blanks.values()) live.remove(part);
    blanks.clear();
    machine.plates.forEach((p) => {
      const sets = plateMats[p.code];
      const mat = sets ? sets[(p.id - 1) % sets.length] : plateMats['OK'][0];
      const part = mesh(live, plateGeo, mat, [p.x, p.y, p.z], true);
      blanks.set(p.id, part);
    });
  }

  function sync(machine, a, drawings, cameraView) {
    const distance = lerp(machine.prev.distance, machine.distance, a);
    c.setBelt(distance);
    loader(machine, a);
    actuator(machine, a);
    o.setOptics(machine, a, distance, drawings, cameraView);
    roll(machine, a);
    for (const p of machine.plates) {
      const part = blanks.get(p.id);
      if (!part) continue;
      const sets = plateMats[p.code];
      const sameSurface = sets ? sets[(p.id - 1) % sets.length] : plateMats['OK'][0];
      if (part.material !== sameSurface) part.material = sameSurface;
      const f = p.state === 'stock' || p.state === 'settled' ? 1 : a;
      part.position.set(lerp(p.prev.x, p.x, f), lerp(p.prev.y, p.y, f), lerp(p.prev.z, p.z, f));
      const angle = lerp(p.prev.angle, p.angle, f);
      const isChute = p.state === 'pushing' || p.state === 'chute' || (['fall', 'landing', 'settled'].includes(p.state) && p.x > 0.5);
      part.rotation.set(isChute ? 0 : -angle, 0, isChute ? angle : 0);
    }
    if (hmiTex) hmiTex.needsUpdate = true;
  }
  return { showRoutes, sync, rebuildBlanks, blanks };
}

export function factoryLighting(scene, renderer) {
  const dome = new T.Scene();
  dome.background = new T.Color(0x2c3439);
  const cards = [];
  for (const [color, power, size, at] of [[0xf4f3e5, 3.7, [12, 3.3], [-2.0, 6.3, 2.2]], [0xd2e1e4, 2.4, [8.2, 2.1], [3.2, 3.8, -.4]],
    [0xffe1bf, 1.75, [7.0, 2.8], [.7, 2.8, -5.5]], [0xc8d6cb, .65, [7, 4], [-6, 1.1, .8]]]) {
    const panel = new T.Mesh(new T.PlaneGeometry(...size), new T.MeshBasicMaterial({ color: new T.Color(color).multiplyScalar(power), side: T.DoubleSide }));
    panel.position.set(...at); panel.lookAt(0, 0, 0); dome.add(panel); cards.push(panel);
  }
  const pmrem = new T.PMREMGenerator(renderer);
  const reflection = pmrem.fromScene(dome, .075, .1, 30, { size: 128 });
  scene.environment = reflection.texture;
  scene.environmentIntensity = .82;
  pmrem.dispose(); cards.forEach((p) => { p.geometry.dispose(); p.material.dispose(); });

  scene.add(new T.HemisphereLight(0xe4edf0, 0x4a4942, 1.05));
  const main = new T.DirectionalLight(0xfff1df, 2.62);
  main.position.set(-4.7, 7.7, -2.4);
  main.target.position.set(0, .35, .45);
  main.castShadow = true;
  main.shadow.mapSize.set(1536, 1536);
  Object.assign(main.shadow.camera, { left: -5.9, right: 5.9, top: 5.65, bottom: -5.65, near: .5, far: 18 });
  main.shadow.camera.updateProjectionMatrix();
  main.shadow.bias = -.00048;
  main.shadow.normalBias = .012;
  main.shadow.intensity = .79;
  scene.add(main, main.target);
  const rim = new T.DirectionalLight(0xbad6e4, 1.64);
  rim.position.set(5.7, 4.5, -.9);
  const front = new T.DirectionalLight(0xe1ebe0, .76);
  front.position.set(-2.4, 3.25, 5.8);
  scene.add(rim, front);
}