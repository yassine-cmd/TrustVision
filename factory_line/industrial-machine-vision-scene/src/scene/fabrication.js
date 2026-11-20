import * as T from 'three';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { carrierLength, carrierPose } from './mechanics.js';
import { texture } from './surfaces.js';

const Y = new T.Vector3(0, 1, 0);
const AXES = { x: new T.Vector3(1, 0, 0), y: Y, z: new T.Vector3(0, 0, 1) };
export const v = (p) => new T.Vector3(...p);
export const matte = (color, roughness = .52, metalness = .22) => new T.MeshStandardMaterial({ color, roughness, metalness });
export const ink = (color) => new T.MeshBasicMaterial({ color, toneMapped: false });

export function mesh(g, geo, material, at = [0, 0, 0], shadow = true) {
  const m = new T.Mesh(geo, material);
  m.position.set(...at);
  m.castShadow = shadow;
  m.receiveShadow = true;
  g.add(m);
  return m;
}

export function box(g, size, at, material, bevel = 0, shadow = true) {
  return mesh(g, bevel ? new RoundedBoxGeometry(...size, 1, bevel) : new T.BoxGeometry(...size), material, at, shadow);
}

export function pin(g, radius, length, at, material, axis = 'y', sides = 20, radiusBottom = radius) {
  const m = mesh(g, new T.CylinderGeometry(radius, radiusBottom, length, sides), material, at);
  if (axis !== 'y') m.quaternion.setFromUnitVectors(Y, AXES[axis]);
  return m;
}

export function bar(g, a, b, width, depth, material, bevel = 0) {
  const start = v(a), end = v(b), delta = end.clone().sub(start);
  const m = box(g, [width, delta.length(), depth], start.clone().lerp(end, .5).toArray(), material, bevel);
  m.quaternion.setFromUnitVectors(Y, delta.normalize());
  return m;
}

export function extrusion(g, size, length, at, material, axis = 'z') {
  const a = size / 2, b = size * .15, d = size * .135;
  const corners = [[-a, -a], [-b, -a], [-b, -a + d], [b, -a + d], [b, -a], [a, -a],
    [a, -b], [a - d, -b], [a - d, b], [a, b], [a, a], [b, a], [b, a - d], [-b, a - d],
    [-b, a], [-a, a], [-a, b], [-a + d, b], [-a + d, -b], [-a, -b]];
  const shape = new T.Shape(corners.map(([x, y]) => new T.Vector2(x, y)));
  shape.closePath();
  const geo = new T.ExtrudeGeometry(shape, { depth: length, bevelEnabled: true, bevelThickness: .0012, bevelSize: .0012, bevelSegments: 1 });
  geo.translate(0, 0, -length / 2);
  const m = mesh(g, geo, material, at);
  if (axis === 'x') m.rotation.y = Math.PI / 2;
  if (axis === 'y') m.rotation.x = -Math.PI / 2;
  return m;
}

export function fitting(g, at, axis, metal, boot, size = .009, direction = 1) {
  const shift = (n) => v(at).addScaledVector(AXES[axis], direction * n).toArray();
  pin(g, size, size * .62, at, metal, axis, 6);
  pin(g, size * .68, size * 1.4, shift(size * .7), boot, axis, 12);
}

export function wire(g, points, radius, material, corner = .034) {
  const path = new T.CurvePath();
  const p = points.map(v);
  let a = p[0];
  for (let i = 1; i < p.length - 1; i++) {
    const bend = p[i];
    const r = Math.min(corner, bend.distanceTo(p[i - 1]) * .45, bend.distanceTo(p[i + 1]) * .45);
    const entry = bend.clone().lerp(p[i - 1], r / bend.distanceTo(p[i - 1]));
    const exit = bend.clone().lerp(p[i + 1], r / bend.distanceTo(p[i + 1]));
    if (a.distanceTo(entry) > .00001) path.add(new T.LineCurve3(a, entry));
    path.add(new T.QuadraticBezierCurve3(entry, bend, exit));
    a = exit;
  }
  path.add(new T.LineCurve3(a, p.at(-1)));
  const geo = new T.TubeGeometry(path, Math.max(14, Math.min(96, Math.ceil(path.getLength() / .027))), radius, 6, false);
  return mesh(g, geo, material, [0, 0, 0], false);
}

export function tag(g, size, at, title, subtitle = '', opts = {}) {
  const { color = '#e5e7e4', fill = '#1a2225', accent = '#e9a343', angleY = 0, angleX = 0, width = 384 } = opts;
  const tex = texture((ctx, w, h) => {
    ctx.fillStyle = fill; ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = accent; ctx.fillRect(0, 0, Math.ceil(w * .032), h);
    ctx.fillStyle = color; ctx.font = `bold ${Math.round(h * (subtitle ? .39 : .59))}px sans-serif`;
    ctx.fillText(title, w * .068, h * (subtitle ? .49 : .71), w * .875);
    if (subtitle) {
      ctx.fillStyle = accent; ctx.font = `bold ${Math.round(h * .20)}px monospace`;
      ctx.fillText(subtitle, w * .070, h * .80, w * .87);
    }
  }, width, Math.max(32, Math.round(width * size[1] / size[0])));
  const material = new T.MeshStandardMaterial({
    map: tex,
    roughness: .73,
    emissive: 0x101412,
    emissiveMap: tex,
    emissiveIntensity: .55,
    transparent: fill === 'transparent',
    side: T.DoubleSide,
  });
  const m = mesh(g, new T.PlaneGeometry(...size), material, at, false);
  m.rotation.set(angleX, angleY, 0, 'YXZ');
  return m;
}

export function batch(g) {
  g.updateMatrixWorld(true);
  const inverse = g.matrixWorld.clone().invert(), buckets = new Map(), old = new Set();
  g.traverse((obj) => {
    if (!obj.isMesh || Array.isArray(obj.material)) return;
    const key = `${obj.material.id}/${obj.castShadow ? 1 : 0}/${obj.receiveShadow ? 1 : 0}`;
    if (!buckets.has(key)) buckets.set(key, { props: obj, list: [] });
    const geometry = obj.geometry.index ? obj.geometry.toNonIndexed() : obj.geometry.clone();
    geometry.applyMatrix4(inverse.clone().multiply(obj.matrixWorld));
    buckets.get(key).list.push(geometry);
    old.add(obj.geometry);
  });
  g.clear();
  for (const { props, list } of buckets.values()) {
    const found = list.length > 1 ? mergeGeometries(list) : list[0];
    mesh(g, found, props.material, [0, 0, 0], props.castShadow).receiveShadow = props.receiveShadow;
    if (list.length > 1) list.forEach((p) => p.dispose());
  }
  old.forEach((geo) => geo.dispose());
}

export function dragChain(g, params, linkWidth, thick, pitchHint, material, side) {
  const length = carrierLength(params);
  const n = Math.ceil(length / pitchHint), pitch = length / n;
  const carrier = new T.Group();
  box(carrier, [pitch * .89, thick, .011], [0, 0, linkWidth / 2 - .005], material, .0024);
  box(carrier, [pitch * .89, thick, .011], [0, 0, -linkWidth / 2 + .005], material, .0024);
  box(carrier, [pitch * .25, thick * .40, linkWidth], [0, 0, 0], material, .0018);
  pin(carrier, thick * .18, linkWidth + .001, [0, 0, 0], side, 'z', 10);
  batch(carrier);
  const instances = carrier.children.map((m) => {
    const links = new T.InstancedMesh(m.geometry, m.material, n);
    links.frustumCulled = false;
    links.castShadow = true;
    links.receiveShadow = true;
    links.instanceMatrix.setUsage(T.DynamicDrawUsage);
    g.add(links);
    return links;
  });
  const part = new T.Object3D();
  let last = NaN;
  return (moving) => {
    if (last === moving) return;
    last = moving;
    for (let i = 0; i < n; i++) {
      const p = carrierPose(params, moving, (i + .5) * pitch);
      part.position.set(p.x, p.y, 0); part.rotation.z = p.angle; part.updateMatrix();
      instances.forEach((m) => m.setMatrixAt(i, part.matrix));
    }
    instances.forEach((m) => { m.instanceMatrix.needsUpdate = true; });
  };
}