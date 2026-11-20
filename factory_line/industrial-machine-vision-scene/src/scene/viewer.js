import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createCell, factoryLighting } from './cell.js';
import { G, STEP, Machine, clamp, ease, lerp } from './mechanics.js';
import { ScanRaster, surfaces } from './surfaces.js';

export const VIEWS = {
  overview: { pos: [-7.9, 5.57, 5.55], target: [.01, 1.04, .44], height: 5.55, width: 8.25 },
  infeed: { pos: [-3.95, 3.48, -4.83], target: [-.79, 1.12, -2.54], height: 2.94, width: 3.55 },
  optics: { pos: [-1.20, 1.59, 2.20], target: [-.055, 1.255, .050], height: 1.19, width: 1.55 },
  sensors: { pos: [-1.78, 1.56, -1.79], target: [-.21, .815, -.58], height: 1.01, width: 1.06 },
  bay1: { pos: [-2.80, 2.30, 2.80], target: [.30, .65, 0.95], height: 2.2, width: 3.2 },
  bay2: { pos: [-3.20, 2.57, 4.67], target: [.30, .65, 1.80], height: 2.45, width: 3.62 },
  bay3: { pos: [-2.80, 2.30, 5.40], target: [.30, .65, 2.65], height: 2.2, width: 3.2 },
  reject: { pos: [-3.80, 2.85, 4.40], target: [.45, .70, 1.80], height: 2.80, width: 4.20 },
  terminal: { pos: [-2.54, 1.37, -0.57], target: [-1.68, 1.13, -0.97], height: 1.15, width: 1.65 },
  top: { pos: [0, 10, .485], target: [0, .55, .48], height: 8.88, width: 6.10 },
  outfeed: { pos: [-2.10, 2.01, 5.43], target: [-.12, .49, 3.53], height: 1.76, width: 2.31 },
};

const MARKS = {
  infeed: [.34, 2.05, G.pickZ + .097],
  optics: [-.03, 1.90, .11],
  reject: [-.78, .995, G.rejectZ + .27],
  terminal: [-1.68, 1.55, -0.97],
  outfeed: [-.02, .375, 4.28],
};

export class Viewer {
  constructor(host, marks, bridge, onFrame, onView) {
    this.host = host;
    this.marks = marks;
    this.bridge = bridge;
    this.onFrame = onFrame;
    this.onView = onView;
    this.playing = !matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.diagrams = true;
    this.labels = true;
    this.accum = 0;
    this.dirty = true;
    this.view = 'overview';
    this.height = VIEWS.overview.height;
    this.width = VIEWS.overview.width;
    this.time = performance.now();
    this.scene = new T.Scene();
    this.scene.background = new T.Color(0x293036);
    this.scene.fog = new T.FogExp2(0x293036, .049);
    this.renderer = new T.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.10;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = T.PCFShadowMap;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
    this.renderer.domElement.setAttribute('aria-label', 'Poste 3D interactif de contrôle qualité et de tri');
    this.renderer.domElement.addEventListener('webglcontextlost', (event) => {
      event.preventDefault(); this.playing = false;
      host.dispatchEvent(new Event('viewercontextlost'));
    });
    host.appendChild(this.renderer.domElement);
    this.perspCamera = new T.PerspectiveCamera(40, 1, .1, 80);
    this.orthoCamera = new T.OrthographicCamera(-5, 5, 3, -3, .05, 65);
    this.projection = 'perspective';
    this.camera = this.perspCamera;
    this.camera.position.set(...VIEWS.overview.pos);
    this.orbit = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbit.target.set(...VIEWS.overview.target);
    this.orbit.enableDamping = true;
    this.orbit.dampingFactor = .08;
    this.orbit.rotateSpeed = 0.85;
    this.orbit.enablePan = true;
    this.orbit.panSpeed = 0.90;
    this.orbit.screenSpacePanning = true;
    this.orbit.minPolarAngle = 0.08;
    this.orbit.maxPolarAngle = Math.PI / 2 - .045;
    this.orbit.minZoom = .50;
    this.orbit.maxZoom = 4.0;
    this.orbit.zoomToCursor = false;
    this.orbit.maxDistance = 13.0;
    this.orbit.minDistance = 0.8;

    // Prevent browser context menu on canvas so right-click panning works cleanly
    this.renderer.domElement.addEventListener('contextmenu', (e) => e.preventDefault());

    // Configure mouse and touch bindings:
    // Left click = Orbit / Rotate, Right click = Pan, Middle click = Pan
    this.orbit.mouseButtons = {
      LEFT: T.MOUSE.ROTATE,
      MIDDLE: T.MOUSE.PAN,
      RIGHT: T.MOUSE.PAN,
    };
    this.orbit.touches = {
      ONE: T.TOUCH.ROTATE,
      TWO: T.TOUCH.DOLLY_PAN,
    };

    this.orbit.addEventListener('start', () => { this.flight = null; });
    this.orbit.addEventListener('change', () => {
      this.clampOrbitLimits();
      this.invalidate();
    });
    this.orbit.update();
    this.resize = this.resize.bind(this);
    this.observer = new ResizeObserver(this.resize);
    this.observer.observe(host);
    this.resize();

    const kit = surfaces();
    this.raster = new ScanRaster(kit.steel);
    this.machine = new Machine((capture) => this.request(capture));
    factoryLighting(this.scene, this.renderer);
    this.cell = createCell(this.scene, kit);
    this.cell.rebuildBlanks(this.machine);
    this.tick(0, true);
    this.renderer.setAnimationLoop((now) => {
      const dt = clamp((now - this.time) / 1000, 0, .080);
      this.time = now;
      this.tick(dt);
    });
    document.addEventListener('visibilitychange', () => { this.time = performance.now(); });
  }

  async request(capture) {
    const payload = this.raster.payload(capture, this.bridge.stage);
    const result = await this.bridge.inspect(payload, capture.plate.code);
    if (this.machine.resolve(capture.id, result)) this.invalidate();
  }

  invalidate() { this.dirty = true; }

  changeView(name) {
    const preset = VIEWS[name];
    if (!preset) return;
    const fromPos = this.camera.position.clone();
    const toPos = new T.Vector3(...preset.pos);
    const fromTarget = this.orbit.target.clone();
    const toTarget = new T.Vector3(...preset.target);
    const dist = fromPos.distanceTo(toPos);
    this.flight = {
      t: 0,
      fromPos,
      toPos,
      fromTarget,
      toTarget,
      dist,
      height: this.height,
      width: this.width,
      zoom: this.camera.zoom,
      preset,
    };
    this.view = name;
    this.onView(name);
    this.invalidate();
  }

  reset() {
    this.bridge.cancel();
    this.machine.reset();
    this.raster.id = '';
    this.raster.rows = 0;
    this.raster.ctx.clearRect(0, 0, this.raster.canvas.width, this.raster.canvas.height);
    this.accum = 0;
    this.cell.rebuildBlanks(this.machine);
    this.invalidate();
  }

  togglePlay() {
    this.freeze = this.playing ? clamp(this.accum / STEP, 0, 1) : undefined;
    this.playing = !this.playing;
    this.invalidate();
  }

  step() {
    this.playing = false;
    for (let i = 0; i < 6; i++) this.machine.step();
    this.freeze = 1;
    this.accum = 0;
    this.invalidate();
  }

  clampOrbitLimits() {
    const t = this.orbit.target;
    const c = this.camera.position;

    // 1. Restrict panning / orbit target strictly to the machine and sorting cell bounds
    const tx = clamp(t.x, -3.2, 2.6);
    const ty = clamp(t.y, 0.40, 2.20);
    const tz = clamp(t.z, -2.8, 3.8);

    const dx = tx - t.x;
    const dy = ty - t.y;
    const dz = tz - t.z;

    if (dx !== 0 || dy !== 0 || dz !== 0) {
      t.set(tx, ty, tz);
      c.x += dx;
      c.y += dy;
      c.z += dz;
    }

    // 2. Prevent camera eye from clipping through walls or dropping under floor
    if (c.y < 0.22) c.y = 0.22;
    if (c.x > 4.10) c.x = 4.10;
    if (c.z < -5.70) c.z = -5.70;
    if (c.z > 7.50) c.z = 7.50;
    if (c.x < -8.50) c.x = -8.50;
  }

  tick(dt, force = false) {
    if (this.playing) {
      this.accum += dt;
      while (this.accum >= STEP) { this.machine.step(); this.accum -= STEP; }
    }
    if (this.flight) {
      this.invalidate();
      const f = this.flight;
      f.t = Math.min(1, f.t + dt / (matchMedia('(prefers-reduced-motion: reduce)').matches ? .01 : .75));
      const a = ease(f.t);
      this.orbit.target.lerpVectors(f.fromTarget, f.toTarget, a);
      this.camera.position.lerpVectors(f.fromPos, f.toPos, a);
      // Subtle elevation lift on long cross-factory transitions for smooth flyover
      const lift = Math.sin(Math.PI * a) * Math.min(0.40, f.dist * 0.05);
      this.camera.position.y += lift;

      this.camera.lookAt(this.orbit.target);
      this.orbit.update();

      this.perspCamera.position.copy(this.camera.position);
      this.orthoCamera.position.copy(this.camera.position);
      this.height = lerp(f.height, f.preset.height, a);
      this.width = lerp(f.width, f.preset.width, a);
      this.camera.zoom = lerp(f.zoom, 1, a);
      this.perspCamera.zoom = this.camera.zoom;
      this.orthoCamera.zoom = this.camera.zoom;
      this.setFrustum();
      if (f.t === 1) {
        this.flight = null;
        this.clampOrbitLimits();
      }
    } else {
      this.orbit.update(dt);
      this.clampOrbitLimits();
      this.perspCamera.position.copy(this.camera.position);
      this.orthoCamera.position.copy(this.camera.position);
    }
    if (!this.playing && !this.dirty && !force) return;
    this.dirty = false;
    const a = this.playing ? clamp(this.accum / STEP, 0, 1) : this.freeze ?? 1;
    this.cell.sync(this.machine, a, this.diagrams, this.view);
    if (this.machine.scan) this.raster.sample(this.machine.scan);
    this.camera.updateMatrixWorld();
    this.renderMarks();
    this.renderer.render(this.scene, this.camera);
    this.onFrame(this, force || !this.playing);
  }

  renderMarks() {
    const enabled = this.labels && this.view === 'overview' && this.w >= 570 && !this.flight;
    for (const marker of this.marks) {
      const v = MARKS[marker.dataset.mark];
      if (!enabled || !v) { marker.hidden = true; continue; }
      const p = new T.Vector3(...v).project(this.camera);
      marker.hidden = Math.abs(p.x) > .89 || p.y < -.62 || p.y > .64 || Math.abs(p.z) > 1;
      if (!marker.hidden) marker.style.transform = `translate3d(${(p.x * .5 + .5) * this.w}px,${(.5 - p.y * .5) * this.h}px,0)`;
    }
  }

  toggleProjection() {
    return this.setProjection(this.projection === 'perspective' ? 'isometric' : 'perspective');
  }

  setProjection(mode) {
    if (mode !== 'perspective' && mode !== 'isometric') return this.projection;
    this.projection = mode;
    const targetCam = mode === 'perspective' ? this.perspCamera : this.orthoCamera;
    targetCam.position.copy(this.camera.position);
    targetCam.zoom = this.camera.zoom;
    this.camera = targetCam;
    this.orbit.object = this.camera;
    this.setFrustum();
    this.invalidate();
    return this.projection;
  }

  setFrustum() {
    const aspect = this.w / this.h;
    const h = Math.max(this.height, this.width / aspect) * .5;
    const adjusted = h * aspect;
    this.orthoCamera.left = -adjusted;
    this.orthoCamera.right = adjusted;
    this.orthoCamera.top = h;
    this.orthoCamera.bottom = -h;
    this.orthoCamera.updateProjectionMatrix();

    this.perspCamera.aspect = aspect;
    this.perspCamera.updateProjectionMatrix();
  }

  resize() {
    this.w = Math.max(1, this.host.clientWidth);
    this.h = Math.max(1, this.host.clientHeight);
    this.setFrustum();
    this.renderer.setSize(this.w, this.h, false);
    this.invalidate();
  }
}