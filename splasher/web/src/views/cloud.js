// 3D point cloud (Three.js): free navigation (orbit/zoom/pan). Colored by height (z),
// overridden by the class color where the point is labeled. Frame: z is up.
//
// Each instance can filter on a specific cloud channel (`setChannel`) or show all.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { viridis, pretty } from "../colors.js";

const SENSOR_COLOR = 0x3b82f6;

// Text label as a camera-facing sprite (canvas texture).
function makeLabel(text) {
  const c = document.createElement("canvas");
  let ctx = c.getContext("2d");
  ctx.font = "bold 40px ui-sans-serif, system-ui, sans-serif";
  const w = Math.ceil(ctx.measureText(text).width) + 26;
  c.width = w; c.height = 56;
  ctx = c.getContext("2d");
  ctx.font = "bold 40px ui-sans-serif, system-ui, sans-serif";
  ctx.fillStyle = "rgba(8,12,20,0.72)"; ctx.fillRect(0, 0, w, 56);
  ctx.fillStyle = "#cfe0ff"; ctx.textBaseline = "middle"; ctx.fillText(text, 13, 30);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({
    map: new THREE.CanvasTexture(c), depthTest: false, transparent: true,
  }));
  sp.scale.set((w / 56) * 0.9, 0.9, 1);
  return sp;
}

// One sensor marker in its LOCAL frame (forward = +x): forward arrow + shape + label.
function makeMarker(sensor) {
  const g = new THREE.Group();
  const mat = new THREE.LineBasicMaterial({ color: SENSOR_COLOR });

  g.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 0, 0), 1.8, SENSOR_COLOR, 0.5, 0.28));

  if (sensor.kind === "image") {
    // small camera frustum opening forward (+x)
    const d = 1.2, hw = 0.5, hh = 0.35;
    const tip = [0, 0, 0];
    const c = [[d, hw, hh], [d, -hw, hh], [d, -hw, -hh], [d, hw, -hh]];
    const pts = [];
    for (const k of c) { pts.push(tip, k); }
    for (let i = 0; i < 4; i++) pts.push(c[i], c[(i + 1) % 4]);
    const geo = new THREE.BufferGeometry().setFromPoints(pts.map((p) => new THREE.Vector3(...p)));
    g.add(new THREE.LineSegments(geo, mat));
  } else {
    g.add(new THREE.Mesh(
      new THREE.OctahedronGeometry(0.3),
      new THREE.MeshBasicMaterial({ color: SENSOR_COLOR, wireframe: true }),
    ));
  }

  const label = makeLabel(pretty(sensor.name));
  label.position.set(0, 0, 0.7);
  g.add(label);
  return g;
}

export class CloudView {
  constructor(container) {
    this.container = container;
    this.palette = { colors: new Map(), ignore: 0 };
    this.view = null;
    this.channel = null;            // null = all channels, otherwise a cloud_keys index
    this.colorBy = "height";        // "height" (z) | feature index i (column 3+i, if present)
    this.running = true;
    this._framed = false;           // auto-fit the camera on the first non-empty cloud

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x07090c);

    this.camera = new THREE.PerspectiveCamera(55, 1, 0.1, 5000);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-30, -30, 25);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio || 1);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;   // no inertia: the camera stops as soon as you do

    const grid = new THREE.GridHelper(200, 40, 0x2a2a30, 0x18181d);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    this.geom = new THREE.BufferGeometry();
    const mat = new THREE.PointsMaterial({ size: 0.18, vertexColors: true, sizeAttenuation: true });
    this.points = new THREE.Points(this.geom, mat);
    this.points.frustumCulled = false;   // never cull the whole cloud (robust to a bad bounding sphere)
    this.scene.add(this.points);

    this.sensorsGroup = new THREE.Group();      // sensor placement markers (reference)
    this.scene.add(this.sensorsGroup);

    // Ego frame at the origin: X red (forward), Y green (left), Z blue (up) — same
    // convention as the BEV's X/Y arrows, so both views read consistently.
    this.scene.add(new THREE.AxesHelper(2.2));
    const ego = makeLabel("ego");
    ego.position.set(0, 0, 0.9);
    this.scene.add(ego);

    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(container);
    this._resize();
    this._loop();
  }

  setPalette(p) { this.palette = p; }
  setBackground(css) { this.scene.background = new THREE.Color(css); }
  setChannel(ch) { this.channel = ch; this._rebuild(); }

  // sensors: [{ name, kind, placement }] — placement is a 4x4 (nested) ego pose, or null.
  setSensors(sensors) {
    this.sensorsGroup.clear();
    for (const s of sensors || []) {
      const g = makeMarker(s);
      if (s.placement) {
        const m = new THREE.Matrix4().set(...s.placement.flat());   // row-major
        const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
        m.decompose(pos, quat, scl);
        g.position.copy(pos); g.quaternion.copy(quat);
      }
      this.sensorsGroup.add(g);
    }
  }
  setColorBy(mode) { this.colorBy = mode; this._rebuild(); }
  setView(view) { this.view = view; this._rebuild(); }

  // Render a standalone point cloud (file viewer): no labels/channels. `refit=false` keeps
  // the camera (e.g. attaching a measure to an already-framed cloud).
  setRawCloud(points, refit = true) {
    if (refit) this._framed = false;
    this.view = { points };
    this._rebuild();
  }

  _rebuild() {
    const p = this.view && this.view.points;
    if (!p || p.shape[0] === 0) { this.geom.setDrawRange(0, 0); return; }
    const [n, stride] = p.shape;
    const labels = this.view.pointLabels ? this.view.pointLabels.data : null;
    const chans = this.view.pointChannels ? this.view.pointChannels.data : null;
    // Scalar driving the gradient: a feature column (3 + index) when present, else height (z).
    const fCol = typeof this.colorBy === "number" ? 3 + this.colorBy : -1;
    const sCol = fCol >= 0 && fCol < stride ? fCol : 2;
    const keep = (i) => this.channel === null || !chans || chans[i] === this.channel;
    const isFin = Number.isFinite;
    // A point is usable only if its x/y/z are finite (lidar returns can carry NaN/Inf).
    const ok = (i) => keep(i) && isFin(p.data[i * stride]) && isFin(p.data[i * stride + 1])
                      && isFin(p.data[i * stride + 2]);

    // First pass: scalar bounds + count of usable points.
    let slo = Infinity, shi = -Infinity, m = 0;
    for (let i = 0; i < n; i++) {
      if (!ok(i)) continue;
      const s = p.data[i * stride + sCol];
      if (isFin(s)) { if (s < slo) slo = s; if (s > shi) shi = s; }
      m++;
    }
    if (!(shi > slo)) shi = slo + 1;

    const pos = new Float32Array(m * 3), col = new Float32Array(m * 3);
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (!ok(i)) continue;
      pos[k * 3] = p.data[i * stride]; pos[k * 3 + 1] = p.data[i * stride + 1]; pos[k * 3 + 2] = p.data[i * stride + 2];
      const lab = labels ? labels[i] : this.palette.ignore;
      let rgb;
      if (lab !== this.palette.ignore && this.palette.colors.has(lab)) {
        rgb = this.palette.colors.get(lab);
      } else {
        const s = p.data[i * stride + sCol];
        rgb = viridis(isFin(s) ? (s - slo) / (shi - slo) : 0);
      }
      col[k * 3] = rgb[0] / 255; col[k * 3 + 1] = rgb[1] / 255; col[k * 3 + 2] = rgb[2] / 255;
      k++;
    }
    this.geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    this.geom.setAttribute("color", new THREE.BufferAttribute(col, 3));
    this.geom.setDrawRange(0, k);
    this.geom.computeBoundingSphere();
    if (!this._framed && k > 0) { this._framed = true; this._fit(this.geom.boundingSphere); }
  }

  // Frame the camera on the cloud once, so points are visible wherever they sit in space.
  _fit(sphere) {
    if (!sphere || !Number.isFinite(sphere.radius) || sphere.radius <= 0) return;
    const c = sphere.center, r = sphere.radius;
    this.controls.target.copy(c);
    const d = r * 2.2 + 1;
    this.camera.position.set(c.x - d * 0.7, c.y - d * 0.7, c.z + d * 0.6);
    this.camera.near = Math.max(0.05, r / 100);
    this.camera.far = r * 20 + 100;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  _resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _loop() {
    if (!this.running) return;
    requestAnimationFrame(() => this._loop());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.running = false;
    this._ro.disconnect();
    this.geom.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
