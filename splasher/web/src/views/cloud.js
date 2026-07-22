// 3D point cloud on projector's shared octree engine (fast LOD): free navigation
// (orbit/zoom/pan). Colored by height (z), overridden by the class color where the
// point is labeled — the labeling stays exactly what the old plain-three view showed,
// only the renderer underneath changed. Frame: z is up.
//
// Each instance can filter on a specific cloud channel (`setChannel`) or show all.
// The engine (../../engine via `/engine`, served by the server from projector's install)
// owns the renderer, camera, LOD octree and ground grid; this view owns splasher's
// semantics: which points to keep, how to color them, and the sensor/ego markers.

import * as THREE from "three";
import { Viewer } from "/engine/viewer.js";
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
    this._framed = false;           // auto-fit the camera on the first non-empty cloud
    this._disposed = false;

    // The engine owns renderer/camera/controls/LOD-octree/ground-grid and renders on
    // demand (parks at idle). Orbit style keeps the world's Z upright, matching the BEV.
    this.viewer = new Viewer(container);
    this.viewer.setControlStyle("orbit");
    this.viewer.setBackground("#07090c");
    this.viewer.setSizeAttenuation(true);   // point size is metres, shrinks with distance
    this.viewer.setPointSize(0.18);

    // Splasher furniture, added onto the engine's scene: sensor placement markers and the
    // ego frame at the origin (X red forward, Y green left, Z blue up — same convention as
    // the BEV's X/Y arrows, so both views read consistently).
    this.sensorsGroup = new THREE.Group();
    this.viewer.scene.add(this.sensorsGroup);

    this._ego = new THREE.AxesHelper(2.2);
    this.viewer.scene.add(this._ego);
    this._egoLabel = makeLabel("ego");
    this._egoLabel.position.set(0, 0, 0.9);
    this.viewer.scene.add(this._egoLabel);
    this.viewer.requestRender();

    // Panels resize via splitters (no window resize), so drive the engine from a
    // container observer.
    this._ro = new ResizeObserver(() => this.viewer.resize());
    this._ro.observe(container);
  }

  setPalette(p) { this.palette = p; this._rebuild(); }
  setBackground(css) { this.viewer.setBackground(css); }
  setChannel(ch) { this.channel = ch; this._rebuild(); }

  // sensors: [{ name, kind, placement }] — placement is a 4x4 (nested) ego pose, or null.
  setSensors(sensors) {
    this._clearSensors();
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
    this.viewer.requestRender();
  }

  // Free GPU resources held by the sensor markers (geometries, materials, label textures);
  // `Group.clear()` alone would leak them on every setSensors call.
  _clearSensors() {
    this.sensorsGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) {
        if (o.material.map) o.material.map.dispose();
        o.material.dispose();
      }
    });
    this.sensorsGroup.clear();
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

  // Rebuild positions + colors from the current view/channel/coloring and hand them to the
  // engine. This is where splasher's labeling shows: a labeled point takes its class color,
  // every other point the viridis height/feature gradient — identical to the old view.
  _rebuild() {
    if (this._disposed) return;
    const p = this.view && this.view.points;
    if (!p || p.shape[0] === 0) {
      this.viewer.setCloud(new Float32Array(0), { octree: false, frame: false });
      return;
    }
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

    // Only the kept points are fed (all visible), so every alpha is 1 — the engine treats
    // alpha < 0.5 as invisible/unpickable, but this view pre-filters instead of masking.
    // Frame the camera once, on the first non-empty cloud; later rebuilds leave it put.
    const doFrame = !this._framed && k > 0;
    this.viewer.setCloud(pos, { frame: doFrame });
    this.viewer.setColors(col, new Float32Array(m).fill(1));
    if (doFrame) this._framed = true;
  }

  dispose() {
    this._disposed = true;
    this._ro.disconnect();
    // Free splasher's furniture (the engine's dispose only frees what it owns).
    this._clearSensors();
    this.viewer.scene.remove(this.sensorsGroup);
    this.viewer.scene.remove(this._ego);
    this.viewer.scene.remove(this._egoLabel);
    this._ego.geometry.dispose(); this._ego.material.dispose();
    this._egoLabel.material.map.dispose(); this._egoLabel.material.dispose();
    this.viewer.dispose();
  }
}
