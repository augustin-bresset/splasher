// Top-down view (BEV) — the labeling surface.
//
// 2D canvas, world frame (y upwards, locked aspect). Stacked layers: height underlay
// (viridis) · label raster · selection · grid lines · top-down points. Click-drag =
// paint/select; otherwise = pan; wheel = zoom.

import { viridis, finiteRange, hexToRgb } from "../colors.js";

const MAX_POINTS = 12000;

export class BevView {
  constructor(stage, cbs) {
    this.cbs = cbs;                  // { getTool, onRect(rect, button) }
    this.base = document.createElement("canvas");
    this.overlay = document.createElement("canvas");
    stage.append(this.base, this.overlay);
    this.bctx = this.base.getContext("2d");
    this.octx = this.overlay.getContext("2d");
    this.stage = stage;
    this.view = null;
    this.palette = { colors: new Map(), ignore: 0 };
    this.preview = null;             // candidate grid (designer) or null
    this._visChan = null;            // Set of visible cloud-channel indices (BEV dots), or null = all
    this._accent = "#00b8d9";        // theme accent (css) for marquee/preview
    this._accentSel = [87, 232, 255]; // theme accent-hi (rgb) for selection fill / ripples
    this._ripples = [];              // active annotation ripples (splash feedback)
    this._raf = null;

    this.scale = 10; this.cx = 0; this.cy = 0; this._sig = null;
    this._drag = null;               // {mode:'rect'|'pan', x0,y0, ...}

    new ResizeObserver(() => this._resize()).observe(stage);
    this._bindMouse();
    this._resize();
  }

  setPalette(p) { this.palette = p; }

  setView(view) {
    this.view = view;
    const g = view.grid;
    const sig = `${g.xmin},${g.ymin},${g.cols},${g.cell_size}`;
    if (sig !== this._sig) { this._sig = sig; this._fit(); }
    this.render();
  }

  setPreviewGrid(spec) { this.preview = spec; this._renderOverlay(); }
  clearPreview() { this.preview = null; this._renderOverlay(); }

  // Cloud channels shown as top-down reference dots (the "Clouds (BEV)" toggles).
  // Store-only: always paired with a following setView() that triggers the render.
  setVisibleChannels(set) { this._visChan = set; }

  setAccent(accentCss, accentHiCss) {
    if (accentCss) this._accent = accentCss;
    if (accentHiCss) this._accentSel = hexToRgb(accentHiCss);
    if (this.view) this.render();
  }

  // Annotation "splash": a short expanding ring at the action's center.
  ripple(wx, wy) {
    this._ripples.push({ wx, wy, t0: performance.now() });
    if (!this._raf) this._animate();
  }
  _animate() {
    this._renderOverlay();
    this._raf = this._ripples.length ? requestAnimationFrame(() => this._animate()) : null;
  }

  // ----------------------------------------------------------- transform
  get W() { return this.stage.clientWidth; }
  get H() { return this.stage.clientHeight; }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    for (const c of [this.base, this.overlay]) {
      c.width = Math.max(1, this.W * dpr);
      c.height = Math.max(1, this.H * dpr);
    }
    this.bctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.octx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.view) this.render();
  }

  _fit() {
    const g = this.view.grid;
    const w = g.cols * g.cell_size, h = g.rows * g.cell_size;
    this.scale = Math.min(this.W / w, this.H / h) * 0.94 || 10;
    this.cx = g.xmin + w / 2;
    this.cy = g.ymin + h / 2;
  }

  _toScreen(x, y) {
    return [this.W / 2 + (x - this.cx) * this.scale, this.H / 2 - (y - this.cy) * this.scale];
  }
  _toWorld(px, py) {
    return [this.cx + (px - this.W / 2) / this.scale, this.cy - (py - this.H / 2) / this.scale];
  }

  // ----------------------------------------------------------- render
  render() {
    const ctx = this.bctx;
    ctx.clearRect(0, 0, this.W, this.H);
    if (!this.view) return;
    const g = this.view.grid;

    this._drawRaster(this.view.bevField, (v) => {
      if (!Number.isFinite(v)) return null;
      const [lo, hi] = this._fieldRange();
      return [...viridis((v - lo) / (hi - lo)), 210];
    });
    this._drawRaster(this.view.gridLabels, (v) => {
      if (v === this.palette.ignore) return null;
      const c = this.palette.colors.get(v) || [200, 200, 200];
      return [c[0], c[1], c[2], 170];
    });
    this._drawRaster(this.view.selection, (v) => (v ? [...this._accentSel, 95] : null));  // theme accent

    this._drawGrid(g);
    this._drawPoints();
    this._drawEgo();
    this._renderOverlay();
  }

  // Ego frame at the world origin: X (red) = forward, Y (green) = left. Same convention
  // as the 3D view's RGB axes, so the BEV and the 3D views line up at a glance.
  _drawEgo() {
    const [ox, oy] = this._toScreen(0, 0);
    const L = 34;                                  // fixed pixel length (HUD-like)
    this._arrow(ox, oy, ox + L, oy, "#ff6b4a", "x");   // +x → right (forward)
    this._arrow(ox, oy, ox, oy - L, "#46c46a", "y");   // +y → up (left)
    const ctx = this.bctx;
    ctx.fillStyle = "#e7eaf0";
    ctx.beginPath(); ctx.arc(ox, oy, 3.5, 0, Math.PI * 2); ctx.fill();
  }

  _arrow(x0, y0, x1, y1, color, label) {
    const ctx = this.bctx;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    const a = Math.atan2(y1 - y0, x1 - x0), h = 7;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x1 - h * Math.cos(a - 0.4), y1 - h * Math.sin(a - 0.4));
    ctx.lineTo(x1 - h * Math.cos(a + 0.4), y1 - h * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fill();
    ctx.font = "bold 11px ui-sans-serif, system-ui, sans-serif";
    ctx.fillText(label, x1 + 3 * Math.cos(a) + 2, y1 + 3 * Math.sin(a) + 4);
  }

  _fieldRange() {
    if (!this._fr || this._frFor !== this.view.bevField) {
      this._fr = this.view.bevField ? finiteRange(this.view.bevField.data) : [0, 1];
      this._frFor = this.view.bevField;
    }
    return this._fr;
  }

  // Draw a [rows,cols] raster via an offscreen canvas (vertical flip = y upwards).
  _drawRaster(arr, colorFn) {
    if (!arr) return;
    const [rows, cols] = arr.shape;
    const off = document.createElement("canvas");
    off.width = cols; off.height = rows;
    const img = off.getContext("2d").createImageData(cols, rows);
    const px = img.data;
    let any = false;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const c = colorFn(arr.data[i * cols + j]);
        if (!c) continue;
        any = true;
        const o = ((rows - 1 - i) * cols + j) * 4;   // flip: row 0 (ymin) at the bottom
        px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = c[3];
      }
    }
    if (!any) return;
    off.getContext("2d").putImageData(img, 0, 0);

    const g = this.view.grid;
    const [x0, y0] = this._toScreen(g.xmin, g.ymin + rows * g.cell_size);  // top-left corner
    const w = cols * g.cell_size * this.scale, h = rows * g.cell_size * this.scale;
    this.bctx.imageSmoothingEnabled = false;
    this.bctx.drawImage(off, x0, y0, w, h);
  }

  _drawGrid(g) {
    const ctx = this.bctx;
    ctx.lineWidth = 1; ctx.strokeStyle = "rgba(122,124,130,0.40)";
    ctx.beginPath();
    const x1 = g.xmin + g.cols * g.cell_size, y1 = g.ymin + g.rows * g.cell_size;
    for (let k = 0; k <= g.cols; k++) {
      const x = g.xmin + k * g.cell_size;
      const a = this._toScreen(x, g.ymin), b = this._toScreen(x, y1);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    }
    for (let k = 0; k <= g.rows; k++) {
      const y = g.ymin + k * g.cell_size;
      const a = this._toScreen(g.xmin, y), b = this._toScreen(x1, y);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
  }

  _drawPoints() {
    const p = this.view.points;
    if (!p || p.shape[0] === 0) return;
    const [n, stride] = p.shape;
    const chans = this.view.pointChannels ? this.view.pointChannels.data : null;
    const vis = this._visChan;
    const step = Math.max(1, Math.ceil(n / MAX_POINTS));
    this.bctx.fillStyle = "rgba(150,170,205,0.30)";  // cool neutral
    for (let i = 0; i < n; i += step) {
      if (vis && chans && !vis.has(chans[i])) continue;   // BEV honors the cloud toggles
      const [sx, sy] = this._toScreen(p.data[i * stride], p.data[i * stride + 1]);
      this.bctx.fillRect(sx, sy, 1.6, 1.6);
    }
  }

  _renderOverlay() {
    const ctx = this.octx;
    ctx.clearRect(0, 0, this.W, this.H);
    if (this.preview && this.view) this._drawPreviewGrid(this.preview);

    const [r, g, b] = this._accentSel;
    if (this._drag && this._drag.mode === "rect") {
      const d = this._drag;
      const x = Math.min(d.x0, d.x1), y = Math.min(d.y0, d.y1);
      const remove = d.button === 2;             // right button = erase / deselect
      ctx.fillStyle = remove ? "rgba(255,107,74,0.16)" : `rgba(${r},${g},${b},0.16)`;
      ctx.strokeStyle = remove ? "#ff6b4a" : this._accent; ctx.lineWidth = 1;
      ctx.fillRect(x, y, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
      ctx.strokeRect(x, y, Math.abs(d.x1 - d.x0), Math.abs(d.y1 - d.y0));
    }

    if (this._ripples.length) {                  // annotation splash feedback
      const now = performance.now();
      this._ripples = this._ripples.filter((rp) => now - rp.t0 < 520);
      for (const rp of this._ripples) {
        const dt = (now - rp.t0) / 520;
        const [sx, sy] = this._toScreen(rp.wx, rp.wy);
        ctx.beginPath(); ctx.arc(sx, sy, 5 + dt * 40, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${r},${g},${b},${(1 - dt) * 0.7})`;
        ctx.lineWidth = 2; ctx.stroke();
      }
    }
  }

  _drawPreviewGrid(spec) {
    const cols = Math.max(1, Math.ceil((spec.xmax - spec.xmin) / spec.cell_size));
    const rows = Math.max(1, Math.ceil((spec.ymax - spec.ymin) / spec.cell_size));
    const ctx = this.octx;
    const [r, g, b] = this._accentSel;
    ctx.lineWidth = 1; ctx.strokeStyle = `rgba(${r},${g},${b},0.7)`;
    ctx.setLineDash([4, 3]); ctx.beginPath();
    const x1 = spec.xmin + cols * spec.cell_size, y1 = spec.ymin + rows * spec.cell_size;
    for (let k = 0; k <= cols; k++) {
      const x = spec.xmin + k * spec.cell_size;
      const a = this._toScreen(x, spec.ymin), b = this._toScreen(x, y1);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    }
    for (let k = 0; k <= rows; k++) {
      const y = spec.ymin + k * spec.cell_size;
      const a = this._toScreen(spec.xmin, y), b = this._toScreen(x1, y);
      ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke(); ctx.setLineDash([]);
  }

  // ----------------------------------------------------------- mouse
  _bindMouse() {
    const el = this.overlay;
    const pos = (e) => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

    // No context menu: the right button is used to erase / deselect on the BEV.
    el.addEventListener("contextmenu", (e) => e.preventDefault());

    el.addEventListener("mousedown", (e) => {
      const [x, y] = pos(e);
      if (e.shiftKey || e.button === 1) {            // pan: Shift+drag, or middle button
        e.preventDefault();
        this._drag = { mode: "pan", x0: x, y0: y };
      } else if (e.button === 0 || e.button === 2) { // action rect: left = apply, right = remove
        this._drag = { mode: "rect", x0: x, y0: y, x1: x, y1: y, button: e.button };
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!this._drag) return;
      const [x, y] = pos(e);
      if (this._drag.mode === "rect") { this._drag.x1 = x; this._drag.y1 = y; this._renderOverlay(); }
      else {
        this.cx -= (x - this._drag.x0) / this.scale;
        this.cy += (y - this._drag.y0) / this.scale;
        this._drag.x0 = x; this._drag.y0 = y; this.render();
      }
    });

    window.addEventListener("mouseup", () => {
      const d = this._drag; this._drag = null;
      if (!d || d.mode !== "rect") { this._renderOverlay(); return; }
      this._renderOverlay();
      const [wx0, wy0] = this._toWorld(d.x0, d.y0), [wx1, wy1] = this._toWorld(d.x1, d.y1);
      this.cbs.onRect([wx0, wy0, wx1, wy1], d.button);   // button 0 = apply, 2 = remove
    });

    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [x, y] = pos(e);
      const [wx, wy] = this._toWorld(x, y);
      this.scale *= Math.exp(-e.deltaY * 0.0015);
      this.cx = wx - (x - this.W / 2) / this.scale;
      this.cy = wy + (y - this.H / 2) / this.scale;
      this.render();
    }, { passive: false });
  }
}
