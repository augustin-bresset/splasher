// View manager: addable/closable/resizable 3D and camera panels, each bound to a
// channel chosen among those available.
//
// Camera panels drive image visibility (the server only sends the cameras actually
// shown) via the `onChange` callback. 3D panels filter points by channel on the
// client side (no server round-trip).

import { CloudView } from "./views/cloud.js";
import { drawCamera } from "./views/cameras.js";
import { growGutter } from "./resize.js";
import { pretty } from "./colors.js";

export class PanelManager {
  constructor(stackEl, { session, palette, onChange, onFiles = null, sensors = [] }) {
    this.stack = stackEl;
    this.session = session;
    this.palette = palette;
    this.onChange = onChange;
    this.onFiles = onFiles;          // called when file views are added/removed
    this.sensors = sensors;
    this.panels = [];
    this.view = null;
    this._bg = null;
    this._seq = 0;
  }

  setPalette(palette) {
    this.palette = palette;
    for (const p of this.panels) if (p.view) p.view.setPalette(palette);
  }

  setBackground(css) {
    this._bg = css;
    for (const p of this.panels) if (p.view) p.view.setBackground(css);
  }

  setSensors(sensors) {
    this.sensors = sensors;
    for (const p of this.panels) if (p.view) p.view.setSensors(sensors);
  }

  // Camera channels actually displayed (for /api/visibility).
  cameraChannels() {
    return [...new Set(this.panels.filter((p) => p.type === "cam").map((p) => p.channel))];
  }

  // Paths of files currently open in views — including attached measure files — to flag
  // them in the file browser.
  openFilePaths() {
    const out = new Set();
    for (const p of this.panels) {
      if (p.path) out.add(p.path);
      for (const f of p.features || []) out.add(f);
    }
    return out;
  }

  // File views currently open (for the browser's "Open views" side list + workspace).
  openFiles() {
    return this.panels.filter((p) => p.path)
      .map((p) => ({ id: p.id, name: p.name, path: p.path, type: p.type, features: p.features || [] }));
  }

  add(spec, silent = false) {
    const id = ++this._seq;
    const type = spec.type;
    const el = document.createElement("section");
    el.className = "vpanel";
    el.style.flex = "1 1 0";

    const head = document.createElement("div");
    head.className = "vpanel-head";
    const tag = document.createElement("span");
    tag.className = "vpanel-title";
    tag.textContent = type === "cloud" ? "3D" : "Cam";
    const sel = document.createElement("select");
    const close = document.createElement("button");
    close.className = "tbtn close";
    close.textContent = "✕";
    close.title = "Close view";

    // 3D panels also get a "color by" selector (height + each per-point feature, if any).
    let colorSel = null;
    if (type === "cloud") {
      colorSel = document.createElement("select");
      colorSel.className = "color-sel";
      colorSel.title = "Color by";
      this._fillSelect(colorSel, this._colorOpts(this.session.feature_names));
    }
    head.append(tag, sel, ...(colorSel ? [colorSel] : []), close);

    const body = document.createElement("div");
    body.className = "vpanel-body " + (type === "cloud" ? "cloud-body" : "cam-body");

    el.append(head, body);

    const panel = { id, type, channel: spec.channel ?? null, el, sel, colorSel, body, view: null, canvas: null, empty: null };

    if (type === "cloud") {
      this._fillSelect(sel, [["", "All clouds"], ...this.session.cloud_keys.map((k, i) => [String(i), pretty(k)])]);
      sel.value = panel.channel === null ? "" : String(panel.channel);
      panel.view = new CloudView(body);
      panel.view.setPalette(this.palette);
      if (this._bg) panel.view.setBackground(this._bg);
      panel.view.setChannel(panel.channel);
      panel.view.setSensors(this.sensors);
      colorSel.onchange = () => panel.view.setColorBy(this._parseColorBy(colorSel.value));
    } else {
      this._fillSelect(sel, this.session.image_keys.map((k) => [k, pretty(k)]));
      if (panel.channel === null) panel.channel = this.session.image_keys[0];
      sel.value = panel.channel;
      panel.canvas = document.createElement("canvas");
      panel.empty = document.createElement("span");
      panel.empty.className = "empty";
      panel.empty.textContent = "(camera unavailable)";
      body.append(panel.canvas, panel.empty);
    }

    sel.onchange = () => this._onSelect(panel);
    close.onclick = () => this.remove(id);

    this.panels.push(panel);
    this._relayout();
    if (this.view) this._render(panel);
    if (type === "cam" && !silent) this.onChange();
    return panel;
  }

  remove(id) {
    const i = this.panels.findIndex((p) => p.id === id);
    if (i < 0) return;
    const [p] = this.panels.splice(i, 1);
    if (p.view) p.view.dispose();
    this._relayout();
    if (p.type === "cam") this.onChange();
    if (p.path && this.onFiles) this.onFiles();
  }

  _onSelect(panel) {
    if (panel.type === "cloud") {
      panel.channel = panel.sel.value === "" ? null : +panel.sel.value;
      panel.view.setChannel(panel.channel);
    } else {
      panel.channel = panel.sel.value;
      this.onChange();            // the chosen camera must be sent by the server
    }
  }

  // Add a standalone view from an opened file (file viewer): not tied to the dataset.
  // `features`: measure files attached to the cloud (already merged into spec.points).
  addFile(spec, features = []) {
    const id = ++this._seq;
    const isCloud = spec.kind === "cloud";
    const el = document.createElement("section");
    el.className = "vpanel"; el.style.flex = "1 1 0";

    const head = document.createElement("div"); head.className = "vpanel-head";
    const tag = document.createElement("span"); tag.className = "vpanel-title";
    tag.textContent = isCloud ? "3D" : "Img";
    const name = document.createElement("span"); name.className = "vpanel-name";
    name.textContent = spec.name; name.title = spec.path || spec.name;
    const close = document.createElement("button");
    close.className = "tbtn close"; close.textContent = "✕"; close.title = "Close view";

    const body = document.createElement("div");
    body.className = "vpanel-body " + (isCloud ? "cloud-body" : "cam-body");
    const panel = { id, type: isCloud ? "file-cloud" : "file-image", el, body, view: null,
                    name: spec.name, path: spec.path };

    if (isCloud) {
      panel.features = features;
      panel.featureNames = spec.feature_names || [];
      panel.nPoints = spec.points ? spec.points.shape[0] : 0;
      const colorSel = document.createElement("select");
      colorSel.className = "color-sel"; colorSel.title = "Color by";
      this._fillSelect(colorSel, this._colorOpts(spec.feature_names));
      panel.colorSel = colorSel;
      head.append(tag, name, colorSel, close);
      panel.view = new CloudView(body);
      panel.view.setPalette(this.palette);
      if (this._bg) panel.view.setBackground(this._bg);
      panel.view.setSensors([]);
      panel.view.setRawCloud(spec.points);
      colorSel.onchange = () => panel.view.setColorBy(this._parseColorBy(colorSel.value));
    } else if (spec.image) {
      // numpy image array (e.g. .npy HxWxC) → draw on a canvas
      head.append(tag, name, close);
      const canvas = document.createElement("canvas");
      canvas.className = "file-img";
      body.appendChild(canvas);
      drawCamera(canvas, spec.image);
    } else {
      // raster image file (.png/.jpg/…) → let the browser decode it
      head.append(tag, name, close);
      const img = document.createElement("img");
      img.className = "file-img";
      const err = document.createElement("span");
      err.className = "empty"; err.style.display = "none";
      img.onerror = () => { img.style.display = "none"; err.style.display = ""; err.textContent = "cannot display this image"; };
      img.src = "/api/fs/raw?path=" + encodeURIComponent(spec.path);
      body.append(img, err);
    }
    close.onclick = () => this.remove(id);
    el.append(head, body);
    this.panels.push(panel);
    this._relayout();
    if (this.onFiles) this.onFiles();
    return panel;
  }

  // Replace an open file-cloud's data (e.g. after attaching a per-point measure file):
  // same panel, same camera; refreshed columns + "color by" options. `colorBy` (a feature
  // name) selects the coloring — e.g. the freshly attached measure.
  updateFileCloud(id, spec, features, colorBy = null) {
    const p = this.panels.find((q) => q.id === id);
    if (!p || p.type !== "file-cloud") return;
    const prev = p.colorSel.value === "height" ? "height" : (p.featureNames || [])[+p.colorSel.value];
    p.features = features;
    p.featureNames = spec.feature_names || [];
    p.nPoints = spec.points ? spec.points.shape[0] : 0;
    this._fillSelect(p.colorSel, this._colorOpts(p.featureNames));
    const want = colorBy && p.featureNames.includes(colorBy) ? colorBy : prev;
    const idx = p.featureNames.indexOf(want);
    p.colorSel.value = idx >= 0 ? String(idx) : "height";
    p.view.setColorBy(this._parseColorBy(p.colorSel.value));
    p.view.setRawCloud(spec.points, false);
    if (this.onFiles) this.onFiles();
  }

  // Open cloud views (for the "Clouds (BEV)" selector → session source, and to match a
  // lone measure file to a cloud by point count).
  openClouds() {
    return this.panels.filter((p) => p.type === "file-cloud")
      .map((p) => ({ id: p.id, name: p.name, path: p.path, features: p.features || [],
                     featureNames: p.featureNames || [], nPoints: p.nPoints }));
  }

  update(view) {
    this.view = view;
    for (const p of this.panels) this._render(p);
  }

  _render(panel) {
    if (panel.type === "cloud") { panel.view.setView(this.view); return; }
    if (panel.type !== "cam") return;     // file-* panels are static, not driven by the dataset
    const img = this.view.images[panel.channel];
    if (img) { drawCamera(panel.canvas, img); panel.canvas.style.display = ""; panel.empty.style.display = "none"; }
    else { panel.canvas.style.display = "none"; panel.empty.style.display = ""; }
  }

  // Rebuild the stack (panels + handles) by re-inserting the existing nodes.
  _relayout() {
    this.stack.replaceChildren();
    this.panels.forEach((p, i) => {
      if (i > 0) {
        const g = document.createElement("div");
        g.className = "gutter gutter-h";
        this.stack.appendChild(g);
        growGutter(g, this.panels[i - 1].el, p.el, "y", 90);
      }
      this.stack.appendChild(p.el);
    });
  }

  _fillSelect(sel, pairs) {
    sel.replaceChildren();
    for (const [value, label] of pairs) {
      const o = document.createElement("option");
      o.value = value; o.textContent = label;
      sel.appendChild(o);
    }
  }

  // "Color by" options: height + one entry per per-point feature (value = its column index).
  _colorOpts(featureNames) {
    const opts = [["height", "Height"]];
    (featureNames || []).forEach((n, i) => opts.push([String(i), pretty(n)]));
    return opts;
  }

  _parseColorBy(value) {
    return value === "height" ? "height" : +value;   // feature index → CloudView column offset
  }
}
