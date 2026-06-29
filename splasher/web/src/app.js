// Web front orchestration: boot, wire the controls to the API commands, render each
// ViewState. The business logic lives in the Session (server); here we only present,
// route, and manage the layout (view panels + resizing).

import { api } from "./api.js";
import { buildLut, rgbCss, pretty, rgbToHex, hexToRgb } from "./colors.js";
import { BevView } from "./views/bev.js";
import { PanelManager } from "./panels.js";
import { basisGutter, growGutter } from "./resize.js";

const $ = (id) => document.getElementById(id);

// Color themes (CSS variable sets live in style.css under html[data-theme="…"]).
const THEMES = [["aqua", "Aqua"], ["pixel", "Pixel"]];

let session = null;
let view = null;
let bev = null, manager = null, lut = null;
let playTimer = null;
let fsPath = null;        // current directory in the file browser
let fileMode = false;     // empty launch → file-viewer mode (clouds = references, grid persists)

function currentTheme() {
  let t = null;
  try { t = localStorage.getItem("splasher-theme"); } catch { /* ignore */ }
  return THEMES.some(([v]) => v === t) ? t : "aqua";   // fall back to aqua (e.g. stale value)
}
function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  try { localStorage.setItem("splasher-theme", name); } catch { /* ignore */ }
  const css = getComputedStyle(document.documentElement);
  const bg = css.getPropertyValue("--canvas-bg").trim();
  if (manager && bg) manager.setBackground(bg);   // 3D scenes paint their own background
  if (bev) bev.setAccent(css.getPropertyValue("--accent").trim(),
                         css.getPropertyValue("--accent-hi").trim());
}
function buildThemeSelect() {
  const sel = $("theme");
  sel.replaceChildren();
  for (const [value, label] of THEMES) {
    const o = document.createElement("option");
    o.value = value; o.textContent = label;
    sel.appendChild(o);
  }
  sel.value = currentTheme();
  sel.onchange = () => applyTheme(sel.value);
}

// Sensor markers (lidars/cameras) for the 3D views, from the channel placements.
function sensorsFromSession() {
  return session.channels
    .filter((c) => c.kind === "pointcloud" || c.kind === "image")
    .map((c) => ({ name: c.name, kind: c.kind, placement: c.placement }));
}

async function boot() {
  session = await api.session();
  lut = buildLut(session.labelset);

  bev = new BevView($("bev-stage"), { getTool: () => (view ? view.tool : "paint"), onRect: onBevRect });
  bev.setPalette(lut);

  fileMode = session.n_frames === 0;   // launched empty → file-viewer mode

  manager = new PanelManager($("views-stack"), {
    session, palette: lut, onChange: refreshVisibility, onFiles: onFilesChanged,
    sensors: sensorsFromSession(),
  });

  buildClasses();
  buildClouds();
  buildAddBar();
  buildThemeSelect();
  applyTheme(currentTheme());     // sets data-theme + 3D background before panels are added
  setupRanges();
  wireControls();
  wireGutters();

  // Default views (dataset mode only): one 3D cloud (all) + one camera (the first).
  if (!fileMode) {
    if (session.cloud_keys.length) manager.add({ type: "cloud", channel: null }, true);
    if (session.image_keys.length) manager.add({ type: "cam", channel: session.image_keys[0] }, true);
  }

  // Initial visibility (checked clouds + shown cameras) → first ViewState.
  const first = await api.cmd("/api/visibility", { clouds: pickClouds(), images: manager.cameraChannels() });
  apply(first);
  fillGridForm(first.grid);
  updateDims(readGridForm());

  await restoreWorkspace();                       // reopen views + restore layout from last session
  window.addEventListener("beforeunload", saveWorkspace);
}

function apply(v) {
  view = v;
  const visIdx = new Set(v.visibleClouds.map((name) => session.cloud_keys.indexOf(name)).filter((i) => i >= 0));
  bev.setVisibleChannels(visIdx);   // BEV dots honor the cloud toggles (3D stays independent)
  bev.setView(v);
  manager.update(v);
  syncControls(v);
}

async function run(promise) {
  try { apply(await promise); }
  catch (e) { $("status").textContent = "⚠ " + e.message; }
}

function refreshVisibility() {
  return run(api.cmd("/api/visibility", { clouds: pickClouds(), images: manager.cameraChannels() }));
}

// BEV rectangle action: left button (0) applies, right button (2) removes.
function onBevRect(rect, button) {
  bev.ripple((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2);    // splash feedback (all themes)
  const tool = view ? view.tool : "paint";
  if (tool === "select") run(api.cmd("/api/select", { rect, op: button === 2 ? "subtract" : "add" }));
  else if (button === 2) run(api.cmd("/api/erase", { rect }));
  else run(api.cmd("/api/paint", { rect }));
}

// ------------------------------------------------------------- UI building
function buildClasses() {
  const box = $("classes");
  box.innerHTML = "";
  for (const c of session.labelset.classes) {
    if (c.id === session.labelset.ignore_id) continue;
    const el = document.createElement("div");
    el.className = "class-item"; el.dataset.id = c.id;
    el.innerHTML = `<span class="swatch" style="background:${rgbCss(c.color)}"></span>
                    <span>${c.id} · ${pretty(c.name)}</span>`;
    el.onclick = () => run(api.cmd("/api/class", { id: c.id }));
    box.appendChild(el);
  }
}

function buildClouds() {
  const box = $("clouds");
  box.innerHTML = "";
  if (fileMode) {
    // file-viewer: pick which open lidar views compose the BEV (combined = accumulate).
    const clouds = manager.openClouds();
    if (!clouds.length) { box.innerHTML = `<span class="muted">open a cloud file</span>`; return; }
    for (const c of clouds) {
      const row = document.createElement("label");
      row.className = "channel-item";
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = true; cb.dataset.path = c.path;
      cb.onchange = updateSourceFromClouds;
      row.append(cb, document.createTextNode(" " + pretty(c.name)));
      box.appendChild(row);
    }
    return;
  }
  if (!session.cloud_keys.length) { box.innerHTML = `<span class="muted">no cloud</span>`; return; }
  for (const name of session.cloud_keys) {
    const row = document.createElement("label");
    row.className = "channel-item";
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true; cb.dataset.name = name;
    cb.onchange = refreshVisibility;
    row.append(cb, document.createTextNode(" " + pretty(name)));
    box.appendChild(row);
  }
}

function pickClouds() {
  return [...$("clouds").querySelectorAll("input[data-name]:checked")].map((cb) => cb.dataset.name);
}

// File-viewer: send the checked open clouds as the (labelable) session source.
function updateSourceFromClouds() {
  const paths = [...$("clouds").querySelectorAll("input[data-path]:checked")].map((cb) => cb.dataset.path);
  const clouds = manager.openClouds();
  const first = clouds.find((c) => paths.includes(c.path));
  if (first) $("export-name").value = first.name.replace(/\.[^.]+$/, "") + "_bev.npy";
  run(api.cmd("/api/source/files", { paths }));
}

// Open/close of a file view → refresh the clouds selector, the source, and the open-list.
function onFilesChanged() {
  if (fileMode) { buildClouds(); updateSourceFromClouds(); }
  renderOpenViews();
  saveWorkspace();
}

// ------------------------------------------------------------- workspace persistence
const WS_KEY = "splasher-workspace";
function saveWorkspace() {
  if (!manager) return;
  try {
    const grow = (sel) => parseFloat(getComputedStyle(document.querySelector(sel)).flexGrow) || 1;
    localStorage.setItem(WS_KEY, JSON.stringify({
      files: manager.openFiles().map((f) => f.path),
      dir: fsPath,
      layout: {
        rail: document.querySelector(".rail").getBoundingClientRect().width,
        bev: grow(".bev-panel"), views: grow(".views"),
      },
    }));
  } catch { /* ignore */ }
}
async function restoreWorkspace() {
  let ws;
  try { ws = JSON.parse(localStorage.getItem(WS_KEY) || "null"); } catch { ws = null; }
  if (!ws) return;
  if (ws.dir) fsPath = ws.dir;
  if (ws.layout) {
    const L = ws.layout;
    if (L.rail) document.querySelector(".rail").style.flex = `0 0 ${L.rail}px`;
    if (L.bev) document.querySelector(".bev-panel").style.flexGrow = L.bev;
    if (L.views) document.querySelector(".views").style.flexGrow = L.views;
  }
  if (fileMode && ws.files && ws.files.length) {
    const cb = manager.onFiles; manager.onFiles = null;        // bulk: refresh once at the end
    for (const path of ws.files) {
      try { manager.addFile(await api.fsOpen(path)); } catch { /* file gone/changed */ }
    }
    manager.onFiles = cb;
    onFilesChanged();
  }
}

function buildAddBar() {
  const sel = $("add-kind");
  sel.replaceChildren();
  const g3d = document.createElement("optgroup"); g3d.label = "3D cloud";
  g3d.append(opt("cloud:", "All clouds"));
  session.cloud_keys.forEach((k, i) => g3d.append(opt("cloud:" + i, pretty(k))));
  sel.append(g3d);
  if (session.image_keys.length) {
    const gc = document.createElement("optgroup"); gc.label = "Camera";
    session.image_keys.forEach((k) => gc.append(opt("cam:" + k, pretty(k))));
    sel.append(gc);
  }
  $("add-view").onclick = () => {
    const v = sel.value, sep = v.indexOf(":");
    const kind = v.slice(0, sep), rest = v.slice(sep + 1);
    if (kind === "cloud") manager.add({ type: "cloud", channel: rest === "" ? null : +rest });
    else manager.add({ type: "cam", channel: rest });
  };

  // No dataset channels (empty / file-viewer mode) → hide the dataset "Add view" control.
  const hasChannels = session.cloud_keys.length || session.image_keys.length;
  $("add-kind").style.display = hasChannels ? "" : "none";
  $("add-view").style.display = hasChannels ? "" : "none";
}
function opt(value, label) { const o = document.createElement("option"); o.value = value; o.textContent = label; return o; }

function setupRanges() {
  const last = Math.max(0, session.n_frames - 1);
  $("frame").max = last;
  $("frame-num").min = 1; $("frame-num").max = session.n_frames;
  $("accum").max = Math.min(last, 50);    // capped: large accumulation is meaningless + costly
  $("accum").disabled = !session.has_pose;
  if (!session.has_pose) $("accum-panel").style.opacity = 0.5;
}

// ------------------------------------------------------------- wiring
function wireControls() {
  $("tool-select").onclick = () =>
    run(api.cmd("/api/tool", { tool: view.tool === "select" ? "paint" : "select" }));
  $("btn-apply-sel").onclick = () => run(api.cmd("/api/selection/apply"));
  $("btn-clear-sel").onclick = () => run(api.cmd("/api/selection/clear"));
  $("btn-clear").onclick = () => run(api.cmd("/api/clear"));
  $("btn-undo").onclick = () => run(api.cmd("/api/undo"));

  $("target-grid").onclick = () => toggleTarget("grid", "target-grid");
  $("target-points").onclick = () => toggleTarget("points", "target-points");

  $("accum").oninput = (e) => { $("accum-val").textContent = "±" + e.target.value; };
  $("accum").onchange = (e) => run(api.cmd("/api/accum", { radius: +e.target.value }));

  $("bev-mode").onchange = (e) => run(api.cmd("/api/bev_mode", { mode: e.target.value }));

  $("classes-edit").onclick = openClassEditor;
  $("modal-close").onclick = closeClassEditor;
  $("class-cancel").onclick = closeClassEditor;
  $("class-add").onclick = () => addClassRow("", "#888888");
  $("class-save").onclick = saveClasses;
  $("modal").addEventListener("click", (e) => { if (e.target.id === "modal") closeClassEditor(); });

  $("open-file").onclick = openFsBrowser;
  $("fs-close").onclick = () => { $("fs-modal").hidden = true; };
  $("fs-modal").addEventListener("click", (e) => { if (e.target.id === "fs-modal") $("fs-modal").hidden = true; });
  $("fs-input").onkeydown = async (e) => {
    if (e.key === "Tab") { e.preventDefault(); await fsComplete(); }
    else if (e.key === "Enter") {
      e.preventDefault();
      const v = $("fs-input").value.trim();
      if (v) try { await fsNavigate(v); } catch { openFile(v); }   // dir → list, else open the file
    }
  };

  $("frame").oninput = (e) => { $("frame-label").textContent = `frame ${+e.target.value + 1} / ${session.n_frames}`; };
  $("frame").onchange = (e) => run(api.cmd("/api/frame", { index: +e.target.value }));
  $("frame-num").onchange = (e) => {
    const i = Math.max(0, Math.min(session.n_frames - 1, ((+e.target.value | 0) - 1)));
    run(api.cmd("/api/frame", { index: i }));
  };
  $("btn-play").onclick = togglePlay;

  for (const id of ["g-xmin", "g-xmax", "g-ymin", "g-ymax", "g-cell"]) $(id).oninput = onGridEdit;
  $("btn-newgrid").onclick = commitGrid;

  $("btn-save").onclick = async () => {
    const dir = $("io-dir").value.trim();
    if (!dir) return ($("status").textContent = "⚠ enter an output folder");
    try { await api.save(dir); $("status").textContent = "saved to " + dir; }
    catch (e) { $("status").textContent = "⚠ " + e.message; }
  };
  $("btn-load").onclick = async () => {
    const dir = $("io-dir").value.trim();
    if (!dir) return ($("status").textContent = "⚠ enter a folder to load");
    try { const v = await api.load(dir); fillGridForm(v.grid); apply(v); $("status").textContent = "loaded from " + dir; }
    catch (e) { $("status").textContent = "⚠ " + e.message; }
  };
  $("btn-export").onclick = async () => {
    const dir = $("io-dir").value.trim();
    const name = $("export-name").value.trim() || "bev.npy";
    if (!dir) return ($("status").textContent = "⚠ enter an output folder");
    try { const r = await api.export(dir, name); $("status").textContent = "exported " + r.path; }
    catch (e) { $("status").textContent = "⚠ " + e.message; }
  };

  // Keyboard shortcuts — ignored while typing in a field.
  window.addEventListener("keydown", (e) => {
    const t = document.activeElement;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;

    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z" && !e.shiftKey) {  // undo
      e.preventDefault(); run(api.cmd("/api/undo")); return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey || !view) return;

    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {                          // frame nav
      e.preventDefault();
      const next = Math.min(session.n_frames - 1, Math.max(0, view.index + (e.key === "ArrowRight" ? 1 : -1)));
      if (next !== view.index) run(api.cmd("/api/frame", { index: next }));
      return;
    }
    if (/^[1-9]$/.test(e.key)) {                                                    // pick class
      const c = session.labelset.classes.filter((c) => c.id !== session.labelset.ignore_id)[+e.key - 1];
      if (c) { e.preventDefault(); run(api.cmd("/api/class", { id: c.id })); }
    }
  });
}

function wireGutters() {
  basisGutter(document.querySelector('[data-gutter="rail"]'), document.querySelector(".rail"), "x", 200);
  growGutter(document.querySelector('[data-gutter="bev"]'),
             document.querySelector(".bev-panel"), document.querySelector(".views"), "x", 220);
}

function toggleTarget(name, id) {
  $(id).classList.toggle("active");
  const targets = ["grid", "points"].filter((t) => $("target-" + t).classList.contains("active"));
  run(api.cmd("/api/targets", { targets }));
}

// ------------------------------------------------------------- file browser
function openFsBrowser() {
  $("fs-modal").hidden = false;
  $("fs-error").textContent = "";
  renderOpenViews();
  fsNavigate(fsPath);
}
function renderOpenViews() {
  const box = $("fs-open-list");
  box.replaceChildren();
  const files = manager.openFiles();
  if (!files.length) { box.innerHTML = `<span class="muted">none yet</span>`; return; }
  for (const f of files) {
    const row = document.createElement("div");
    row.className = "fs-open-entry"; row.title = f.path;
    const tag = document.createElement("span");
    tag.className = "fs-open-tag"; tag.textContent = f.type === "file-cloud" ? "3D" : "Img";
    const nm = document.createElement("span"); nm.className = "fs-open-name"; nm.textContent = f.name;
    const x = document.createElement("button");
    x.className = "icon-btn"; x.textContent = "✕"; x.title = "Close view";
    x.onclick = () => { manager.remove(f.id); renderOpenViews(); fsNavigate(fsPath); };
    row.append(tag, nm, x);
    box.appendChild(row);
  }
}
function renderListing(d) {
  fsPath = d.path;
  const open = manager.openFilePaths();            // files already shown in views
  const list = $("fs-list");
  list.replaceChildren();
  if (d.parent) list.appendChild(fsEntry({ name: "..", path: d.parent, is_dir: true, openable: true }, open));
  for (const e of d.entries) list.appendChild(fsEntry(e, open));
}
async function fsNavigate(path) {
  try {
    const d = await api.fsList(path);
    renderListing(d);
    $("fs-input").value = d.path.replace(/\/?$/, "/");   // show current dir, ready to type a child
    $("fs-error").textContent = "";
  } catch (e) { $("fs-error").textContent = "⚠ " + e.message; throw e; }
}
// Tab-complete the typed path against the directory it points into.
async function fsComplete() {
  const v = $("fs-input").value;
  const slash = v.lastIndexOf("/");
  const dir = slash <= 0 ? "/" : v.slice(0, slash);
  const partial = v.slice(slash + 1);
  let d;
  try { d = await api.fsList(dir); } catch { return; }
  renderListing(d);
  const matches = d.entries.filter((e) => e.name.startsWith(partial));
  if (!matches.length) return;
  let lcp = matches[0].name;                         // longest common prefix
  for (const m of matches) while (!m.name.startsWith(lcp)) lcp = lcp.slice(0, -1);
  let done = (dir === "/" ? "" : dir) + "/" + lcp;
  if (matches.length === 1 && matches[0].is_dir) done += "/";
  $("fs-input").value = done;
}
function fsEntry(e, open) {
  const loaded = !e.is_dir && open.has(e.path);
  const row = document.createElement("div");
  row.className = "fs-entry" + (e.is_dir ? " dir" : "") + (e.openable ? "" : " off") + (loaded ? " loaded" : "");
  row.append(document.createTextNode(e.name + (e.is_dir ? "/" : "")));
  if (loaded) {
    const badge = document.createElement("span");
    badge.className = "fs-open"; badge.textContent = "open";
    row.appendChild(badge);
  }
  if (e.is_dir) row.onclick = () => fsNavigate(e.path);
  else if (e.openable) row.onclick = () => openFile(e.path);
  return row;
}
async function openFile(path) {
  try {
    manager.addFile(await api.fsOpen(path));   // triggers onFiles (clouds + source + open-list)
    $("fs-error").textContent = "";
    fsNavigate(fsPath);                        // refresh the "open" markers
  } catch (e) { $("fs-error").textContent = "⚠ " + e.message; }
}

// ------------------------------------------------------------- class editor
function openClassEditor() {
  $("class-editor").replaceChildren();
  for (const c of session.labelset.classes) {
    if (c.id === session.labelset.ignore_id) continue;
    addClassRow(c.name, rgbToHex(c.color));
  }
  $("modal").hidden = false;
}
function closeClassEditor() { $("modal").hidden = true; }

function addClassRow(name, hex) {
  const row = document.createElement("div");
  row.className = "class-row";
  const color = document.createElement("input"); color.type = "color"; color.value = hex;
  const txt = document.createElement("input"); txt.type = "text"; txt.placeholder = "class name"; txt.value = name;
  const rm = document.createElement("button"); rm.className = "icon-btn"; rm.textContent = "✕"; rm.title = "Remove";
  rm.onclick = () => row.remove();
  row.append(color, txt, rm);
  $("class-editor").appendChild(row);
}

async function saveClasses() {
  const rows = [...$("class-editor").querySelectorAll(".class-row")];
  const classes = [{ id: 0, name: "unlabeled", color: [0, 0, 0] }];
  rows.forEach((row, i) => {
    const [color, txt] = row.querySelectorAll("input");
    classes.push({ id: i + 1, name: txt.value.trim() || `class ${i + 1}`, color: hexToRgb(color.value) });
  });
  const labelset = { ignore_id: 0, classes };
  try {
    const v = await api.cmd("/api/labelset", labelset);
    session.labelset = labelset;
    lut = buildLut(session.labelset);
    bev.setPalette(lut); manager.setPalette(lut);
    buildClasses();
    closeClassEditor();
    apply(v);
  } catch (e) { $("status").textContent = "⚠ " + e.message; }
}

// ------------------------------------------------------------- grid
function readGridForm() {
  return {
    xmin: +$("g-xmin").value, xmax: +$("g-xmax").value,
    ymin: +$("g-ymin").value, ymax: +$("g-ymax").value, cell_size: +$("g-cell").value,
  };
}
function fillGridForm(g) {
  $("g-xmin").value = g.xmin; $("g-xmax").value = g.xmax;
  $("g-ymin").value = g.ymin; $("g-ymax").value = g.ymax; $("g-cell").value = g.cell_size;
}
function gridValid(s) {
  return s.cell_size > 0 && s.xmax > s.xmin && s.ymax > s.ymin && [s.xmin, s.xmax, s.ymin, s.ymax].every(Number.isFinite);
}
function updateDims(s) {
  const cols = Math.max(1, Math.ceil((s.xmax - s.xmin) / s.cell_size));
  const rows = Math.max(1, Math.ceil((s.ymax - s.ymin) / s.cell_size));
  $("g-dims").textContent = `${cols} × ${rows} cells  (${cols * rows} total)`;
}
function onGridEdit() {
  const s = readGridForm();
  if (!gridValid(s)) { $("g-dims").textContent = "invalid extent"; $("btn-newgrid").disabled = true; bev.clearPreview(); return; }
  $("btn-newgrid").disabled = false;
  updateDims(s);
  bev.setPreviewGrid(s);
}
async function commitGrid() {
  const s = readGridForm();
  if (!gridValid(s)) return;
  const { count } = await api.gridLabelledCount();
  if (count > 0 && !confirm(
    `A grid labeling already exists (${count} frame(s)).\n` +
    "Creating a new grid will clear it (per-point labels are kept). Continue?")) return;
  bev.clearPreview();
  run(api.cmd("/api/grid", s));
}

// ------------------------------------------------------------- timeline
function togglePlay() {
  if (playTimer) { clearInterval(playTimer); playTimer = null; $("btn-play").textContent = "▶"; $("btn-play").classList.remove("active"); return; }
  $("btn-play").textContent = "⏸"; $("btn-play").classList.add("active");
  playTimer = setInterval(() => run(api.cmd("/api/frame", { index: (view.index + 1) % session.n_frames })), 140);
}

// ------------------------------------------------------------- control sync
function className(id) {
  const c = session.labelset.classes.find((c) => c.id === id);
  return c ? c.name : String(id);
}
function updateBevHint() {
  $("bev-hint").textContent = (view && view.tool === "select")
    ? "drag: select · right-drag: deselect · shift/middle: pan"
    : "drag: paint · right-drag: erase · shift/middle: pan";
}
function syncControls(v) {
  for (const el of $("classes").children) el.classList.toggle("active", +el.dataset.id === v.activeClass);
  $("tool-select").classList.toggle("active", v.tool === "select");
  $("target-grid").classList.toggle("active", v.activeTargets.includes("grid"));
  $("target-points").classList.toggle("active", v.activeTargets.includes("points"));

  const hasSel = v.selection != null;
  $("btn-apply-sel").disabled = !hasSel;
  $("btn-clear-sel").disabled = !hasSel;

  if (document.activeElement !== $("frame")) $("frame").value = v.index;
  if (document.activeElement !== $("frame-num")) $("frame-num").value = v.index + 1;
  $("frame-label").textContent = `frame ${v.index + 1} / ${v.nFrames}`;
  $("accum").value = v.accumRadius; $("accum-val").textContent = "±" + v.accumRadius;
  if (v.bevMode) $("bev-mode").value = v.bevMode;
  updateBevHint();

  const accum = v.accumRadius ? "±" + v.accumRadius : "off";
  const tool = v.tool === "select" ? "select" : "paint";
  const npts = v.points ? v.points.shape[0] : 0;
  $("status").textContent =
    `${v.nFrames} frames · ${npts} pts · clouds: ${v.visibleClouds.join(", ") || "—"} · ` +
    `class: ${className(v.activeClass)} · target: ${v.activeTargets.join(", ") || "—"} · ` +
    `accum: ${accum} · tool: ${tool}`;
}

boot().catch((e) => { $("status").textContent = "⚠ " + e.message; console.error(e); });
