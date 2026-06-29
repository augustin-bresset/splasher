// Front-side colorization (the engine only delivers semantic primitives).
// Faithful mirror of splasher.core.colormap / engine.render.

const VIRIDIS = [
  [68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37],
];

const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);

// t in [0,1] → [r,g,b] 0..255 (linear interpolation between viridis anchors).
export function viridis(t) {
  t = clamp01(t);
  const n = VIRIDIS.length - 1;
  const pos = t * n;
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n);
  const f = pos - lo;
  const a = VIRIDIS[lo], b = VIRIDIS[hi];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// Finite [min,max] bounds of a TypedArray (NaNs ignored).
export function finiteRange(arr) {
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  }
  if (!Number.isFinite(lo)) return [0, 1];
  if (hi <= lo) hi = lo + 1;
  return [lo, hi];
}

// Build the id → [r,g,b] table from the labelset (ignore_id excluded = transparent).
export function buildLut(labelset) {
  const lut = new Map();
  for (const c of labelset.classes) {
    if (c.id === labelset.ignore_id) continue;
    lut.set(c.id, c.color);
  }
  return { colors: lut, ignore: labelset.ignore_id };
}

export const rgbCss = ([r, g, b]) => `rgb(${r | 0},${g | 0},${b | 0})`;

// Display name: channel keys are identifiers; never show raw underscores in the UI.
export const pretty = (s) => String(s).replace(/_/g, " ");

const hx = (n) => Math.max(0, Math.min(255, n | 0)).toString(16).padStart(2, "0");
export const rgbToHex = ([r, g, b]) => `#${hx(r)}${hx(g)}${hx(b)}`;
export const hexToRgb = (h) => {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h);
  return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [0, 0, 0];
};
