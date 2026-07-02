// Splasher API client: /api/* calls + numpy array decoding.
//
// Arrays travel as {dtype, shape, data(base64)}: we decode them into a TypedArray
// straight over the buffer (no extra copy, semantics preserved).

const TYPED = {
  float32: Float32Array, float64: Float64Array,
  int8: Int8Array, int16: Int16Array, int32: Int32Array,
  uint8: Uint8Array, uint16: Uint16Array, uint32: Uint32Array,
  int64: BigInt64Array, uint64: BigUint64Array,
};

export function decodeArray(o) {
  if (!o) return null;
  const bin = atob(o.data);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const Ctor = TYPED[o.dtype] || Uint8Array;
  return { data: new Ctor(bytes.buffer), shape: o.shape, dtype: o.dtype };
}

// Turn the ViewState dict into a usable object (decoded arrays).
export function decodeView(d) {
  const images = {};
  for (const [name, arr] of Object.entries(d.images || {})) images[name] = decodeArray(arr);
  return {
    grid: d.grid,
    points: decodeArray(d.points),          // {data: Float32Array, shape: [N, stride]}
    pointLabels: decodeArray(d.point_labels),
    pointChannels: decodeArray(d.point_channels),  // {data: Int16Array, shape: [N]} → cloud_keys idx
    bevField: decodeArray(d.bev_field),      // {data, shape: [rows, cols]} per bevMode
    gridLabels: decodeArray(d.grid_labels),  // or null
    selection: decodeArray(d.selection),     // or null (uint8)
    images,
    index: d.index,
    nFrames: d.n_frames,
    activeClass: d.active_class,
    activeTargets: d.active_targets,
    accumRadius: d.accum_radius,
    bevMode: d.bev_mode,
    tool: d.tool,
    visibleClouds: d.visible_clouds,
    visibleImages: d.visible_images,
  };
}

async function getJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

async function postJson(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status} ${await r.text()}`);
  return r.json();
}

export const api = {
  session: () => getJson("/api/session"),
  view: async () => decodeView(await getJson("/api/view")),
  gridLabelledCount: () => getJson("/api/grid/labelled_count"),

  // Every command returns the updated ViewState (decoded).
  cmd: async (path, body) => decodeView(await postJson(path, body)),

  // I/O: non-ViewState responses.
  save: (dir) => postJson("/api/save", { dir }),
  load: async (dir) => decodeView(await postJson("/api/load", { dir })),
  export: (dir, name) => postJson("/api/export", { dir, name }),

  // File viewer: browse the filesystem and open single files.
  fsList: async (path) => {
    const r = await fetch("/api/fs" + (path ? "?path=" + encodeURIComponent(path) : ""));
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `list failed (${r.status})`);
    return d;
  },
  // `features`: per-point measure files (any location) to attach to the opened cloud.
  fsOpen: async (path, features = []) => {
    const r = await fetch("/api/fs/open", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(features.length ? { path, features } : { path }),
    });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || `open failed (${r.status})`);
    if (d.points) d.points = decodeArray(d.points);
    if (d.image) d.image = decodeArray(d.image);    // numpy image array (.npy HxWxC)
    return d;
  },
  fsRawUrl: (path) => "/api/fs/raw?path=" + encodeURIComponent(path),
};
