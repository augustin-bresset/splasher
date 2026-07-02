"""Filesystem browsing + single-file loading for the *file viewer* mode.

Lets the front browse the (local) filesystem and open individual files into views:
point clouds are decoded to `(N, 3+F)` float32 arrays here; images are streamed raw and
decoded by the browser. Unreadable / unsupported files raise with a clear message.

Per-point scalar features may live in sibling `<base>_<suffix>.npy` files (shape `(N,)`) —
the same `<cloud>_<suffix>` convention apairo uses for suffixed sub-channels (e.g. a Tartan
`000000_intensity.npy` beside `000000.npy`). Opening any member of the group loads the
coordinate cloud plus every sibling feature as trailing named columns (`feature_names`), so
the viewer can color by any of them. A native 4th column is surfaced as `intensity`.

Measures living *anywhere else* (labels, intensity… — e.g. `ground_truth/00123.npy` for the
cloud `00123.npy`) can be attached explicitly: `open_file(cloud, features=[...])` merges the
given per-point files into the group (see `_extra_feature_name` for how they are named).
Opening a lone per-point file with no coordinate sibling returns `kind="feature"` so a front
can offer to attach it to an already-open cloud of matching length.

Local use only (the server binds to 127.0.0.1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core.source import ordered_features

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
CLOUD_EXTS = {".npy", ".bin", ".pcd"}


def list_dir(path: str | None = None) -> dict:
    """List a directory: dirs first, then files (alphabetical). `None` → the user's home."""
    base = (Path(path).expanduser() if path else Path.home()).resolve()
    if not base.is_dir():
        raise NotADirectoryError(f"not a directory: {base}")
    entries = []
    for child in sorted(base.iterdir(), key=lambda c: (not _is_dir(c), c.name.lower())):
        is_dir = _is_dir(child)
        if not is_dir and child.suffix.lower() not in IMAGE_EXTS | CLOUD_EXTS:
            openable = False
        else:
            openable = True
        entries.append({"name": child.name, "path": str(child), "is_dir": is_dir,
                        "openable": is_dir or openable})
    parent = str(base.parent) if base.parent != base else None
    return {"path": str(base), "parent": parent, "entries": entries}


def _is_dir(p: Path) -> bool:
    try:
        return p.is_dir()
    except OSError:
        return False


def open_file(path: str, features: list[str] | None = None) -> dict:
    """Open a file.

    Returns one of:
    - `{kind:"cloud", points: (N,3+F) ndarray, feature_names: [...]}`,
    - `{kind:"feature", length: N}`  (a per-point measure with no coordinate cloud beside it —
      the caller may attach it to an open cloud of length N via `features`),
    - `{kind:"image", image: (H,W,C) ndarray}`  (decoded array, e.g. a `.npy` image),
    - `{kind:"image"}`  (raster file like .png/.jpg — streamed raw, decoded by the browser).

    A cloud's `points` are `[x, y, z, *feature_names]`: the trailing columns are per-point
    scalar features gathered from sibling `<base>_<suffix>.npy` files (and a native 4th
    column, as `intensity`), plus the `features` files attached explicitly (per-point measures
    living anywhere, e.g. `ground_truth/00123.npy`). See `_cloud_group`.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    ext = p.suffix.lower()
    base = {"name": p.name, "path": str(p)}

    if ext == ".npy":                      # a .npy can hold an image, a cloud, or a scalar feature
        arr = np.asarray(np.load(p))
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            return {"kind": "image", "image": arr, **base}
        if arr.ndim == 2 and arr.shape[1] >= 3:
            pts, names = _cloud_group(p, coords=arr, extra=features)
            return {"kind": "cloud", "points": pts, "feature_names": names, **base}
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            # a per-point scalar (e.g. 000000_intensity.npy) → pair with its coordinate cloud
            coord = _coord_sibling(p)
            if coord is None:
                # no sibling: a standalone measure — attachable to an open cloud of that length
                return {"kind": "feature", "length": int(arr.reshape(-1).shape[0]), **base}
            pts, names = _cloud_group(coord, extra=features)
            return {"kind": "cloud", "points": pts, "feature_names": names, **base}
        raise ValueError(f".npy shape {arr.shape} is neither an (N,>=3) cloud, a (N,) feature, "
                         f"nor an (H,W,1/3/4) image")
    if ext in IMAGE_EXTS:
        return {"kind": "image", **base}   # raster file → streamed raw to the browser
    if ext in CLOUD_EXTS:                  # .bin, .pcd
        pts, names = _cloud_group(p, extra=features)
        return {"kind": "cloud", "points": pts, "feature_names": names, **base}
    raise ValueError(f"unsupported format: {ext or p.name}")


def _feature_siblings(base_path: Path) -> list[tuple[str, Path]]:
    """`(feature_name, path)` for `<base>_<suffix>.npy` files beside the coordinate cloud."""
    stem = base_path.stem
    out = []
    for p in sorted(base_path.parent.glob(f"{stem}_*.npy")):
        suffix = p.stem[len(stem) + 1:]
        if suffix:
            out.append((suffix, p))
    return out


def _coord_sibling(scalar_path: Path) -> Path | None:
    """`<base>_<suffix>.npy` → the coordinate cloud `<base>.<ext>` beside it, if any.

    `<base>` is the stem up to the first `_` (frame stems carry no `_`, per the convention)."""
    stem = scalar_path.stem
    if "_" not in stem:
        return None
    base = stem.split("_", 1)[0]
    for ext in (".npy", ".bin", ".pcd"):
        cand = scalar_path.with_name(base + ext)
        if cand != scalar_path and cand.is_file():
            return cand
    return None


def _extra_feature_name(base_path: Path, feature_path: Path) -> str:
    """Feature name for an explicitly attached measure file.

    `<cloudstem>_<suffix>` → the suffix (sibling convention, wherever the file lives); the
    same stem as the cloud (e.g. `ground_truth/00123.npy` on `00123.npy`) → the holding
    directory's name; anything else → the file's stem.
    """
    stem, base = feature_path.stem, base_path.stem
    if stem.startswith(base + "_") and len(stem) > len(base) + 1:
        return stem[len(base) + 1:]
    if stem == base:
        return feature_path.parent.name or stem
    return stem


def _load_extra_feature(fp: Path, n: int) -> np.ndarray:
    """Load an explicitly attached per-point measure: `(n,)` or `(n,1)` .npy → `(n,) float32`.

    Unlike siblings (best-effort, skipped on mismatch), an explicit attach fails loudly."""
    if fp.suffix.lower() != ".npy":
        raise ValueError(f"{fp.name}: per-point measures must be .npy files")
    arr = np.asarray(np.load(fp))
    if not (arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1)):
        raise ValueError(f"{fp.name}: expected an (N,) or (N,1) per-point array, "
                         f"got shape {arr.shape}")
    vals = arr.reshape(-1)
    if vals.shape[0] != n:
        raise ValueError(f"{fp.name}: {vals.shape[0]} values for a {n}-point cloud")
    return vals.astype(np.float32)


def _cloud_group(base_path: Path, coords: np.ndarray | None = None,
                 extra: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Load a coordinate cloud plus its sibling scalar features into `(N, 3+F)` float32.

    Features: a native 4th column (as `intensity`) and every `<base>_<suffix>.npy` sibling of
    matching length (a sibling `intensity` overrides the native one), then the `extra` files
    (explicitly attached measures — any location, named by `_extra_feature_name`, they win
    over both). Returns the array and the ordered `feature_names` (aligned with columns 3..).
    No feature → a plain `(N, 3)`.
    """
    if coords is None:
        coords = _load_cloud(base_path, base_path.suffix.lower())
    coords = np.asarray(coords)
    n = len(coords)

    feats: dict[str, np.ndarray] = {}
    if coords.ndim == 2 and coords.shape[1] >= 4:
        feats["intensity"] = coords[:, 3].astype(np.float32)   # KITTI-style x,y,z,intensity
    for name, sp in _feature_siblings(base_path):
        try:
            arr = np.asarray(np.load(sp)).reshape(-1)
        except (OSError, ValueError):
            continue
        if arr.shape[0] == n:
            feats[name] = arr.astype(np.float32)               # sibling wins over native
    for fpath in extra or ():
        fp = Path(fpath).expanduser()
        feats[_extra_feature_name(base_path, fp)] = _load_extra_feature(fp, n)

    names = ordered_features(feats.keys())
    out = np.empty((n, 3 + len(names)), np.float32)
    out[:, :3] = coords[:, :3]
    for i, name in enumerate(names):
        out[:, 3 + i] = feats[name]
    return out, names


def combine_clouds(clouds: list[tuple[np.ndarray, list[str]]]) -> tuple[np.ndarray, list[str]]:
    """Concatenate `(points, feature_names)` clouds into one `(M, 3+F)` float32 array.

    `F` = the union of the feature names (ordered by `ordered_features`); each cloud fills
    the features it has, `NaN` elsewhere (the NaN-excluding convention of the BEV mean).
    """
    names = ordered_features({n for _, ns in clouds for n in ns})
    total = sum(len(c) for c, _ in clouds)
    col = {name: 3 + i for i, name in enumerate(names)}
    out = np.full((total, 3 + len(names)), np.nan, dtype=np.float32)
    k = 0
    for c, ns in clouds:
        n = len(c)
        out[k:k + n, :3] = c[:, :3]
        for i, name in enumerate(ns):
            out[k:k + n, col[name]] = c[:, 3 + i]
        k += n
    return out, names


def _load_cloud(p: Path, ext: str) -> np.ndarray:
    if ext == ".npy":
        arr = np.asarray(np.load(p))
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"expected an (N, >=3) array, got shape {arr.shape}")
        return arr.astype(np.float32)
    if ext == ".bin":
        raw = np.fromfile(p, dtype=np.float32)
        for w in (4, 3):
            if raw.size and raw.size % w == 0:
                return raw.reshape(-1, w).astype(np.float32)
        raise ValueError("raw .bin is not a multiple of 3 or 4 float32 (KITTI-style expected)")
    if ext == ".pcd":
        return _load_pcd_ascii(p)
    raise ValueError(f"unsupported point-cloud format: {ext}")


def _load_pcd_ascii(p: Path) -> np.ndarray:
    """Minimal ASCII-PCD reader (x/y/z columns). Binary PCD is rejected with a message."""
    lines = p.read_text(errors="replace").splitlines()
    fields: list[str] = []
    data_at = None
    for i, line in enumerate(lines):
        t = line.strip()
        up = t.upper()
        if up.startswith("FIELDS"):
            fields = t.split()[1:]
        elif up.startswith("DATA"):
            if "ascii" not in t.lower():
                raise ValueError("only ASCII PCD is supported (this file is binary)")
            data_at = i + 1
            break
    if data_at is None or not fields:
        raise ValueError("invalid PCD header (no FIELDS / DATA)")
    try:
        xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError:
        raise ValueError("PCD has no x/y/z fields") from None
    rows = []
    for line in lines[data_at:]:
        parts = line.split()
        if len(parts) <= max(xi, yi, zi):
            continue
        try:
            rows.append((float(parts[xi]), float(parts[yi]), float(parts[zi])))
        except ValueError:
            continue
    if not rows:
        raise ValueError("no points parsed from PCD")
    return np.asarray(rows, dtype=np.float32)
