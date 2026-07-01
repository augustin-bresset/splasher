"""Filesystem browsing + single-file loading for the *file viewer* mode.

Lets the front browse the (local) filesystem and open individual files into views:
point clouds are decoded to `(N, 3+F)` float32 arrays here; images are streamed raw and
decoded by the browser. Unreadable / unsupported files raise with a clear message.

Per-point scalar features may live in sibling `<base>_<suffix>.npy` files (shape `(N,)`) —
the same `<cloud>_<suffix>` convention apairo uses for suffixed sub-channels (e.g. a Tartan
`000000_intensity.npy` beside `000000.npy`). Opening any member of the group loads the
coordinate cloud plus every sibling feature as trailing named columns (`feature_names`), so
the viewer can color by any of them. A native 4th column is surfaced as `intensity`.

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


def open_file(path: str) -> dict:
    """Open a file.

    Returns one of:
    - `{kind:"cloud", points: (N,3+F) ndarray, feature_names: [...]}`,
    - `{kind:"image", image: (H,W,C) ndarray}`  (decoded array, e.g. a `.npy` image),
    - `{kind:"image"}`  (raster file like .png/.jpg — streamed raw, decoded by the browser).

    A cloud's `points` are `[x, y, z, *feature_names]`: the trailing columns are per-point
    scalar features gathered from sibling `<base>_<suffix>.npy` files (and a native 4th
    column, as `intensity`). See `_cloud_group`.
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
            pts, names = _cloud_group(p, coords=arr)
            return {"kind": "cloud", "points": pts, "feature_names": names, **base}
        if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 1):
            # a per-point scalar (e.g. 000000_intensity.npy) → pair with its coordinate cloud
            coord = _coord_sibling(p)
            if coord is None:
                raise ValueError(
                    f"{p.name} is a 1-D per-point array (shape {arr.shape}) with no coordinate "
                    f"cloud beside it (expected a sibling '<base>.npy')")
            pts, names = _cloud_group(coord)
            return {"kind": "cloud", "points": pts, "feature_names": names, **base}
        raise ValueError(f".npy shape {arr.shape} is neither an (N,>=3) cloud, a (N,) feature, "
                         f"nor an (H,W,1/3/4) image")
    if ext in IMAGE_EXTS:
        return {"kind": "image", **base}   # raster file → streamed raw to the browser
    if ext in CLOUD_EXTS:                  # .bin, .pcd
        pts, names = _cloud_group(p)
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


def _cloud_group(base_path: Path, coords: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Load a coordinate cloud plus its sibling scalar features into `(N, 3+F)` float32.

    Features: a native 4th column (as `intensity`) and every `<base>_<suffix>.npy` sibling of
    matching length (a sibling `intensity` overrides the native one). Returns the array and
    the ordered `feature_names` (aligned with columns 3..). No sibling → a plain `(N, 3)`.
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

    names = ordered_features(feats.keys())
    out = np.empty((n, 3 + len(names)), np.float32)
    out[:, :3] = coords[:, :3]
    for i, name in enumerate(names):
        out[:, 3 + i] = feats[name]
    return out, names


def combine_clouds(clouds: list[np.ndarray]) -> np.ndarray:
    """Concatenate clouds into a single `(M, 4)` array [x, y, z, intensity] (0 if absent)."""
    total = sum(len(c) for c in clouds)
    out = np.zeros((total, 4), dtype=np.float32)
    k = 0
    for c in clouds:
        n = len(c)
        out[k:k + n, :3] = c[:, :3]
        if c.shape[1] >= 4:
            out[k:k + n, 3] = c[:, 3]
        k += n
    return out


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
