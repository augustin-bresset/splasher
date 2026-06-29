"""Filesystem browsing + single-file loading for the *file viewer* mode.

Lets the front browse the (local) filesystem and open individual files into views:
point clouds are decoded to `(N, >=3)` float32 arrays here; images are streamed raw and
decoded by the browser. Unreadable / unsupported files raise with a clear message.

Local use only (the server binds to 127.0.0.1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

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
    - `{kind:"cloud", points: (N,>=3) ndarray}`,
    - `{kind:"image", image: (H,W,C) ndarray}`  (decoded array, e.g. a `.npy` image),
    - `{kind:"image"}`  (raster file like .png/.jpg — streamed raw, decoded by the browser).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"not a file: {p}")
    ext = p.suffix.lower()
    base = {"name": p.name, "path": str(p)}

    if ext == ".npy":                      # a .npy can hold either an image or a cloud
        arr = np.asarray(np.load(p))
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            return {"kind": "image", "image": arr, **base}
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return {"kind": "cloud", "points": arr.astype(np.float32), **base}
        raise ValueError(f".npy shape {arr.shape} is neither an (N,>=3) cloud nor an (H,W,1/3/4) image")
    if ext in IMAGE_EXTS:
        return {"kind": "image", **base}   # raster file → streamed raw to the browser
    if ext in CLOUD_EXTS:                  # .bin, .pcd
        return {"kind": "cloud", "points": _load_cloud(p, ext), **base}
    raise ValueError(f"unsupported format: {ext or p.name}")


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
        raise ValueError("PCD has no x/y/z fields")
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
