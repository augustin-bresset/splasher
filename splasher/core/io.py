"""Save/load a label session — self-contained files.

On-disk format (output directory):

    session.json          # grid (extent, cell_size) + class set
    grid/frame_00007.npy  # id raster (rows, cols), one per labeled sample
    grid/frame_00007.png  # same raster colorized (preview, y upwards)
    grid/global.npy       # global raster (if labeled)
    points/frame_00007.npy# labels (N,) int64 per frame (segmentation)

Everything uses numpy + stdlib only (the `.png` preview is encoded via `zlib`, with no
UI dependency).
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

from .grid import Grid
from .labels import LabelSet


def _save_png(path: Path, rgba: np.ndarray) -> bool:
    """Encode an RGBA `(h, w, 3|4)` uint8 array as PNG (minimal stdlib encoder)."""
    arr = np.ascontiguousarray(np.flipud(np.asarray(rgba)).astype(np.uint8))  # y upwards
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[2] == 3:
        arr = np.concatenate([arr, np.full((h, w, 1), 255, np.uint8)], axis=2)

    # Raw data: one filter byte (0) at the start of each row.
    raw = np.zeros((h, 1 + w * 4), np.uint8)
    raw[:, 1:] = arr.reshape(h, w * 4)

    def _chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8 bits, type 6 = RGBA
    png = (b"\x89PNG\r\n\x1a\n"
           + _chunk(b"IHDR", ihdr)
           + _chunk(b"IDAT", zlib.compress(raw.tobytes(), 9))
           + _chunk(b"IEND", b""))
    Path(path).write_bytes(png)
    return True


def save_session(out_dir, *, grid: Grid, labelset: LabelSet,
                 grid_target=None, point_target=None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "grid": {
            "xmin": grid.xmin, "xmax": grid.xmax,
            "ymin": grid.ymin, "ymax": grid.ymax,
            "cell_size": grid.cell_size, "rows": grid.rows, "cols": grid.cols,
        },
        "labels": labelset.to_dict(),
    }
    (out / "session.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if grid_target is not None:
        gd = out / "grid"
        gd.mkdir(exist_ok=True)
        for fi, raster in grid_target.rasters().items():
            np.save(gd / f"frame_{fi:05d}.npy", raster)
            _save_png(gd / f"frame_{fi:05d}.png", labelset.colorize(raster, alpha=255))
        glob = grid_target.global_raster()
        if (glob != labelset.ignore_id).any():
            np.save(gd / "global.npy", glob)
            _save_png(gd / "global.png", labelset.colorize(glob, alpha=255))

    if point_target is not None:
        pd = out / "points"
        pd.mkdir(exist_ok=True)
        for fi, lab in point_target.all_labels().items():
            np.save(pd / f"frame_{fi:05d}.npy", lab)

    return out


def _frame_idx(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_session(out_dir) -> dict:
    """Return `{grid, labelset, grid_labels: {i: raster}, point_labels: {i: (N,)}}`."""
    out = Path(out_dir)
    meta = json.loads((out / "session.json").read_text())
    g = meta["grid"]
    grid = Grid(g["xmin"], g["xmax"], g["ymin"], g["ymax"], g["cell_size"])
    labelset = LabelSet.from_dict(meta["labels"])

    grid_labels: dict[int, np.ndarray] = {}
    gd = out / "grid"
    if gd.is_dir():
        for p in sorted(gd.glob("frame_*.npy")):
            grid_labels[_frame_idx(p)] = np.load(p)

    point_labels: dict[int, np.ndarray] = {}
    pd = out / "points"
    if pd.is_dir():
        for p in sorted(pd.glob("frame_*.npy")):
            point_labels[_frame_idx(p)] = np.load(p)

    return {
        "grid": grid,
        "labelset": labelset,
        "grid_labels": grid_labels,
        "point_labels": point_labels,
    }
