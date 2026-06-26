"""Sauvegarde/chargement d'une session de labels — fichiers autonomes.

Format sur disque (dossier de sortie) :

    session.json          # grille (étendue, cell_size) + jeu de classes
    grid/frame_00007.npy  # raster d'ids (rows, cols), un par sample labélisé
    grid/frame_00007.png  # même raster colorisé (aperçu, y vers le haut)
    grid/global.npy       # raster global (si labélisé)
    points/frame_00007.npy# labels (N,) int64 par frame (segmentation)

Les `.npy`/`.json` n'utilisent que numpy + stdlib. Le `.png` (aperçu) passe par Qt
si disponible, sinon il est simplement omis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .grid import Grid
from .labels import LabelSet


def _save_png(path: Path, rgba: np.ndarray) -> bool:
    try:
        from PySide6.QtGui import QImage
    except Exception:
        return False
    arr = np.ascontiguousarray(np.flipud(rgba).astype(np.uint8))  # y vers le haut à l'écran
    h, w = arr.shape[:2]
    img = QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888)
    return bool(img.copy().save(str(path)))


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
    """Renvoie `{grid, labelset, grid_labels: {i: raster}, point_labels: {i: (N,)}}`."""
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
