"""Projection BEV : nuage de points -> cellules de la grille, et sélection par rectangle.

Tout est vectorisé numpy. Sert :
- la sous-couche de la vue de dessus (densité / hauteur par cellule),
- la cible Grid (cellules couvertes par un rectangle),
- la cible Points + le surlignage (masque des points dans un rectangle).
"""

from __future__ import annotations

import numpy as np

from .colormap import colormap
from .grid import Grid

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1) en coords monde


def points_to_cells(xy: np.ndarray, grid: Grid):
    """Raccourci vers `grid.world_to_cell` : `xy` (N, 2) -> (`ij` (N, 2), `valid`)."""
    return grid.world_to_cell(xy)


def _reduce_per_cell(points: np.ndarray, grid: Grid, op: str) -> np.ndarray:
    """Réduit une grandeur par cellule. `op` = 'max_z' ou 'count'. Cellules vides = NaN."""
    out = np.full(grid.rows * grid.cols, -np.inf if op == "max_z" else 0.0, dtype=np.float64)
    if len(points):
        ij, valid = grid.world_to_cell(points[:, :2])
        flat = ij[valid, 0] * grid.cols + ij[valid, 1]
        if op == "max_z":
            np.maximum.at(out, flat, points[valid, 2])
        else:
            np.add.at(out, flat, 1.0)
    out = out.reshape(grid.shape)
    if op == "max_z":
        out[~np.isfinite(out)] = np.nan
    else:
        out[out == 0.0] = np.nan
    return out


def bev_max_height(points: np.ndarray, grid: Grid) -> np.ndarray:
    """`(rows, cols)` float : hauteur max (z) par cellule, NaN si vide."""
    return _reduce_per_cell(points, grid, "max_z")


def bev_count(points: np.ndarray, grid: Grid) -> np.ndarray:
    """`(rows, cols)` float : nombre de points par cellule, NaN si vide."""
    return _reduce_per_cell(points, grid, "count")


def bev_image(scalar_field: np.ndarray, *, alpha: int = 210) -> np.ndarray:
    """Colorise un champ `(rows, cols)` (NaN = transparent) en RGBA uint8 `(rows, cols, 4)`."""
    rows, cols = scalar_field.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)
    filled = np.isfinite(scalar_field)
    if filled.any():
        colors = colormap(scalar_field[filled])
        rgba[filled, :3] = (colors[:, :3] * 255).astype(np.uint8)
        rgba[filled, 3] = alpha
    return rgba


def cells_in_rect(rect: Rect, grid: Grid) -> tuple[slice, slice]:
    """Cellules (lignes, colonnes) couvertes par un rectangle monde -> slices `(i, j)`."""
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    j0 = int(np.clip(np.floor((x0 - grid.xmin) / grid.cell_size), 0, grid.cols))
    j1 = int(np.clip(np.ceil((x1 - grid.xmin) / grid.cell_size), 0, grid.cols))
    i0 = int(np.clip(np.floor((y0 - grid.ymin) / grid.cell_size), 0, grid.rows))
    i1 = int(np.clip(np.ceil((y1 - grid.ymin) / grid.cell_size), 0, grid.rows))
    return slice(i0, i1), slice(j0, j1)


def points_in_rect(xy: np.ndarray, rect: Rect) -> np.ndarray:
    """Masque booléen `(N,)` des points dont `(x, y)` tombe dans le rectangle monde."""
    if xy is None or len(xy) == 0:
        return np.zeros(0, dtype=bool)
    x0, y0, x1, y1 = rect
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    xy = np.asarray(xy)
    return (
        (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
    )
