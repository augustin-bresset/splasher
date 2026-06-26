"""`Grid` — la grille de carrés vue-de-dessus (BEV), définie en unités monde.

C'est la première chose qu'on conçoit : une étendue monde `(xmin..xmax, ymin..ymax)`
et la taille d'un carré `cell_size` (mètres). On en déduit `cols × rows`. La grille
fournit le mapping monde <-> cellule, un raster vide, et les segments de lignes pour
l'affichage.

Conventions :
- `j` (colonne) indexe `x` : `j = floor((x - xmin) / cell_size)`
- `i` (ligne)   indexe `y` : `i = floor((y - ymin) / cell_size)`
- un raster a la forme `(rows, cols)`, indexé `raster[i, j]`
- l'origine `(xmin, ymin)` est en bas-gauche, `y` vers le haut.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_size: float

    def __post_init__(self) -> None:
        if self.cell_size <= 0:
            raise ValueError("cell_size doit être > 0")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("étendue invalide (xmax > xmin et ymax > ymin requis)")

    # --- dimensions -------------------------------------------------------
    @property
    def cols(self) -> int:
        return max(1, math.ceil((self.xmax - self.xmin) / self.cell_size))

    @property
    def rows(self) -> int:
        return max(1, math.ceil((self.ymax - self.ymin) / self.cell_size))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def width(self) -> float:
        """Largeur réelle couverte par les carrés (cols * cell_size)."""
        return self.cols * self.cell_size

    @property
    def height(self) -> float:
        return self.rows * self.cell_size

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    # --- mapping monde <-> cellule ---------------------------------------
    def world_to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`xy` (N, 2) -> (`ij` (N, 2) int [ligne, colonne], `valid` (N,) bool)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        j = np.floor((xy[:, 0] - self.xmin) / self.cell_size).astype(np.intp)
        i = np.floor((xy[:, 1] - self.ymin) / self.cell_size).astype(np.intp)
        valid = (j >= 0) & (j < self.cols) & (i >= 0) & (i < self.rows)
        return np.stack([i, j], axis=1), valid

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        """Centre monde de la cellule `(i, j)`."""
        x = self.xmin + (j + 0.5) * self.cell_size
        y = self.ymin + (i + 0.5) * self.cell_size
        return (x, y)

    def empty_raster(self, fill: int = 0, dtype=np.int32) -> np.ndarray:
        return np.full((self.rows, self.cols), fill, dtype=dtype)

    # --- affichage --------------------------------------------------------
    def image_rect(self) -> tuple[float, float, float, float]:
        """`(x, y, w, h)` pour positionner un `ImageItem (rows, cols)` en coords monde."""
        return (self.xmin, self.ymin, self.width, self.height)

    def line_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Segments des lignes de la grille pour un tracé `connect='pairs'`."""
        x0, y0 = self.xmin, self.ymin
        x1, y1 = self.xmin + self.width, self.ymin + self.height
        vx = self.xmin + self.cell_size * np.arange(self.cols + 1)
        hy = self.ymin + self.cell_size * np.arange(self.rows + 1)

        xs_v = np.repeat(vx, 2)
        ys_v = np.tile([y0, y1], self.cols + 1)
        xs_h = np.tile([x0, x1], self.rows + 1)
        ys_h = np.repeat(hy, 2)

        xs = np.concatenate([xs_v, xs_h]).astype(np.float64)
        ys = np.concatenate([ys_v, ys_h]).astype(np.float64)
        return xs, ys


def grid_from_points(xy: np.ndarray, cell_size: float = 1.0,
                     margin: float = 2.0) -> Grid:
    """Grille par défaut englobant un nuage de points top-down `xy` (N, 2)."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(xy) == 0:
        return Grid(-10.0, 10.0, -10.0, 10.0, cell_size)
    lo = np.floor(xy.min(axis=0) - margin)
    hi = np.ceil(xy.max(axis=0) + margin)
    return Grid(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), cell_size)
