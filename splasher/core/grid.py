"""`Grid` — the top-down (BEV) grid of cells, defined in world units.

It is the first thing you design: a world extent `(xmin..xmax, ymin..ymax)` and the cell
size `cell_size` (meters). From it we derive `cols × rows`. The grid provides the
world <-> cell mapping, an empty raster, and the line segments for display.

Conventions:
- `j` (column) indexes `x`: `j = floor((x - xmin) / cell_size)`
- `i` (row)    indexes `y`: `i = floor((y - ymin) / cell_size)`
- a raster has shape `(rows, cols)`, indexed `raster[i, j]`
- the origin `(xmin, ymin)` is bottom-left, `y` upwards.
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
        if not math.isfinite(self.cell_size) or self.cell_size <= 0:
            raise ValueError("cell_size must be a finite number > 0")
        if not all(math.isfinite(v) for v in (self.xmin, self.xmax, self.ymin, self.ymax)):
            raise ValueError("extent must be finite (no NaN/inf bounds)")
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise ValueError("invalid extent (xmax > xmin and ymax > ymin required)")

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
        """Actual width covered by the cells (cols * cell_size)."""
        return self.cols * self.cell_size

    @property
    def height(self) -> float:
        return self.rows * self.cell_size

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    # --- world <-> cell mapping ------------------------------------------
    def world_to_cell(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`xy` (N, 2) -> (`ij` (N, 2) int [row, column], `valid` (N,) bool)."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        j = np.floor((xy[:, 0] - self.xmin) / self.cell_size).astype(np.intp)
        i = np.floor((xy[:, 1] - self.ymin) / self.cell_size).astype(np.intp)
        valid = (j >= 0) & (j < self.cols) & (i >= 0) & (i < self.rows)
        return np.stack([i, j], axis=1), valid

    def cell_to_world(self, i: int, j: int) -> tuple[float, float]:
        """World center of cell `(i, j)`."""
        x = self.xmin + (j + 0.5) * self.cell_size
        y = self.ymin + (i + 0.5) * self.cell_size
        return (x, y)

    def empty_raster(self, fill: int = 0, dtype=np.int32) -> np.ndarray:
        return np.full((self.rows, self.cols), fill, dtype=dtype)

    # --- display ----------------------------------------------------------
    def image_rect(self) -> tuple[float, float, float, float]:
        """`(x, y, w, h)` to place an `ImageItem (rows, cols)` in world coordinates."""
        return (self.xmin, self.ymin, self.width, self.height)

    def line_segments(self) -> tuple[np.ndarray, np.ndarray]:
        """Grid line segments for a `connect='pairs'` plot."""
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
    """Default grid enclosing a top-down point cloud `xy` (N, 2).

    Non-finite points (NaN/inf, common for invalid lidar returns) are ignored; if none
    remain, a neutral default extent is returned rather than a NaN grid.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        return Grid(-10.0, 10.0, -10.0, 10.0, cell_size)
    lo = np.floor(xy.min(axis=0) - margin)
    hi = np.ceil(xy.max(axis=0) + margin)
    return Grid(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), cell_size)
