"""Colorization of a `ViewState` — pure numpy helpers, optional.

The view-state is *semantic*: the front does the colorization. These helpers provide
the reference rendering (viridis on height + label override) shared by Python fronts.
A web front reimplements the equivalent in JS.
"""

from __future__ import annotations

import numpy as np

from ..core.colormap import colormap
from ..core.labels import LabelSet


def cloud_colors(points: np.ndarray, point_labels: np.ndarray,
                 labelset: LabelSet) -> np.ndarray | None:
    """RGBA colors `(N, 4)` float [0,1]: viridis on z, overridden by the class color."""
    if points is None or len(points) == 0:
        return None
    colors = colormap(points[:, 2])
    mask = point_labels != labelset.ignore_id
    if mask.any():
        lut = labelset.lut(alpha=255, max_id=int(point_labels.max()))
        colors[mask] = lut[point_labels[mask]].astype(np.float32) / 255.0
    return colors


def selection_rgba(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Translucent highlight of the selected cells (`(rows, cols, 4)` uint8)."""
    if mask is None or not mask.any():
        return None
    rgba = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    rgba[mask] = (225, 210, 165, 90)
    return rgba
