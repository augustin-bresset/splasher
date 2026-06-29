"""Label targets: what a rectangle selection produces.

- `GridTarget`  : id raster per sample — "grid as-is" output.
- `PointTarget` : `(N,)` labels per frame (cloud segmentation).

**Per-frame history**: each frame has its own undo stack. `undo` takes the current frame
and only undoes its last action. An accumulated brush stroke (de-accumulated over several
frames) is recorded, atomically, under the reference frame where it was painted.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .grid import Grid
from .projection import Rect, cells_in_rect, points_in_rect


class LabelTarget(Protocol):
    name: str

    def undo(self, frame_idx: int): ...


class GridTarget:
    """Id rasters per sample (+ a global raster). "grid" target."""

    name = "grid"

    def __init__(self, grid: Grid, ignore_id: int = 0) -> None:
        self.grid = grid
        self.ignore_id = ignore_id
        self._rasters: dict[int, np.ndarray] = {}
        self._global = grid.empty_raster(ignore_id)
        self._undo: dict[int, list] = {}

    def raster(self, frame_idx: int, scope: str = "sample") -> np.ndarray:
        if scope == "global":
            return self._global
        r = self._rasters.get(frame_idx)
        if r is None:
            r = self.grid.empty_raster(self.ignore_id)
            self._rasters[frame_idx] = r
        return r

    def has(self, frame_idx: int, scope: str = "sample") -> bool:
        return scope == "global" or frame_idx in self._rasters

    def rasters(self) -> dict[int, np.ndarray]:
        return self._rasters

    def global_raster(self) -> np.ndarray:
        return self._global

    def load_rasters(self, rasters: dict[int, np.ndarray], global_raster=None) -> None:
        self._rasters = {int(k): np.asarray(v) for k, v in rasters.items()}
        if global_raster is not None:
            self._global = np.asarray(global_raster)
        self._undo.clear()

    def apply(self, frame_idx: int, rect: Rect, class_id: int, scope: str = "sample") -> bool:
        si, sj = cells_in_rect(rect, self.grid)
        if si.start >= si.stop or sj.start >= sj.stop:
            return False
        return self._set(frame_idx, (si, sj), class_id, scope)

    def apply_mask(self, frame_idx: int, mask: np.ndarray, class_id: int,
                   scope: str = "sample") -> bool:
        """Paint a boolean cell mask `(rows, cols)` (selection)."""
        if mask is None or not mask.any():
            return False
        return self._set(frame_idx, mask, class_id, scope)

    def _set(self, frame_idx: int, sel, class_id: int, scope: str) -> bool:
        target = self.raster(frame_idx, scope)
        self._undo.setdefault(frame_idx, []).append((scope, sel, target[sel].copy()))
        target[sel] = class_id
        return True

    def clear(self, frame_idx: int, scope: str = "sample") -> None:
        target = self.raster(frame_idx, scope)
        self._undo.setdefault(frame_idx, []).append((scope, (slice(None), slice(None)), target.copy()))
        target[:] = self.ignore_id

    def undo(self, frame_idx: int):
        stack = self._undo.get(frame_idx)
        if not stack:
            return None
        scope, sel, prev = stack.pop()
        self.raster(frame_idx, scope)[sel] = prev
        return scope, frame_idx


class PointTarget:
    """`(N,) int64` labels per frame. "points" target (cloud segmentation).

    Labels are sized on the **full concatenation** of the frame's cloud channels (fixed
    order), hence independent of channel visibility.
    """

    name = "points"

    def __init__(self, ignore_id: int = 0) -> None:
        self.ignore_id = ignore_id
        self._labels: dict[int, np.ndarray] = {}
        self._undo: dict[int, list] = {}

    def has(self, frame_idx: int) -> bool:
        return frame_idx in self._labels

    def labels(self, frame_idx: int, n: int | None = None) -> np.ndarray | None:
        lab = self._labels.get(frame_idx)
        if (lab is None or (n is not None and len(lab) != n)) and n is not None:
            lab = np.full(n, self.ignore_id, dtype=np.int64)
            self._labels[frame_idx] = lab
        return lab

    def all_labels(self) -> dict[int, np.ndarray]:
        return self._labels

    def load_labels(self, labels: dict[int, np.ndarray]) -> None:
        self._labels = {int(k): np.asarray(v, dtype=np.int64) for k, v in labels.items()}
        self._undo.clear()

    def apply(self, frame_idx: int, rect: Rect, class_id: int, xy: np.ndarray) -> bool:
        mask = points_in_rect(xy, rect)
        if not mask.any():
            return False
        lab = self.labels(frame_idx, len(xy))
        self._undo.setdefault(frame_idx, []).append([(frame_idx, mask, lab[mask].copy())])
        lab[mask] = class_id
        return True

    def apply_scatter(self, ref_frame: int, frame_to_sel: dict[int, tuple[np.ndarray, int]],
                      class_id: int) -> bool:
        """Assign `class_id` to points spread over several frames (de-accumulation).

        `frame_to_sel`: `{frame_idx: (indices, n_points_of_the_frame)}`. The operation is
        recorded atomically under `ref_frame` (the current frame).
        """
        changes: list[tuple] = []
        for frame_idx, (indices, n) in frame_to_sel.items():
            if len(indices) == 0:
                continue
            lab = self.labels(frame_idx, n)
            changes.append((frame_idx, indices, lab[indices].copy()))
            lab[indices] = class_id
        if not changes:
            return False
        self._undo.setdefault(ref_frame, []).append(changes)
        return True

    def clear(self, frame_idx: int, xy: np.ndarray | None = None) -> None:
        lab = self._labels.get(frame_idx)
        if lab is None:
            return
        self._undo.setdefault(frame_idx, []).append([(frame_idx, slice(None), lab.copy())])
        lab[:] = self.ignore_id

    def undo(self, frame_idx: int):
        stack = self._undo.get(frame_idx)
        if not stack:
            return None
        for f, sel, prev in stack.pop():
            self._labels[f][sel] = prev
        return self.name, frame_idx
