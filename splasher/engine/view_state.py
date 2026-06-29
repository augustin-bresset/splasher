"""Data exchanged between the engine and a front: `SessionInfo` (static) and
`ViewState` (dynamic, per request).

Deliberately *semantic*: we carry primitives (points, labels, scalar fields, ID
raster, boolean selection, raw images), **not** already-colorized pixels. Each front
decides how to draw them (the web front and any other may diverge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.grid import Grid
from ..core.labels import LabelSet
from ..core.source import ChannelSpec


@dataclass(frozen=True)
class SessionInfo:
    """~Static description of the session: what a front needs only once."""

    n_frames: int
    channels: list[ChannelSpec]
    labelset: LabelSet
    cloud_keys: list[str]
    image_keys: list[str]
    pose_key: str | None

    @property
    def has_pose(self) -> bool:
        return self.pose_key is not None


@dataclass
class ViewState:
    """Everything needed to render the current state (frame + accumulation + targets…).

    The arrays are aligned with each other: `point_labels[k]` is the label of point
    `points[k]` (already de-accumulated/aggregated for the current reference frame).
    """

    # --- geometry / content ---------------------------------------------
    grid: Grid
    points: np.ndarray                  # (N, 3+) accumulated *visible* points, reference frame
    point_labels: np.ndarray            # (N,) int64 aligned with `points` (ignore_id = unlabeled)
    point_channels: np.ndarray          # (N,) cloud channel index (→ cloud_keys), to filter per view
    bev_field: np.ndarray               # (rows, cols) float: scalar field per cell (NaN empty), per bev_mode
    grid_labels: np.ndarray | None      # (rows, cols) int: current frame raster, or None
    selection: np.ndarray | None        # (rows, cols) bool: selected cells, or None
    images: dict[str, np.ndarray] = field(default_factory=dict)  # *visible* image channels

    # --- dynamic UI state (to re-sync a front) --------------------------
    index: int = 0
    n_frames: int = 0
    active_class: int = 0
    active_targets: list[str] = field(default_factory=list)
    accum_radius: int = 0
    bev_mode: str = "height"
    tool: str = "paint"
    visible_clouds: list[str] = field(default_factory=list)
    visible_images: list[str] = field(default_factory=list)
