"""`Session` — the labeling engine, with no UI dependency whatsoever.

Holds all the state of a session (current frame, grid, targets, accumulation,
visibility, tool, selection, active class) and exposes the **operations** as pure
methods (numpy only). No method draws: the rendered state is read via
`view_state()` / `info()`.

Plugging in any front means driving this `Session` (the web front does it over HTTP).
"""

from __future__ import annotations

import numpy as np

from ..core.accumulate import Accumulation, accumulate, window_indices
from ..core.grid import Grid, grid_from_points
from ..core.io import load_session, save_session
from ..core.labels import LabelSet
from ..core.projection import (
    Rect,
    bev_field,
    cells_in_rect,
    points_in_rect,
)
from ..core.source import (
    ChannelKind,
    Source,
    channels_of_kind,
    ordered_features,
    point_features,
)
from ..core.target import GridTarget, PointTarget
from .view_state import SessionInfo, ViewState

Tool = str  # "paint" (direct painting) | "select" (marquee selection)


class Session:
    """State + operations of a labeling session, drivable by any front."""

    def __init__(self, source: Source, labelset: LabelSet | None = None) -> None:
        self.labelset = labelset or LabelSet.default()
        self.tool: Tool = "paint"
        self.bev_mode = "height"   # BEV underlay: "height" | "intensity" | "density" | "normal"
        self.active_targets: set[str] = {"grid"}
        paintable = self.labelset.paintable
        self.active_class = paintable[0].id if paintable else self.labelset.ignore_id
        self.set_source(source)

    def set_source(self, source: Source, keep_grid: bool = False) -> None:
        """(Re)bind the dataset source and reset cloud-derived state.

        With `keep_grid` the grid **and its labels persist** (the BEV grid is independent of
        any single lidar): a file-viewer can swap/combine the displayed cloud as a reference
        without wiping the labeling. The point target is always reset (it is cloud-sized).
        """
        self.source = source
        self.index = 0
        self.cloud_keys = channels_of_kind(source, ChannelKind.POINTCLOUD)
        self.image_keys = channels_of_kind(source, ChannelKind.IMAGE)
        pose_keys = channels_of_kind(source, ChannelKind.POSE)
        self.pose_key = pose_keys[0] if pose_keys else None
        # Per-point scalar features (`<cloud>_<suffix>` sibling channels) → trailing columns.
        self.point_features = point_features(source, self.cloud_keys)
        self.feature_names = self._feature_names()
        self.set_bev_mode(self.bev_mode)   # drop a feature underlay the new source no longer has
        self.accum_radius = 0
        self.visible_clouds: set[str] = set(self.cloud_keys)
        self.visible_images: set[str] = set(self.image_keys)
        self.selection: np.ndarray | None = None
        if not keep_grid:
            self.grid = self._default_grid()
            self.grid_target = GridTarget(self.grid, ignore_id=self.labelset.ignore_id)
        self.point_target = PointTarget(ignore_id=self.labelset.ignore_id)

    def _feature_names(self) -> list[str]:
        """Ordered per-point scalar features the views can color by. Sibling scalar channels
        (apairo convention) plus, if any cloud has a native 4th column and no sibling one, an
        `intensity` feature (KITTI-style x,y,z,intensity)."""
        names = {feat for feats in self.point_features.values() for feat in feats}
        if "intensity" not in names and len(self.source) and self.cloud_keys:
            frame = self.source[0]
            for k in self.cloud_keys:
                a = frame.channels.get(k)
                if a is not None and np.asarray(a).ndim == 2 and np.asarray(a).shape[1] >= 4:
                    names.add("intensity")
                    break
        return ordered_features(names)

    # ================================================================ info
    def info(self) -> SessionInfo:
        return SessionInfo(
            n_frames=len(self.source),
            channels=self.source.channels(),
            labelset=self.labelset,
            cloud_keys=list(self.cloud_keys),
            image_keys=list(self.image_keys),
            pose_key=self.pose_key,
            feature_names=list(self.feature_names),
        )

    @property
    def accum_enabled(self) -> bool:
        return self.pose_key is not None and len(self.source) > 1

    def grid_labelled_count(self) -> int:
        """Number of frames that own a grid raster (to confirm a grid reset)."""
        return len(self.grid_target.rasters())

    # ============================================================ commands
    def set_frame(self, index: int) -> None:
        self.index = int(np.clip(index, 0, max(0, len(self.source) - 1)))

    def set_active_class(self, class_id: int) -> None:
        self.active_class = int(class_id)

    def set_tool(self, tool: Tool) -> None:
        self.tool = "select" if tool == "select" else "paint"
        if self.tool != "select":
            self.selection = None

    def set_active_targets(self, targets) -> None:
        self.active_targets = {t for t in targets if t in ("grid", "points")}

    def toggle_target(self, name: str, on: bool) -> None:
        if name not in ("grid", "points"):
            return
        (self.active_targets.add if on else self.active_targets.discard)(name)

    MAX_ACCUM = 50              # accumulating more frames is both meaningless and very expensive
    MAX_VIEW_POINTS = 300_000  # cap points sent for *display* (BEV/labeling stay full-res server-side)

    def set_accum_radius(self, radius: int) -> None:
        cap = min(self.MAX_ACCUM, max(0, len(self.source) - 1))
        self.accum_radius = max(0, min(int(radius), cap))

    def set_bev_mode(self, mode: str) -> None:
        """BEV underlay: 'height' | 'density' | 'normal', or any per-point feature name."""
        valid = mode in ("height", "density", "normal") or mode in self.feature_names
        self.bev_mode = mode if valid else "height"

    def set_labelset(self, data: dict) -> None:
        """Replace the labeling class set (ids/names/colors). `ignore_id` stays the unlabeled id.

        Existing rasters/labels keep their integer ids; only the class set/colors change.
        The active class is reset if it no longer exists.
        """
        self.labelset = LabelSet.from_dict(data)
        self.grid_target.ignore_id = self.labelset.ignore_id
        self.point_target.ignore_id = self.labelset.ignore_id
        ids = {c.id for c in self.labelset.classes}
        if self.active_class not in ids or self.active_class == self.labelset.ignore_id:
            paintable = self.labelset.paintable
            self.active_class = paintable[0].id if paintable else self.labelset.ignore_id

    def set_visible_clouds(self, names) -> None:
        self.visible_clouds = {n for n in names if n in self.cloud_keys}

    def set_visible_images(self, names) -> None:
        self.visible_images = {n for n in names if n in self.image_keys}

    def paint_rect(self, rect: Rect) -> bool:
        """Apply the active class to the active targets over `rect`. Returns True if changed."""
        return self._apply_class_rect(rect, self.active_class)

    def erase_rect(self, rect: Rect) -> bool:
        """Unlabel (set to `ignore_id`) the active targets over `rect`. Returns True if changed."""
        return self._apply_class_rect(rect, self.labelset.ignore_id)

    def _apply_class_rect(self, rect: Rect, cls: int) -> bool:
        changed = False
        if "grid" in self.active_targets:
            changed |= self.grid_target.apply(self.index, rect, cls)
        if "points" in self.active_targets:
            acc = self._accumulated()
            changed |= self._paint_points(acc, points_in_rect(acc.xy, rect), cls)
        return changed

    def select_rect(self, rect: Rect, op: str = "add") -> bool:
        """Update the cell selection with `rect`. `op` = 'add' | 'subtract' | 'replace'."""
        si, sj = cells_in_rect(rect, self.grid)
        if si.start >= si.stop or sj.start >= sj.stop:
            return False
        rect_mask = np.zeros(self.grid.shape, dtype=bool)
        rect_mask[si, sj] = True
        if op == "replace" or self.selection is None:
            self.selection = rect_mask if op != "subtract" else np.zeros(self.grid.shape, dtype=bool)
        elif op == "subtract":
            self.selection = self.selection & ~rect_mask
        else:  # add
            self.selection = self.selection | rect_mask
        return True

    def apply_selection(self) -> bool:
        """Apply the active class to the whole current selection, then clear it."""
        if self.selection is None or not self.selection.any():
            return False
        cls = self.active_class
        changed = False
        if "grid" in self.active_targets:
            changed |= self.grid_target.apply_mask(self.index, self.selection, cls)
        if "points" in self.active_targets:
            acc = self._accumulated()
            ij, valid = self.grid.world_to_cell(acc.xy)
            hit = np.zeros(len(acc.xy), dtype=bool)
            hit[valid] = self.selection[ij[valid, 0], ij[valid, 1]]
            changed |= self._paint_points(acc, hit, cls)
        if changed:
            self.selection = None
        return changed

    def clear_selection(self) -> None:
        self.selection = None

    def clear_frame(self) -> None:
        """Clear the current frame's labels for the active targets."""
        if "grid" in self.active_targets:
            self.grid_target.clear(self.index)
        if "points" in self.active_targets:
            self.point_target.clear(self.index)

    def undo(self) -> None:
        """Undo the last action (per frame) on the active targets."""
        if "grid" in self.active_targets:
            self.grid_target.undo(self.index)
        if "points" in self.active_targets:
            self.point_target.undo(self.index)

    def commit_grid(self, grid: Grid) -> None:
        """Replace the active grid (resets the grid rasters, keeps the point labels).

        The optional confirmation (`grid_labelled_count() > 0`) is left to the front.
        """
        self.grid = grid
        self.grid_target = GridTarget(grid, ignore_id=self.labelset.ignore_id)
        self.selection = None

    def save(self, out_dir):
        return save_session(out_dir, grid=self.grid, labelset=self.labelset,
                            grid_target=self.grid_target, point_target=self.point_target)

    def load(self, out_dir) -> None:
        data = load_session(out_dir)
        self.grid = data["grid"]
        self.grid_target = GridTarget(self.grid, ignore_id=self.labelset.ignore_id)
        self.grid_target.load_rasters(data["grid_labels"])
        self.point_target = PointTarget(ignore_id=self.labelset.ignore_id)
        self.point_target.load_labels(data["point_labels"])
        self.selection = None

    # ============================================================== render
    def view_state(self) -> ViewState:
        """Build the current render state (a single accumulation).

        `points` carries **all** cloud channels (with `point_channels`): each front view
        filters on its own. `visible_clouds` only drives the BEV layer (height map) and
        which clouds get labeled — never what the 3D views can show.
        """
        acc = self._accumulated()
        vis = acc.visible_mask(self._visible_cloud_indices())
        labels = self._accumulated_labels(acc)

        # Cap points sent for display; BEV field below still uses the full accumulation.
        pts, plabels, pchans = acc.points, labels, acc.chan_id
        if len(pts) > self.MAX_VIEW_POINTS:
            step = int(np.ceil(len(pts) / self.MAX_VIEW_POINTS))
            pts, plabels, pchans = pts[::step], plabels[::step], pchans[::step]

        grid_labels = (
            self.grid_target.raster(self.index)
            if self.grid_target.has(self.index)
            else None
        )

        images: dict[str, np.ndarray] = {}
        if self.image_keys:
            frame = self.source[self.index]
            for key in self.image_keys:
                if key in self.visible_images:
                    img = frame.channels.get(key)
                    if img is not None:
                        images[key] = img

        return ViewState(
            grid=self.grid,
            points=pts,
            point_labels=plabels,
            point_channels=pchans,
            bev_field=bev_field(acc.points[vis], self.grid, self.bev_mode, self.feature_names),
            grid_labels=grid_labels,
            selection=self.selection,
            images=images,
            index=self.index,
            n_frames=len(self.source),
            active_class=self.active_class,
            active_targets=sorted(self.active_targets),
            accum_radius=self.accum_radius,
            bev_mode=self.bev_mode,
            tool=self.tool,
            visible_clouds=sorted(self.visible_clouds),
            visible_images=sorted(self.visible_images),
        )

    # ============================================================== internals
    def _frame_points(self, frame) -> np.ndarray:
        parts = [frame.channels[k] for k in self.cloud_keys
                 if frame.channels.get(k) is not None and len(frame.channels[k])]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), np.float32)

    def _default_grid(self) -> Grid:
        if len(self.source) == 0 or not self.cloud_keys:
            return Grid(-20.0, 20.0, -20.0, 20.0, 1.0)
        return grid_from_points(self._frame_points(self.source[0])[:, :2], cell_size=1.0)

    def _visible_cloud_indices(self) -> list[int]:
        return [i for i, k in enumerate(self.cloud_keys) if k in self.visible_clouds]

    def _accumulated(self) -> Accumulation:
        """Accumulate over **all** cloud channels (stable point_id); ±radius via poses."""
        n = len(self.source)
        if n == 0:
            return accumulate(self.source, self.index, [], self.cloud_keys, self.pose_key,
                              self.point_features, self.feature_names)
        if self.accum_radius > 0 and self.pose_key is not None:
            idx = window_indices(self.index, self.accum_radius, n)
            return accumulate(self.source, self.index, idx, self.cloud_keys, self.pose_key,
                              self.point_features, self.feature_names)
        return accumulate(self.source, self.index, [self.index], self.cloud_keys, self.pose_key,
                          self.point_features, self.feature_names)

    def _accumulated_labels(self, acc: Accumulation) -> np.ndarray:
        """Per-point labels aligned with `acc.points` (ignore_id by default, reverse de-accumulation)."""
        lab = np.full(len(acc.points), self.labelset.ignore_id, np.int64)
        for f in np.unique(acc.frame_id):
            f = int(f)
            if not self.point_target.has(f):
                continue
            src = self.point_target.labels(f)
            sel = acc.frame_id == f
            pid = acc.point_id[sel]
            if len(pid) and int(pid.max()) < len(src):
                lab[sel] = src[pid]
        return lab

    def _paint_points(self, acc: Accumulation, point_mask: np.ndarray, cls: int) -> bool:
        """De-accumulate a point mask (from the `acc` accumulation) back to each source frame."""
        mask = point_mask & acc.visible_mask(self._visible_cloud_indices())
        if not mask.any():
            return False
        fids, pids = acc.frame_id[mask], acc.point_id[mask]
        frame_to_sel = {int(f): (pids[fids == f], acc.counts[int(f)]) for f in np.unique(fids)}
        return self.point_target.apply_scatter(self.index, frame_to_sel, cls)
