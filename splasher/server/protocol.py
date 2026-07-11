"""JSON (de)serialization of the engine — no FastAPI dependency.

numpy arrays are encoded as `{dtype, shape, data(base64)}`: compact, lossless, and
trivially decodable on the JS side (`atob` + a `TypedArray` over the buffer). We keep
the `ViewState` semantics (points + labels + scalar fields), never pixels.
"""

from __future__ import annotations

import base64

import numpy as np

from ..core.grid import Grid
from ..core.poses import pose_to_matrix
from ..core.source import ChannelSpec
from ..engine.view_state import SessionInfo, ViewState


# --------------------------------------------------------------- arrays
def encode_array(arr) -> dict | None:
    if arr is None:
        return None
    arr = np.ascontiguousarray(arr)
    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes()).decode("ascii"),
    }


def decode_array(d: dict | None):
    if d is None:
        return None
    buf = base64.b64decode(d["data"])
    return np.frombuffer(buf, dtype=np.dtype(d["dtype"])).reshape(d["shape"])


# ------------------------------------------------------------------ grid
def grid_to_dict(grid: Grid) -> dict:
    return {
        "xmin": grid.xmin, "xmax": grid.xmax,
        "ymin": grid.ymin, "ymax": grid.ymax,
        "cell_size": grid.cell_size, "rows": grid.rows, "cols": grid.cols,
    }


def grid_from_dict(d: dict) -> Grid:
    return Grid(d["xmin"], d["xmax"], d["ymin"], d["ymax"], d["cell_size"])


# ------------------------------------------------------------------ channels
def channelspec_to_dict(spec: ChannelSpec) -> dict:
    placement = None if spec.placement is None else pose_to_matrix(spec.placement).tolist()
    return {
        "name": spec.name,
        "kind": spec.kind.value,
        "dtype": None if spec.dtype is None else str(spec.dtype),
        "shape": None if spec.shape is None else list(spec.shape),
        "placement": placement,   # 4x4 ego-frame pose, or null (defaults to origin, forward)
    }


# --------------------------------------------------------------- session info
def session_info_to_dict(info: SessionInfo) -> dict:
    return {
        "n_frames": info.n_frames,
        "cloud_keys": info.cloud_keys,
        "image_keys": info.image_keys,
        "pose_key": info.pose_key,
        "has_pose": info.has_pose,
        "feature_names": list(info.feature_names),
        "labelset": info.labelset.to_dict(),
        "channels": [channelspec_to_dict(s) for s in info.channels],
    }


# --------------------------------------------------------------- view state
def view_state_to_dict(view: ViewState) -> dict:
    grid_labels = None if view.grid_labels is None else view.grid_labels.astype(np.int32)
    selection = None if view.selection is None else view.selection.astype(np.uint8)
    return {
        "grid": grid_to_dict(view.grid),
        "points": encode_array(np.asarray(view.points, dtype=np.float32)),
        "point_labels": encode_array(np.asarray(view.point_labels, dtype=np.int32)),
        "point_channels": encode_array(np.asarray(view.point_channels, dtype=np.int16)),
        "bev_field": encode_array(np.asarray(view.bev_field, dtype=np.float32)),
        "grid_labels": encode_array(grid_labels),
        "selection": encode_array(selection),
        "images": {k: encode_array(v) for k, v in view.images.items()},
        "index": view.index,
        "n_frames": view.n_frames,
        "active_class": view.active_class,
        "active_targets": view.active_targets,
        "accum_radius": view.accum_radius,
        "bev_mode": view.bev_mode,
        "tool": view.tool,
        "visible_clouds": view.visible_clouds,
        "visible_images": view.visible_images,
    }
