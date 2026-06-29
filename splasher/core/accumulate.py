"""Frame accumulation by pose registration, with traceability for de-accumulation.

We accumulate a window of frames **into the reference frame's frame of reference** (the
current frame): each frame `j` is brought back by `inv(pose_ref) @ pose_j`. Each
accumulated point keeps:
- `frame_id`: source frame,
- `chan_id` : index of the source cloud channel (within `cloud_keys`),
- `point_id`: index of the point within the **full concatenation** of that frame's cloud
  channels (order = `cloud_keys`), fixed regardless of channel visibility.

This makes it possible to **de-accumulate** labels painted on the accumulated cloud back
to each source frame, and to **filter by channel** without ever misaligning the point
labels (which stay sized on the full concatenation).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .poses import invert, pose_to_matrix, transform_points


@dataclass
class Accumulation:
    points: np.ndarray          # (M, 3+) in the reference frame
    frame_id: np.ndarray        # (M,) source frame
    chan_id: np.ndarray         # (M,) source cloud channel index
    point_id: np.ndarray        # (M,) index within the frame's full concatenation
    counts: dict[int, int] = field(default_factory=dict)  # frame -> total number of points

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    def visible_mask(self, visible_chan_indices) -> np.ndarray:
        """Mask of the points whose channel is in `visible_chan_indices`."""
        if len(self.chan_id) == 0:
            return np.zeros(0, dtype=bool)
        return np.isin(self.chan_id, np.asarray(list(visible_chan_indices), dtype=np.int64))


def window_indices(ref_idx: int, radius: int, n_frames: int) -> list[int]:
    """Window `[ref-radius, ref+radius]` clamped to `[0, n_frames)`."""
    lo = max(0, ref_idx - radius)
    hi = min(n_frames, ref_idx + radius + 1)
    return list(range(lo, hi))


def accumulate(source, ref_idx: int, indices: list[int], cloud_keys: list[str],
               pose_key: str | None = None) -> Accumulation:
    """Accumulate `indices` into the frame of `ref_idx`. `pose_key=None` -> identity."""
    p_ref_inv = None
    if pose_key is not None:
        ref_pose = source[ref_idx].channels.get(pose_key)
        if ref_pose is not None:
            p_ref_inv = invert(pose_to_matrix(ref_pose))

    pts, fids, cids, pids = [], [], [], []
    counts: dict[int, int] = {}
    for j in indices:
        # Pose-based accumulation but no reference pose → can't register neighbors; keep ref only.
        if pose_key is not None and p_ref_inv is None and j != ref_idx:
            continue
        frame = source[j]
        T = None  # identity (reference frame, or no pose)
        if p_ref_inv is not None and j != ref_idx:
            pj = frame.channels.get(pose_key)
            if pj is None:
                continue  # this frame lacks its pose → can't register it, skip its points
            T = p_ref_inv @ pose_to_matrix(pj)

        offset = 0
        for ci, key in enumerate(cloud_keys):
            p = frame.channels.get(key)
            if p is None or len(p) == 0:
                continue
            q = np.asarray(p, dtype=np.float64).copy() if T is None else transform_points(p, T)
            m = len(p)
            pts.append(q)
            fids.append(np.full(m, j, dtype=np.int64))
            cids.append(np.full(m, ci, dtype=np.int64))
            pids.append(offset + np.arange(m, dtype=np.int64))
            offset += m
        counts[j] = offset

    if pts:
        return Accumulation(
            np.concatenate(pts, axis=0),
            np.concatenate(fids),
            np.concatenate(cids),
            np.concatenate(pids),
            counts,
        )
    return Accumulation(
        np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), counts
    )
