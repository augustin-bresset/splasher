"""Cumul de frames par registration des poses, avec traçabilité pour le décumul.

On accumule une fenêtre de frames **dans le repère du frame de référence** (le frame
courant) : chaque frame `j` est ramené par `inv(pose_ref) @ pose_j`. Chaque point
accumulé garde :
- `frame_id` : frame source,
- `chan_id`  : indice du canal nuage source (dans `cloud_keys`),
- `point_id` : indice du point dans **la concaténation complète** des canaux nuage de
  ce frame (ordre = `cloud_keys`), fixe quelle que soit la visibilité des canaux.

Cela permet de **décumuler** des labels peints sur le nuage cumulé vers chaque frame
d'origine, et de **filtrer par canal** sans jamais désaligner les labels points
(qui restent dimensionnés sur la concaténation complète).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .poses import invert, pose_to_matrix, transform_points


@dataclass
class Accumulation:
    points: np.ndarray          # (M, 3+) dans le repère de référence
    frame_id: np.ndarray        # (M,) frame source
    chan_id: np.ndarray         # (M,) indice de canal nuage source
    point_id: np.ndarray        # (M,) indice dans la concaténation complète du frame
    counts: dict[int, int] = field(default_factory=dict)  # frame -> nb total de points

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    def visible_mask(self, visible_chan_indices) -> np.ndarray:
        """Masque des points dont le canal est dans `visible_chan_indices`."""
        if len(self.chan_id) == 0:
            return np.zeros(0, dtype=bool)
        return np.isin(self.chan_id, np.asarray(list(visible_chan_indices), dtype=np.int64))


def window_indices(ref_idx: int, radius: int, n_frames: int) -> list[int]:
    """Fenêtre `[ref-radius, ref+radius]` bornée à `[0, n_frames)`."""
    lo = max(0, ref_idx - radius)
    hi = min(n_frames, ref_idx + radius + 1)
    return list(range(lo, hi))


def accumulate(source, ref_idx: int, indices: list[int], cloud_keys: list[str],
               pose_key: str | None = None) -> Accumulation:
    """Accumule `indices` dans le repère de `ref_idx`. `pose_key=None` -> identité."""
    p_ref_inv = None
    if pose_key is not None:
        p_ref_inv = invert(pose_to_matrix(source[ref_idx].channels[pose_key]))

    pts, fids, cids, pids = [], [], [], []
    counts: dict[int, int] = {}
    for j in indices:
        frame = source[j]
        if pose_key is not None and j != ref_idx:
            T = p_ref_inv @ pose_to_matrix(frame.channels[pose_key])
        else:
            T = None  # identité (frame de référence, ou pas de pose)

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
