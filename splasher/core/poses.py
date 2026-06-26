"""Poses rigides : conversion en matrice 4x4, inversion, transformation de points.

Réimplémenté ici (≈30 lignes) pour ne pas dépendre d'apairo_visu / Open3D.
Formats acceptés : `(4, 4)`, `(3, 4)`, ou vecteur `(7,)` `[x, y, z, qx, qy, qz, qw]`.
"""

from __future__ import annotations

import numpy as np


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    """Normalise une pose en matrice homogène `(4, 4)` float64."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape == (4, 4):
        return pose.copy()
    if pose.shape == (3, 4):
        T = np.eye(4)
        T[:3, :4] = pose
        return T
    if pose.shape == (7,):
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(pose[3:])
        T[:3, 3] = pose[:3]
        return T
    raise ValueError(f"forme de pose non supportée : {pose.shape}")


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse d'une transformation rigide `(4, 4)`."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Applique `T` aux colonnes xyz de `points` (N, 3+) ; les colonnes en plus sont conservées."""
    if len(points) == 0:
        return points.copy()
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, :3] = points[:, :3] @ T[:3, :3].T + T[:3, 3]
    return out
