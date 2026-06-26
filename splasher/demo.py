"""Source synthétique multi-canaux pour démos/tests — aucune donnée externe.

Scène « conduite » : l'ego avance en +x. Canaux fournis :
- `lidar`        : sol bruité + obstacles (nuage dense),
- `lidar_haut`   : points en hauteur au-dessus des obstacles (nuage épars, distinct),
- `camera_avant` / `camera_arriere` : deux images factices,
- `pose`         : matrice 4x4 par frame (pour le cumul).
"""

from __future__ import annotations

import numpy as np

from .core.array_source import ArraySource
from .core.source import ChannelKind, ChannelSpec


def _make_image(h: int, w: int, t: int, n_frames: int, *, rear: bool = False) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    horizon = h // 2
    img[:horizon] = (140, 110, 90) if rear else (90, 120, 200)  # ciel (teinte différente derrière)
    img[horizon:] = (70, 110, 70)
    for r in range(horizon, h):
        frac = (r - horizon) / max(1, h - horizon)
        half = int((0.05 + 0.45 * frac) * w)
        c = w // 2
        img[r, max(0, c - half):min(w, c + half)] = (60, 60, 64)
    # « obstacle » mobile (sens inverse derrière)
    prog = t / max(1, n_frames - 1)
    bx = int((1 - prog if rear else prog) * (w - 50))
    img[horizon - 30:horizon + 10, bx:bx + 40] = (80, 150, 210) if rear else (210, 80, 60)
    return img


def make_demo_source(n_frames: int = 40, seed: int = 0) -> ArraySource:
    rng = np.random.default_rng(seed)
    h, w = 200, 360
    speed = 1.2

    n_obs = 9
    obs_x = rng.uniform(8.0, 70.0, n_obs)
    obs_y = rng.uniform(-12.0, 12.0, n_obs)
    obs_r = rng.uniform(0.6, 1.8, n_obs)
    obs_h = rng.uniform(1.0, 3.0, n_obs)

    specs = [
        ChannelSpec("lidar", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 4)),
        ChannelSpec("lidar_haut", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 4)),
        ChannelSpec("camera_avant", ChannelKind.IMAGE, np.dtype("uint8"), (h, w, 3)),
        ChannelSpec("camera_arriere", ChannelKind.IMAGE, np.dtype("uint8"), (h, w, 3)),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]

    frames: list[dict[str, np.ndarray]] = []
    for t in range(n_frames):
        ex = speed * t

        ng = 14000
        gx = rng.uniform(1.0, 40.0, ng)
        gy = rng.uniform(-20.0, 20.0, ng)
        gz = rng.normal(0.0, 0.03, ng)
        ground = [np.stack([gx, gy, gz, rng.uniform(0.1, 0.3, ng)], axis=1)]
        high = []

        for k in range(n_obs):
            xr = obs_x[k] - ex
            if 0.0 < xr < 40.0:
                m = 500
                px = xr + rng.normal(0.0, obs_r[k], m)
                py = obs_y[k] + rng.normal(0.0, obs_r[k], m)
                pz = rng.uniform(0.0, obs_h[k], m)
                ground.append(np.stack([px, py, pz, rng.uniform(0.6, 1.0, m)], axis=1))
                # nuage "haut" : points épars au-dessus de l'obstacle
                mh = 80
                hx = xr + rng.normal(0.0, obs_r[k] * 0.5, mh)
                hy = obs_y[k] + rng.normal(0.0, obs_r[k] * 0.5, mh)
                hz = obs_h[k] + rng.uniform(1.0, 3.0, mh)
                high.append(np.stack([hx, hy, hz, rng.uniform(0.2, 0.5, mh)], axis=1))

        lidar = np.concatenate(ground, axis=0).astype(np.float32)
        lidar_haut = (np.concatenate(high, axis=0) if high
                      else np.zeros((0, 4))).astype(np.float32)

        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = ex

        frames.append({
            "lidar": lidar,
            "lidar_haut": lidar_haut,
            "camera_avant": _make_image(h, w, t, n_frames),
            "camera_arriere": _make_image(h, w, t, n_frames, rear=True),
            "pose": pose,
        })

    return ArraySource(specs, frames)
