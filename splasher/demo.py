"""Synthetic multi-channel source for demos/tests — no external data.

"Natural terrain" scene: the ego drives forward over Perlin-noise hills scattered
with trees and patchy grass (`terrain.py` builds the world as a heightfield mesh).
`lidar`, `lidar_top` and both cameras are *simulated* against that one consistent 3D
world — a real ray scan and a real ray-marched render — not hand-painted.

Point clouds are emitted in the **ego frame** (as `pose` composes them into world
space during accumulation): the terrain's own elevation change is what's left in
after subtracting the ego's current world position, so a hill ahead shows up as
rising ground exactly like a real ego-relative lidar frame would. Channels:

- `lidar`         : ray-cast scan of the terrain + trees (dense cloud), (N, 4)
                     [x, y, z, intensity], ego frame
- `lidar_top`     : sparse sample of tree-canopy points (points above the
                     obstacles), ego frame
- `camera_front` / `camera_rear` : ray-marched renders of the same scene,
                     (H, W, 3) uint8
- `pose`          : 4x4 matrix per frame (world placement, for accumulation) —
                     rides the terrain elevation
"""

from __future__ import annotations

import numpy as np

from . import terrain
from .core.array_source import ArraySource
from .core.source import ChannelKind, ChannelSpec

CAM_H, CAM_W = 170, 300
LIDAR_MOUNT = 1.8
LIDAR_TOP_MOUNT = 2.4
CAM_FRONT_PLACEMENT = np.array([1.6, 0.0, 1.5], np.float32)
# camera_rear faces backward: 180° about z, quaternion [x, y, z, qx, qy, qz, qw].
CAM_REAR_PLACEMENT = np.array([-1.6, 0.0, 1.5, 0.0, 0.0, 1.0, 0.0], np.float32)


def _canopy_top_points(
    scene: terrain.Scene, ego_xy: np.ndarray, rng: np.random.Generator, radius: float = 45.0
):
    """Sparse sample of points on the upper half of nearby canopies — a decorative
    "above the obstacles" cloud, ego-relative."""
    trees = scene.trees
    pts = []
    for i in range(len(trees.pos)):
        d = np.hypot(*(trees.pos[i] - ego_xy))
        if d > radius:
            continue
        cx, cy = trees.pos[i]
        cz = trees.ground_z[i] + trees.trunk_h[i] + trees.canopy_r[i] * 0.65
        m = 80
        # points on the upper hemisphere of the canopy sphere
        theta = rng.uniform(0, 2 * np.pi, m)
        phi = rng.uniform(0, np.pi / 2, m)  # upper half only
        r = trees.canopy_r[i]
        x = cx + r * np.sin(phi) * np.cos(theta)
        y = cy + r * np.sin(phi) * np.sin(theta)
        z = cz + r * np.cos(phi)
        intensity = rng.uniform(0.2, 0.5, m)
        pts.append(np.stack([x, y, z, intensity], axis=1))
    if not pts:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(pts, axis=0).astype(np.float32)


def make_demo_source(n_frames: int = 40, seed: int = 0) -> ArraySource:
    rng = np.random.default_rng(seed)
    speed = 1.2
    span = speed * max(n_frames - 1, 1)

    scene = terrain.build_scene(
        seed=seed,
        x_range=(-8.0, span + 26.0),
        y_range=(-22.0, 22.0),
        n_trees=max(3, min(9, n_frames // 4 + 3)),
        hill_amplitude=2.8,
        grid_shape=(192, 192),
    )

    specs = [
        ChannelSpec(
            "lidar",
            ChannelKind.POINTCLOUD,
            np.dtype("float32"),
            (None, 4),
            placement=np.array([0.0, 0.0, LIDAR_MOUNT], np.float32),
        ),
        ChannelSpec(
            "lidar_top",
            ChannelKind.POINTCLOUD,
            np.dtype("float32"),
            (None, 4),
            placement=np.array([0.0, 0.0, LIDAR_TOP_MOUNT], np.float32),
        ),
        ChannelSpec(
            "camera_front",
            ChannelKind.IMAGE,
            np.dtype("uint8"),
            (CAM_H, CAM_W, 3),
            placement=CAM_FRONT_PLACEMENT,
        ),
        ChannelSpec(
            "camera_rear",
            ChannelKind.IMAGE,
            np.dtype("uint8"),
            (CAM_H, CAM_W, 3),
            placement=CAM_REAR_PLACEMENT,
        ),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]

    frames: list[dict[str, np.ndarray]] = []
    for t in range(n_frames):
        ex = speed * t
        ez = float(scene.field.height(np.array([ex]), np.array([0.0]))[0])
        ego = np.array([ex, 0.0, ez])

        lidar_pos = ego + np.array([0.0, 0.0, LIDAR_MOUNT])
        cam_front_pos = ego + CAM_FRONT_PLACEMENT
        cam_rear_pos = ego + CAM_REAR_PLACEMENT[:3]

        points_world, _materials = terrain.scan_lidar(
            scene, lidar_pos, rng, n_rings=26, n_az=480, march_steps=36
        )
        lidar = points_world.copy()
        lidar[:, :3] -= ego  # world -> ego frame

        lidar_top = _canopy_top_points(scene, ego[:2], rng)
        lidar_top[:, :3] -= ego

        camera_front = terrain.render_camera(scene, cam_front_pos, CAM_H, CAM_W, march_steps=34, rng=rng)
        camera_rear = terrain.render_camera(
            scene, cam_rear_pos, CAM_H, CAM_W, march_steps=34, yaw_deg=180.0, rng=rng
        )

        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = ex
        pose[2, 3] = ez

        frames.append(
            {
                "lidar": lidar.astype(np.float32),
                "lidar_top": lidar_top.astype(np.float32),
                "camera_front": camera_front,
                "camera_rear": camera_rear,
                "pose": pose,
            }
        )

    return ArraySource(specs, frames)
