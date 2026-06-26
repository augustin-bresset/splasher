"""Test headless du round-trip de session (save -> load)."""

import numpy as np

from splasher import Grid
from splasher.core.io import load_session, save_session
from splasher.core.labels import LabelSet
from splasher.core.target import GridTarget, PointTarget


def test_session_roundtrip(tmp_path):
    grid = Grid(0.0, 8.0, 0.0, 6.0, 1.0)
    ls = LabelSet.default()

    gt = GridTarget(grid, ignore_id=0)
    gt.apply(3, (1.0, 1.0, 4.0, 4.0), class_id=2)

    pt = PointTarget(ignore_id=0)
    xy = np.array([[0.5, 0.5], [2.0, 2.0]])
    pt.apply(3, (1.0, 1.0, 3.0, 3.0), class_id=1, xy=xy)

    out = save_session(tmp_path, grid=grid, labelset=ls, grid_target=gt, point_target=pt)
    assert (out / "session.json").exists()
    assert (out / "grid" / "frame_00003.npy").exists()
    assert (out / "points" / "frame_00003.npy").exists()

    data = load_session(out)
    assert data["grid"].shape == grid.shape
    assert data["labelset"].name_of(2) == ls.name_of(2)

    np.testing.assert_array_equal(data["grid_labels"][3], gt.raster(3))
    np.testing.assert_array_equal(data["point_labels"][3], pt.labels(3))

    # ré-injection dans des cibles neuves
    gt2 = GridTarget(grid)
    gt2.load_rasters(data["grid_labels"])
    assert (gt2.raster(3) == gt.raster(3)).all()
