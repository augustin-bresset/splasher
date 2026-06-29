"""Headless tests for the BEV projection and rectangle selection."""

import numpy as np

from splasher import Grid
from splasher.core.projection import (
    bev_count,
    bev_image,
    bev_max_height,
    cells_in_rect,
    points_in_rect,
)


def _grid():
    return Grid(0.0, 4.0, 0.0, 4.0, 1.0)  # 4x4


def test_bev_max_height_and_empty():
    g = _grid()
    pts = np.array([[0.5, 0.5, 1.0], [0.6, 0.6, 3.0], [3.5, 3.5, 2.0]])
    h = bev_max_height(pts, g)
    assert h.shape == (4, 4)
    assert h[0, 0] == 3.0  # max of the two points in cell (0,0)
    assert h[3, 3] == 2.0
    assert np.isnan(h[2, 2])  # empty cell


def test_bev_count():
    g = _grid()
    pts = np.array([[0.5, 0.5, 0.0], [0.6, 0.6, 0.0]])
    c = bev_count(pts, g)
    assert c[0, 0] == 2.0
    assert np.isnan(c[1, 1])


def test_bev_image_alpha():
    g = _grid()
    pts = np.array([[0.5, 0.5, 1.0]])
    img = bev_image(bev_max_height(pts, g))
    assert img.shape == (4, 4, 4)
    assert img[0, 0, 3] > 0  # filled cell, opaque
    assert img[2, 2, 3] == 0  # empty cell, transparent


def test_cells_in_rect():
    g = _grid()
    si, sj = cells_in_rect((1.2, 0.5, 2.9, 3.4), g)
    assert (si.start, si.stop) == (0, 4)
    assert (sj.start, sj.stop) == (1, 3)


def test_points_in_rect():
    xy = np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]])
    mask = points_in_rect(xy, (1.0, 1.0, 3.0, 3.0))
    assert mask.tolist() == [False, True, False]
    # corner order does not matter
    assert points_in_rect(xy, (3.0, 3.0, 1.0, 1.0)).tolist() == [False, True, False]
