"""Tests headless de la grille (numpy pur)."""

import numpy as np
import pytest

from splasher import Grid, grid_from_points


def test_dims():
    g = Grid(0.0, 10.0, 0.0, 4.0, 1.0)
    assert g.cols == 10
    assert g.rows == 4
    assert g.shape == (4, 10)
    assert g.empty_raster(fill=-1).shape == (4, 10)


def test_dims_ceil():
    g = Grid(0.0, 10.0, 0.0, 5.0, 3.0)
    assert g.cols == 4  # ceil(10/3)
    assert g.rows == 2  # ceil(5/3)


def test_world_to_cell():
    g = Grid(0.0, 10.0, 0.0, 10.0, 1.0)
    xy = np.array([[0.5, 0.5], [9.5, 9.5], [-1.0, 5.0], [5.0, 11.0]])
    ij, valid = g.world_to_cell(xy)
    assert ij[0].tolist() == [0, 0]  # [i(y), j(x)]
    assert ij[1].tolist() == [9, 9]
    assert valid.tolist() == [True, True, False, False]


def test_cell_to_world_roundtrip():
    g = Grid(-5.0, 5.0, -5.0, 5.0, 2.0)
    x, y = g.cell_to_world(0, 0)
    ij, valid = g.world_to_cell(np.array([[x, y]]))
    assert valid[0]
    assert ij[0].tolist() == [0, 0]


def test_invalid():
    with pytest.raises(ValueError):
        Grid(0.0, 0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        Grid(0.0, 1.0, 0.0, 1.0, 0.0)


def test_grid_from_points():
    xy = np.array([[1.0, 2.0], [3.0, 8.0]])
    g = grid_from_points(xy, cell_size=1.0, margin=1.0)
    assert g.xmin <= 0.0 and g.ymin <= 1.0
    assert g.xmax >= 4.0 and g.ymax >= 9.0


def test_line_segments_shape():
    g = Grid(0.0, 4.0, 0.0, 2.0, 1.0)
    xs, ys = g.line_segments()
    # (cols+1) verticales + (rows+1) horizontales, 2 points chacune
    assert len(xs) == len(ys) == 2 * ((g.cols + 1) + (g.rows + 1))
