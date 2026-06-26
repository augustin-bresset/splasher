"""Tests headless de PointTarget (labels par point)."""

import numpy as np

from splasher.core.target import PointTarget


def test_apply_assigns_class_to_points_in_rect():
    xy = np.array([[0.0, 0.0], [2.0, 2.0], [5.0, 5.0]])
    t = PointTarget(ignore_id=0)
    assert t.apply(0, (1.0, 1.0, 3.0, 3.0), class_id=7, xy=xy)
    lab = t.labels(0)
    assert lab.tolist() == [0, 7, 0]


def test_apply_no_points_returns_false():
    xy = np.array([[0.0, 0.0]])
    t = PointTarget()
    assert t.apply(0, (10.0, 10.0, 20.0, 20.0), class_id=1, xy=xy) is False
    assert not t.has(0)


def test_undo_restores():
    xy = np.array([[0.0, 0.0], [2.0, 2.0]])
    t = PointTarget()
    t.apply(0, (1.0, 1.0, 3.0, 3.0), class_id=3, xy=xy)
    t.apply(0, (-1.0, -1.0, 1.0, 1.0), class_id=4, xy=xy)
    assert t.labels(0).tolist() == [4, 3]
    t.undo(0)
    assert t.labels(0).tolist() == [0, 3]


def test_clear():
    xy = np.array([[2.0, 2.0]])
    t = PointTarget()
    t.apply(0, (1.0, 1.0, 3.0, 3.0), class_id=5, xy=xy)
    t.clear(0)
    assert t.labels(0).tolist() == [0]
