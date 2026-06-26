"""Tests headless : LabelSet (colorize, IO) et GridTarget (apply/undo/clear)."""

import numpy as np

from splasher import Grid
from splasher.core.labels import LabelClass, LabelSet
from splasher.core.target import GridTarget


def test_labelset_colorize_ignore_transparent():
    ls = LabelSet.default()
    raster = np.array([[0, 1], [2, 3]], dtype=np.int32)
    rgba = ls.colorize(raster)
    assert rgba.shape == (2, 2, 4)
    assert rgba[0, 0, 3] == 0  # ignore -> transparent
    assert tuple(rgba[0, 1, :3]) == (60, 200, 70)  # classe 1
    assert rgba[1, 0, 3] == 255  # classe 2 opaque


def test_labelset_json_roundtrip(tmp_path):
    ls = LabelSet([LabelClass(0, "void", (0, 0, 0)), LabelClass(5, "x", (1, 2, 3))], ignore_id=0)
    p = tmp_path / "labels.json"
    ls.save(p)
    back = LabelSet.load(p)
    assert back.ignore_id == 0
    assert back.name_of(5) == "x"
    assert back.color_of(5) == (1, 2, 3)


def test_gridtarget_apply_and_undo():
    g = Grid(0.0, 4.0, 0.0, 4.0, 1.0)
    t = GridTarget(g, ignore_id=0)
    assert not t.has(0)
    assert t.apply(0, (1.0, 1.0, 3.0, 3.0), class_id=2)
    r = t.raster(0)
    assert (r[1:3, 1:3] == 2).all()
    assert r[0, 0] == 0  # hors rectangle
    t.undo(0)
    assert (t.raster(0) == 0).all()


def test_gridtarget_apply_mask_and_undo():
    g = Grid(0.0, 4.0, 0.0, 4.0, 1.0)  # 4x4
    t = GridTarget(g, ignore_id=0)
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = mask[3, 3] = mask[1, 2] = True  # cellules non contiguës (sélection)
    assert t.apply_mask(0, mask, class_id=3)
    r = t.raster(0)
    assert r[0, 0] == 3 and r[3, 3] == 3 and r[1, 2] == 3
    assert r[2, 2] == 0
    t.undo(0)
    assert (t.raster(0) == 0).all()


def test_gridtarget_apply_mask_empty_false():
    g = Grid(0.0, 4.0, 0.0, 4.0, 1.0)
    t = GridTarget(g)
    assert t.apply_mask(0, np.zeros((4, 4), dtype=bool), class_id=1) is False


def test_gridtarget_rect_outside_returns_false():
    g = Grid(0.0, 4.0, 0.0, 4.0, 1.0)
    t = GridTarget(g)
    assert t.apply(0, (100.0, 100.0, 200.0, 200.0), class_id=1) is False


def test_gridtarget_clear():
    g = Grid(0.0, 4.0, 0.0, 4.0, 1.0)
    t = GridTarget(g)
    t.apply(0, (0.0, 0.0, 4.0, 4.0), class_id=1)
    t.clear(0)
    assert (t.raster(0) == 0).all()
