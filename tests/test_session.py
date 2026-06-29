"""The `Session` engine must be drivable *without any UI toolkit* (numpy only)."""

from __future__ import annotations

import numpy as np
import pytest

from splasher.demo import make_demo_source
from splasher.engine import Session


@pytest.fixture
def session() -> Session:
    return Session(make_demo_source(n_frames=5, seed=1))


def test_info_describes_demo_channels(session: Session) -> None:
    info = session.info()
    assert info.n_frames == 5
    assert info.cloud_keys == ["lidar", "lidar_top"]
    assert info.image_keys == ["camera_front", "camera_rear"]
    assert info.pose_key == "pose"
    assert info.has_pose


def test_view_state_shapes_are_aligned(session: Session) -> None:
    v = session.view_state()
    assert v.points.ndim == 2 and v.points.shape[1] >= 3
    assert v.point_labels.shape == (len(v.points),)
    assert v.point_channels.shape == (len(v.points),)
    assert v.bev_field.shape == v.grid.shape
    assert v.grid_labels is None          # nothing labeled at start
    assert set(v.images) == {"camera_front", "camera_rear"}


def test_paint_grid_then_undo(session: Session) -> None:
    rect = (-5.0, -5.0, 5.0, 5.0)
    assert session.paint_rect(rect) is True
    v = session.view_state()
    assert v.grid_labels is not None
    assert (v.grid_labels == session.active_class).any()

    session.undo()
    assert session.view_state().grid_labels is not None  # raster exists, reset to ignore
    assert not (session.view_state().grid_labels == session.active_class).any()


def test_select_then_apply(session: Session) -> None:
    session.set_tool("select")
    assert session.select_rect((-3.0, -3.0, 3.0, 3.0)) is True
    assert session.view_state().selection.any()

    session.set_active_targets({"grid"})
    assert session.apply_selection() is True
    v = session.view_state()
    assert v.selection is None            # selection consumed
    assert (v.grid_labels == session.active_class).any()


def test_point_target_labels_reflected_in_view(session: Session) -> None:
    session.set_active_targets({"points"})
    cls = session.active_class
    assert session.paint_rect((-40.0, -40.0, 40.0, 40.0)) is True
    v = session.view_state()
    assert (v.point_labels == cls).any()


def test_accumulation_grows_with_radius(session: Session) -> None:
    session.set_frame(2)
    n0 = len(session.view_state().points)
    session.set_accum_radius(2)
    n1 = len(session.view_state().points)
    assert n1 > n0


def test_bev_visibility_does_not_affect_3d_points(session: Session) -> None:
    n_all = len(session.view_state().points)
    filled_all = int(np.isfinite(session.view_state().bev_field).sum())

    session.set_visible_clouds({"lidar_top"})      # keep only the sparse cloud in the BEV
    v = session.view_state()
    assert len(v.points) == n_all                  # 3D keeps every channel (decoupled)
    assert set(v.point_channels.tolist()) == {0, 1}
    assert int(np.isfinite(v.bev_field).sum()) < filled_all  # BEV underlay reflects visibility


def test_set_bev_mode(session: Session) -> None:
    session.set_bev_mode("density")
    assert session.view_state().bev_mode == "density"
    session.set_bev_mode("nonsense")
    assert session.bev_mode == "height"     # invalid falls back


def test_set_labelset_updates_classes_and_resets_active(session: Session) -> None:
    session.set_active_class(2)
    session.set_labelset({
        "ignore_id": 0,
        "classes": [
            {"id": 0, "name": "unlabeled", "color": [0, 0, 0]},
            {"id": 1, "name": "road", "color": [1, 2, 3]},
        ],
    })
    assert [c.name for c in session.labelset.classes] == ["unlabeled", "road"]
    assert session.active_class == 1        # id 2 no longer exists -> first paintable


def test_commit_grid_resets_grid_labels_only(session: Session) -> None:
    session.set_active_targets({"grid", "points"})
    session.paint_rect((-5.0, -5.0, 5.0, 5.0))
    assert session.grid_labelled_count() == 1

    from splasher.core.grid import Grid
    session.commit_grid(Grid(-10, 10, -10, 10, 2.0))
    assert session.grid_labelled_count() == 0
    # per-point labels preserved
    assert (session.view_state().point_labels == session.active_class).any()


def test_save_load_roundtrip(session: Session, tmp_path) -> None:
    session.set_active_targets({"grid", "points"})
    session.paint_rect((-5.0, -5.0, 5.0, 5.0))
    before = session.view_state().grid_labels.copy()

    session.save(tmp_path)

    fresh = Session(make_demo_source(n_frames=5, seed=1))
    fresh.load(tmp_path)
    after = fresh.view_state().grid_labels
    assert np.array_equal(before, after)
