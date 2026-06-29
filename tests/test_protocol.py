"""The (de)serialization protocol must preserve the `ViewState` (numpy only, no FastAPI)."""

from __future__ import annotations

import numpy as np

from splasher.demo import make_demo_source
from splasher.engine import Session
from splasher.server import protocol as P


def test_array_roundtrip_preserves_dtype_shape_values() -> None:
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    back = P.decode_array(P.encode_array(arr))
    assert back.dtype == arr.dtype and back.shape == arr.shape
    assert np.array_equal(back, arr)


def test_encode_none_is_none() -> None:
    assert P.encode_array(None) is None
    assert P.decode_array(None) is None


def test_nan_survives_in_bev_field() -> None:
    field = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float32)
    back = P.decode_array(P.encode_array(field))
    assert np.isnan(back[0, 1]) and back[1, 1] == 2.0


def test_session_info_dict_shape() -> None:
    s = Session(make_demo_source(n_frames=3))
    d = P.session_info_to_dict(s.info())
    assert d["n_frames"] == 3
    assert d["cloud_keys"] == ["lidar", "lidar_top"]
    assert {c["name"] for c in d["channels"]} >= {"lidar", "camera_front", "pose"}
    assert "classes" in d["labelset"]


def test_channelspec_placement_serialized() -> None:
    d = P.session_info_to_dict(Session(make_demo_source(n_frames=2)).info())
    by = {c["name"]: c for c in d["channels"]}
    assert by["camera_front"]["placement"] is not None
    assert len(by["camera_front"]["placement"]) == 4          # 4x4 matrix
    assert by["pose"]["placement"] is None                    # non-sensor


def test_view_state_dict_decodes_back_to_arrays() -> None:
    s = Session(make_demo_source(n_frames=3))
    s.paint_rect((-5.0, -5.0, 5.0, 5.0))
    d = P.view_state_to_dict(s.view_state())

    points = P.decode_array(d["points"])
    channels = P.decode_array(d["point_channels"])
    grid_labels = P.decode_array(d["grid_labels"])
    assert points.shape[1] >= 3
    assert channels.shape[0] == points.shape[0]
    assert grid_labels.shape == (d["grid"]["rows"], d["grid"]["cols"])
    assert d["index"] == 0 and d["tool"] == "paint"
    assert P.grid_from_dict(d["grid"]).cell_size == s.grid.cell_size
