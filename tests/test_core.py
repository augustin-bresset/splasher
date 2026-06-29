"""Headless core tests (pure numpy) — never import the UI."""

import numpy as np
import pytest

from splasher import ArraySource, ChannelKind, ChannelSpec, channels_of_kind
from splasher.core.colormap import colormap


def _tiny_source():
    specs = [
        ChannelSpec("lidar", ChannelKind.POINTCLOUD, np.dtype("float32"), (None, 4)),
        ChannelSpec("cam", ChannelKind.IMAGE, np.dtype("uint8"), (4, 4, 3)),
        ChannelSpec("pose", ChannelKind.POSE, np.dtype("float32"), (4, 4)),
    ]
    frames = [
        {
            "lidar": np.zeros((10, 4), np.float32),
            "cam": np.zeros((4, 4, 3), np.uint8),
            "pose": np.eye(4, dtype=np.float32),
        }
        for _ in range(3)
    ]
    return ArraySource(specs, frames)


def test_array_source_basic():
    src = _tiny_source()
    assert len(src) == 3
    frame = src[0]
    assert set(frame.keys()) == {"lidar", "cam", "pose"}
    assert frame.timestamp is None  # synchronous
    assert frame["lidar"].shape == (10, 4)


def test_channels_of_kind():
    src = _tiny_source()
    assert channels_of_kind(src, ChannelKind.POINTCLOUD) == ["lidar"]
    assert channels_of_kind(src, ChannelKind.IMAGE) == ["cam"]
    assert channels_of_kind(src, ChannelKind.POSE) == ["pose"]


def test_array_source_missing_channel_raises():
    specs = [ChannelSpec("lidar", ChannelKind.POINTCLOUD)]
    with pytest.raises(ValueError):
        ArraySource(specs, [{"other": np.zeros((1, 3))}])


def test_colormap_rgba_range():
    rgba = colormap(np.array([0.0, 1.0, 2.0, np.nan]))
    assert rgba.shape == (4, 4)
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0
    assert np.allclose(rgba[:, 3], 1.0)
