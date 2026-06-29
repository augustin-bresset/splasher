"""Headless test for the apairo adapter WITHOUT apairo installed.

We simulate a synchronous apairo dataset with a duck-typed object: the adapter must only
depend on the interface (is_synchronous / keys / __len__ / __getitem__).
"""

from dataclasses import dataclass

import numpy as np
import pytest

from splasher.adapters.apairo_source import ApairoSource, _kind_of
from splasher.core.source import ChannelKind


@dataclass
class _FakeSample:
    data: dict
    timestamp: float | None = None


class _FakeDataset:
    is_synchronous = True

    def __init__(self):
        self.keys = ["lidar", "labels", "cam", "pose"]
        self._frames = [
            {
                "lidar": np.zeros((100, 4), np.float32),
                "labels": np.zeros((100,), np.int64),
                "cam": np.zeros((8, 8, 3), np.uint8),
                "pose": np.eye(4, dtype=np.float32),
            }
            for _ in range(2)
        ]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return _FakeSample(self._frames[i], timestamp=None)


def test_kind_of():
    assert _kind_of(np.zeros((10, 4))) is ChannelKind.POINTCLOUD
    assert _kind_of(np.zeros((8, 8, 3), np.uint8)) is ChannelKind.IMAGE
    assert _kind_of(np.eye(4)) is ChannelKind.POSE
    assert _kind_of(np.zeros((7,))) is ChannelKind.POSE
    assert _kind_of(np.zeros((100,))) is ChannelKind.SCALAR  # labels


def test_adapter_classifies_and_reads():
    src = ApairoSource(_FakeDataset())
    kinds = {s.name: s.kind for s in src.channels()}
    assert kinds["lidar"] is ChannelKind.POINTCLOUD
    assert kinds["cam"] is ChannelKind.IMAGE
    assert kinds["pose"] is ChannelKind.POSE
    assert kinds["labels"] is ChannelKind.SCALAR
    assert len(src) == 2
    assert src[0]["lidar"].shape == (100, 4)


def test_adapter_rejects_async():
    class Async(_FakeDataset):
        is_synchronous = False

    with pytest.raises(ValueError):
        ApairoSource(Async())
