"""Splasher's input protocol — the central point of genericity.

A `Source` is a *synchronous dataset*: at each time index it yields a `Frame` = a pack
of named channels (numpy arrays), each typed by a `ChannelKind`. That is all the tool
requires. apairo, files, in-memory arrays… are just ways to produce a `Source`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


class ChannelKind(Enum):
    """Nature of a channel — drives the view that shows it and the way to label it."""

    POINTCLOUD = "pointcloud"  # (N, 3) or (N, 3+C): [x, y, z, ...]
    IMAGE = "image"  # (H, W) or (H, W, C) uint8
    POSE = "pose"  # (4, 4) or (7,) [x,y,z, qx,qy,qz,qw]: world placement
    SCALAR = "scalar"  # other array (reserved / extensible)

    def __repr__(self) -> str:  # compact display
        return f"ChannelKind.{self.name}"


@dataclass(frozen=True)
class ChannelSpec:
    """Channel metadata. `shape` may contain `None` for variable dimensions.

    `placement` is the sensor's pose in the ego frame: a `(4, 4)` matrix, a `(7,)`
    `[x, y, z, qx, qy, qz, qw]` pose, or a `(3,)` position (orientation defaults to
    forward, +x). `None` for non-sensor channels (or unknown placement).
    """

    name: str
    kind: ChannelKind
    dtype: np.dtype | None = None
    shape: tuple[int | None, ...] | None = None
    placement: np.ndarray | None = None


@dataclass
class Frame:
    """A synchronized time step: all channels at the same instant."""

    channels: dict[str, np.ndarray]
    timestamp: float | None = None  # None = synchronous (apairo convention)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def __contains__(self, key: str) -> bool:
        return key in self.channels

    def keys(self):
        return self.channels.keys()


@runtime_checkable
class Source(Protocol):
    """Synchronous dataset: length, indexed access to a `Frame`, and channel description."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Frame: ...

    def channels(self) -> list[ChannelSpec]: ...


def channels_of_kind(source_or_specs, kind: ChannelKind) -> list[str]:
    """Names of the channels of a given `ChannelKind`, from a `Source` or a list of specs."""
    specs: Iterable[ChannelSpec]
    if hasattr(source_or_specs, "channels"):
        specs = source_or_specs.channels()
    else:
        specs = source_or_specs
    return [s.name for s in specs if s.kind == kind]
