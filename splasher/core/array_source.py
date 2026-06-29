"""`ArraySource` — a `Source` built from in-memory numpy arrays.

This is the default input, with no dependency: handy for demos, tests, and any pipeline
that already produces arrays. No hidden I/O.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .source import ChannelSpec, Frame, Source


class ArraySource(Source):
    """In-memory synchronous source.

    Parameters
    ----------
    specs:
        Channel description (`ChannelSpec`). Order is preserved.
    frames:
        A sequence of dicts `{channel_name: np.ndarray}`, one per time step.
    timestamps:
        Optional; per-frame timestamps. `None` (default) = synchronous.
    """

    def __init__(
        self,
        specs: Sequence[ChannelSpec],
        frames: Sequence[Mapping[str, np.ndarray]],
        timestamps: Sequence[float] | None = None,
    ) -> None:
        self._specs = list(specs)
        self._frames = [dict(f) for f in frames]
        if timestamps is not None and len(timestamps) != len(self._frames):
            raise ValueError("timestamps must have the same length as frames")
        self._timestamps = None if timestamps is None else [float(t) for t in timestamps]

        names = {s.name for s in self._specs}
        for i, fr in enumerate(self._frames):
            missing = names - set(fr.keys())
            if missing:
                raise ValueError(f"frame {i}: missing channels {sorted(missing)}")

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> Frame:
        ts = None if self._timestamps is None else self._timestamps[index]
        return Frame(channels=dict(self._frames[index]), timestamp=ts)

    def channels(self) -> list[ChannelSpec]:
        return list(self._specs)
