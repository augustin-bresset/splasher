"""**Optional** adapter: a synchronous apairo dataset -> Splasher `Source`.

apairo is not imported at module level: only `from_path` loads it. The class works by
duck-typing on any apairo-like object
(`is_synchronous`, `keys`, `__len__`, `__getitem__` -> object with `.data`/`.timestamp`).
Install via the extra: `uv sync --extra apairo`.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core.source import ChannelKind, ChannelSpec, Frame


def _kind_of(arr: np.ndarray) -> ChannelKind:
    """Guess the `ChannelKind` of an apairo channel from the array shape."""
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return ChannelKind.IMAGE
    if arr.shape in ((4, 4), (3, 4)) or arr.shape == (7,):
        return ChannelKind.POSE
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return ChannelKind.POINTCLOUD
    return ChannelKind.SCALAR  # e.g. labels (N,)


class ApairoSource:
    """Wraps a **synchronous** apairo dataset into a `Source`."""

    def __init__(self, dataset, keys: list[str] | None = None) -> None:
        if not getattr(dataset, "is_synchronous", False):
            raise ValueError(
                "ApairoSource requires a synchronous apairo dataset — "
                "call ds.synchronize(reference=..., tolerance=...) first."
            )
        self._ds = dataset
        self._keys = list(keys) if keys is not None else list(dataset.keys)
        self._specs = self._classify()

    def _classify(self) -> list[ChannelSpec]:
        """Discover each channel's kind/shape by scanning the first frames.

        A synchronized dataset may list channels (in `dataset.keys`) that are absent from a
        given sample (e.g. nothing within tolerance for that timestamp). We scan a few frames
        to find each one; channels never seen are dropped (with a warning), and `__getitem__`
        tolerates per-frame gaps.
        """
        requested = list(self._keys)
        found: dict[str, ChannelSpec] = {}
        for i in range(min(len(self._ds), 25)):
            data = self._ds[i].data
            for k in requested:
                if k in found or k not in data:
                    continue
                arr = np.asarray(data[k])
                found[k] = ChannelSpec(k, _kind_of(arr), arr.dtype, tuple(arr.shape))
            if len(found) == len(requested):
                break

        dropped = [k for k in requested if k not in found]
        if dropped:
            warnings.warn(f"apairo: channels absent from the synchronized samples, skipped: {dropped}",
                          stacklevel=2)
        self._keys = [k for k in requested if k in found]   # keep order, only present ones
        return [found[k] for k in self._keys]

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, index: int) -> Frame:
        s = self._ds[index]
        channels = {k: np.asarray(s.data[k]) for k in self._keys if k in s.data}
        return Frame(channels=channels, timestamp=getattr(s, "timestamp", None))

    def channels(self) -> list[ChannelSpec]:
        return list(self._specs)

    @classmethod
    def from_path(cls, path: str, *, keys: list[str] | None = None,
                  reference: str | None = None, tolerance: float = 0.1,
                  split: str | None = None, start: int = 0,
                  count: int | None = None) -> ApairoSource:
        """Open an apairo `RawDataset`, synchronize/split/window it, and wrap it.

        `split` selects a built-in split (`ds.split(name)`); `start`/`count` keep a frame
        window (`ds.filter(range(...))`) — handy to work on a slice of a very large dataset.
        """
        import apairo  # lazy import — the `apairo` extra must be installed

        ds = apairo.RawDataset(path, keys=keys) if keys else apairo.RawDataset(path)
        if not ds.is_synchronous:
            if reference is None:
                raise ValueError(
                    "asynchronous dataset: pass reference=<channel> for synchronization."
                )
            ds = ds.synchronize(reference=reference, tolerance=tolerance)

        if split:
            if not hasattr(ds, "split"):
                raise ValueError("this dataset does not support splits (.split())")
            ds = ds.split(split)
        if start or count is not None:
            if not hasattr(ds, "filter"):
                raise ValueError("this dataset does not support windowing (.filter())")
            n = len(ds)
            lo = min(max(0, start), n)
            hi = n if count is None else min(n, lo + max(0, count))
            ds = ds.filter(list(range(lo, hi)))

        return cls(ds, keys=keys)
