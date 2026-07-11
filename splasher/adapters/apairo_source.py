"""**Optional** adapter: a synchronous apairo dataset -> Splasher `Source`.

apairo is not imported at module level: only `from_path` loads it. The class works by
duck-typing on any apairo-like object
(`is_synchronous`, `keys`, `__len__`, `__getitem__` -> object with `.data`/`.timestamp`).
Install via the extra: `uv sync --extra apairo`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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

    def __init__(self, dataset, keys: list[str] | None = None, *,
                 dataset_root: str | None = None, sequence: str | None = None,
                 reference: str | None = None, tolerance: float = 0.1) -> None:
        if not getattr(dataset, "is_synchronous", False):
            raise ValueError(
                "ApairoSource requires a synchronous apairo dataset — "
                "call ds.synchronize(reference=..., tolerance=...) first."
            )
        self._ds = dataset
        self._keys = list(keys) if keys is not None else list(dataset.keys)
        # Provenance (kept so the labeling can be written back). `dataset_root` marks this
        # source as apairo-managed; it is `None` for a directly-wrapped (e.g. test) dataset.
        self._dataset_root = dataset_root
        self._sequence = sequence
        self._reference = reference
        self._tolerance = tolerance
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

    # ---------------------------------------------------------- write-back
    @property
    def point_channels(self) -> list[str]:
        """Names of the point-cloud channels — the candidate reference channels for a save."""
        return [s.name for s in self._specs if s.kind is ChannelKind.POINTCLOUD]

    def apairo_meta(self) -> dict:
        """What a front needs to offer 'browse sequences' + 'write back as a channel'.

        `is_apairo` is False for a source not opened from a path (nothing to write back to).
        `write_root` is the directory a save writes into (the sequence dir when one is
        loaded, else the dataset root — `run_preprocess` fans a root out to its sequences).
        """
        if not self._dataset_root:
            return {"is_apairo": False}
        point_channels = self.point_channels
        sequences = self._list_sequences(self._dataset_root)
        write_root = self._dataset_root
        if self._sequence:
            write_root = str(Path(self._dataset_root) / self._sequence)
        reference = self._reference or (point_channels[0] if point_channels else None)
        return {
            "is_apairo": True,
            "dataset_root": str(self._dataset_root),
            "write_root": str(write_root),
            "name": Path(self._dataset_root).name,
            "sequences": sequences,
            "sequence": self._sequence,
            "point_channels": point_channels,
            "reference": reference,
            "tolerance": self._tolerance,
        }

    @staticmethod
    def _list_sequences(dataset_root: str) -> list[str]:
        """Sequence ids of a dataset root, or `[]` for a lone sequence directory."""
        try:
            import apairo

            ds = apairo.RawDataset(str(dataset_root))
            return list(ds.sequence_ids)
        except Exception:  # noqa: BLE001 — a lone sequence has no sequence_ids; treat as none
            return []

    def for_sequence(self, sequence: str | None) -> ApairoSource:
        """A sibling source for another `sequence` of the same dataset (or the whole root).

        `sequence=None` loads the dataset root as-is (every sequence, flat timeline).
        """
        return self.from_path(
            self._dataset_root, keys=self._keys if self._keys else None,
            reference=self._reference, tolerance=self._tolerance, sequence=sequence,
        )

    @classmethod
    def from_path(cls, path: str, *, keys: list[str] | None = None,
                  reference: str | None = None, tolerance: float = 0.1,
                  split: str | None = None, start: int = 0,
                  count: int | None = None, sequence: str | None = None) -> ApairoSource:
        """Open an apairo `RawDataset`, synchronize/split/window it, and wrap it.

        `sequence` loads a single named sequence of `path` (a dataset root); `None` loads
        `path` as-is. `split` selects a built-in split (`ds.split(name)`); `start`/`count`
        keep a frame window (`ds.filter(range(...))`) — a slice of a very large dataset.
        """
        import apairo  # lazy import — the `apairo` extra must be installed

        dataset_root = str(path)
        open_path = str(Path(path) / sequence) if sequence else dataset_root

        ds = apairo.RawDataset(open_path, keys=keys) if keys else apairo.RawDataset(open_path)
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

        return cls(ds, keys=keys, dataset_root=dataset_root, sequence=sequence,
                   reference=reference, tolerance=tolerance)
