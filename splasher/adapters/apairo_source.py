"""Adaptateur **optionnel** : un dataset apairo synchrone -> `Source` Splasher.

apairo n'est pas importé au niveau module : seul `from_path` le charge. La classe
fonctionne en duck-typing sur n'importe quel objet façon apairo
(`is_synchronous`, `keys`, `__len__`, `__getitem__` -> objet avec `.data`/`.timestamp`).
Installer via l'extra : `uv sync --extra apairo`.
"""

from __future__ import annotations

import numpy as np

from ..core.source import ChannelKind, ChannelSpec, Frame


def _kind_of(arr: np.ndarray) -> ChannelKind:
    """Devine le `ChannelKind` d'un canal apairo d'après la forme du tableau."""
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return ChannelKind.IMAGE
    if arr.shape in ((4, 4), (3, 4)) or arr.shape == (7,):
        return ChannelKind.POSE
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return ChannelKind.POINTCLOUD
    return ChannelKind.SCALAR  # ex. labels (N,)


class ApairoSource:
    """Enveloppe un dataset apairo **synchrone** en `Source`."""

    def __init__(self, dataset, keys: list[str] | None = None) -> None:
        if not getattr(dataset, "is_synchronous", False):
            raise ValueError(
                "ApairoSource requiert un dataset apairo synchrone — "
                "appelez ds.synchronize(reference=..., tolerance=...) d'abord."
            )
        self._ds = dataset
        self._keys = list(keys) if keys is not None else list(dataset.keys)
        self._specs = self._classify()

    def _classify(self) -> list[ChannelSpec]:
        sample = self._ds[0]
        specs = []
        for k in self._keys:
            arr = np.asarray(sample.data[k])
            specs.append(ChannelSpec(k, _kind_of(arr), arr.dtype, tuple(arr.shape)))
        return specs

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
                  reference: str | None = None, tolerance: float = 0.1) -> "ApairoSource":
        """Ouvre un `RawDataset` apairo et le synchronise si besoin."""
        import apairo  # import paresseux — l'extra `apairo` doit être installé

        ds = apairo.RawDataset(path, keys=keys) if keys else apairo.RawDataset(path)
        if not ds.is_synchronous:
            if reference is None:
                raise ValueError(
                    "dataset asynchrone : précisez reference=<canal> pour la synchronisation."
                )
            ds = ds.synchronize(reference=reference, tolerance=tolerance)
        return cls(ds, keys=keys)
