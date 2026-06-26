"""`ArraySource` — une `Source` construite depuis des tableaux numpy en mémoire.

C'est l'entrée par défaut, sans aucune dépendance : utile pour les démos, les
tests, et tout pipeline qui produit déjà des arrays. Aucune I/O cachée.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .source import ChannelSpec, Frame, Source


class ArraySource(Source):
    """Source synchrone en mémoire.

    Parameters
    ----------
    specs:
        Description des canaux (`ChannelSpec`). L'ordre est conservé.
    frames:
        Une séquence de dicts `{nom_canal: np.ndarray}`, un par pas de temps.
    timestamps:
        Optionnel ; horodatages par frame. `None` (défaut) = synchrone.
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
            raise ValueError("timestamps doit avoir la même longueur que frames")
        self._timestamps = None if timestamps is None else [float(t) for t in timestamps]

        names = {s.name for s in self._specs}
        for i, fr in enumerate(self._frames):
            missing = names - set(fr.keys())
            if missing:
                raise ValueError(f"frame {i}: canaux manquants {sorted(missing)}")

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> Frame:
        ts = None if self._timestamps is None else self._timestamps[index]
        return Frame(channels=dict(self._frames[index]), timestamp=ts)

    def channels(self) -> list[ChannelSpec]:
        return list(self._specs)
