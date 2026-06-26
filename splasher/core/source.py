"""Protocole d'entrée de Splasher — le point central de la généricité.

Une `Source` est un *dataset synchrone* : à chaque index temporel, elle rend un
`Frame` = un pack de canaux nommés (tableaux numpy), chacun typé par un
`ChannelKind`. C'est tout ce que l'outil exige. apairo, des fichiers, des arrays
en mémoire… ne sont que des manières de produire une `Source`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol, runtime_checkable

import numpy as np


class ChannelKind(Enum):
    """Nature d'un canal — pilote la vue qui l'affiche et la manière de le labéliser."""

    POINTCLOUD = "pointcloud"  # (N, 3) ou (N, 3+C) : [x, y, z, ...]
    IMAGE = "image"  # (H, W) ou (H, W, C) uint8
    POSE = "pose"  # (4, 4) ou (7,) [x,y,z, qx,qy,qz,qw] : placement monde
    SCALAR = "scalar"  # autre tableau (réservé / extensible)

    def __repr__(self) -> str:  # affichage compact
        return f"ChannelKind.{self.name}"


@dataclass(frozen=True)
class ChannelSpec:
    """Métadonnées d'un canal. `shape` peut contenir des `None` pour les dims variables."""

    name: str
    kind: ChannelKind
    dtype: np.dtype | None = None
    shape: tuple[int | None, ...] | None = None


@dataclass
class Frame:
    """Un pas de temps synchronisé : tous les canaux à un même instant."""

    channels: dict[str, np.ndarray]
    timestamp: float | None = None  # None = synchrone (convention apairo)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.channels[key]

    def __contains__(self, key: str) -> bool:
        return key in self.channels

    def keys(self):
        return self.channels.keys()


@runtime_checkable
class Source(Protocol):
    """Dataset synchrone : longueur, accès indexé à un `Frame`, et description des canaux."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Frame: ...

    def channels(self) -> list[ChannelSpec]: ...


def channels_of_kind(source_or_specs, kind: ChannelKind) -> list[str]:
    """Noms des canaux d'un `ChannelKind` donné, depuis une `Source` ou une liste de specs."""
    specs: Iterable[ChannelSpec]
    if hasattr(source_or_specs, "channels"):
        specs = source_or_specs.channels()
    else:
        specs = source_or_specs
    return [s.name for s in specs if s.kind == kind]
