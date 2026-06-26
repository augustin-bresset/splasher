"""Splasher — labélisation de canaux synchronisés vers une grille 2D BEV ou des labels par point.

Le cœur (`splasher.core`) ne dépend que de numpy et reste importable sans toolkit UI.
L'UI (PySide6/pyqtgraph) n'est chargée qu'au moment de `launch()`.
"""

from __future__ import annotations

from .core.source import ChannelKind, ChannelSpec, Frame, Source, channels_of_kind
from .core.array_source import ArraySource
from .core.grid import Grid, grid_from_points

__version__ = "0.1.0"

__all__ = [
    "ChannelKind",
    "ChannelSpec",
    "Frame",
    "Source",
    "channels_of_kind",
    "ArraySource",
    "Grid",
    "grid_from_points",
    "launch",
    "__version__",
]


def launch(*args, **kwargs):
    """Ouvre la fenêtre Splasher sur une `Source`. Importe l'UI paresseusement."""
    from .ui.app import launch as _launch

    return _launch(*args, **kwargs)
