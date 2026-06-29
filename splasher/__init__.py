"""Splasher — labeling of synchronized channels into a 2D BEV grid or per-point labels.

Three layers, from most generic to most specific:
- `splasher.core`   : pure numpy model (grid, targets, projection, accumulation…).
- `splasher.engine` : *headless* `Session` (state + operations + semantic `ViewState`),
  drivable indifferently by the API or any front. No UI dependency.
- `splasher.server` : FastAPI backend + web front (`web/`) + desktop app.
  `serve()` runs the web server (`api` extra); `app()` opens the desktop window
  (web front inside a native webview, `app` extra).
"""

from __future__ import annotations

from .core.array_source import ArraySource
from .core.grid import Grid, grid_from_points
from .core.source import ChannelKind, ChannelSpec, Frame, Source, channels_of_kind
from .engine import Session, SessionInfo, ViewState

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
    "Session",
    "SessionInfo",
    "ViewState",
    "serve",
    "app",
    "__version__",
]


def serve(*args, **kwargs):
    """Run the web + API server on a `Source`/`Session` (`api` extra). Lazy import."""
    from .server import serve as _serve

    return _serve(*args, **kwargs)


def app(*args, **kwargs):
    """Open the desktop app (web front in a native webview, `app` extra). Lazy import."""
    from .server.desktop import run_desktop

    return run_desktop(*args, **kwargs)
