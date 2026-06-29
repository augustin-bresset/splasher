"""Splasher *headless* engine (no UI toolkit).

`Session` holds all the state of a labeling session and exposes the operations
(paint, select, accumulate, change grid, undo, save/load…). It renders nothing
itself: it produces a *semantic* `ViewState` (numpy primitives: points + per-point
labels, BEV scalar field, grid raster, selection, images) that any front (web via
the API, …) colorizes/draws its own way.

No Qt dependency: importable from the server and from tests alike.
"""

from __future__ import annotations

from .session import Session
from .view_state import SessionInfo, ViewState

__all__ = ["Session", "SessionInfo", "ViewState"]
