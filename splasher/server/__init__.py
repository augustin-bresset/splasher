"""Splasher backend: exposes an `engine.Session` over HTTP (REST + JSON) and serves
the web front (`web/`). A desktop app (`desktop.run_desktop`) opens it in a webview.

Optional — `serve()` needs the `api` extra (FastAPI + uvicorn), `run_desktop()` the
`app` extra (+ pywebview). Everything is imported lazily so `splasher` stays usable
without these dependencies.
"""

from __future__ import annotations


def create_app(session_or_source, *, labels=None):
    """Build the FastAPI app around a `Session` (or a `Source`)."""
    from .app import create_app as _create_app

    return _create_app(session_or_source, labels=labels)


def serve(session_or_source, *, host: str = "127.0.0.1", port: int = 8000, labels=None):
    """Run the web/API server (uvicorn) on a `Session` (or a `Source`)."""
    from .app import serve as _serve

    return _serve(session_or_source, host=host, port=port, labels=labels)


__all__ = ["create_app", "serve"]
