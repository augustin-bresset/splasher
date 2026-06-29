"""Desktop app: the web front rendered in a native window (Spotify/Electron style).

Starts the FastAPI server in a background thread, then opens the OS webview
(pywebview) on the local URL. No Qt toolkit of our own: this is exactly the web
front, rendered by the platform's native web engine. Requires the `app` extra.

If no native webview backend is available (common on a bare Linux virtualenv,
which needs GTK+WebKit or Qt WebEngine Python bindings), it falls back to opening
the URL in the default browser and keeps serving — so it works everywhere.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
import webbrowser


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_ready(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)


def _preferred_gui() -> str | None:
    """Pick a webview backend explicitly to skip pywebview's probing (and its noisy
    GTK import traceback when GTK bindings are absent). Returns None to auto-select."""
    try:
        import qtpy  # noqa: F401  (provided by the `app` extra → Qt WebEngine)
        return "qt"
    except Exception:
        return None


def run_desktop(session_or_source, *, title: str = "Splasher", labels=None,
                host: str = "127.0.0.1", port: int = 0,
                width: int = 1480, height: int = 880, quiet: bool = False) -> None:
    import uvicorn

    from .app import create_app

    app = create_app(session_or_source, labels=labels)
    if port == 0:
        port = _free_port(host)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    _wait_ready(host, port)
    url = f"http://{host}:{port}"

    try:
        gui = _preferred_gui()
        if gui == "qt":
            # Qt's event loop swallows SIGINT; restore the default so Ctrl+C quits at once.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if quiet:
                # Opt-in: hush QtWebEngine/Chromium noise (GPU "Vulkan fallback" notice, the
                # harmless "profile still not deleted" teardown warning). GPU stays on, so
                # WebGL / the 3D view keep working. By default logs are kept for diagnostics.
                os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--log-level=2")
                os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

        import webview

        webview.create_window(title, url, width=width, height=height)
        webview.start(gui=gui)                  # blocks until the window is closed
    except Exception as exc:
        # pywebview missing, or no native backend (GTK/Qt) — fall back to the browser.
        print(f"[splasher] native window unavailable ({type(exc).__name__}); opening browser.")
        print("[splasher] for a native window, install a webview backend "
              "(system GTK+WebKit, or `pip install 'pywebview[qt]'`).")
        print(f"[splasher] serving at {url} — press Ctrl+C to stop.")
        webbrowser.open(url)
        try:
            while not server.should_exit:
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass

    server.should_exit = True
