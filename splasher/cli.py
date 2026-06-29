"""Splasher CLI.

    splasher                                        # empty: file-viewer mode (browse + open files)
    splasher demo                                   # desktop app on the synthetic source
    splasher <path> --adapter apairo                # apairo dataset, all channels
    splasher <path> --adapter apairo --channels lidar,cam_front,pose   # pick channels by hand
    splasher <path> --adapter apairo --reference lidar --tolerance 0.05  # sync an async dataset
    splasher <path> --adapter apairo --serve        # browser instead of the desktop window

Extras: `app` (desktop), `api` (browser/--serve), `apairo` (the adapter). Combine them:
    uv sync --extra apairo --extra app
"""

from __future__ import annotations

import argparse
import sys


def _make_source(args):
    """Build the requested `Source`, or (None, None) if the invocation is invalid."""
    if args.path is None:
        from .core.array_source import ArraySource

        return ArraySource([], []), "Splasher"   # empty: file-viewer mode (browse + open files)
    if args.path == "demo":
        from .demo import make_demo_source

        return make_demo_source(), "Splasher — demo"
    if args.adapter == "apairo":
        from .adapters.apairo_source import ApairoSource

        keys = [k.strip() for k in args.channels.split(",")] if args.channels else None
        src = ApairoSource.from_path(args.path, keys=keys,
                                     reference=args.reference, tolerance=args.tolerance,
                                     split=args.split, start=args.start, count=args.count)
        return src, f"Splasher — {args.path}"
    return None, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splasher", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", help="dataset path, or 'demo'")
    parser.add_argument("--adapter", choices=["apairo"], help="input adapter")
    parser.add_argument("--channels", help="apairo: comma-separated channels to load (default: all)")
    parser.add_argument("--reference", help="apairo: reference channel to synchronize an async dataset")
    parser.add_argument("--tolerance", type=float, default=0.1, help="apairo: sync tolerance in seconds")
    parser.add_argument("--split", help="apairo: select a built-in split (e.g. train/val/test)")
    parser.add_argument("--start", type=int, default=0, help="apairo: first frame of the working window")
    parser.add_argument("--count", type=int, help="apairo: number of frames to keep from --start")
    parser.add_argument("--serve", action="store_true",
                        help="run the headless web server (open it in a browser) instead of the desktop app")
    parser.add_argument("--host", default="127.0.0.1", help="server host")
    parser.add_argument("--port", type=int, default=8077, help="server port (--serve mode)")
    parser.add_argument("--quiet", action="store_true",
                        help="desktop app: hush QtWebEngine/Chromium console logs")
    args = parser.parse_args(argv)

    try:
        source, title = _make_source(args)
    except ImportError:
        parser.error("apairo is not installed — run: uv sync --extra apairo")
        return 2
    except (ValueError, OSError, KeyError) as e:
        parser.error(str(e))
        return 2

    if source is None:
        parser.error("specify an --adapter (e.g. --adapter apairo) or use 'demo'")
        return 2

    if args.serve:
        from . import serve

        print(f"Splasher → http://{args.host}:{args.port}  (web front + API, docs at /docs)")
        serve(source, host=args.host, port=args.port)
        return 0

    from . import app

    app(source, title=title, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
