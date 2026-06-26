"""CLI Splasher.

    splasher demo                      # source synthétique (zéro donnée externe)
    splasher <chemin> --adapter apairo # via l'adaptateur apairo (extra optionnel)
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="splasher", description=__doc__)
    parser.add_argument("path", nargs="?", help="chemin du dataset, ou 'demo'")
    parser.add_argument("--adapter", choices=["apairo"], help="adaptateur d'entrée")
    args = parser.parse_args(argv)

    from . import launch

    if args.path in (None, "demo"):
        from .demo import make_demo_source

        launch(make_demo_source(), title="Splasher — démo")
        return 0

    if args.adapter == "apairo":
        from .adapters.apairo_source import ApairoSource

        launch(ApairoSource.from_path(args.path), title=f"Splasher — {args.path}")
        return 0

    parser.error("précise un --adapter (ex. --adapter apairo) ou utilise 'demo'")
    return 2


if __name__ == "__main__":
    sys.exit(main())
