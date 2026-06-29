"""Launch Splasher (desktop app) on a synthetic source — no external data required.

    uv sync --extra app
    uv run python examples/demo_arraysource.py

For browser mode instead: `splasher demo --serve` (`api` extra).
"""

from splasher import app
from splasher.demo import make_demo_source

if __name__ == "__main__":
    app(make_demo_source(), title="Splasher — demo")
