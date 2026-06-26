"""Lance Splasher sur une source synthétique — aucune donnée externe requise.

    uv run python examples/demo_arraysource.py
"""

from splasher import launch
from splasher.demo import make_demo_source

if __name__ == "__main__":
    launch(make_demo_source(), title="Splasher — démo")
