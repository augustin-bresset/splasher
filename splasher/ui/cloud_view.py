"""Vue 3D du/des nuage(s) de points : navigation libre (orbit/pan/zoom)."""

from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl

from ..core.colormap import colormap


class CloudView(gl.GLViewWidget):
    """Affiche un ou plusieurs nuages nommés. Couleur par hauteur (z) par défaut."""

    def __init__(self) -> None:
        super().__init__()
        self.setBackgroundColor("#101014")
        self.setCameraPosition(distance=60.0, elevation=30.0, azimuth=-60.0)

        grid = gl.GLGridItem()
        grid.setSize(x=100, y=100)
        grid.setSpacing(x=5, y=5)
        grid.setColor((255, 255, 255, 40))
        self.addItem(grid)
        self._grid = grid

        self._scatters: dict[str, gl.GLScatterPlotItem] = {}

    def set_cloud(self, name: str, points: np.ndarray, colors: np.ndarray | None = None,
                  size: float = 2.0) -> None:
        """Met à jour (ou crée) le nuage `name`. `points` (N, 3+), `colors` (N, 4) RGBA [0,1]."""
        if points is None or len(points) == 0:
            if name in self._scatters:
                self._scatters[name].setData(pos=np.zeros((0, 3), np.float32))
            return

        xyz = np.ascontiguousarray(points[:, :3], dtype=np.float32)
        if colors is None:
            colors = colormap(xyz[:, 2])

        item = self._scatters.get(name)
        if item is None:
            item = gl.GLScatterPlotItem(pxMode=True)
            # 'opaque' (et non l'additif par défaut) : les couleurs ne saturent pas
            # vers le blanc quand des dizaines de milliers de points se superposent (cumul).
            item.setGLOptions("opaque")
            self.addItem(item)
            self._scatters[name] = item
        item.setData(pos=xyz, color=colors, size=size)

    def clear_clouds(self) -> None:
        for item in self._scatters.values():
            item.setData(pos=np.zeros((0, 3), np.float32))
