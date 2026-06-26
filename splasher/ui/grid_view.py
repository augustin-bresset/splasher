"""Vue de dessus (BEV) : grille de carrés, sous-couche densité, raster de labels,
et sélection par rectangle (rubber-band) pour peindre.
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core.grid import Grid


class _PaintViewBox(pg.ViewBox):
    """ViewBox avec un mode « peindre » : clic-glisser gauche dessine un rectangle."""

    rectDrawn = QtCore.Signal(object)  # (x0, y0, x1, y1) en coords monde

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._paint = True
        self._preview = QtWidgets.QGraphicsRectItem()
        self._preview.setPen(pg.mkPen(255, 213, 74, width=1))
        self._preview.setBrush(pg.mkBrush(255, 213, 74, 50))
        self._preview.setZValue(50)
        self._preview.hide()
        self.addItem(self._preview, ignoreBounds=True)

    def set_paint_mode(self, on: bool) -> None:
        self._paint = on

    def mouseDragEvent(self, ev, axis=None):
        if self._paint and ev.button() == QtCore.Qt.LeftButton:
            ev.accept()
            p0 = self.mapSceneToView(ev.buttonDownScenePos())
            p1 = self.mapSceneToView(ev.scenePos())
            x0, y0, x1, y1 = p0.x(), p0.y(), p1.x(), p1.y()
            self._preview.setRect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            self._preview.show()
            if ev.isFinish():
                self._preview.hide()
                self.rectDrawn.emit((x0, y0, x1, y1))
        else:
            super().mouseDragEvent(ev, axis)


class GridView(QtWidgets.QWidget):
    rectDrawn = QtCore.Signal(object)

    def __init__(self) -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)
        self._vb = _PaintViewBox()
        self._vb.setAspectLocked(True)  # carrés vraiment carrés, y vers le haut
        self._vb.setBackgroundColor("#0c0c10")
        self._glw.addItem(self._vb)
        self._vb.rectDrawn.connect(self.rectDrawn)

        self._under = pg.ImageItem()       # densité/hauteur BEV (M2)
        self._under.setZValue(-10)
        self._labels = pg.ImageItem()      # raster de labels colorisé (M3)
        self._labels.setZValue(-5)
        self._sel = pg.ImageItem()         # surbrillance de la sélection (mode select)
        self._sel.setZValue(8)
        self._pts = pg.ScatterPlotItem(    # points top-down (référence)
            size=2, pen=None, brush=pg.mkBrush(170, 175, 205, 70), pxMode=True
        )
        self._grid_lines = pg.PlotCurveItem(pen=pg.mkPen(120, 125, 150, 140, width=1))
        self._grid_lines.setZValue(5)
        for item in (self._under, self._labels, self._pts, self._grid_lines, self._sel):
            self._vb.addItem(item)

        self._grid: Grid | None = None

    # ------------------------------------------------------------------ API
    def set_paint_mode(self, on: bool) -> None:
        self._vb.set_paint_mode(on)

    def set_grid(self, grid: Grid, autorange: bool = True) -> None:
        self._grid = grid
        xs, ys = grid.line_segments()
        self._grid_lines.setData(xs, ys, connect="pairs")
        if autorange:
            self._vb.setRange(
                xRange=(grid.xmin, grid.xmin + grid.width),
                yRange=(grid.ymin, grid.ymin + grid.height),
                padding=0.03,
            )

    def set_topdown_points(self, xy: np.ndarray | None, max_points: int = 12000) -> None:
        if xy is None or len(xy) == 0:
            self._pts.setData(x=[], y=[])
            return
        xy = np.asarray(xy)
        if len(xy) > max_points:
            xy = xy[np.linspace(0, len(xy) - 1, max_points).astype(int)]
        self._pts.setData(x=xy[:, 0], y=xy[:, 1])

    def set_underlay(self, image: np.ndarray | None, grid: Grid) -> None:
        self._set_raster(self._under, image, grid, levels=(0, 255))

    def set_labels(self, image: np.ndarray | None, grid: Grid) -> None:
        self._set_raster(self._labels, image, grid, levels=(0, 255))

    def set_selection(self, mask: np.ndarray | None, grid: Grid) -> None:
        """Surligne les cellules sélectionnées (`mask` booléen `(rows, cols)`)."""
        if mask is None or not mask.any():
            self._sel.clear()
            return
        rgba = np.zeros((grid.rows, grid.cols, 4), dtype=np.uint8)
        rgba[mask] = (90, 200, 255, 110)  # cyan translucide
        self._set_raster(self._sel, rgba, grid, levels=(0, 255))

    def _set_raster(self, item: pg.ImageItem, image, grid: Grid, levels) -> None:
        if image is None:
            item.clear()
            return
        item.setImage(image, autoLevels=False, levels=levels)
        x, y, w, h = grid.image_rect()
        item.setRect(QtCore.QRectF(x, y, w, h))
