"""Panneau d'affichage d'un canal image (caméra)."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets


class ImageView(QtWidgets.QWidget):
    """ViewBox verrouillé en aspect avec un `ImageItem`. Orientation image naturelle."""

    def __init__(self, title: str = "") -> None:
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        if title:
            label = QtWidgets.QLabel(title)
            label.setStyleSheet("color:#ccc; padding:2px;")
            layout.addWidget(label)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)
        self._vb = self._glw.addViewBox()
        self._vb.setAspectLocked(True)
        self._vb.invertY(True)  # (row 0) en haut, image à l'endroit
        self._img = pg.ImageItem()
        self._vb.addItem(self._img)

    def set_image(self, image: np.ndarray) -> None:
        if image is None:
            return
        self._img.setImage(np.ascontiguousarray(image), autoLevels=True)
        self._vb.autoRange(padding=0)
