"""Panneau de conception de la grille : étendue monde + taille de carré.

Deux temps, pour que ce soit clair :
- éditer les champs met à jour l'**aperçu** des lignes (`previewChanged`) et l'affichage
  `cols × rows`, sans rien effacer ;
- le bouton **« Nouvelle grille »** crée/applique la grille (`newGridRequested`), ce qui
  **réinitialise** les rasters de grille (ils sont liés à la géométrie).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core.grid import Grid


def _spin(value, lo, hi, step, decimals) -> QtWidgets.QDoubleSpinBox:
    sb = QtWidgets.QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setSingleStep(step)
    sb.setDecimals(decimals)
    sb.setValue(value)
    return sb


class GridDesigner(QtWidgets.QWidget):
    previewChanged = QtCore.Signal(object)     # Grid (aperçu, n'efface rien)
    newGridRequested = QtCore.Signal(object)   # Grid (création/commit)

    def __init__(self, grid: Grid) -> None:
        super().__init__()
        self._committed = grid

        self._xmin = _spin(grid.xmin, -2000, 2000, 1.0, 1)
        self._xmax = _spin(grid.xmax, -2000, 2000, 1.0, 1)
        self._ymin = _spin(grid.ymin, -2000, 2000, 1.0, 1)
        self._ymax = _spin(grid.ymax, -2000, 2000, 1.0, 1)
        self._cell = _spin(grid.cell_size, 0.05, 100.0, 0.5, 2)

        form = QtWidgets.QFormLayout()
        form.addRow("x min", self._xmin)
        form.addRow("x max", self._xmax)
        form.addRow("y min", self._ymin)
        form.addRow("y max", self._ymax)
        form.addRow("taille carré (m)", self._cell)

        self._dims = QtWidgets.QLabel()
        self._dims.setStyleSheet("color:#9cf; font-weight:bold;")

        self._btn = QtWidgets.QPushButton("➕ Nouvelle grille")
        self._btn.setToolTip("Créer/appliquer la grille à cette taille (réinitialise la grille labélisée)")
        self._btn.clicked.connect(self._on_commit)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(self._dims)
        root.addWidget(self._btn)
        root.addStretch(1)

        for sb in (self._xmin, self._xmax, self._ymin, self._ymax, self._cell):
            sb.valueChanged.connect(self._on_edit)

        self._update_dims(grid)

    def committed_grid(self) -> Grid:
        return self._committed

    def set_grid(self, grid: Grid) -> None:
        """Recale les champs sur `grid` sans rien émettre (ex. après chargement)."""
        self._committed = grid
        for sb, val in ((self._xmin, grid.xmin), (self._xmax, grid.xmax),
                        (self._ymin, grid.ymin), (self._ymax, grid.ymax), (self._cell, grid.cell_size)):
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)
        self._update_dims(grid)

    def _build(self) -> Grid | None:
        try:
            return Grid(self._xmin.value(), self._xmax.value(),
                        self._ymin.value(), self._ymax.value(), self._cell.value())
        except ValueError:
            return None

    def _update_dims(self, grid: Grid) -> None:
        self._dims.setText(f"{grid.cols} × {grid.rows} carrés  ({grid.cols * grid.rows} cellules)")

    def _on_edit(self) -> None:
        grid = self._build()
        if grid is None:
            self._dims.setText("étendue invalide")
            self._btn.setEnabled(False)
            return
        self._btn.setEnabled(True)
        self._update_dims(grid)
        self.previewChanged.emit(grid)

    def _on_commit(self) -> None:
        grid = self._build()
        if grid is None:
            return
        self._committed = grid
        self.newGridRequested.emit(grid)
