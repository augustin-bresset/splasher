"""Palette de classes : sélection de la classe active (pastille couleur + nom)."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.labels import LabelSet


def _swatch(color) -> QtGui.QIcon:
    pix = QtGui.QPixmap(16, 16)
    pix.fill(QtGui.QColor(*color))
    return QtGui.QIcon(pix)


class Palette(QtWidgets.QWidget):
    classChanged = QtCore.Signal(int)  # id de classe active

    def __init__(self, labelset: LabelSet) -> None:
        super().__init__()
        self._labelset = labelset

        self._list = QtWidgets.QListWidget()
        for c in labelset.paintable:
            item = QtWidgets.QListWidgetItem(_swatch(c.color), f"{c.id} · {c.name}")
            item.setData(QtCore.Qt.UserRole, c.id)
            self._list.addItem(item)
        self._list.currentItemChanged.connect(self._on_change)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._list)

        if self._list.count():
            self._list.setCurrentRow(0)

    def active_id(self) -> int:
        item = self._list.currentItem()
        if item is None:
            return self._labelset.ignore_id
        return int(item.data(QtCore.Qt.UserRole))

    def _on_change(self, current, _previous) -> None:
        if current is not None:
            self.classChanged.emit(int(current.data(QtCore.Qt.UserRole)))
