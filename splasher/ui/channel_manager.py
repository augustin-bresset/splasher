"""Gestionnaire de canaux : afficher/masquer les canaux disponibles de la `Source`.

Liste tous les canaux (nuages, caméras, poses…). Les nuages et les images sont
cochables (afficher/masquer la vue correspondante). Les poses/scalaires sont listés
pour information (non cochables).
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..core.source import ChannelKind, ChannelSpec

_KIND_LABEL = {
    ChannelKind.POINTCLOUD: "nuage",
    ChannelKind.IMAGE: "caméra",
    ChannelKind.POSE: "pose",
    ChannelKind.SCALAR: "scalaire",
}


class ChannelManager(QtWidgets.QWidget):
    visibilityChanged = QtCore.Signal()

    def __init__(self, specs: list[ChannelSpec]) -> None:
        super().__init__()
        self._boxes: dict[str, QtWidgets.QCheckBox] = {}
        self._kind: dict[str, ChannelKind] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)

        for spec in specs:
            self._kind[spec.name] = spec.kind
            toggleable = spec.kind in (ChannelKind.POINTCLOUD, ChannelKind.IMAGE)
            row = QtWidgets.QCheckBox(f"{spec.name}  ·  {_KIND_LABEL.get(spec.kind, '?')}")
            row.setChecked(toggleable)
            row.setEnabled(toggleable)
            if toggleable:
                row.toggled.connect(lambda _checked: self.visibilityChanged.emit())
            self._boxes[spec.name] = row
            layout.addWidget(row)

        layout.addStretch(1)

    def _visible_of_kind(self, kind: ChannelKind) -> set[str]:
        return {name for name, box in self._boxes.items()
                if self._kind[name] == kind and box.isChecked()}

    def visible_clouds(self) -> set[str]:
        return self._visible_of_kind(ChannelKind.POINTCLOUD)

    def visible_images(self) -> set[str]:
        return self._visible_of_kind(ChannelKind.IMAGE)
