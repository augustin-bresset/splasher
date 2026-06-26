"""Point d'entrée UI : `launch(source)`."""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6 import QtWidgets

from ..core.source import Source
from .main_window import MainWindow


def launch(source: Source, *, labels=None, title: str = "Splasher", block: bool = True):
    """Ouvre la fenêtre Splasher sur `source`.

    `labels` : un `LabelSet` optionnel (défaut = `LabelSet.default()`).
    Réutilise une `QApplication` existante si présente. Si `block` et qu'aucune
    boucle n'est active, lance `app.exec()`. Renvoie la fenêtre.
    """
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if owns_app:
        app = QtWidgets.QApplication(sys.argv[:1])

    window = MainWindow(source, title=title, labelset=labels)
    window.show()

    if block and owns_app:
        app.exec()
    return window
