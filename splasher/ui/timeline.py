"""Curseur temporel : slider sur les frames + lecture (play/pause)."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class Timeline(QtWidgets.QWidget):
    frameChanged = QtCore.Signal(int)

    def __init__(self, n_frames: int) -> None:
        super().__init__()
        self._n = max(1, n_frames)

        self._play_btn = QtWidgets.QToolButton()
        self._play_btn.setText("▶")
        self._play_btn.setCheckable(True)
        self._play_btn.toggled.connect(self._on_play_toggled)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._slider.setRange(0, self._n - 1)
        self._slider.valueChanged.connect(self._on_slider)

        self._label = QtWidgets.QLabel()
        self._label.setMinimumWidth(90)
        self._update_label(0)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(100)  # ~10 fps
        self._timer.timeout.connect(self._advance)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._slider)
        layout.addWidget(self._label)

    @property
    def index(self) -> int:
        return self._slider.value()

    def _update_label(self, i: int) -> None:
        self._label.setText(f"frame {i + 1} / {self._n}")

    def _on_slider(self, i: int) -> None:
        self._update_label(i)
        self.frameChanged.emit(i)

    def _on_play_toggled(self, on: bool) -> None:
        self._play_btn.setText("⏸" if on else "▶")
        (self._timer.start if on else self._timer.stop)()

    def _advance(self) -> None:
        self._slider.setValue((self._slider.value() + 1) % self._n)
