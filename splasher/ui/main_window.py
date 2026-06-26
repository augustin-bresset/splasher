"""Fenêtre principale.

Disposition : nuage 3D (navigation libre) | vue de dessus (grille BEV).
Docks : conception de grille + palette + canaux (gauche), caméras (droite), temps (bas).

- La grille se crée explicitement (bouton « Nouvelle grille ») ; l'undo est par frame.
- Les canaux disponibles s'affichent/se masquent via le gestionnaire de canaux.
- La sélection d'un rectangle nourrit les cibles actives (Grille / Points) ; en mode
  cumulé, les labels points sont décumulés vers chaque frame source.
"""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core.accumulate import Accumulation, accumulate, window_indices
from ..core.colormap import colormap
from ..core.grid import Grid, grid_from_points
from ..core.io import load_session, save_session
from ..core.labels import LabelSet
from ..core.projection import bev_image, bev_max_height, cells_in_rect, points_in_rect
from ..core.source import ChannelKind, Source, channels_of_kind
from ..core.target import GridTarget, PointTarget
from .channel_manager import ChannelManager
from .cloud_view import CloudView
from .grid_designer import GridDesigner
from .grid_view import GridView
from .image_view import ImageView
from .palette import Palette
from .timeline import Timeline


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, source: Source, title: str = "Splasher",
                 labelset: LabelSet | None = None) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1480, 860)
        self._source = source
        self._index = 0
        self._labelset = labelset or LabelSet.default()

        self._cloud_keys = channels_of_kind(source, ChannelKind.POINTCLOUD)
        self._image_keys = channels_of_kind(source, ChannelKind.IMAGE)
        pose_keys = channels_of_kind(source, ChannelKind.POSE)
        self._pose_key = pose_keys[0] if pose_keys else None
        self._accum_radius = 0
        self._visible_clouds: set[str] = set(self._cloud_keys)
        self._visible_images: set[str] = set(self._image_keys)
        self._tool = "paint"  # "paint" (peinture directe) | "select" (sélection marquee)
        self._selection: np.ndarray | None = None  # masque (rows, cols) ou None

        # --- canvases centraux : 3D | vue de dessus -------------------------
        self._cloud_view = CloudView()
        self._grid_view = GridView()
        self._grid_view.rectDrawn.connect(self._on_rect)
        central = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        central.addWidget(self._cloud_view)
        central.addWidget(self._grid_view)
        central.setStretchFactor(0, 1)
        central.setStretchFactor(1, 1)
        self.setCentralWidget(central)

        # --- grille + cibles ------------------------------------------------
        self._grid = self._default_grid()
        self._grid_view.set_grid(self._grid)
        self._grid_target = GridTarget(self._grid, ignore_id=self._labelset.ignore_id)
        self._point_target = PointTarget(ignore_id=self._labelset.ignore_id)
        self._active_targets: set[str] = {"grid"}

        # --- docks ----------------------------------------------------------
        self._designer = GridDesigner(self._grid)
        self._designer.previewChanged.connect(self._on_grid_preview)
        self._designer.newGridRequested.connect(self._on_grid_commit)
        self._add_dock("Grille", self._designer, QtCore.Qt.LeftDockWidgetArea)

        self._palette = Palette(self._labelset)
        self._palette.classChanged.connect(lambda _id: self._update_status())
        self._add_dock("Classes", self._palette, QtCore.Qt.LeftDockWidgetArea)

        self._channels = ChannelManager(source.channels())
        self._channels.visibilityChanged.connect(self._on_visibility)
        self._add_dock("Canaux", self._channels, QtCore.Qt.LeftDockWidgetArea)

        self._build_toolbar()

        self._image_views: dict[str, ImageView] = {}
        if self._image_keys:
            self._image_panel = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(self._image_panel)
            v.setContentsMargins(0, 0, 0, 0)
            for key in self._image_keys:
                iv = ImageView(title=key)
                self._image_views[key] = iv
                v.addWidget(iv)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self._image_panel)
            self._add_dock("Caméras", scroll, QtCore.Qt.RightDockWidgetArea)

        self._timeline = Timeline(len(source))
        self._timeline.frameChanged.connect(self._show_frame)
        self._add_dock("Temps", self._timeline, QtCore.Qt.BottomDockWidgetArea, fixed=True)

        self._update_status()
        if len(source) > 0:
            self._show_frame(0)

    # ------------------------------------------------------------------ build
    def _add_dock(self, title, widget, area, fixed: bool = False) -> None:
        dock = QtWidgets.QDockWidget(title, self)
        dock.setWidget(widget)
        if fixed:
            dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(area, dock)

    def _tool_button(self, tb, text, *, checkable=False, checked=False, on=None, tip=""):
        btn = QtWidgets.QToolButton()
        btn.setText(text)
        btn.setCheckable(checkable)
        btn.setChecked(checked)
        if tip:
            btn.setToolTip(tip)
        if on is not None:
            (btn.toggled if checkable else btn.clicked).connect(on)
        tb.addWidget(btn)
        return btn

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Outils")
        tb.setMovable(False)
        self._tool_button(tb, "✏ Peindre", checkable=True, checked=True,
                          on=self._grid_view.set_paint_mode,
                          tip="Peindre (clic-glisser) vs naviguer (décoché)")
        self._tool_button(tb, "Effacer frame", on=self._on_clear)
        self._tool_button(tb, "↶ Annuler (frame)", on=self._on_undo)
        tb.addSeparator()
        self._tool_button(tb, "⬚ Sélection", checkable=True, checked=False,
                          on=self._on_tool_changed,
                          tip="Sélectionner des cellules (Shift = ajouter), puis appliquer.\n"
                              "Décoché = peinture directe.")
        self._apply_sel_btn = self._tool_button(tb, "✓ Appliquer sél.", on=self._on_apply_selection,
                                                tip="Appliquer la classe active à la sélection")
        self._clear_sel_btn = self._tool_button(tb, "✗ Vider sél.", on=self._on_clear_selection)
        self._apply_sel_btn.setEnabled(False)
        self._clear_sel_btn.setEnabled(False)
        tb.addSeparator()
        lbl = QtWidgets.QLabel(" cible : ")
        lbl.setStyleSheet("color:#aaa;")
        tb.addWidget(lbl)
        self._tool_button(tb, "▦ Grille", checkable=True, checked=True,
                          on=lambda on: self._toggle_target("grid", on),
                          tip="Sortie : raster de la grille BEV")
        self._tool_button(tb, "• Points", checkable=True, checked=False,
                          on=lambda on: self._toggle_target("points", on),
                          tip="Sortie : labels par point (le nuage se colorise)")
        tb.addSeparator()
        lbl2 = QtWidgets.QLabel(" cumul ± ")
        lbl2.setStyleSheet("color:#aaa;")
        tb.addWidget(lbl2)
        self._accum_spin = QtWidgets.QSpinBox()
        self._accum_spin.setRange(0, max(0, len(self._source) - 1))
        self._accum_spin.setSuffix(" frames")
        self._accum_spin.setToolTip(
            "Cumuler ±N frames recalées par leurs poses dans le repère du frame courant.\n"
            "La grille et les labels restent par frame (décumul)."
        )
        self._accum_spin.setEnabled(self._pose_key is not None and len(self._source) > 1)
        self._accum_spin.valueChanged.connect(self._on_accum_changed)
        tb.addWidget(self._accum_spin)
        tb.addSeparator()
        self._tool_button(tb, "💾 Enregistrer", on=self._on_save,
                          tip="Exporter labels (.npy/.png/.json)")
        self._tool_button(tb, "📂 Charger", on=self._on_load,
                          tip="Recharger une session de labels")

    # ------------------------------------------------------------------ utils
    def _frame_points(self, frame) -> np.ndarray:
        parts = [frame.channels[k] for k in self._cloud_keys
                 if frame.channels.get(k) is not None and len(frame.channels[k])]
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 3), np.float32)

    def _default_grid(self) -> Grid:
        if len(self._source) == 0 or not self._cloud_keys:
            return Grid(-20.0, 20.0, -20.0, 20.0, 1.0)
        return grid_from_points(self._frame_points(self._source[0])[:, :2], cell_size=1.0)

    def _visible_cloud_indices(self) -> list[int]:
        return [i for i, k in enumerate(self._cloud_keys) if k in self._visible_clouds]

    def _update_status(self) -> None:
        active = ", ".join(sorted(self._active_targets)) or "—"
        cls = self._labelset.name_of(self._palette.active_id())
        cumul = f"±{self._accum_radius}" if self._accum_radius else "off"
        outil = "sélection" if self._tool == "select" else "peinture"
        self.statusBar().showMessage(
            f"{len(self._source)} frames · nuages: {sorted(self._visible_clouds) or '—'} · "
            f"caméras: {sorted(self._visible_images) or '—'} · classe: {cls} · "
            f"cible: {active} · cumul: {cumul} · outil: {outil}"
        )

    # --------------------------------------------------------------- handlers
    def _toggle_target(self, name: str, on: bool) -> None:
        (self._active_targets.add if on else self._active_targets.discard)(name)
        self._update_status()

    def _on_visibility(self) -> None:
        self._visible_clouds = self._channels.visible_clouds()
        self._visible_images = self._channels.visible_images()
        for key, view in self._image_views.items():
            view.setVisible(key in self._visible_images)
        acc = self._accumulated()
        self._refresh_cloud(acc)
        self._refresh_bev(acc)
        self._grid_view.set_topdown_points(self._visible_xy(acc))
        self._update_status()

    def _on_grid_preview(self, grid: Grid) -> None:
        # aperçu des lignes seulement (n'efface rien, ne change pas la grille active)
        self._grid_view.set_grid(grid, autorange=False)

    def _on_grid_commit(self, grid: Grid) -> None:
        n_labelled = len(self._grid_target.rasters())
        if n_labelled:
            reply = QtWidgets.QMessageBox.warning(
                self, "Nouvelle grille",
                f"Une labélisation de grille existe déjà ({n_labelled} frame(s)).\n"
                "Créer une nouvelle grille l'effacera (les labels par point sont conservés).\n\n"
                "Continuer ?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                # annulation : on restaure l'affichage sur la grille active
                self._designer.set_grid(self._grid)
                self._grid_view.set_grid(self._grid, autorange=False)
                return
        self._grid = grid
        self._grid_view.set_grid(grid, autorange=True)
        self._grid_target = GridTarget(grid, ignore_id=self._labelset.ignore_id)
        self._set_selection(None)
        self._refresh_bev()
        self._refresh_labels()
        msg = f"nouvelle grille {grid.cols}×{grid.rows}"
        if n_labelled:
            msg += " · grille labélisée réinitialisée"
        self.statusBar().showMessage(msg, 6000)

    def _on_accum_changed(self, value: int) -> None:
        self._accum_radius = int(value)
        acc = self._accumulated()
        self._refresh_cloud(acc)
        self._refresh_bev(acc)
        self._grid_view.set_topdown_points(self._visible_xy(acc))
        self._update_status()

    def _on_tool_changed(self, on: bool) -> None:
        self._tool = "select" if on else "paint"
        if not on:
            self._set_selection(None)
        self._update_status()

    def _set_selection(self, mask: np.ndarray | None) -> None:
        self._selection = mask
        self._grid_view.set_selection(mask, self._grid)
        active = mask is not None and bool(mask.any())
        self._apply_sel_btn.setEnabled(active)
        self._clear_sel_btn.setEnabled(active)

    def _on_clear_selection(self) -> None:
        self._set_selection(None)

    def _on_rect(self, rect) -> None:
        if self._tool == "select":
            self._select_rect(rect)
            return
        cls = self._palette.active_id()
        changed = False
        if "grid" in self._active_targets:
            changed |= self._grid_target.apply(self._index, rect, cls)
        if "points" in self._active_targets:
            acc = self._accumulated()
            changed |= self._paint_points(acc, points_in_rect(acc.xy, rect), cls)
        if changed:
            self._refresh_labels()
            self._refresh_cloud()

    def _select_rect(self, rect) -> None:
        si, sj = cells_in_rect(rect, self._grid)
        if si.start >= si.stop or sj.start >= sj.stop:
            return
        rect_mask = np.zeros(self._grid.shape, dtype=bool)
        rect_mask[si, sj] = True
        additive = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier)
        if additive and self._selection is not None:
            self._set_selection(self._selection | rect_mask)
        else:
            self._set_selection(rect_mask)

    def _paint_points(self, acc: Accumulation, point_mask: np.ndarray, cls: int) -> bool:
        """Décumule un masque de points (du cumul `acc`) vers chaque frame source."""
        mask = point_mask & acc.visible_mask(self._visible_cloud_indices())
        if not mask.any():
            return False
        fids, pids = acc.frame_id[mask], acc.point_id[mask]
        frame_to_sel = {int(f): (pids[fids == f], acc.counts[int(f)]) for f in np.unique(fids)}
        return self._point_target.apply_scatter(self._index, frame_to_sel, cls)

    def _on_apply_selection(self) -> None:
        if self._selection is None or not self._selection.any():
            return
        cls = self._palette.active_id()
        changed = False
        if "grid" in self._active_targets:
            changed |= self._grid_target.apply_mask(self._index, self._selection, cls)
        if "points" in self._active_targets:
            acc = self._accumulated()
            ij, valid = self._grid.world_to_cell(acc.xy)
            hit = np.zeros(len(acc.xy), dtype=bool)
            hit[valid] = self._selection[ij[valid, 0], ij[valid, 1]]
            changed |= self._paint_points(acc, hit, cls)
        if changed:
            self._set_selection(None)
            self._refresh_labels()
            self._refresh_cloud()

    def _on_clear(self) -> None:
        if "grid" in self._active_targets:
            self._grid_target.clear(self._index)
        if "points" in self._active_targets:
            self._point_target.clear(self._index)
        self._refresh_labels()
        self._refresh_cloud()

    def _on_undo(self) -> None:
        if "grid" in self._active_targets:
            self._grid_target.undo(self._index)
        if "points" in self._active_targets:
            self._point_target.undo(self._index)
        self._refresh_labels()
        self._refresh_cloud()

    def _on_save(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Dossier de sortie des labels")
        if not d:
            return
        save_session(d, grid=self._grid, labelset=self._labelset,
                     grid_target=self._grid_target, point_target=self._point_target)
        self.statusBar().showMessage(
            f"enregistré dans {d} · grille: {len(self._grid_target.rasters())} frame(s) · "
            f"points: {len(self._point_target.all_labels())} frame(s)", 8000)

    def _on_load(self) -> None:
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Charger une session de labels")
        if not d:
            return
        data = load_session(d)
        self._grid = data["grid"]
        self._designer.set_grid(self._grid)
        self._grid_view.set_grid(self._grid, autorange=False)
        self._grid_target = GridTarget(self._grid, ignore_id=self._labelset.ignore_id)
        self._grid_target.load_rasters(data["grid_labels"])
        self._point_target = PointTarget(ignore_id=self._labelset.ignore_id)
        self._point_target.load_labels(data["point_labels"])
        self._set_selection(None)
        self._refresh_bev()
        self._refresh_labels()
        self._refresh_cloud()
        self.statusBar().showMessage(f"chargé depuis {d}", 8000)

    # ------------------------------------------------------------- rendering
    def _accumulated(self) -> Accumulation:
        """Cumul sur **tous** les canaux nuage (point_id stable) ; cumul ±radius via poses."""
        n = len(self._source)
        if self._accum_radius > 0 and self._pose_key is not None:
            idx = window_indices(self._index, self._accum_radius, n)
            return accumulate(self._source, self._index, idx, self._cloud_keys, self._pose_key)
        return accumulate(self._source, self._index, [self._index], self._cloud_keys, self._pose_key)

    def _visible_xy(self, acc: Accumulation) -> np.ndarray:
        vis = acc.visible_mask(self._visible_cloud_indices())
        return acc.xy[vis]

    def _acc_colors(self, acc: Accumulation) -> np.ndarray | None:
        if len(acc.points) == 0:
            return None
        colors = colormap(acc.points[:, 2])
        lab = np.full(len(acc.points), self._labelset.ignore_id, np.int64)
        for f in np.unique(acc.frame_id):
            f = int(f)
            if not self._point_target.has(f):
                continue
            src = self._point_target.labels(f)
            sel = acc.frame_id == f
            pid = acc.point_id[sel]
            if len(pid) and int(pid.max()) < len(src):
                lab[sel] = src[pid]
        mask = lab != self._labelset.ignore_id
        if mask.any():
            lut = self._labelset.lut(alpha=255, max_id=int(lab.max()))
            colors[mask] = lut[lab[mask]].astype(np.float32) / 255.0
        return colors

    def _refresh_cloud(self, acc: Accumulation | None = None) -> None:
        acc = acc if acc is not None else self._accumulated()
        vis = acc.visible_mask(self._visible_cloud_indices())
        colors = self._acc_colors(acc)
        self._cloud_view.set_cloud(
            "scene", acc.points[vis], None if colors is None else colors[vis]
        )

    def _refresh_bev(self, acc: Accumulation | None = None) -> None:
        acc = acc if acc is not None else self._accumulated()
        vis = acc.visible_mask(self._visible_cloud_indices())
        height = bev_max_height(acc.points[vis], self._grid)
        self._grid_view.set_underlay(bev_image(height), self._grid)

    def _refresh_labels(self) -> None:
        if self._grid_target.has(self._index):
            raster = self._grid_target.raster(self._index)
            self._grid_view.set_labels(self._labelset.colorize(raster, alpha=170), self._grid)
        else:
            self._grid_view.set_labels(None, self._grid)

    def _show_frame(self, i: int) -> None:
        self._index = i
        frame = self._source[i]
        acc = self._accumulated()
        self._refresh_cloud(acc)
        self._refresh_bev(acc)
        self._grid_view.set_topdown_points(self._visible_xy(acc))
        for key in self._image_keys:
            img = frame.channels.get(key)
            if img is not None:
                self._image_views[key].set_image(img)
        self._refresh_labels()
