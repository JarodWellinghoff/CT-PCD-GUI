from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ct_pcd_gui.shared.qt.dicom_viewer import DicomSliceViewer, window_hu_to_uint8

from ..models import OverlayEntry, SeriesPreview
from .image_view import OverlayImageView


class LesionExtractionWorkspace(QWidget):
    series_changed = Signal(int)
    entity_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._preview: SeriesPreview | None = None
        self._entry_by_label: dict[int, OverlayEntry] = {}
        self._entry_by_entity: dict[str, OverlayEntry] = {}
        self._muted_entities: set[str] = set()
        self._selected_entity: str | None = None
        self._build_ui()
        self._connect()
        self.clear_preview()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        series_row = QHBoxLayout()
        series_row.addWidget(QLabel("Series:"))
        self.series_combo = QComboBox(self)
        self.series_combo.setMinimumWidth(260)
        series_row.addWidget(self.series_combo, 1)
        root.addLayout(series_row)

        self.image_view = OverlayImageView()
        self.image_view.setMinimumSize(480, 420)
        self.viewer = DicomSliceViewer(self, image_view=self.image_view)
        self.viewer.add_toolbar_separator()
        self.show_overlay = QCheckBox("Show overlay", self.viewer.toolbar)
        self.show_overlay.setChecked(True)
        self.viewer.add_toolbar_widget(self.show_overlay)
        self.viewer.add_toolbar_widget(QLabel("Opacity:", self.viewer.toolbar))
        self.opacity = QSlider(Qt.Orientation.Horizontal, self.viewer.toolbar)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setFixedWidth(110)
        self.viewer.add_toolbar_widget(self.opacity)

        # Compatibility aliases for integrations that use the workspace directly.
        self.slice_slider = self.viewer.slice_slider
        self.slice_label = self.viewer.slice_position_label
        self.summary = self.viewer.info_label

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.addWidget(self.viewer)
        legend_container = QWidget(splitter)
        legend_layout = QVBoxLayout(legend_container)
        legend_layout.setContentsMargins(6, 0, 0, 0)
        legend_layout.addWidget(QLabel("Overlay legend"))
        self.legend = QTreeWidget(legend_container)
        self.legend.setHeaderLabels(["Name", "State"])
        self.legend.setRootIsDecorated(False)
        self.legend.setAlternatingRowColors(True)
        self.legend.setMinimumWidth(220)
        legend_layout.addWidget(self.legend, 1)
        help_label = QLabel(
            "Select a component to jump to its center. Click an overlay to select it. "
            "Use the mouse wheel for slices, Ctrl+wheel to zoom, middle drag to pan, "
            "and right drag to adjust window/level.",
            legend_container,
        )
        help_label.setWordWrap(True)
        legend_layout.addWidget(help_label)
        splitter.addWidget(legend_container)
        splitter.setStretchFactor(0, 1)
        root.addWidget(splitter, 1)

    def _connect(self) -> None:
        self.series_combo.currentIndexChanged.connect(self.series_changed)
        self.viewer.slice_changed.connect(self._render)
        self.viewer.window_level_changed.connect(self._render)
        self.show_overlay.toggled.connect(self._render)
        self.opacity.valueChanged.connect(self._render)
        self.image_view.label_clicked.connect(self._select_label)
        self.legend.itemSelectionChanged.connect(self._legend_changed)

    def set_series_names(self, names: Sequence[str], selected_index: int = 0) -> None:
        with QSignalBlocker(self.series_combo):
            self.series_combo.clear()
            self.series_combo.addItems([str(name) for name in names])
            if names:
                self.series_combo.setCurrentIndex(
                    max(0, min(selected_index, len(names) - 1))
                )
        self.series_combo.setEnabled(bool(names))

    def current_series_index(self) -> int:
        return max(0, self.series_combo.currentIndex())

    def set_preview(self, preview: SeriesPreview) -> None:
        self._preview = preview
        self._entry_by_label = {entry.label_id: entry for entry in preview.overlay_entries}
        self._entry_by_entity = {
            entry.entity_id: entry for entry in preview.overlay_entries
        }
        self._selected_entity = None
        self._populate_legend()
        slices = preview.volume_hu_zyx.shape[0]
        visible = np.flatnonzero(np.any(preview.overlay_labels_zyx > 0, axis=(1, 2)))
        initial = int(visible[len(visible) // 2]) if visible.size else slices // 2
        self.viewer.clear_image()
        self.viewer.set_slice_count(slices, initial_index=initial, emit=False)
        self.viewer.set_window(
            preview.window_width,
            preview.window_center,
            emit=False,
            remember_for_reset=True,
        )
        self.viewer.set_info_text(
            f"{preview.volume_hu_zyx.shape[2]} x "
            f"{preview.volume_hu_zyx.shape[1]} x {slices}"
        )
        self._render()

    def clear_preview(self, message: str = "No inputs loaded") -> None:
        self._preview = None
        self._entry_by_label.clear()
        self._entry_by_entity.clear()
        self.viewer.clear_image()
        self.viewer.set_slice_count(0)
        self.viewer.set_info_text(message)
        self.legend.clear()
        self.series_combo.setEnabled(False)

    def set_muted_entities(self, entity_ids: set[str]) -> None:
        self._muted_entities = set(entity_ids)
        self._populate_legend()
        self._render()

    def update_entity_names(self, names: Mapping[str, str]) -> None:
        updated: dict[int, OverlayEntry] = {}
        for label_id, entry in self._entry_by_label.items():
            updated[label_id] = replace(
                entry, display_name=names.get(entry.entity_id, entry.display_name)
            )
        self._entry_by_label = updated
        self._entry_by_entity = {
            entry.entity_id: entry for entry in updated.values()
        }
        self._populate_legend()

    def select_entity(self, entity_id: str, *, emit: bool = False) -> None:
        entry = self._entry_by_entity.get(entity_id)
        if entry is None:
            return
        self._selected_entity = entity_id
        if entry.center_index_xyz is not None:
            self.viewer.set_slice(entry.center_index_xyz[2])
        self._select_legend(entity_id)
        self._render()
        if emit:
            self.entity_selected.emit(entity_id)

    def _populate_legend(self) -> None:
        selected = self._selected_entity
        with QSignalBlocker(self.legend):
            self.legend.clear()
            for entry in self._entry_by_label.values():
                muted = entry.entity_id in self._muted_entities
                item = QTreeWidgetItem(
                    [entry.display_name, "Omitted" if muted else "Included"]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, entry.entity_id)
                item.setForeground(
                    0,
                    QBrush(
                        QColor(150, 150, 150)
                        if muted
                        else QColor(*entry.color_rgb)
                    ),
                )
                self.legend.addTopLevelItem(item)
        if selected:
            self._select_legend(selected)
        self.legend.resizeColumnToContents(0)

    def _select_legend(self, entity_id: str) -> None:
        with QSignalBlocker(self.legend):
            for index in range(self.legend.topLevelItemCount()):
                item = self.legend.topLevelItem(index)
                if item.data(0, Qt.ItemDataRole.UserRole) == entity_id:
                    self.legend.setCurrentItem(item)
                    break

    def _legend_changed(self) -> None:
        items = self.legend.selectedItems()
        if items:
            entity_id = str(items[0].data(0, Qt.ItemDataRole.UserRole))
            self.select_entity(entity_id)
            self.entity_selected.emit(entity_id)

    def _select_label(self, label_id: int) -> None:
        entry = self._entry_by_label.get(label_id)
        if entry:
            self.select_entity(entry.entity_id)
            self.entity_selected.emit(entry.entity_id)

    def _render(self, *_args) -> None:
        preview = self._preview
        if preview is None:
            return
        index = self.viewer.current_slice
        source = preview.volume_hu_zyx[index]
        gray = window_hu_to_uint8(
            source,
            self.viewer.window_width,
            self.viewer.window_level,
        )
        rgb = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
        labels = preview.overlay_labels_zyx[index]
        if self.show_overlay.isChecked() and self.opacity.value() > 0:
            max_label = max(self._entry_by_label, default=0)
            colors = np.zeros((max_label + 1, 3), dtype=np.float32)
            alphas = np.zeros(max_label + 1, dtype=np.float32)
            base_alpha = self.opacity.value() / 100.0
            for label_id, entry in self._entry_by_label.items():
                muted = entry.entity_id in self._muted_entities
                selected = entry.entity_id == self._selected_entity
                colors[label_id] = (145, 145, 145) if muted else entry.color_rgb
                alpha = base_alpha * (0.28 if muted else 1.0)
                alphas[label_id] = max(alpha, 0.68) if selected else alpha
            clipped = np.minimum(labels, max_label)
            alpha_image = alphas[clipped][..., None]
            rgb = rgb * (1.0 - alpha_image) + colors[clipped] * alpha_image
        self.image_view.set_rgb_array(np.clip(rgb, 0, 255).astype(np.uint8), labels)
