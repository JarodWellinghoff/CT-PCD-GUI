from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ct_pcd_gui.shared.qt.dicom_viewer import DicomSliceViewer

from ..domain.models import DicomVolume, LesionInstance, PreviewResult
from .slice_view import SliceMarker, SliceView


class LesionInsertionWorkspace(QWidget):
    slice_changed = Signal(int)
    location_selected = Signal(float, float, int)
    preview_toggled = Signal(bool)
    preview_mode_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._volume: DicomVolume | None = None
        self._preview: PreviewResult | None = None
        self._lesions: tuple[LesionInstance, ...] = ()
        self._selected_id = ""
        self._cursor: tuple[float, float, float] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        research = QLabel(
            "Research use only — image-domain preview is approximate; final insertion "
            "operates on DICOM-CT-PD raw data."
        )
        research.setWordWrap(True)
        research.setObjectName("ResearchNotice")
        layout.addWidget(research)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.view = SliceView()
        self.viewer = DicomSliceViewer(self, image_view=self.view)
        self.viewer.slice_changed.connect(self._slice_selected)
        self.view.location_selected.connect(self._location_clicked)

        self.viewer.add_toolbar_separator()
        self.preview_checkbox = QCheckBox("Approximate preview", self.viewer.toolbar)
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.setToolTip(
            "Toggle the responsive image-domain approximation. This is not the final "
            "raw-data reconstruction."
        )
        self.preview_checkbox.toggled.connect(self.preview_toggled)
        self.viewer.add_toolbar_widget(self.preview_checkbox)
        self.mode_combo = QComboBox(self.viewer.toolbar)
        self.mode_combo.addItem("Before", "before")
        self.mode_combo.addItem("Overlay", "overlay")
        self.mode_combo.addItem("After (approx.)", "after")
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentIndexChanged.connect(
            lambda _index: self.preview_mode_changed.emit(
                str(self.mode_combo.currentData())
            )
        )
        self.viewer.add_toolbar_widget(self.mode_combo)

        # Compatibility aliases used by the presenter and existing integrations.
        self.preset_combo = self.viewer.preset_combo
        self.slice_slider = self.viewer.slice_slider
        self.slice_spin = self.viewer.slice_spin
        self.coordinate_label = self.viewer.info_label
        splitter.addWidget(self.viewer)

        log_container = QWidget()
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(0, 0, 0, 0)
        status_row = QHBoxLayout()
        self.status_label = QLabel("Load a reconstruction to begin.")
        self.status_label.setWordWrap(True)
        status_row.addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        status_row.addWidget(self.progress)
        log_layout.addLayout(status_row)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        log_layout.addWidget(self.log)
        splitter.addWidget(log_container)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    @property
    def current_slice(self) -> int:
        return self.viewer.current_slice

    @property
    def cursor_position(self) -> tuple[float, float, float] | None:
        return self._cursor

    def set_volume(self, volume: DicomVolume) -> None:
        self._volume = volume
        slice_count = volume.hu.shape[0]
        middle = max(0, slice_count - 1) // 2
        self._cursor = (
            (volume.hu.shape[2] - 1) / 2.0,
            (volume.hu.shape[1] - 1) / 2.0,
            float(middle),
        )
        self._preview = None
        self.viewer.clear_image()
        self.viewer.set_slice_count(
            slice_count,
            initial_index=middle,
            emit=False,
        )
        self.render()

    def set_scene_state(
        self,
        *,
        lesions: tuple[LesionInstance, ...],
        selected_id: str,
        cursor: tuple[float, float, float] | None = None,
    ) -> None:
        self._lesions = lesions
        self._selected_id = selected_id
        if cursor is not None:
            self._cursor = cursor
        self.render()

    def set_preview(self, result: PreviewResult | None) -> None:
        self._preview = result
        self.render()

    def set_preview_state(self, text: str) -> None:
        self.status_label.setText(text)

    def render(self) -> None:
        if self._volume is None:
            self.viewer.clear_image()
            self.viewer.set_info_text("No volume loaded")
            return
        z = self.current_slice
        before = np.asarray(self._volume.hu[z], dtype=np.float32)
        display = before
        mode = str(self.mode_combo.currentData())
        preview_valid = self._preview is not None and self._preview.slice_index == z
        if self.preview_checkbox.isChecked() and preview_valid:
            assert self._preview is not None
            if mode == "after":
                display = self._preview.approximate_after_hu
            elif mode == "overlay":
                alpha = self._preview.overlay_alpha
                display = (
                    before * (1.0 - alpha)
                    + self._preview.approximate_after_hu * alpha
                )
        self.viewer.set_hu_image(display)
        markers = [
            SliceMarker(
                instance_id=lesion.instance_id,
                column=lesion.center_voxel_crs[0],
                row=lesion.center_voxel_crs[1],
                selected=lesion.instance_id == self._selected_id,
                enabled=lesion.enabled,
            )
            for lesion in self._lesions
            if lesion.visible and abs(lesion.center_voxel_crs[2] - z) <= 0.51
        ]
        cursor = None
        if self._cursor is not None and abs(self._cursor[2] - z) <= 0.51:
            cursor = (self._cursor[0], self._cursor[1])
        self.view.set_annotations(cursor_position=cursor, markers=markers)
        if self._cursor is not None:
            self.viewer.set_info_text(
                f"C/R/S: {self._cursor[0]:.1f}, {self._cursor[1]:.1f}, "
                f"{self._cursor[2]:.1f}"
            )

    @Slot(int)
    def _slice_selected(self, value: int) -> None:
        if self._cursor is not None:
            self._cursor = (self._cursor[0], self._cursor[1], float(value))
        self.render()
        self.slice_changed.emit(value)

    @Slot(float, float)
    def _location_clicked(self, column: float, row: float) -> None:
        self._cursor = (column, row, float(self.current_slice))
        self.render()
        self.location_selected.emit(column, row, self.current_slice)

    def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_checkbox.setChecked(enabled)

    def set_preview_mode(self, mode: str) -> None:
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

    def show_busy(self, message: str, *, indeterminate: bool = True) -> None:
        self.status_label.setText(message)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0 if indeterminate else 100)

    def show_progress(self, completed: int, total: int, message: str) -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(completed)
        self.status_label.setText(message)

    def show_ready(self, message: str) -> None:
        self.progress.setVisible(False)
        self.progress.setRange(0, 100)
        self.status_label.setText(message)

    def append_log(self, message: str) -> None:
        self.log.appendPlainText(message)
