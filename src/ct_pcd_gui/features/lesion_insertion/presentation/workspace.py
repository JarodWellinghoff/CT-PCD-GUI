from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QImage, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..domain.models import DicomVolume, LesionInstance, PreviewResult
from .slice_view import SliceMarker, SliceView

WINDOW_PRESETS = {
    "Soft tissue": (400.0, 40.0),
    "Liver": (150.0, 70.0),
    "Lung": (1500.0, -600.0),
    "Bone": (2000.0, 500.0),
}


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
        self._window = 400.0
        self._level = 40.0
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

        toolbar = QToolBar("Lesion placement", self)
        fit_action = toolbar.addAction("Fit")
        fit_action.setShortcut(QKeySequence("F"))
        fit_action.triggered.connect(self._fit)
        reset_action = toolbar.addAction("Reset view")
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(self._reset_view)
        toolbar.addSeparator()
        toolbar.addWidget(QLabel(" Window preset: "))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(WINDOW_PRESETS)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        toolbar.addWidget(self.preset_combo)
        toolbar.addSeparator()
        self.preview_checkbox = QCheckBox("Approximate preview")
        self.preview_checkbox.setChecked(True)
        self.preview_checkbox.setToolTip(
            "Toggle the responsive image-domain approximation. This is not the final "
            "raw-data reconstruction."
        )
        self.preview_checkbox.toggled.connect(self.preview_toggled)
        toolbar.addWidget(self.preview_checkbox)
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Before", "before")
        self.mode_combo.addItem("Overlay", "overlay")
        self.mode_combo.addItem("After (approx.)", "after")
        self.mode_combo.setCurrentIndex(1)
        self.mode_combo.currentIndexChanged.connect(
            lambda _index: self.preview_mode_changed.emit(str(self.mode_combo.currentData()))
        )
        toolbar.addWidget(self.mode_combo)
        layout.addWidget(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.view = SliceView()
        self.view.location_selected.connect(self._location_clicked)
        self.view.slice_delta_requested.connect(self._step_slice)
        self.view.window_level_dragged.connect(self._adjust_window_level)
        viewer_layout.addWidget(self.view, 1)

        navigation = QHBoxLayout()
        navigation.addWidget(QLabel("Slice"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._slice_selected)
        navigation.addWidget(self.slice_slider, 1)
        self.slice_spin = QSpinBox()
        self.slice_spin.setRange(0, 0)
        self.slice_spin.valueChanged.connect(self.slice_slider.setValue)
        self.slice_slider.valueChanged.connect(self.slice_spin.setValue)
        navigation.addWidget(self.slice_spin)
        self.coordinate_label = QLabel("No volume loaded")
        navigation.addWidget(self.coordinate_label)
        viewer_layout.addLayout(navigation)
        splitter.addWidget(viewer_container)

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
        return self.slice_slider.value()

    @property
    def cursor_position(self) -> tuple[float, float, float] | None:
        return self._cursor

    def set_volume(self, volume: DicomVolume) -> None:
        self._volume = volume
        maximum = volume.hu.shape[0] - 1
        self.slice_slider.setRange(0, maximum)
        self.slice_spin.setRange(0, maximum)
        middle = maximum // 2
        self._cursor = (
            (volume.hu.shape[2] - 1) / 2.0,
            (volume.hu.shape[1] - 1) / 2.0,
            float(middle),
        )
        self.slice_slider.setValue(middle)
        self._preview = None
        self.render()
        self.view.fit_image()

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
            self.view.clear_image()
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
                display = before * (1.0 - alpha) + self._preview.approximate_after_hu * alpha
        image = self._to_qimage(display)
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
        self.view.set_image(image, cursor_position=cursor, markers=markers)
        if self._cursor is not None:
            self.coordinate_label.setText(
                f"C/R/S: {self._cursor[0]:.1f}, {self._cursor[1]:.1f}, "
                f"{self._cursor[2]:.1f}"
            )

    def _to_qimage(self, values: np.ndarray) -> QImage:
        lower = self._level - self._window / 2.0
        scaled = np.clip((values - lower) / max(self._window, 1.0), 0.0, 1.0)
        pixels = np.ascontiguousarray(np.rint(scaled * 255.0).astype(np.uint8))
        image = QImage(
            pixels.data,
            pixels.shape[1],
            pixels.shape[0],
            pixels.strides[0],
            QImage.Format.Format_Grayscale8,
        )
        return image.copy()

    @Slot()
    def _fit(self) -> None:
        self.view.fit_image()

    @Slot()
    def _reset_view(self) -> None:
        self.view.reset_view()
        self._apply_preset(self.preset_combo.currentText())

    @Slot(str)
    def _apply_preset(self, name: str) -> None:
        self._window, self._level = WINDOW_PRESETS.get(name, (400.0, 40.0))
        self.render()

    @Slot(float, float)
    def _adjust_window_level(self, window_delta: float, level_delta: float) -> None:
        self._window = max(1.0, self._window + window_delta * 2.0)
        self._level += level_delta * 2.0
        self.render()

    @Slot(int)
    def _step_slice(self, delta: int) -> None:
        self.slice_slider.setValue(
            max(
                self.slice_slider.minimum(),
                min(self.slice_slider.maximum(), self.current_slice + delta),
            )
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
