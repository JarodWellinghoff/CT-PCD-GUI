from __future__ import annotations

import json
from pathlib import Path
import vtkmodules.all as vtk
from PySide6.QtCore import Qt, QSettings, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)
from ..domain.inspection import SliceChannel
from ..domain.models import LesionFileData, VolumeCandidate
from .slice_view import LesionSliceViews
from .view_state import AppearanceState
from .vtk_scene import LesionVtkScene


class LesionViewerWorkspace(QWidget):
    view_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = QSettings()
        self._has_hu_channels = False
        self.scene = LesionVtkScene(self)
        self.stack = QStackedWidget(self)
        self.placeholder = QLabel("No 3-D surface.\n\nSelect a MAT or NPZ file from the module panel.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.placeholder)
        self.stack.addWidget(self.scene)
        self.slices = LesionSliceViews(self)
        self.views = QSplitter(Qt.Orientation.Horizontal)
        self.views.addWidget(self.slices)
        self.views.addWidget(self.stack)
        self.views.setStretchFactor(0, 1)
        self.views.setStretchFactor(1, 1)
        self.views.setSizes([600, 600])
        self.info = QPlainTextEdit()
        self.info.setReadOnly(True)
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        metadata_page = QWidget()
        metadata_layout = QVBoxLayout(metadata_page)
        search_row = QHBoxLayout()
        self.metadata_search = QLineEdit()
        self.metadata_search.setPlaceholderText("Find a field or value in the file metadata...")
        find_button = QPushButton("Find next")
        copy_button = QPushButton("Copy metadata")
        search_row.addWidget(self.metadata_search, 1)
        search_row.addWidget(find_button)
        search_row.addWidget(copy_button)
        metadata_layout.addLayout(search_row)
        metadata_layout.addWidget(self.metadata)
        find_button.clicked.connect(self._find_metadata)
        self.metadata_search.returnPressed.connect(self._find_metadata)
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(self.metadata.toPlainText()))
        self.details = QTabWidget()
        self.details.addTab(self.info, "Summary / surface metrics")
        self.details.addTab(metadata_page, "File metadata / DICOM headers")
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.views)
        split.addWidget(self.details)
        split.setStretchFactor(0, 1)
        split.setSizes([700, 210])
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Views:"))
        self.mode_combo = QComboBox()
        for text, value in (("2-D and 3-D", "both"), ("2-D only", "2d"), ("3-D only", "3d")):
            self.mode_combo.addItem(text, value)
        controls.addWidget(self.mode_combo)
        controls.addStretch(1)
        details_check = QCheckBox("Show file information")
        details_check.setChecked(True)
        details_check.toggled.connect(self.details.setVisible)
        controls.addWidget(details_check)
        status = QFrame()
        status_row = QHBoxLayout(status)
        status_row.setContentsMargins(2, 2, 2, 2)
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.progress.hide()
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.progress)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addLayout(controls)
        layout.addWidget(split, 1)
        layout.addWidget(status)
        mode = self.settings.value("lesion_viewer/view_mode", "both", type=str)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(mode)))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self._mode_changed()

    @property
    def show_3d(self) -> bool:
        return self.mode_combo.currentData() != "2d"

    @property
    def surface(self) -> vtk.vtkPolyData | None:
        return self.scene.surface

    def _mode_changed(self, *_args) -> None:
        mode = str(self.mode_combo.currentData())
        self.slices.set_side_by_side(mode == "2d")
        self.slices.setVisible(mode != "3d")
        self.stack.setVisible(mode != "2d")
        self.settings.setValue("lesion_viewer/view_mode", mode)
        self.view_mode_changed.emit(mode)

    def _find_metadata(self) -> None:
        text = self.metadata_search.text()
        if text and not self.metadata.find(text):
            self.metadata.moveCursor(QTextCursor.MoveOperation.Start)
            self.metadata.find(text)

    def set_file_data(self, data: LesionFileData) -> None:
        channels = data.file_metadata.get("slice_channels", [])
        self._has_hu_channels = bool(channels)
        self.slices.set_channels(channels)
        metadata = {key: value for key, value in data.file_metadata.items() if key != "slice_channels"}
        metadata["file"] = str(data.path)
        metadata["warnings"] = list(data.warnings)
        self.metadata.setPlainText(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))

    def set_candidate(self, candidate: VolumeCandidate) -> None:
        if not self._has_hu_channels:
            self.slices.set_channels([SliceChannel(candidate.key, candidate.array, units="Stored value")])

    def clear_file(self) -> None:
        self.clear_surface()
        self.slices.clear()
        self._has_hu_channels = False
        self.info.clear()
        self.metadata.clear()

    def set_surface(self, polydata: vtk.vtkPolyData, *, reset_camera: bool) -> None:
        self.scene.set_surface(polydata, reset_camera=reset_camera)
        self.stack.setCurrentWidget(self.scene)

    def clear_surface(self) -> None:
        self.scene.clear_surface()
        self.stack.setCurrentWidget(self.placeholder)

    def apply_appearance(self, state: AppearanceState) -> None:
        self.scene.apply_appearance(state)

    def set_info(self, text: str) -> None:
        self.info.setPlainText(text)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_busy(self, text: str) -> None:
        self.status_label.setText(text)
        self.progress.setRange(0, 0)
        self.progress.show()

    def set_ready(self, text: str = "Ready") -> None:
        self.status_label.setText(text)
        self.progress.hide()
        self.progress.setRange(0, 100)

    def save_screenshot(self, path: str | Path) -> Path:
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.views.grab().save(str(output), "PNG"):
            raise OSError(f"Could not save screenshot: {output}")
        return output

    def cleanup(self) -> None:
        self.slices.clear()
        self.scene.cleanup()
