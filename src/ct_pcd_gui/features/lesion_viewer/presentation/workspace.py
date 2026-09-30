from __future__ import annotations

from pathlib import Path
import vtkmodules.all as vtk
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar,
    QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)
from .view_state import AppearanceState
from .vtk_scene import LesionVtkScene


class LesionViewerWorkspace(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.scene = LesionVtkScene(self)
        self.stack = QStackedWidget(self)
        self.placeholder = QLabel(
            "No lesion loaded.\n\nSelect a MAT or NPZ file from the module panel."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stack.addWidget(self.placeholder); self.stack.addWidget(self.scene)
        self.info = QPlainTextEdit(); self.info.setReadOnly(True)
        self.info.setMaximumHeight(210)
        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(self.stack); split.addWidget(self.info)
        split.setStretchFactor(0, 1); split.setSizes([700, 180])
        status = QFrame(); status_row = QHBoxLayout(status)
        status_row.setContentsMargins(2, 2, 2, 2)
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar(); self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False); self.progress.hide()
        status_row.addWidget(self.status_label, 1); status_row.addWidget(self.progress)
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(split, 1); layout.addWidget(status)

    @property
    def surface(self) -> vtk.vtkPolyData | None: return self.scene.surface
    def set_surface(self, polydata: vtk.vtkPolyData, *, reset_camera: bool) -> None:
        self.scene.set_surface(polydata, reset_camera=reset_camera); self.stack.setCurrentWidget(self.scene)
    def clear_surface(self) -> None: self.scene.clear_surface(); self.stack.setCurrentWidget(self.placeholder)
    def apply_appearance(self, state: AppearanceState) -> None: self.scene.apply_appearance(state)
    def set_info(self, text: str) -> None: self.info.setPlainText(text)
    def set_status(self, text: str) -> None: self.status_label.setText(text)
    def set_busy(self, text: str) -> None:
        self.status_label.setText(text); self.progress.setRange(0, 0); self.progress.show()
    def set_ready(self, text: str = "Ready") -> None:
        self.status_label.setText(text); self.progress.hide(); self.progress.setRange(0, 100)
    def save_screenshot(self, path: str | Path) -> Path: return self.scene.save_screenshot(path)
    def cleanup(self) -> None: self.scene.cleanup()
