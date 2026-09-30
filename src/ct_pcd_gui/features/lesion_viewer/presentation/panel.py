from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget
from .file_model import LesionFileFilterProxy, LesionFileTableModel
from .panel_behavior import PanelBehaviorMixin
from .panel_ui import PanelUiMixin


class LesionViewerPanel(PanelUiMixin, PanelBehaviorMixin, QWidget):
    source_requested = Signal(str)
    refresh_requested = Signal()
    file_requested = Signal(object)
    candidate_changed = Signal(int)
    surface_settings_changed = Signal()
    appearance_changed = Signal(object)
    reset_camera_requested = Signal()
    axis_view_requested = Signal(str)
    screenshot_requested = Signal(str)
    export_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._configuring = False
        self._current_file: Path | None = None
        self._color = QColor(230, 128, 76)
        self._threshold_bounds = (0.0, 1.0)
        self.file_model = LesionFileTableModel(self)
        self.file_proxy = LesionFileFilterProxy(self)
        self.file_proxy.setSourceModel(self.file_model)
        self._build_ui()
        self._connect()
        self._style_color_button()
        self.set_controls_enabled(False)
