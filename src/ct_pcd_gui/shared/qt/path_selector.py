from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QWidget,
)


class PathSelector(QWidget):
    pathChanged = Signal(str)
    userPathChanged = Signal(str)

    def __init__(
        self,
        placeholder: str,
        *,
        allow_file: bool,
        file_caption: str = "Select file",
        file_tooltip: str = "",
        folder_caption: str = "Select folder",
        folder_tooltip: str = "",
        file_filter: str = "All files (*)",
        allow_all_files: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.allow_file = allow_file
        self.file_caption = file_caption
        self.file_tooltip = file_tooltip
        self.folder_caption = folder_caption
        self.folder_tooltip = folder_tooltip
        self.file_filter = file_filter
        self.allow_all_files = allow_all_files

        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.textChanged.connect(self.pathChanged)
        self.line_edit.textEdited.connect(self.userPathChanged)

        self.folder_button = QPushButton("Folder...", self)
        self.folder_button.setToolTip(self.folder_tooltip)
        self.folder_button.clicked.connect(self._choose_folder)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.folder_button)

        self.file_button: QPushButton | None = None
        if allow_file:
            self.file_button = QPushButton("File...", self)
            self.file_button.setToolTip(self.file_tooltip or self.file_caption)
            self.file_button.clicked.connect(self._choose_file)
            layout.addWidget(self.file_button)

    def text(self) -> str:
        return self.line_edit.text().strip()

    def set_text(self, value: str) -> None:
        self.line_edit.setText(value)

    @Slot()
    def _choose_folder(self) -> None:
        start = self.text() or str(Path.home())
        if Path(start).is_file():
            start = str(Path(start).parent)
        chosen = QFileDialog.getExistingDirectory(
            self,
            self.folder_caption,
            start,
        )
        if chosen:
            self.set_text(chosen)
            self.userPathChanged.emit(chosen)

    @Slot()
    def _choose_file(self) -> None:
        start = self.text() or str(Path.home())
        if Path(start).is_dir():
            start = str(Path(start))
        selected_filter = self.file_filter
        if self.allow_all_files and "All files (*)" not in selected_filter:
            selected_filter = f"{selected_filter};;All files (*)"
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            self.file_caption,
            start,
            selected_filter,
        )
        if chosen:
            self.set_text(chosen)
            self.userPathChanged.emit(chosen)
