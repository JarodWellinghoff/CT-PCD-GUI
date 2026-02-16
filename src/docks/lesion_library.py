from pathlib import Path
from typing import Optional
from PyQt6.QtWidgets import (
    QFileDialog,
    QLabel,
    QDockWidget,
    QWidget,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt
from src.widgets import LesionViewerWidget


# ===========================================================================
# Lesion Library dock
# ===========================================================================


class LesionLibraryDock(QDockWidget):
    """Dock for browsing .mat files and controlling 3-D spacing."""

    def __init__(self, parent=None):
        super().__init__("Lesion Library", parent)
        self.setMinimumWidth(250)

        container = QWidget()
        lay = QVBoxLayout(container)

        # ---- Directory selector ----
        dir_grp = QGroupBox("Library Directory")
        dir_lay = QHBoxLayout(dir_grp)
        self.dir_field = QLineEdit()
        self.dir_field.setReadOnly(True)
        self.dir_field.setPlaceholderText("(not set)")
        dir_lay.addWidget(self.dir_field)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_dir)
        dir_lay.addWidget(btn_browse)
        lay.addWidget(dir_grp)

        # ---- File list ----
        files_grp = QGroupBox("Models (.mat)")
        files_lay = QVBoxLayout(files_grp)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(120)
        self.file_list.itemDoubleClicked.connect(self._on_file_double_click)
        files_lay.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Selected")
        self.btn_load.clicked.connect(self._load_selected)
        btn_row.addWidget(self.btn_load)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self._refresh_list)
        btn_row.addWidget(self.btn_refresh)
        files_lay.addLayout(btn_row)
        lay.addWidget(files_grp)

        # ---- Spacing controls ----
        spacing_grp = QGroupBox("3-D Spacing")
        spacing_lay = QFormLayout(spacing_grp)

        self.use_dicom_cb = QCheckBox("Use DICOM spacing (if loaded)")
        self.use_dicom_cb.setChecked(True)
        self.use_dicom_cb.toggled.connect(self._on_spacing_toggle)
        spacing_lay.addRow(self.use_dicom_cb)

        self.spin_x = QDoubleSpinBox()
        self.spin_y = QDoubleSpinBox()
        self.spin_z = QDoubleSpinBox()
        for s in (self.spin_x, self.spin_y, self.spin_z):
            s.setRange(0.01, 50.0)
            s.setDecimals(4)
            s.setSingleStep(0.1)
            s.setValue(1.0)
            s.valueChanged.connect(self._on_custom_spacing_changed)
        spacing_lay.addRow("X:", self.spin_x)
        spacing_lay.addRow("Y:", self.spin_y)
        spacing_lay.addRow("Z:", self.spin_z)

        self.dicom_spacing_label = QLabel("DICOM spacing: (not loaded)")
        self.dicom_spacing_label.setStyleSheet("color: #888; font-size: 11px;")
        spacing_lay.addRow(self.dicom_spacing_label)

        lay.addWidget(spacing_grp)
        lay.addStretch()
        self.setWidget(container)

        # State
        self._library_dir: Optional[str] = None
        self._lesion_viewer: Optional["LesionViewerWidget"] = None
        self._dicom_spacing: Optional[list] = None
        self._on_spacing_toggle(self.use_dicom_cb.isChecked())

    def bind_lesion_viewer(self, viewer: LesionViewerWidget):
        self._lesion_viewer = viewer

    def set_library_dir(self, path: str):
        self._library_dir = path
        self.dir_field.setText(path)
        self._refresh_list()

    def set_dicom_spacing(self, spacing: list):
        self._dicom_spacing = spacing
        self.dicom_spacing_label.setText(
            f"DICOM spacing: ({spacing[0]:.4f}, {spacing[1]:.4f}, {spacing[2]:.4f})"
        )
        if self.use_dicom_cb.isChecked():
            self._apply_spacing(spacing)

    def get_custom_spacing(self) -> list:
        return [self.spin_x.value(), self.spin_y.value(), self.spin_z.value()]

    def set_custom_spacing(self, spacing: list):
        self.spin_x.setValue(spacing[0])
        self.spin_y.setValue(spacing[1])
        self.spin_z.setValue(spacing[2])

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Lesion Library Folder")
        if d:
            self.set_library_dir(d)

    def _refresh_list(self):
        self.file_list.clear()
        if not self._library_dir:
            return
        lib = Path(self._library_dir)
        if not lib.is_dir():
            return
        mat_files = sorted(lib.glob("*.mat"))
        for f in mat_files:
            item = QListWidgetItem(f.name)
            item.setData(Qt.ItemDataRole.UserRole, str(f))
            self.file_list.addItem(item)

    def _load_selected(self):
        item = self.file_list.currentItem()
        if item and self._lesion_viewer:
            self._lesion_viewer.load_mat(item.data(Qt.ItemDataRole.UserRole))

    def _on_file_double_click(self, item):
        if self._lesion_viewer:
            self._lesion_viewer.load_mat(item.data(Qt.ItemDataRole.UserRole))

    def _on_spacing_toggle(self, use_dicom: bool):
        custom_enabled = not use_dicom
        self.spin_x.setEnabled(custom_enabled)
        self.spin_y.setEnabled(custom_enabled)
        self.spin_z.setEnabled(custom_enabled)
        if use_dicom and self._dicom_spacing:
            self._apply_spacing(self._dicom_spacing)
        elif not use_dicom:
            self._apply_spacing(self.get_custom_spacing())

    def _on_custom_spacing_changed(self):
        if not self.use_dicom_cb.isChecked():
            self._apply_spacing(self.get_custom_spacing())

    def _apply_spacing(self, spacing: list):
        if self._lesion_viewer:
            self._lesion_viewer.spacing = spacing
