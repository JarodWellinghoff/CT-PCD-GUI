from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QTransform
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from ct_pcd_gui.shared.qt.dicom_viewer import DicomImageView, DicomSliceViewer


class ReconstructionPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll.setWidget(content)
        root.addWidget(scroll)
        self.inputs = QGroupBox("1. Prepare one CTPD acquisition")
        paths = QFormLayout(self.inputs)
        self.executable = QLineEdit()
        self.executable.setPlaceholderText("fct-wfbp or your trusted fct-wfbp_ZY_*.exe")
        self.source = QLineEdit()
        self.source.setPlaceholderText("Projection folder or one multi-frame DICOM file")
        self.output = QLineEdit()
        self.output.setPlaceholderText("Working/output folder outside the input tree")
        paths.addRow("FreeCT executable", self._path_row(self.executable, "executable"))
        paths.addRow("CTPD input", self._path_row(self.source, "input"))
        paths.addRow("Output folder", self._path_row(self.output, "directory"))
        self.spectrum = QSpinBox()
        self.source_index = QSpinBox()
        for control in (self.spectrum, self.source_index):
            control.setRange(-1, 65535)
            control.setValue(-1)
            control.setSpecialValueText("Auto (one only)")
        paths.addRow("Spectrum index", self.spectrum)
        paths.addRow("Source index", self.source_index)
        note = QLabel(
            "Select one series/threshold. Subfolders are included. Preparation copies "
            "all selected projections into single-frame files; allow enough disk space."
        )
        note.setWordWrap(True)
        paths.addRow(note)
        layout.addWidget(self.inputs)
        self.prepare_button = QPushButton("Prepare data and configuration")
        layout.addWidget(self.prepare_button)

        self.settings = QGroupBox("2. Reconstruction settings")
        form = QFormLayout(self.settings)
        self.volume_controls: dict[str, QDoubleSpinBox | QSpinBox] = {}
        for key, label, low, high, default in (
            ("StartPosition", "Start Z (mm)", -100000, 100000, 0),
            ("EndPosition", "End Z (mm)", -100000, 100000, 0),
            ("FieldOfView", "Field of view (mm)", 0.001, 10000, 300),
            ("SliceThickness", "Slice thickness (mm)", 0.001, 1000, 1),
            ("SlicePitch", "Slice spacing (mm)", 0.001, 1000, 1),
            ("Nx", "Image columns", 16, 4096, 512),
            ("Ny", "Image rows", 16, 4096, 512),
        ):
            if key in {"Nx", "Ny"}:
                control = QSpinBox()
                control.setRange(int(low), int(high))
                control.setValue(int(default))
            else:
                control = QDoubleSpinBox()
                control.setDecimals(6)
                control.setRange(float(low), float(high))
                control.setValue(float(default))
            self.volume_controls[key] = control
            form.addRow(label, control)
        self.kernel_a = QComboBox()
        for label, value in (("Hann (0.5)", 0.5), ("Hamming (0.54)", 0.54), ("Ramp (1.0)", 1.0)):
            self.kernel_a.addItem(label, value)
        self.kernel_c = QDoubleSpinBox()
        self.kernel_c.setDecimals(6)
        self.kernel_c.setRange(0.000001, 1.0)
        self.kernel_c.setValue(0.5)
        form.addRow("Kernel A", self.kernel_a)
        form.addRow("Kernel C / cutoff", self.kernel_c)
        self.rescale = QCheckBox("Rescale to HU (requires calibration)")
        self.native_dicom = QCheckBox("Also write native DICOM (experimental)")
        form.addRow(self.rescale)
        form.addRow(self.native_dicom)
        layout.addWidget(self.settings)
        self.run_button = QPushButton("Reconstruct images")
        self.cancel_button = QPushButton("Cancel")
        self.open_button = QPushButton("Open job/output folder")
        layout.addWidget(self.run_button)
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.open_button)
        warning = QLabel(
            "RESEARCH ONLY - not for clinical use. Validate scanner geometry, HU "
            "calibration and orientation against a known reconstruction. Telemetry is disabled."
        )
        warning.setWordWrap(True)
        layout.addWidget(warning)
        layout.addStretch()
        self.set_state(False, False, False)

    def _path_row(self, edit: QLineEdit, kind: str) -> QWidget:
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        if kind in {"input", "directory"}:
            button = QPushButton("Folder")
            button.clicked.connect(lambda: self._choose(edit, True))
            row.addWidget(button)
        if kind != "directory":
            button = QPushButton("File")
            button.clicked.connect(lambda: self._choose(edit, False))
            row.addWidget(button)
        return widget

    def _choose(self, edit: QLineEdit, directory: bool) -> None:
        if directory:
            path = QFileDialog.getExistingDirectory(self, "Choose folder", edit.text())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", edit.text())
        if path:
            edit.setText(path)

    def set_state(self, busy: bool, prepared: bool, has_directory: bool) -> None:
        self.inputs.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy)
        self.settings.setEnabled(prepared and not busy)
        self.run_button.setEnabled(prepared and not busy)
        self.cancel_button.setEnabled(busy)
        self.open_button.setEnabled(has_directory)

    def load_settings(self, config: dict) -> None:
        for key, control in self.volume_controls.items():
            value = config["ReconVolume"].get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Generated {key} is invalid; check the CTPD scan geometry.")
            if not control.minimum() <= value <= control.maximum():
                raise ValueError(f"Generated {key}={value} is outside the GUI's supported range.")
            control.setValue(int(value) if isinstance(control, QSpinBox) else float(value))
        index = self.kernel_a.findData(config["ReconSpecific"].get("KernelA"))
        if index < 0:
            raise ValueError("Generated KernelA is not supported.")
        self.kernel_a.setCurrentIndex(index)
        cutoff = config["ReconSpecific"].get("KernelC")
        if (isinstance(cutoff, bool) or not isinstance(cutoff, (int, float))
                or not math.isfinite(cutoff) or not self.kernel_c.minimum() <= cutoff <= 1):
            raise ValueError("Generated KernelC is outside the supported range.")
        self.kernel_c.setValue(float(cutoff))
        self.rescale.setChecked(bool(config["ImageOutput"]["RescaleToHU"]))

    def apply_settings(self, config: dict) -> dict:
        for key, control in self.volume_controls.items():
            config["ReconVolume"][key] = control.value()
        config["ReconSpecific"] = {
            "KernelA": float(self.kernel_a.currentData()), "KernelC": self.kernel_c.value(),
        }
        config["ImageOutput"]["RescaleToHU"] = self.rescale.isChecked()
        return config


class ReconstructionImageView(DicomImageView):
    """Preserve physical pixel aspect when Nx and Ny differ for a square FOV."""

    def __init__(self) -> None:
        super().__init__()
        self.pixel_aspect = 1.0

    def set_qimage(self, image: QImage) -> None:
        super().set_qimage(image)
        self.pixmap_item.setTransform(QTransform.fromScale(self.pixel_aspect, 1.0))
        self.scene().setSceneRect(self.pixmap_item.sceneBoundingRect())
        if self._auto_fit:
            self.fit_image()

    def fit_image(self) -> None:
        if self.has_image:
            self.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._auto_fit = True


class ReconstructionWorkspace(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        self.status = QLabel("Choose FreeCT, CTPD input and an output folder, then prepare the data.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.image_view = ReconstructionImageView()
        self.viewer = DicomSliceViewer(image_view=self.image_view)
        self.tabs.addTab(self.viewer, "Reconstructed images")
        config_tab = QWidget()
        config_layout = QVBoxLayout(config_tab)
        explanation = QLabel(
            "Review scan geometry and preprocessing here. The panel's volume, kernel and HU "
            "controls override those YAML values when you run. Input/output paths are managed "
            "by the job. No patient-space orientation is inferred."
        )
        explanation.setWordWrap(True)
        config_layout.addWidget(explanation)
        self.config = QPlainTextEdit()
        self.config.setPlaceholderText("FreeCT-generated YAML will appear after preparation.")
        config_layout.addWidget(self.config)
        self.tabs.addTab(config_tab, "Advanced FreeCT configuration")
        splitter.addWidget(self.tabs)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2500)
        self.log.setMinimumHeight(100)
        splitter.addWidget(self.log)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)
