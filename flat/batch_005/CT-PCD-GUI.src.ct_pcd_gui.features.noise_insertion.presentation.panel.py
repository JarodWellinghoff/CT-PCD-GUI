"""PySide6 widgets and worker for the DICOM-CT-PD noise insertion module."""

from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


from ct_pcd_gui.shared.qt.collapsible_section import CollapsibleSection
from ct_pcd_gui.shared.qt.path_selector import PathSelector
from .view_state import NoiseJobDraft, NoiseViewState, NoiseJobPhase


class NoiseInsertionPanel(QWidget):
    run_requested = Signal(object)
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_was_edited = False

        title = QLabel("DICOM-CT-PD Noise Insertion", self)
        title_font = QFont()
        title_font.setPointSize(13)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setContentsMargins(6, 7, 6, 2)

        subtitle = QLabel(
            "Batch-process single-frame or multi-frame projection data without "
            "loading the full study into an interactive viewer.",
            self,
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #bdbdbd;")
        subtitle.setContentsMargins(6, 0, 6, 5)

        inner = QWidget(self)
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 0, 4, 4)
        inner_layout.setSpacing(5)

        input_section = CollapsibleSection("Input", expanded=True)
        self.input_selector = PathSelector(
            "Folder containing DICOMs, or one multi-frame DICOM",
            allow_file=True,
        )
        input_section.add_widget(self.input_selector)

        input_form = QFormLayout()
        input_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.input_mode = QComboBox(self)
        self.input_mode.addItem("Auto-detect each file", "auto")
        self.input_mode.addItem("Single-frame DICOM files", "single")
        self.input_mode.addItem("Multi-frame DICOM files", "multi")
        input_form.addRow("DICOM organization:", self.input_mode)

        self.recursive = QCheckBox("Include DICOMs in subfolders", self)
        input_form.addRow("", self.recursive)
        input_section.add_layout(input_form)
        inner_layout.addWidget(input_section)

        output_section = CollapsibleSection("Output", expanded=True)
        self.output_selector = PathSelector(
            "Output folder",
            allow_file=False,
        )
        output_section.add_widget(self.output_selector)

        output_form = QFormLayout()
        output_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.file_suffix = QLineEdit("_noise", self)
        self.file_suffix.setToolTip(
            "Appended before .dcm/.ima. Recursive jobs preserve the relative folder tree."
        )
        output_form.addRow("Filename suffix:", self.file_suffix)

        self.overwrite_existing = QCheckBox("Overwrite existing output files", self)
        output_form.addRow("", self.overwrite_existing)
        output_section.add_layout(output_form)
        inner_layout.addWidget(output_section)

        settings_section = CollapsibleSection("Noise Settings", expanded=True)
        settings_form = QFormLayout()
        settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        factor_row = QWidget(self)
        factor_layout = QHBoxLayout(factor_row)
        factor_layout.setContentsMargins(0, 0, 0, 0)
        factor_layout.setSpacing(6)
        self.mas_factor = QDoubleSpinBox(self)
        self.mas_factor.setRange(0.0001, 1.0)
        self.mas_factor.setDecimals(4)
        self.mas_factor.setSingleStep(0.05)
        self.mas_factor.setValue(0.25)
        self.mas_factor.setKeyboardTracking(False)
        self.dose_label = QLabel("25% of input dose", self)
        self.dose_label.setStyleSheet("color: #bdbdbd;")
        factor_layout.addWidget(self.mas_factor)
        factor_layout.addWidget(self.dose_label, 1)
        settings_form.addRow("mAs factor:", factor_row)

        self.electronic_noise = QDoubleSpinBox(self)
        self.electronic_noise.setRange(0.0, 1.0e12)
        self.electronic_noise.setDecimals(6)
        self.electronic_noise.setValue(0.0)
        settings_form.addRow("Electronic noise (Ne):", self.electronic_noise)

        seed_row = QWidget(self)
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.setSpacing(6)
        self.use_seed = QCheckBox("Use fixed seed", self)
        self.use_seed.setChecked(True)
        self.seed = QSpinBox(self)
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setValue(42)
        seed_layout.addWidget(self.use_seed)
        seed_layout.addWidget(self.seed, 1)
        settings_form.addRow("Randomization:", seed_row)

        self.update_tube_current = QCheckBox(
            "Scale XrayTubeCurrent by the mAs factor when present",
            self,
        )
        self.update_tube_current.setChecked(True)
        settings_form.addRow("", self.update_tube_current)
        settings_section.add_layout(settings_form)
        inner_layout.addWidget(settings_section)

        processing_section = CollapsibleSection("Processing and Preview", expanded=True)
        processing_form = QFormLayout()
        processing_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self.parallel_mode = QComboBox(self)
        self.parallel_mode.addItem(
            "Auto (process files / thread frames)",
            "auto",
        )
        self.parallel_mode.addItem(
            "Multiprocessing across independent files",
            "processes",
        )
        self.parallel_mode.addItem("Multithreading", "threads")
        self.parallel_mode.addItem("Sequential (debugging)", "sequential")
        self.parallel_mode.setToolTip(
            "Auto uses spawned processes for folders of independent single-frame "
            "DICOMs and threads for frames inside a multi-frame DICOM."
        )
        processing_form.addRow("Parallel engine:", self.parallel_mode)

        self.max_workers = QSpinBox(self)
        cpu_count = os.cpu_count() or 1
        self.max_workers.setRange(0, max(4, min(64, cpu_count * 2)))
        self.max_workers.setSpecialValueText("Auto")
        self.max_workers.setValue(0)
        self.max_workers.setToolTip(
            f"0 lets the backend choose a conservative worker count. "
            f"This system reports {cpu_count} logical CPU(s). Higher values can "
            "increase memory use, especially for large projection frames."
        )
        processing_form.addRow("Parallel workers:", self.max_workers)

        self.preview_interval = QSpinBox(self)
        self.preview_interval.setRange(0, 1_000_000)
        self.preview_interval.setSpecialValueText("Off")
        self.preview_interval.setValue(100)
        self.preview_interval.setToolTip(
            "Generate a live before/after preview on every Nth ordered projection. "
            "Each single-frame DICOM counts as one iteration; each frame in a "
            "multi-frame DICOM counts as one iteration. Set 0 to disable previews."
        )
        processing_form.addRow("Preview every N iterations:", self.preview_interval)

        self.continue_on_error = QCheckBox(
            "Continue when an individual file fails", self
        )
        self.continue_on_error.setChecked(True)
        processing_form.addRow("", self.continue_on_error)
        processing_section.add_layout(processing_form)
        inner_layout.addWidget(processing_section)
        inner_layout.addStretch(1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(5, 3, 5, 5)
        self.start_button = QPushButton("Run Noise Insertion", self)
        self.start_button.setDefault(True)
        self.start_button.setMinimumHeight(32)
        self.start_button.setStyleSheet(
            "QPushButton { background: #4d6a8c; border: 1px solid #6f8fb0; "
            "border-radius: 3px; padding: 6px; font-weight: bold; }"
            "QPushButton:hover { background: #58789f; }"
            "QPushButton:disabled { background: #444; color: #888; border-color: #555; }"
        )
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumHeight(32)
        button_row.addWidget(self.start_button, 1)
        button_row.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(scroll, 1)
        layout.addLayout(button_row)

        self.mas_factor.valueChanged.connect(self._update_dose_label)
        self.use_seed.toggled.connect(self.seed.setEnabled)
        self.parallel_mode.currentIndexChanged.connect(
            lambda _index: self._update_parallel_controls()
        )
        self.input_selector.pathChanged.connect(self._suggest_output_folder)
        self.output_selector.userPathChanged.connect(self._mark_output_edited)
        self.start_button.clicked.connect(self._emit_run_requested)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self._update_parallel_controls()

    @Slot(str)
    def _mark_output_edited(self, _text: str) -> None:
        self._output_was_edited = True

    @Slot(str)
    def _suggest_output_folder(self, source_text: str) -> None:
        if self._output_was_edited or not source_text.strip():
            return
        source = Path(source_text).expanduser()
        if source.is_file() or source.suffix.lower() in {".dcm", ".ima"}:
            suggestion = source.parent / f"{source.stem}_noise_output"
        else:
            suggestion = source.parent / f"{source.name}_noise_output"
        self.output_selector.set_text(str(suggestion))

    @Slot(float)
    def _update_dose_label(self, factor: float) -> None:
        self.dose_label.setText(f"{factor * 100.0:.4g}% of input dose")

    @Slot()
    def _update_parallel_controls(self) -> None:
        self.max_workers.setEnabled(
            self.parallel_mode.currentData() != "sequential"
            and self.start_button.isEnabled()
        )

    def read_draft(self) -> NoiseJobDraft:
        input_path = self.input_selector.text()
        output_dir = self.output_selector.text()
        if not input_path:
            raise ValueError("Select an input DICOM folder or file.")
        if not output_dir:
            raise ValueError("Select an output folder.")

        return NoiseJobDraft(
            input_path=input_path,
            output_dir=output_dir,
            input_mode=str(self.input_mode.currentData()),
            mas_factor=float(self.mas_factor.value()),
            electronic_noise=float(self.electronic_noise.value()),
            seed=int(self.seed.value()) if self.use_seed.isChecked() else None,
            file_suffix=self.file_suffix.text(),
            update_tube_current=self.update_tube_current.isChecked(),
            overwrite_existing=self.overwrite_existing.isChecked(),
            recursive=self.recursive.isChecked(),
            continue_on_error=self.continue_on_error.isChecked(),
            preview_interval=int(self.preview_interval.value()),
            parallel_mode=str(self.parallel_mode.currentData()),
            max_workers=int(self.max_workers.value()),
        )

    @Slot()
    def _emit_run_requested(self) -> None:
        self.run_requested.emit(self.read_draft())

    def render_state(self, state: NoiseViewState) -> None:
        running = state.phase == NoiseJobPhase.RUNNING
        self.start_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        for widget in (
            self.input_selector,
            self.output_selector,
            self.input_mode,
            self.recursive,
            self.file_suffix,
            self.overwrite_existing,
            self.mas_factor,
            self.electronic_noise,
            self.use_seed,
            self.update_tube_current,
            self.parallel_mode,
            self.max_workers,
            self.preview_interval,
            self.continue_on_error,
        ):
            widget.setEnabled(not running)
        self.seed.setEnabled(not running and self.use_seed.isChecked())
        self.max_workers.setEnabled(
            not running and self.parallel_mode.currentData() != "sequential"
        )
