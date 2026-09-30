"""PySide6 widgets and worker for the DICOM-CT-PD noise insertion module."""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

try:
    from .ctpd_noise_backend import (
        JobSummary,
        NoiseInsertionCancelled,
        NoiseJobConfig,
        PreviewPayload,
        run_noise_job,
    )
except ImportError:  # Allows ``python main.py`` from this directory.
    from archive.legacy.original_code.ctpd_noise_backend import (
        JobSummary,
        NoiseInsertionCancelled,
        NoiseJobConfig,
        PreviewPayload,
        run_noise_job,
    )

from ct_pcd_gui.shared.qt.collapsible_section import CollapsibleSection
from ct_pcd_gui.shared.qt.path_selector import PathSelector
from ct_pcd_gui.shared.qt.scaled_image_label import ScaledImageLabel


class NoiseInsertionPanel(QWidget):
    startRequested = Signal(object)
    cancelRequested = Signal()

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
        self.start_button.clicked.connect(self._emit_start)
        self.cancel_button.clicked.connect(self.cancelRequested.emit)
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

    def build_config(self) -> NoiseJobConfig:
        input_path = self.input_selector.text()
        output_dir = self.output_selector.text()
        if not input_path:
            raise ValueError("Select an input DICOM folder or file.")
        if not output_dir:
            raise ValueError("Select an output folder.")

        return NoiseJobConfig(
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
    def _emit_start(self) -> None:
        try:
            config = self.build_config()
        except ValueError as exc:
            QMessageBox.warning(self, "Noise Insertion", str(exc))
            return
        self.startRequested.emit(config)

    def set_running(self, running: bool) -> None:
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


class PreviewDisplay(QFrame):
    """Persistent before/after display that is updated in place."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "PreviewDisplay, QFrame { background: #2d2d2d; border: 1px solid #505050; "
            "border-radius: 3px; }"
        )

        self.title_label = QLabel("No preview generated", self)
        self.title_label.setWordWrap(True)
        self.title_label.setStyleSheet("font-weight: bold; border: 0;")

        before_heading = QLabel("Before", self)
        after_heading = QLabel("After", self)
        for heading in (before_heading, after_heading):
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            heading.setStyleSheet("font-weight: bold; border: 0;")

        self.before_image = ScaledImageLabel(self)
        self.after_image = ScaledImageLabel(self)

        image_grid = QGridLayout()
        image_grid.setContentsMargins(0, 0, 0, 0)
        image_grid.setSpacing(5)
        image_grid.addWidget(before_heading, 0, 0)
        image_grid.addWidget(after_heading, 0, 1)
        image_grid.addWidget(self.before_image, 1, 0)
        image_grid.addWidget(self.after_image, 1, 1)
        image_grid.setColumnStretch(0, 1)
        image_grid.setColumnStretch(1, 1)

        self.metrics_label = QLabel("Waiting for a scheduled preview iteration.", self)
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("color: #bdbdbd; font-size: 10px; border: 0;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 7, 8, 7)
        layout.setSpacing(5)
        layout.addWidget(self.title_label)
        layout.addLayout(image_grid, 1)
        layout.addWidget(self.metrics_label)

        self.clear_preview()

    def clear_preview(
        self, message: str = "Waiting for a scheduled preview iteration."
    ) -> None:
        self.title_label.setText("No preview generated")
        self.before_image.clear_image("Waiting")
        self.after_image.clear_image("Waiting")
        self.metrics_label.setText(message)

    def set_payload(self, payload: PreviewPayload) -> None:
        self.title_label.setText(payload.title)
        self.before_image.set_array(payload.before_u8)
        self.after_image.set_array(payload.after_u8)
        self.metrics_label.setText(
            f"Display window: {payload.window_low:.6g} to {payload.window_high:.6g} | "
            f"Before mean/std: {payload.before_mean:.6g} / {payload.before_std:.6g} | "
            f"After mean/std: {payload.after_mean:.6g} / {payload.after_std:.6g}"
        )


class NoiseInsertionWorkspace(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        heading = QLabel("Live before/after preview", self)
        heading_font = QFont()
        heading_font.setPointSize(12)
        heading_font.setBold(True)
        heading.setFont(heading_font)

        self.preview_note = QLabel(
            "The same image pair is replaced at each configured preview interval.",
            self,
        )
        self.preview_note.setWordWrap(True)
        self.preview_note.setStyleSheet("color: #bdbdbd;")

        preview_header = QHBoxLayout()
        preview_header.addWidget(heading)
        preview_header.addStretch(1)
        preview_header.addWidget(self.preview_note)

        self.preview_display = PreviewDisplay(self)

        preview_panel = QWidget(self)
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(7, 7, 7, 4)
        preview_layout.addLayout(preview_header)
        preview_layout.addWidget(self.preview_display, 1)

        output_group = QGroupBox("Process output and status", self)
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(7, 7, 7, 7)
        output_layout.setSpacing(5)

        log_actions = QHBoxLayout()
        log_actions.addStretch(1)
        clear_button = QPushButton("Clear", self)
        copy_button = QPushButton("Copy", self)
        clear_button.clicked.connect(self.clear_log)
        copy_button.clicked.connect(self._copy_log)
        log_actions.addWidget(copy_button)
        log_actions.addWidget(clear_button)
        output_layout.addLayout(log_actions)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.document().setMaximumBlockCount(20_000)
        fixed_font = QFont("Consolas")
        fixed_font.setStyleHint(QFont.StyleHint.Monospace)
        self.log.setFont(fixed_font)
        self.log.setStyleSheet("background: #181818; border: 1px solid #505050;")
        output_layout.addWidget(self.log, 1)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(preview_panel)
        splitter.addWidget(output_group)
        splitter.setSizes([560, 250])

        self.progress_status = QLabel("Ready", self)
        self.progress_status.setMinimumWidth(230)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(8, 4, 8, 7)
        progress_row.setSpacing(8)
        progress_row.addWidget(self.progress_status)
        progress_row.addWidget(self.progress_bar, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(splitter, 1)
        layout.addLayout(progress_row)

        self._preview_interval = 0
        self._latest_preview_iteration = 0

    @Slot()
    def clear_log(self) -> None:
        self.log.clear()

    @Slot()
    def _copy_log(self) -> None:
        self.log.selectAll()
        self.log.copy()
        cursor = self.log.textCursor()
        cursor.clearSelection()
        self.log.setTextCursor(cursor)

    def reset_for_job(self, preview_interval: int) -> None:
        self.clear_log()
        self.clear_previews(preview_interval)
        self.progress_bar.setRange(0, 0)
        self.progress_status.setText("Inspecting input...")

    def clear_previews(self, preview_interval: int = 0) -> None:
        self._preview_interval = max(0, int(preview_interval))
        self._latest_preview_iteration = 0
        if self._preview_interval == 0:
            self.preview_note.setText("Live preview is disabled for this run.")
            self.preview_display.clear_preview("Preview generation is disabled.")
            return

        unit = "iteration" if self._preview_interval == 1 else "iterations"
        self.preview_note.setText(
            f"This display is replaced every {self._preview_interval:,} {unit}."
        )
        self.preview_display.clear_preview(
            f"Waiting for projection iteration {self._preview_interval:,}."
        )

    def append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        for line in str(message).splitlines() or [""]:
            self.log.appendPlainText(f"[{timestamp}] {line}")
        scroll_bar = self.log.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def configure_progress(
        self, total_units: int, file_count: int, frame_count: int
    ) -> None:
        self.progress_bar.setRange(0, max(1, total_units))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m work units (%p%)")
        self.progress_status.setText(
            f"Starting {file_count} file(s), {frame_count} frame(s)..."
        )

    def set_progress(self, completed: int, total: int, status: str) -> None:
        if self.progress_bar.maximum() != max(1, total):
            self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(min(completed, max(1, total)))
        self.progress_status.setText(status)

    @Slot(object)
    def update_preview(self, payload: PreviewPayload) -> None:
        # Parallel file/frame tasks can finish out of order. Do not let a stale
        # lower-numbered projection replace a newer preview already on screen.
        if int(payload.iteration) <= self._latest_preview_iteration:
            return

        self._latest_preview_iteration = int(payload.iteration)
        self.preview_display.set_payload(payload)

        if self._preview_interval > 0:
            next_iteration = payload.iteration + self._preview_interval
            if next_iteration <= payload.total_iterations:
                self.preview_note.setText(
                    f"Showing iteration {payload.iteration:,}/{payload.total_iterations:,}; "
                    f"next scheduled preview: {next_iteration:,}."
                )
            else:
                self.preview_note.setText(
                    f"Showing the final scheduled preview at iteration "
                    f"{payload.iteration:,}/{payload.total_iterations:,}."
                )

    def mark_finished(self, summary: JobSummary) -> None:
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.progress_status.setText(
            f"Finished: {summary.written_files} written, "
            f"{summary.failed_files} failed, {summary.skipped_files} skipped"
        )
        if self._preview_interval > 0 and self._latest_preview_iteration == 0:
            self.preview_note.setText(
                "Finished without generating a scheduled preview."
            )
            self.preview_display.clear_preview(
                "No scheduled preview target completed. The interval may exceed the "
                "projection count, or the target input may have been skipped or failed."
            )

    def mark_cancelled(self) -> None:
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
        self.progress_status.setText("Cancelled")
        self.progress_bar.setFormat("Cancelled at %v / %m work units")

    def mark_failed(self) -> None:
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
        self.progress_status.setText("Stopped by an error")
        self.progress_bar.setFormat("Stopped at %v / %m work units")


class NoiseInsertionWorker(QObject):
    jobStarted = Signal(int, int, int)
    progressChanged = Signal(int, int, str)
    logMessage = Signal(str)
    previewReady = Signal(object)
    completed = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)

    def __init__(self, config: NoiseJobConfig) -> None:
        super().__init__()
        self.config = config
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def run(self) -> None:
        try:
            summary = run_noise_job(
                self.config,
                cancel_event=self.cancel_event,
                log_callback=self.logMessage.emit,
                started_callback=self.jobStarted.emit,
                progress_callback=self.progressChanged.emit,
                preview_callback=self.previewReady.emit,
            )
        except NoiseInsertionCancelled as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.logMessage.emit(f"FATAL: {exc}")
            self.failed.emit(str(exc))
        else:
            self.completed.emit(summary)


class NoiseInsertionController(QObject):
    """Own the QThread lifecycle and connect the module panel to its workspace."""

    runningChanged = Signal(bool)

    def __init__(
        self,
        panel: NoiseInsertionPanel,
        workspace: NoiseInsertionWorkspace,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.panel = panel
        self.workspace = workspace
        self.thread: QThread | None = None
        self.worker: NoiseInsertionWorker | None = None

        panel.startRequested.connect(self.start)
        panel.cancelRequested.connect(self.cancel)

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    @Slot(object)
    def start(self, config: NoiseJobConfig) -> None:
        if self.is_running:
            return

        self.workspace.reset_for_job(config.preview_interval)
        self.panel.set_running(True)
        self.runningChanged.emit(True)

        self.thread = QThread(self)
        self.worker = NoiseInsertionWorker(config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.jobStarted.connect(self.workspace.configure_progress)
        self.worker.progressChanged.connect(self.workspace.set_progress)
        self.worker.logMessage.connect(self.workspace.append_log)
        self.worker.previewReady.connect(self.workspace.update_preview)
        self.worker.completed.connect(self._completed)
        self.worker.cancelled.connect(self._cancelled)
        self.worker.failed.connect(self._failed)

        self.worker.completed.connect(self.thread.quit)
        self.worker.cancelled.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @Slot()
    def cancel(self) -> None:
        if self.worker is None or not self.is_running:
            return
        self.workspace.append_log(
            "Cancellation requested. Active frame/file tasks will stop at the next "
            "safe boundary; a task already writing a DICOM will finish first."
        )
        self.workspace.progress_status.setText("Cancelling...")
        self.panel.cancel_button.setEnabled(False)
        self.worker.cancel()

    @Slot(object)
    def _completed(self, summary: JobSummary) -> None:
        self.workspace.mark_finished(summary)
        self.workspace.append_log(
            f"Summary: {summary.written_files} output file(s), "
            f"{summary.failed_files} failed, {summary.skipped_files} skipped, "
            f"{summary.clipped_pixels:,} clipped pixel(s), maximum "
            f"{summary.max_workers} parallel worker(s)."
        )

    @Slot(str)
    def _cancelled(self, message: str) -> None:
        self.workspace.mark_cancelled()
        self.workspace.append_log(message or "Noise insertion was cancelled.")

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.workspace.mark_failed()
        self.workspace.append_log(f"Job stopped: {message}")
        QMessageBox.critical(self.panel, "Noise Insertion", message)

    @Slot()
    def _thread_finished(self) -> None:
        self.panel.set_running(False)
        self.runningChanged.emit(False)
        self.worker = None
        self.thread = None  # pyright: ignore[reportIncompatibleMethodOverride]
