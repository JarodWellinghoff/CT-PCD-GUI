from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ct_pcd_gui.shared.qt.scaled_image_label import ScaledImageLabel

from ..application.models import JobProgress, JobStarted, JobSummary, PreviewPayload


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

    def show_inspecting(self, preview_interval: int) -> None:
        self.reset_for_job(preview_interval)
        self.append_log("Inspecting input files and preparing the processing plan.")

    def show_cancelling(self) -> None:
        self.progress_status.setText("Cancelling...")
        if self.progress_bar.maximum() > 0:
            self.progress_bar.setFormat("Cancelling at %v / %m work units")
        self.append_log(
            "Cancellation requested. Active work will stop at the next safe boundary."
        )

    def show_started(self, event: JobStarted) -> None:
        self.configure_progress(
            event.total_units,
            event.file_count,
            event.frame_count,
        )

    def show_progress(self, event: JobProgress) -> None:
        self.set_progress(event.completed_units, event.total_units, event.status)

    def show_completed(self, summary: JobSummary) -> None:
        self.mark_finished(summary)
        self.append_log(
            f"Summary: {summary.written_files} output file(s), "
            f"{summary.failed_files} failed, {summary.skipped_files} skipped, "
            f"{summary.clipped_pixels:,} clipped pixel(s), maximum "
            f"{summary.max_workers} parallel worker(s)."
        )

    def show_cancelled(self, message: str) -> None:
        self.mark_cancelled()
        self.append_log(message or "Noise insertion was cancelled.")

    def show_failed(self, message: str) -> None:
        self.mark_failed()
        self.append_log(f"Job stopped: {message}")

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
        maximum = max(1, total)
        if self.progress_bar.maximum() != maximum:
            self.progress_bar.setRange(0, maximum)
        self.progress_bar.setValue(min(completed, maximum))
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
            self.preview_note.setText("Finished without generating a scheduled preview.")
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
