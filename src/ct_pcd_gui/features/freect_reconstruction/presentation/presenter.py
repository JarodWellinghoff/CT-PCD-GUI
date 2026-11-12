from __future__ import annotations

import shutil
import threading
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QUrl, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QMessageBox

from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner, TaskCancelled

from ..config import dump_config, load_config
from ..images import read_slice
from ..models import (
    JobContext, PreparedJob, ReconstructionCancelled, ReconstructionResult, Update,
)
from ..service import prepare, reconstruct
from .view import ReconstructionPanel, ReconstructionWorkspace


class ReconstructionPresenter(QObject):
    def __init__(self, panel: ReconstructionPanel, workspace: ReconstructionWorkspace,
                 task_runner: QtTaskRunner) -> None:
        super().__init__(panel)
        self.panel, self.workspace, self.runner = panel, workspace, task_runner
        self.prepared: PreparedJob | None = None
        self.result: ReconstructionResult | None = None
        self.handle = None
        self._busy = False
        self.settings = QSettings()
        panel.executable.setText(str(self.settings.value("freect/executable", shutil.which("fct-wfbp") or "")))
        panel.prepare_button.clicked.connect(self.prepare)
        panel.run_button.clicked.connect(self.run)
        panel.cancel_button.clicked.connect(self.cancel)
        panel.open_button.clicked.connect(self.open_folder)
        for edit in (panel.executable, panel.source, panel.output):
            edit.textChanged.connect(self.invalidate)
        panel.spectrum.valueChanged.connect(self.invalidate)
        panel.source_index.valueChanged.connect(self.invalidate)
        workspace.viewer.slice_changed.connect(self.show_slice)

    @property
    def is_running(self) -> bool:
        return self._busy

    @Slot()
    def invalidate(self) -> None:
        if self._busy:
            return
        self.prepared = None
        self._clear_result()
        self.workspace.config.clear()
        self.panel.set_state(False, False, False)
        self.workspace.status.setText("Input changed. Prepare the acquisition before reconstructing.")

    def _clear_result(self) -> None:
        self.result = None
        self.workspace.viewer.clear_image()
        self.workspace.viewer.set_slice_count(0)
        self.workspace.viewer.set_info_text("No completed reconstruction loaded")

    @Slot()
    def prepare(self) -> None:
        if self._busy:
            return
        if not all(edit.text().strip() for edit in (self.panel.executable, self.panel.source, self.panel.output)):
            self._error("Select a FreeCT executable, CTPD input and output folder.")
            return
        executable = self.panel.executable.text().strip()
        source, output = Path(self.panel.source.text().strip()), Path(self.panel.output.text().strip())
        spectrum, source_index = self.panel.spectrum.value(), self.panel.source_index.value()
        self.settings.setValue("freect/executable", executable)
        self.prepared = None
        self._clear_result()
        self.workspace.config.clear()
        self._start(lambda context: prepare(executable, source, output, context,
                                            spectrum=spectrum, source_index=source_index))

    @Slot()
    def run(self) -> None:
        if self._busy or self.prepared is None:
            return
        try:
            config = self.panel.apply_settings(load_config(self.workspace.config.toPlainText()))
            text = dump_config(config)
        except (ValueError, TypeError, KeyError) as exc:
            self._error(str(exc))
            return
        self.workspace.config.setPlainText(text)
        prepared, native = self.prepared, self.panel.native_dicom.isChecked()
        self._clear_result()
        self._start(lambda context: reconstruct(prepared, text, native, context))

    def _start(self, operation) -> None:
        self._busy = True
        self.panel.set_state(True, self.prepared is not None, self.prepared is not None)
        self.workspace.config.setReadOnly(True)
        self.workspace.log.clear()
        self.workspace.progress.setRange(0, 0)
        self.workspace.status.setText("Running...")
        # QtTaskRunner starts the thread immediately. Do not let a fast operation
        # emit its terminal signals until every queued UI connection is installed.
        ready = threading.Event()

        def work(task):
            ready.wait()
            context = JobContext(task.cancel_event, task.emit)
            try:
                return operation(context)
            except ReconstructionCancelled as exc:
                raise TaskCancelled(str(exc)) from exc

        try:
            self.handle = self.runner.start(work)
            signals = self.handle.context.signals
            signals.event.connect(self.updated, Qt.ConnectionType.QueuedConnection)
            signals.succeeded.connect(self.succeeded, Qt.ConnectionType.QueuedConnection)
            signals.failed.connect(self.failed, Qt.ConnectionType.QueuedConnection)
            signals.cancelled.connect(self.cancelled, Qt.ConnectionType.QueuedConnection)
            self.handle.thread.finished.connect(self.finished, Qt.ConnectionType.QueuedConnection)
        except Exception as exc:
            self._busy = False
            self._error(str(exc))
            self.finished()
        finally:
            ready.set()

    @Slot(object)
    def updated(self, update: Update) -> None:
        self.workspace.log.appendPlainText(update.message)
        self.workspace.status.setText(update.message)
        if update.percent is None:
            self.workspace.progress.setRange(0, 0)
        else:
            self.workspace.progress.setRange(0, 100)
            self.workspace.progress.setValue(update.percent)

    @Slot(object)
    def succeeded(self, value) -> None:
        if isinstance(value, PreparedJob):
            try:
                self.panel.load_settings(load_config(value.config_text))
            except (ValueError, TypeError, KeyError) as exc:
                self._error(str(exc))
                return
            self.prepared = value
            self.workspace.config.setPlainText(value.config_text)
            self.workspace.tabs.setCurrentIndex(1)
            self.workspace.status.setText(
                f"Prepared {value.projections} projections. Review settings; then reconstruct. "
                f"Job: {value.directory}"
            )
        elif isinstance(value, ReconstructionResult):
            self.result = value
            viewer = self.workspace.viewer
            self.workspace.image_view.pixel_aspect = value.shape[1] / value.shape[2]
            viewer.preset_combo.setEnabled(value.rescaled_to_hu)
            viewer.set_slice_count(value.shape[0], initial_index=value.shape[0] // 2)
            if value.rescaled_to_hu:
                viewer.set_window(400, 40, preset_name="Soft tissue")
            self.show_slice(viewer.current_slice)
            self.workspace.tabs.setCurrentIndex(0)
            self.workspace.status.setText(f"Completed: {value.directory}")
        self.workspace.progress.setRange(0, 100)
        self.workspace.progress.setValue(100)

    @Slot(int)
    def show_slice(self, index: int) -> None:
        if self.result is None:
            return
        try:
            pixels = read_slice(self.result.image_path, index)
            viewer = self.workspace.viewer
            if self.result.rescaled_to_hu:
                viewer.set_hu_image(pixels)
                units = "HU"
            else:
                low, high = float(pixels.min()), float(pixels.max())
                gray = np.rint(np.clip((pixels - low) / max(high-low, 1e-12), 0, 1) * 255).astype(np.uint8)
                viewer.set_rgb_image(gray)
                viewer.window_label.setText("Auto-scaled attenuation (not HU)")
                units = f"native attenuation; range {low:.5g} to {high:.5g}"
            viewer.set_info_text(
                f"{units}; scanner Z={self.result.start + index*self.result.pitch:.3f} mm; "
                "patient orientation unknown"
            )
        except (OSError, ValueError, IndexError) as exc:
            self.workspace.viewer.clear_image()
            self._error(f"Cannot read reconstructed slice: {exc}")

    @Slot(str)
    def failed(self, message: str) -> None:
        self.workspace.log.appendPlainText(message)
        self.workspace.status.setText("Failed. Partial outputs are not completed images; see the job logs.")
        self._error(message.strip().splitlines()[-1] if message.strip() else "FreeCT failed.")

    @Slot(str)
    def cancelled(self, message: str) -> None:
        self.workspace.status.setText(message + " Partial files are retained for inspection.")
        self.workspace.log.appendPlainText(message)

    @Slot()
    def finished(self) -> None:
        self.handle = None
        self._busy = False
        self.workspace.config.setReadOnly(False)
        self.panel.set_state(False, self.prepared is not None,
                             self.result is not None or self.prepared is not None)
        if self.workspace.progress.maximum() == 0:
            self.workspace.progress.setRange(0, 100)
            self.workspace.progress.setValue(0)

    @Slot()
    def cancel(self) -> None:
        if self.handle is not None:
            self.handle.cancel()
            self.panel.cancel_button.setEnabled(False)
            self.workspace.status.setText("Cancelling and stopping FreeCT...")

    def cleanup(self) -> None:
        self.cancel()

    @Slot()
    def open_folder(self) -> None:
        directory = self.result.directory if self.result else (
            self.prepared.directory if self.prepared else None
        )
        if directory is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    def _error(self, message: str) -> None:
        QMessageBox.critical(self.panel, "FreeCT reconstruction", message)
