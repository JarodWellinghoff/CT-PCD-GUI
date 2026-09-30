from __future__ import annotations

import threading
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QMessageBox
from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner, TaskHandle
from ..application.use_case import RunNoiseInsertion
from ..infrastructure.executor import run_noise_job
from ..application.models import NoiseJobConfig
from ..application.errors import NoiseInsertionCancelled
from .panel import NoiseInsertionPanel
from .workspace import NoiseInsertionWorkspace
from .view_state import NoiseJobDraft


class NoiseInsertionPresenter(QObject):
    running_changed = Signal(bool)
    error_requested = Signal(str, str)

    def __init__(
        self,
        *,
        panel: NoiseInsertionPanel,
        workspace: NoiseInsertionWorkspace,
        use_case: RunNoiseInsertion,
        task_runner: QtTaskRunner,
        parent: QObject | None = None,
    ) -> None:
        # The parent must be a persistent QObject or None.
        super().__init__(parent)

        self._panel = panel
        self._workspace = workspace
        self._use_case = use_case
        self._task_runner = task_runner
        self._task_handle: TaskHandle | None = None

        panel.run_requested.connect(self.start)
        panel.cancel_requested.connect(self.cancel)

    @property
    def is_running(self) -> bool:
        return self._task_handle is not None and self._task_handle.is_running

    @Slot(object)
    def start(self, draft: NoiseJobDraft) -> None:

        if self.is_running:
            return

        self._workspace.reset_for_job(draft.preview_interval)
        self._panel.render_state(True)
        self.running_changed.emit(True)

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

    @Slot()
    def cancel(self) -> None:
        if self._task_handle is not None:
            self._task_handle.cancel()

    def cleanup(self) -> None:
        if self._task_handle is not None:
            self._task_handle.cancel()


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

        panel.run_requested.connect(self.start)
        panel.cancel_requested.connect(self.cancel)

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
