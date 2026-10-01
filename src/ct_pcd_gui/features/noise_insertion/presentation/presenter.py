from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Signal, Slot

from ct_pcd_gui.shared.qt.task_runner import (
    QtTaskRunner,
    TaskCancelled,
    TaskContext,
    TaskHandle,
)

from ..application.errors import NoiseInsertionCancelled
from ..application.models import (
    JobLog,
    JobProgress,
    JobStarted,
    JobSummary,
    NoiseJobConfig,
    PreviewPayload,
)
from ..application.ports import NoiseJobCallbacks
from ..application.use_case import RunNoiseInsertion
from ..application.validation import validate_config
from .panel import NoiseInsertionPanel
from .view_state import NoiseJobDraft, NoiseJobPhase, NoiseViewState
from .workspace import NoiseInsertionWorkspace


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
        super().__init__(parent)
        self._panel = panel
        self._workspace = workspace
        self._use_case = use_case
        self._task_runner = task_runner
        self._task_handle: TaskHandle | None = None
        self._generation = 0
        self._cleaned = False

        panel.run_requested.connect(self.start)
        panel.cancel_requested.connect(self.cancel)
        panel.validation_failed.connect(self._show_validation_error)

    @property
    def is_running(self) -> bool:
        return self._task_handle is not None and self._task_handle.is_running

    @Slot(object)
    def start(self, draft: object) -> None:
        if self._task_handle is not None or self._cleaned:
            return
        if not isinstance(draft, NoiseJobDraft):
            self._show_validation_error("The noise-insertion settings are invalid.")
            return

        config = _config_from_draft(draft)
        try:
            validate_config(config)
        except (OSError, TypeError, ValueError) as exc:
            self._show_validation_error(str(exc))
            return

        self._generation += 1
        generation = self._generation

        self._workspace.show_inspecting(config.preview_interval)
        self._render_active(NoiseJobPhase.INSPECTING, cancel_enabled=True)
        callbacks_connected = threading.Event()

        def operation(context: TaskContext) -> JobSummary:
            # QtTaskRunner starts its thread before returning the handle. Hold the
            # operation until this presenter has connected every result signal so
            # even an immediate backend failure reaches the UI.
            callbacks_connected.wait()
            callbacks = NoiseJobCallbacks(
                log=lambda message: context.emit(JobLog(str(message))),
                started=lambda total, files, frames: context.emit(
                    JobStarted(int(total), int(files), int(frames))
                ),
                progress=lambda completed, total, status: context.emit(
                    JobProgress(int(completed), int(total), str(status))
                ),
                preview=context.emit,
            )
            try:
                summary = self._use_case.execute(
                    config,
                    cancel_event=context.cancel_event,
                    callbacks=callbacks,
                )
            except NoiseInsertionCancelled as exc:
                raise TaskCancelled(str(exc)) from exc

            context.raise_if_cancelled()
            return summary

        try:
            handle = self._task_runner.start(operation)
        except Exception as exc:
            message = str(exc) or type(exc).__name__
            self._workspace.show_failed(message)
            self._panel.render_state(NoiseViewState(phase=NoiseJobPhase.FAILED))
            self.error_requested.emit("Noise Insertion", message)
            return

        self._task_handle = handle
        self.running_changed.emit(True)

        signals = handle.context.signals
        signals.event.connect(lambda event, g=generation: self._handle_event(g, event))
        signals.succeeded.connect(
            lambda result, g=generation: self._completed(g, result)
        )
        signals.cancelled.connect(
            lambda message, g=generation: self._cancelled(g, message)
        )
        signals.failed.connect(lambda message, g=generation: self._failed(g, message))
        signals.finished.connect(lambda h=handle, g=generation: self._finished(g, h))
        callbacks_connected.set()

    @Slot()
    def cancel(self) -> None:
        handle = self._task_handle
        if handle is None or not handle.is_running:
            return

        self._workspace.show_cancelling()
        self._render_active(NoiseJobPhase.CANCELLING, cancel_enabled=False)
        handle.cancel()

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.cancel()

    def _handle_event(self, generation: int, event: object) -> None:
        if generation != self._generation:
            return
        if isinstance(event, JobLog):
            self._workspace.append_log(event.message)
        elif isinstance(event, JobStarted):
            self._workspace.show_started(event)
            self._render_active(NoiseJobPhase.RUNNING, cancel_enabled=True)
        elif isinstance(event, JobProgress):
            self._workspace.show_progress(event)
        elif isinstance(event, PreviewPayload):
            self._workspace.update_preview(event)

    def _completed(self, generation: int, result: object) -> None:
        if generation != self._generation:
            return
        if not isinstance(result, JobSummary):
            self._failed(generation, "Noise insertion returned an invalid summary.")
            return

        self._workspace.show_completed(result)
        self._render_active(NoiseJobPhase.COMPLETED, cancel_enabled=False)

    def _cancelled(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        self._workspace.show_cancelled(message)
        self._render_active(NoiseJobPhase.CANCELLING, cancel_enabled=False)

    def _failed(self, generation: int, message: str) -> None:
        if generation != self._generation:
            return
        friendly = _friendly_error(message)
        self._workspace.show_failed(friendly)
        self._render_active(NoiseJobPhase.FAILED, cancel_enabled=False)
        self.error_requested.emit("Noise Insertion", friendly)

    def _finished(self, generation: int, handle: TaskHandle) -> None:
        if generation != self._generation or self._task_handle is not handle:
            return
        self._task_handle = None
        self._panel.render_state(NoiseViewState())
        self.running_changed.emit(False)

    def _render_active(
        self,
        phase: NoiseJobPhase,
        *,
        cancel_enabled: bool,
    ) -> None:
        self._panel.render_state(
            NoiseViewState(
                phase=phase,
                inputs_enabled=False,
                start_enabled=False,
                cancel_enabled=cancel_enabled,
            )
        )

    @Slot(str)
    def _show_validation_error(self, message: str) -> None:
        self.error_requested.emit("Noise Insertion", str(message))


def _config_from_draft(draft: NoiseJobDraft) -> NoiseJobConfig:
    return NoiseJobConfig(
        input_path=draft.input_path,
        output_dir=draft.output_dir,
        input_mode=draft.input_mode,
        mas_factor=draft.mas_factor,
        fine_tune_factor=draft.fine_tune_factor,
        electronic_noise=draft.electronic_noise,
        seed=draft.seed,
        file_suffix=draft.file_suffix,
        overwrite_existing=draft.overwrite_existing,
        recursive=draft.recursive,
        continue_on_error=draft.continue_on_error,
        preview_interval=draft.preview_interval,
        parallel_mode=draft.parallel_mode,
        max_workers=draft.max_workers,
    )


def _friendly_error(message: str) -> str:
    lines = [line.strip() for line in str(message).splitlines() if line.strip()]
    return lines[-1] if lines else "Noise insertion failed for an unknown reason."
