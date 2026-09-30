from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from PySide6.QtCore import QObject, Signal
from pydicom.dataset import Dataset

from ct_pcd_gui.features.noise_insertion.application.models import (
    JobSummary,
    NoiseJobConfig,
)
from ct_pcd_gui.features.noise_insertion.application.ports import NoiseJobCallbacks
from ct_pcd_gui.features.noise_insertion.application.use_case import RunNoiseInsertion
from ct_pcd_gui.features.noise_insertion.application.validation import validate_config
from ct_pcd_gui.features.noise_insertion.infrastructure import executor_adapter
from ct_pcd_gui.features.noise_insertion.infrastructure.dicom_io import (
    TAG_TUBE_CURRENT,
    update_tube_current,
)
from ct_pcd_gui.features.noise_insertion.presentation.panel import NoiseInsertionPanel
from ct_pcd_gui.features.noise_insertion.presentation.presenter import (
    NoiseInsertionPresenter,
)
from ct_pcd_gui.features.noise_insertion.presentation.view_state import (
    NoiseJobDraft,
    NoiseJobPhase,
    NoiseViewState,
)
from ct_pcd_gui.features.noise_insertion.presentation.workspace import (
    NoiseInsertionWorkspace,
)
from ct_pcd_gui.shared.qt.task_runner import (
    QtTaskRunner,
    TaskContext,
    TaskSignals,
)


class _Panel(QObject):
    run_requested = Signal(object)
    cancel_requested = Signal()
    validation_failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.states: list[NoiseViewState] = []

    def render_state(self, state: NoiseViewState) -> None:
        self.states.append(state)


class _Workspace:
    def __init__(self) -> None:
        self.inspection_interval: int | None = None
        self.logs: list[str] = []
        self.started: list[object] = []
        self.progress: list[object] = []
        self.completed: JobSummary | None = None
        self.cancelling = False

    def show_inspecting(self, preview_interval: int) -> None:
        self.inspection_interval = preview_interval

    def show_cancelling(self) -> None:
        self.cancelling = True

    def append_log(self, message: str) -> None:
        self.logs.append(message)

    def show_started(self, event: object) -> None:
        self.started.append(event)

    def show_progress(self, event: object) -> None:
        self.progress.append(event)

    def update_preview(self, _event: object) -> None:
        pass

    def show_completed(self, summary: JobSummary) -> None:
        self.completed = summary

    def show_cancelled(self, _message: str) -> None:
        pass

    def show_failed(self, _message: str) -> None:
        pass


class _Handle:
    def __init__(self) -> None:
        self.context = TaskContext(threading.Event(), TaskSignals())
        self.running = True

    @property
    def is_running(self) -> bool:
        return self.running

    def cancel(self) -> None:
        self.context.cancel()


class _Runner:
    def __init__(self) -> None:
        self.operation: Callable[[TaskContext], object] | None = None
        self.handle: _Handle | None = None

    def start(self, operation: Callable[[TaskContext], object]) -> _Handle:
        self.operation = operation
        self.handle = _Handle()
        return self.handle


class _UseCase:
    def execute(self, config, *, cancel_event, callbacks) -> JobSummary:
        assert not cancel_event.is_set()
        assert not hasattr(config, "update_tube_current")
        callbacks.log("Processing test input")
        callbacks.started(3, 1, 2)
        callbacks.progress(1, 3, "Generated noise")
        return JobSummary(total_files=1, total_frames=2, written_files=1)


def _draft(input_path: Path, output_dir: Path) -> NoiseJobDraft:
    return NoiseJobDraft(
        input_path=str(input_path),
        output_dir=str(output_dir),
        input_mode="auto",
        mas_factor=0.25,
        electronic_noise=0.0,
        seed=42,
        file_suffix="_noise",
        overwrite_existing=False,
        recursive=False,
        continue_on_error=True,
        preview_interval=10,
        parallel_mode="sequential",
        max_workers=0,
    )


def test_panel_reports_missing_paths_instead_of_raising(qtbot) -> None:
    panel = NoiseInsertionPanel()
    qtbot.addWidget(panel)
    messages: list[str] = []
    panel.validation_failed.connect(messages.append)

    assert not hasattr(panel, "update_tube_current")

    panel.start_button.click()

    assert messages == ["Select an input DICOM folder or file."]


def test_presenter_runs_job_and_restores_idle_state(qtbot, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    runner = _Runner()
    panel = _Panel()
    workspace = _Workspace()
    presenter = NoiseInsertionPresenter(
        panel=cast(NoiseInsertionPanel, panel),
        workspace=cast(NoiseInsertionWorkspace, workspace),
        use_case=cast(RunNoiseInsertion, _UseCase()),
        task_runner=cast(QtTaskRunner, runner),
    )

    presenter.start(_draft(input_dir, tmp_path / "output"))

    assert runner.operation is not None
    assert runner.handle is not None
    summary = runner.operation(runner.handle.context)
    runner.handle.context.signals.succeeded.emit(summary)
    runner.handle.running = False
    runner.handle.context.signals.finished.emit()

    assert workspace.inspection_interval == 10
    assert workspace.logs == ["Processing test input"]
    assert len(workspace.started) == 1
    assert len(workspace.progress) == 1
    assert workspace.completed is summary
    assert [state.phase for state in panel.states] == [
        NoiseJobPhase.INSPECTING,
        NoiseJobPhase.RUNNING,
        NoiseJobPhase.COMPLETED,
        NoiseJobPhase.IDLE,
    ]
    assert presenter.is_running is False


def test_presenter_cancel_sets_shared_event(qtbot, tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    runner = _Runner()
    panel = _Panel()
    workspace = _Workspace()
    presenter = NoiseInsertionPresenter(
        panel=cast(NoiseInsertionPanel, panel),
        workspace=cast(NoiseInsertionWorkspace, workspace),
        use_case=cast(RunNoiseInsertion, _UseCase()),
        task_runner=cast(QtTaskRunner, runner),
    )

    presenter.start(_draft(input_dir, tmp_path / "output"))
    presenter.cancel()

    assert runner.handle is not None
    assert runner.handle.context.cancel_event.is_set()
    assert workspace.cancelling is True
    assert panel.states[-1].phase is NoiseJobPhase.CANCELLING
    assert panel.states[-1].cancel_enabled is False


def test_executor_adapter_supplies_live_cancel_event(
    monkeypatch, tmp_path: Path
) -> None:
    cancel_event = threading.Event()
    observed: dict[str, object] = {}

    def discover(config, received_cancel_event, log_callback):
        observed["discovery_config"] = config
        observed["discovery_cancel_event"] = received_cancel_event
        observed["log_callback"] = log_callback
        return []

    def run_noise_job(config, **kwargs):
        observed["run_cancel_event"] = kwargs["cancel_event"]
        executor_adapter.executor.discover_work_items(
            config,
            object(),
            kwargs["log_callback"],
        )
        return JobSummary()

    monkeypatch.setattr(executor_adapter.executor, "discover_work_items", discover)
    monkeypatch.setattr(executor_adapter.executor, "run_noise_job", run_noise_job)

    config = NoiseJobConfig(
        input_path=str(tmp_path),
        output_dir=str(tmp_path / "output"),
    )
    callbacks = NoiseJobCallbacks(
        log=lambda _message: None,
        started=lambda _total, _files, _frames: None,
        progress=lambda _completed, _total, _status: None,
        preview=lambda _payload: None,
    )

    summary = executor_adapter.DefaultNoiseJobExecutor().run(
        config,
        cancel_event=cancel_event,
        callbacks=callbacks,
    )

    assert summary == JobSummary()
    assert observed["run_cancel_event"] is cancel_event
    assert observed["discovery_cancel_event"] is cancel_event


def test_tube_current_is_always_scaled() -> None:
    dataset = Dataset()
    dataset.add_new(TAG_TUBE_CURRENT, "IS", 200)

    update_tube_current(dataset, 0.25)

    assert int(dataset[TAG_TUBE_CURRENT].value) == 50


def test_validation_rejects_blank_input_and_file_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="input DICOM"):
        validate_config(
            NoiseJobConfig(input_path="   ", output_dir=str(tmp_path / "output"))
        )

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_file = tmp_path / "not-a-directory"
    output_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        validate_config(
            NoiseJobConfig(input_path=str(input_dir), output_dir=str(output_file))
        )
