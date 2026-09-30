from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QMessageBox

from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from ct_pcd_gui.shell.module_registry import ModuleDescriptor

from ..application.use_case import RunNoiseInsertion
from ..infrastructure.executor_adapter import DefaultNoiseJobExecutor
from .panel import NoiseInsertionPanel
from .presenter import NoiseInsertionPresenter
from .workspace import NoiseInsertionWorkspace


@dataclass(slots=True)
class NoiseInsertionModule:
    panel: NoiseInsertionPanel
    workspace: NoiseInsertionWorkspace
    presenter: NoiseInsertionPresenter

    def descriptor(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            module_id="noise-insertion",
            display_name="Noise Insertion",
            panel=self.panel,
            workspace=self.workspace,
            status_text="Configure a noise insertion job",
            owner=self.presenter,
            show_panel=True,
            is_busy=lambda: self.presenter.is_running,
            request_cancel=self.presenter.cancel,
            cleanup=self.presenter.cleanup,
        )


def build_noise_insertion_module(
    task_runner: QtTaskRunner,
) -> ModuleDescriptor:
    panel = NoiseInsertionPanel()
    workspace = NoiseInsertionWorkspace()
    presenter = NoiseInsertionPresenter(
        panel=panel,
        workspace=workspace,
        use_case=RunNoiseInsertion(DefaultNoiseJobExecutor()),
        task_runner=task_runner,
        parent=panel,
    )
    presenter.error_requested.connect(
        lambda title, message: QMessageBox.critical(panel, title, message)
    )
    return NoiseInsertionModule(panel, workspace, presenter).descriptor()
