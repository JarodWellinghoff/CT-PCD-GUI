from dataclasses import dataclass

from ct_pcd_gui.shell.module_registry import ModuleDescriptor
from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from .panel import NoiseInsertionPanel
from .workspace import NoiseInsertionWorkspace
from .presenter import NoiseInsertionPresenter
from ..infrastructure.executor import DefaultNoiseJobExecutor
from ..application.use_case import RunNoiseInsertion
from PySide6.QtWidgets import QMessageBox


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
    executor = DefaultNoiseJobExecutor()
    use_case = RunNoiseInsertion(executor)
    presenter = NoiseInsertionPresenter(
        panel=panel,
        workspace=workspace,
        use_case=use_case,
        task_runner=task_runner,
        parent=panel,
    )

    presenter.error_requested.connect(
        lambda title, message: QMessageBox.critical(panel, title, message)
    )

    return ModuleDescriptor(
        module_id="noise-insertion",
        display_name="Noise Insertion",
        panel=panel,
        workspace=workspace,
        status_text="Configure a noise insertion job",
        show_panel=True,
        is_busy=lambda: presenter.is_running,
        request_cancel=presenter.cancel,
        cleanup=presenter.cleanup,
        # Add this field to ModuleDescriptor as described below.
        owner=presenter,
    )
