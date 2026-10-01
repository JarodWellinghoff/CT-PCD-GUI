from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QMessageBox

from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from ct_pcd_gui.shell.module_registry import ModuleDescriptor

from ..batch import LesionExtractionService
from .panel import LesionExtractionPanel
from .presenter import LesionExtractionPresenter
from .workspace import LesionExtractionWorkspace


@dataclass(slots=True)
class LesionExtractionModule:
    panel: LesionExtractionPanel
    workspace: LesionExtractionWorkspace
    presenter: LesionExtractionPresenter

    def descriptor(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            module_id="lesion-extraction",
            display_name="Lesion Extraction",
            panel=self.panel,
            workspace=self.workspace,
            status_text="Extract segmented lesions into compatible NPZ models",
            owner=self.presenter,
            show_panel=True,
            is_busy=lambda: self.presenter.is_running,
            request_cancel=self.presenter.cancel,
            activate=self.presenter.activate,
            deactivate=self.presenter.deactivate,
            cleanup=self.presenter.cleanup,
        )


def build_lesion_extraction_module(task_runner: QtTaskRunner) -> ModuleDescriptor:
    panel = LesionExtractionPanel()
    workspace = LesionExtractionWorkspace()
    presenter = LesionExtractionPresenter(
        panel=panel,
        workspace=workspace,
        service=LesionExtractionService(),
        task_runner=task_runner,
        parent=panel,
    )
    presenter.error_requested.connect(
        lambda title, message: QMessageBox.critical(panel, title, message)
    )
    return LesionExtractionModule(panel, workspace, presenter).descriptor()
