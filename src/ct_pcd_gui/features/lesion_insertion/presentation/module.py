from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QMessageBox

from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from ct_pcd_gui.shell.module_registry import ModuleDescriptor

from ..infrastructure.dicom_service import PydicomLesionDicomService
from ..infrastructure.legacy_adapter import LegacyLesionInsertionAdapter
from ..infrastructure.lesion_library import LocalLesionModelLibrary
from ..infrastructure.preview import ApproximatePreviewGenerator
from .panel import LesionInsertionPanel
from .presenter import LesionInsertionPresenter
from .workspace import LesionInsertionWorkspace


@dataclass(slots=True)
class LesionInsertionModule:
    panel: LesionInsertionPanel
    workspace: LesionInsertionWorkspace
    presenter: LesionInsertionPresenter

    def descriptor(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            module_id="lesion-insertion",
            display_name="Lesion Insertion",
            panel=self.panel,
            workspace=self.workspace,
            status_text="Place lesions and insert them into DICOM-CT-PD raw data",
            owner=self.presenter,
            show_panel=True,
            is_busy=lambda: self.presenter.is_running,
            request_cancel=self.presenter.cancel,
            activate=self.presenter.activate,
            deactivate=self.presenter.deactivate,
            cleanup=self.presenter.cleanup,
        )


def build_lesion_insertion_module(task_runner: QtTaskRunner) -> ModuleDescriptor:
    panel = LesionInsertionPanel()
    workspace = LesionInsertionWorkspace()
    library = LocalLesionModelLibrary()
    presenter = LesionInsertionPresenter(
        panel=panel,
        workspace=workspace,
        dicom_service=PydicomLesionDicomService(),
        lesion_library=library,
        preview_generator=ApproximatePreviewGenerator(library.load_model),
        insertion_runner=LegacyLesionInsertionAdapter(),
        task_runner=task_runner,
        parent=panel,
    )
    presenter.error_requested.connect(
        lambda title, message: QMessageBox.critical(panel, title, message)
    )
    return LesionInsertionModule(panel, workspace, presenter).descriptor()
