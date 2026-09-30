from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtWidgets import QMessageBox

from ct_pcd_gui.features.lesion_viewer.application.use_cases import (
    BuildLesionSurface,
    LoadLesionFile,
    ScanLesionLibrary,
)
from ct_pcd_gui.features.lesion_viewer.infrastructure.file_loader import (
    MatNpzLesionFileLoader,
)
from ct_pcd_gui.features.lesion_viewer.infrastructure.mesh_exporter import (
    VtkMeshExporter,
)
from ct_pcd_gui.features.lesion_viewer.infrastructure.vtk_surface_builder import (
    VtkSurfaceBuilder,
)
from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from ct_pcd_gui.shell.module_registry import ModuleDescriptor

from .presenter import LesionViewerPresenter
from .view import LesionViewerPanel, LesionViewerWorkspace


@dataclass(slots=True)
class LesionViewerModule:
    panel: LesionViewerPanel
    workspace: LesionViewerWorkspace
    presenter: LesionViewerPresenter

    def descriptor(self) -> ModuleDescriptor:
        return ModuleDescriptor(
            module_id="lesion-viewer",
            display_name="Lesion Viewer",
            panel=self.panel,
            workspace=self.workspace,
            status_text="Browse MAT/NPZ lesions and reconstruct 3-D surfaces",
            owner=self.presenter,
            show_panel=True,
            is_busy=lambda: self.presenter.is_running,
            request_cancel=self.presenter.cancel,
            activate=self.presenter.activate,
            deactivate=self.presenter.deactivate,
            cleanup=self.presenter.cleanup,
        )


def build_lesion_viewer_module(
    task_runner: QtTaskRunner,
) -> ModuleDescriptor:
    panel = LesionViewerPanel()
    workspace = LesionViewerWorkspace()
    presenter = LesionViewerPresenter(
        panel=panel,
        workspace=workspace,
        scan_library=ScanLesionLibrary(),
        load_file=LoadLesionFile(MatNpzLesionFileLoader()),
        build_surface=BuildLesionSurface(VtkSurfaceBuilder()),
        exporter=VtkMeshExporter(),
        task_runner=task_runner,
        parent=panel,
    )
    presenter.error_requested.connect(
        lambda title, message: QMessageBox.critical(panel, title, message)
    )
    return LesionViewerModule(panel, workspace, presenter).descriptor()
