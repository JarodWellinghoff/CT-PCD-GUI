from __future__ import annotations

from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from ct_pcd_gui.shell.module_registry import ModuleDescriptor

from .presenter import ReconstructionPresenter
from .view import ReconstructionPanel, ReconstructionWorkspace


def build_freect_reconstruction_module(task_runner: QtTaskRunner) -> ModuleDescriptor:
    panel = ReconstructionPanel()
    workspace = ReconstructionWorkspace()
    presenter = ReconstructionPresenter(panel, workspace, task_runner)
    return ModuleDescriptor(
        module_id="freect-reconstruction",
        display_name="FreeCT Reconstruction",
        panel=panel,
        workspace=workspace,
        status_text="Reconstruct CTPD projections with FreeCT v2 (research only)",
        owner=presenter,
        show_panel=True,
        is_busy=lambda: presenter.is_running,
        request_cancel=presenter.cancel,
        cleanup=presenter.cleanup,
    )
