from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from importlib.resources import files

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtWidgets import QApplication

# from src.ct_pcd_gui.features.lesion_viewer.presentation.module import (
#     build_lesion_viewer_module,
# )
from src.ct_pcd_gui.features.noise_insertion.presentation.module import (
    build_noise_insertion_module,
)
from src.ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from src.ct_pcd_gui.shell.main_window import MainWindow
from src.ct_pcd_gui.shell.module_registry import ModuleRegistry


def load_stylesheet() -> str:
    resource = files("ct_pcd_gui.resources.styles").joinpath("dark.qss")
    return resource.read_text(encoding="utf-8")


def build_module_registry(task_runner: QtTaskRunner) -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(build_noise_insertion_module(task_runner))
    # registry.register(build_lesion_viewer_module(task_runner))
    return registry


def main(argv: Sequence[str] | None = None) -> int:
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setOrganizationName("DICOM CTPD")
    app.setApplicationName("CT-PCD-GUI")
    app.setStyle("Fusion")
    app.setStyleSheet(load_stylesheet())

    task_runner = QtTaskRunner(app)
    registry = build_module_registry(task_runner)
    window = MainWindow(registry)
    window.show()

    return app.exec()
