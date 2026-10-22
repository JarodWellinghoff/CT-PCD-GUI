"""Cancellable, selection-scoped loading for the compact library preview."""
from __future__ import annotations

from collections.abc import Callable
from threading import Event

from PySide6.QtCore import QObject, Qt, Signal

from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner, TaskContext, TaskHandle

from ..application.library_preview import LibraryPreviewData, prepare_library_preview
from ..domain.models import LesionLibraryItem
from ..infrastructure.lesion_library import LocalLesionModelLibrary
from .library_preview import LibraryLesionPreview


class LibraryPreviewController(QObject):
    running_changed = Signal(bool)
    diagnostic = Signal(str)

    def __init__(
        self, view: LibraryLesionPreview, library: LocalLesionModelLibrary,
        task_runner: QtTaskRunner, parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.view, self.library, self.task_runner = view, library, task_runner
        self._data: LibraryPreviewData | None = None
        self._versions = {"load": 0, "surface": 0}
        self._latest: dict[str, TaskHandle] = {}
        self._active: set[TaskHandle] = set()
        self._closed = False
        view.surface_requested.connect(self.request_surface)

    @property
    def is_running(self) -> bool:
        # Keep cancelled/superseded workers counted until their finished signal.
        return bool(self._active)

    def _invalidate(self, kind: str) -> None:
        self._versions[kind] += 1
        handle = self._latest.pop(kind, None)
        if handle is not None:
            handle.cancel()

    def select(self, item: object) -> None:
        if self._closed:
            return
        self._invalidate("load")
        self._invalidate("surface")
        self._data = None
        source = item if isinstance(item, LesionLibraryItem) else None
        self.view.set_item(source)
        if source is None or not source.compatible:
            return

        def operation(context: TaskContext) -> LibraryPreviewData:
            context.raise_if_cancelled()
            model = self.library.load_model(source.path)
            context.raise_if_cancelled()
            return prepare_library_preview(
                model, source, check_cancelled=context.raise_if_cancelled
            )

        def success(result: object) -> None:
            if isinstance(result, LibraryPreviewData):
                self._data = result
                self.view.set_data(result)

        self._start("load", operation, success, self.view.show_load_error)

    def request_surface(self, channel_index: int) -> None:
        data = self._data
        if self._closed or data is None:
            return

        def operation(context: TaskContext) -> object:
            context.raise_if_cancelled()
            from ct_pcd_gui.features.lesion_viewer.infrastructure.vtk_surface_builder import (
                VtkSurfaceBuilder,
            )

            result = VtkSurfaceBuilder().build(data.surface_request(channel_index))
            context.raise_if_cancelled()
            return result.polydata

        self._start(
            "surface", operation,
            lambda result: self.view.show_surface(channel_index, result),
            self.view.show_surface_error,
        )

    def _start(
        self, kind: str, operation: Callable[[TaskContext], object],
        success: Callable[[object], None], error: Callable[[str], None],
    ) -> None:
        self._invalidate(kind)
        version = self._versions[kind]
        callbacks_ready = Event()

        def guarded_operation(context: TaskContext) -> object:
            # QtTaskRunner starts immediately; do not lose an instant success/error
            # before the queued callbacks below have been connected.
            callbacks_ready.wait()
            context.raise_if_cancelled()
            return operation(context)

        handle = self.task_runner.start(guarded_operation)
        self._latest[kind] = handle
        self._active.add(handle)

        def current() -> bool:
            return not self._closed and self._versions[kind] == version

        def on_success(result: object) -> None:
            if current():
                try:
                    success(result)
                except Exception as exc:
                    error(str(exc))

        def on_error(details: str) -> None:
            if current():
                self.diagnostic.emit(details)
                message = (
                    details.strip().splitlines()[-1] if details.strip() else "Unknown preview error"
                )
                error(message)

        def on_cancelled(message: str) -> None:
            if current():
                error(message or "Preview cancelled.")

        def on_finished() -> None:
            self._active.discard(handle)
            if self._latest.get(kind) is handle:
                self._latest.pop(kind, None)
            if not self._closed:
                self.running_changed.emit(self.is_running)

        try:
            queued = Qt.ConnectionType.QueuedConnection
            signals = handle.context.signals
            signals.succeeded.connect(on_success, queued)
            signals.failed.connect(on_error, queued)
            signals.cancelled.connect(on_cancelled, queued)
            signals.finished.connect(on_finished, queued)
            self.running_changed.emit(True)
        finally:
            callbacks_ready.set()

    def cancel(self) -> None:
        loading = "load" in self._latest
        surface = "surface" in self._latest
        self._invalidate("load")
        self._invalidate("surface")
        for handle in tuple(self._active):
            handle.cancel()
        if not self._closed:
            if loading:
                self.view.show_load_error("Loading cancelled. Reselect the lesion to retry.")
            elif surface:
                self.view.show_surface_error("3-D loading cancelled. Use Refresh to retry.")

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.cancel()
        self._data = None
        self.view.cleanup()
