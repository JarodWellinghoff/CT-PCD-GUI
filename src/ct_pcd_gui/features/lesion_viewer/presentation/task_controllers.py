from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QObject, Signal

from ct_pcd_gui.features.lesion_viewer.application.use_cases import (
    BuildLesionSurface, LoadLesionFile, ScanLesionLibrary,
)
from ct_pcd_gui.features.lesion_viewer.infrastructure.local_source import LocalDirectoryLesionSource
from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner, TaskContext, TaskHandle


class LibraryTaskController(QObject):
    batch_ready = Signal(object)
    scan_completed = Signal(object)
    file_loaded = Signal(object)
    status = Signal(str)
    failed = Signal(str, str)
    running_changed = Signal(bool)

    def __init__(self, scan: ScanLesionLibrary, load: LoadLesionFile,
                 runner: QtTaskRunner, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._scan, self._load, self._runner = scan, load, runner
        self._scan_handle: TaskHandle | None = None
        self._load_handle: TaskHandle | None = None
        self._scan_generation = 0; self._load_generation = 0

    @property
    def is_running(self) -> bool:
        return any(h is not None and h.is_running for h in (self._scan_handle, self._load_handle))

    def scan(self, directory: Path) -> None:
        self.cancel(); self._scan_generation += 1; generation = self._scan_generation
        source = LocalDirectoryLesionSource(directory)
        def operation(context: TaskContext):
            records = self._scan.execute(
                source, cancel_event=context.cancel_event,
                on_batch=lambda batch: context.emit(tuple(batch)), batch_size=100,
            )
            context.raise_if_cancelled(); return records
        handle = self._runner.start(operation); self._scan_handle = handle; self._emit_running()
        signals = handle.context.signals
        signals.event.connect(lambda batch, g=generation: self._batch(g, batch))
        signals.succeeded.connect(lambda result, g=generation: self._scan_done(g, result))
        signals.failed.connect(lambda message, g=generation: self._fail_scan(g, message))
        signals.cancelled.connect(lambda _message, g=generation: self._cancelled_scan(g))
        signals.finished.connect(lambda h=handle: self._finished_scan(h))

    def load(self, path: Path) -> None:
        self._cancel(self._load_handle); self._load_generation += 1; generation = self._load_generation
        def operation(context: TaskContext):
            result = self._load.execute(path); context.raise_if_cancelled(); return result
        handle = self._runner.start(operation); self._load_handle = handle; self._emit_running()
        signals = handle.context.signals
        signals.succeeded.connect(lambda result, g=generation: self._loaded(g, result))
        signals.failed.connect(lambda message, g=generation: self._fail_load(g, message))
        signals.cancelled.connect(lambda _message, g=generation: self._cancelled_load(g))
        signals.finished.connect(lambda h=handle: self._finished_load(h))

    def cancel(self) -> None:
        self._scan_generation += 1; self._load_generation += 1
        self._cancel(self._scan_handle); self._cancel(self._load_handle)

    def _batch(self, generation: int, batch: object) -> None:
        if generation == self._scan_generation: self.batch_ready.emit(list(batch))
    def _scan_done(self, generation: int, result: object) -> None:
        if generation == self._scan_generation: self.scan_completed.emit(result)
    def _loaded(self, generation: int, result: object) -> None:
        if generation == self._load_generation: self.file_loaded.emit(result)
    def _fail_scan(self, generation: int, message: str) -> None:
        if generation == self._scan_generation: self.failed.emit("Could not scan lesion library", _friendly(message))
    def _fail_load(self, generation: int, message: str) -> None:
        if generation == self._load_generation: self.failed.emit("Could not load lesion file", _friendly(message))
    def _cancelled_scan(self, generation: int) -> None:
        if generation == self._scan_generation: self.status.emit("Lesion-library scan cancelled.")
    def _cancelled_load(self, generation: int) -> None:
        if generation == self._load_generation: self.status.emit("Lesion-file load cancelled.")
    def _finished_scan(self, handle: TaskHandle ) -> None:
        if self._scan_handle is handle: self._scan_handle = None
        self._emit_running()
    def _finished_load(self, handle: TaskHandle ) -> None:
        if self._load_handle is handle: self._load_handle = None
        self._emit_running()
    def _emit_running(self) -> None: self.running_changed.emit(self.is_running)
    @staticmethod
    def _cancel(handle: TaskHandle | None) -> None:
        if handle is not None and handle.is_running: handle.cancel()


class SurfaceTaskController(QObject):
    completed = Signal(object)
    status = Signal(str)
    failed = Signal(str, str)
    running_changed = Signal(bool)

    def __init__(self, build: BuildLesionSurface, runner: QtTaskRunner,
                 parent: QObject | None = None) -> None:
        super().__init__(parent); self._build, self._runner = build, runner
        self._handle: TaskHandle | None = None; self._generation = 0

    @property
    def is_running(self) -> bool: return self._handle is not None and self._handle.is_running

    def start(self, request) -> None:
        self.cancel(); self._generation += 1; generation = self._generation
        def operation(context: TaskContext):
            context.raise_if_cancelled(); result = self._build.execute(request)
            context.raise_if_cancelled(); return result
        handle = self._runner.start(operation); self._handle = handle; self.running_changed.emit(True)
        signals = handle.context.signals
        signals.succeeded.connect(lambda result, g=generation: self._done(g, result))
        signals.failed.connect(lambda message, g=generation: self._failed(g, message))
        signals.cancelled.connect(lambda _message, g=generation: self._cancelled(g))
        signals.finished.connect(lambda h=handle: self._finished(h))

    def cancel(self) -> None:
        self._generation += 1
        if self._handle is not None and self._handle.is_running: self._handle.cancel()
    def _done(self, generation: int, result: object) -> None:
        if generation == self._generation: self.completed.emit(result)
    def _failed(self, generation: int, message: str) -> None:
        if generation == self._generation: self.failed.emit("Could not build lesion surface", _friendly(message))
    def _cancelled(self, generation: int) -> None:
        if generation == self._generation: self.status.emit("Surface build cancelled.")
    def _finished(self, handle: TaskHandle) -> None:
        if self._handle is handle: self._handle = None
        self.running_changed.emit(self.is_running)


def _friendly(message: str) -> str:
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return lines[-1] if lines else "Unknown error"
