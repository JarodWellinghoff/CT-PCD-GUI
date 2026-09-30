from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot


class TaskCancelled(RuntimeError):
    pass


class TaskSignals(QObject):
    event = Signal(object)
    succeeded = Signal(object)
    cancelled = Signal(str)
    failed = Signal(str)
    finished = Signal()


@dataclass(slots=True)
class TaskContext:
    cancel_event: threading.Event
    signals: TaskSignals

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TaskCancelled("Operation was cancelled.")

    def emit(self, event: object) -> None:
        self.signals.event.emit(event)


class TaskWorker(QObject):
    def __init__(
        self,
        operation: Callable[[TaskContext], object],
        context: TaskContext,
    ) -> None:
        super().__init__()
        self._operation = operation
        self._context = context

    @Slot()
    def run(self) -> None:
        try:
            result = self._operation(self._context)
        except TaskCancelled as exc:
            self._context.signals.cancelled.emit(str(exc))
        except Exception:
            self._context.signals.failed.emit(traceback.format_exc())
        else:
            self._context.signals.succeeded.emit(result)
        finally:
            self._context.signals.finished.emit()


class TaskHandle(QObject):
    def __init__(
        self,
        thread: QThread,
        worker: TaskWorker,
        context: TaskContext,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.thread = thread
        self.worker = worker
        self.context = context

    @property
    def is_running(self) -> bool:
        return self.thread.isRunning()

    def cancel(self) -> None:
        self.context.cancel()


class QtTaskRunner(QObject):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._active: set[TaskHandle] = set()

    def start(
        self,
        operation: Callable[[TaskContext], object],
    ) -> TaskHandle:
        thread = QThread(self)
        signals = TaskSignals()
        context = TaskContext(threading.Event(), signals)
        worker = TaskWorker(operation, context)
        worker.moveToThread(thread)

        handle = TaskHandle(thread, worker, context, self)
        self._active.add(handle)

        thread.started.connect(worker.run)
        signals.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda: self._release(handle))
        thread.start()

        return handle

    def _release(self, handle: TaskHandle) -> None:
        self._active.discard(handle)
        handle.deleteLater()
