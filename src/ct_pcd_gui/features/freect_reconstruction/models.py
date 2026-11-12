from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


class ReconstructionCancelled(RuntimeError):
    """Cooperative cancellation, including termination of a running FreeCT process."""


@dataclass(frozen=True, slots=True)
class Update:
    message: str
    # None means indeterminate. FreeCT percentages describe individual stages,
    # not overall job completion, so they are deliberately left in the log.
    percent: int | None = None


@dataclass(slots=True)
class JobContext:
    cancel_event: threading.Event
    emit: Callable[[Update], None]

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ReconstructionCancelled("FreeCT operation cancelled.")

    def report(self, message: str, percent: int | None = None) -> None:
        self.check()
        self.emit(Update(message, percent))


@dataclass(frozen=True, slots=True)
class PreparedJob:
    directory: Path
    executable: Path
    projections: int
    config_text: str


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    directory: Path
    image_path: Path
    shape: tuple[int, int, int]  # z, y, x
    rescaled_to_hu: bool
    start: float
    pitch: float  # signed, for display only; not a patient-space transform
