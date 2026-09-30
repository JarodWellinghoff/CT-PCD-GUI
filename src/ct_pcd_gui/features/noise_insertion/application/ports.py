from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .models import JobSummary, NoiseJobConfig, PreviewPayload

LogCallback = Callable[[str], None]
StartedCallback = Callable[[int, int, int], None]
ProgressCallback = Callable[[int, int, str], None]
PreviewCallback = Callable[[PreviewPayload], None]


class CancellationFlag(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class NoiseJobCallbacks:
    log: LogCallback
    started: StartedCallback
    progress: ProgressCallback
    preview: PreviewCallback


class NoiseJobExecutor(Protocol):
    def run(
        self,
        config: NoiseJobConfig,
        *,
        cancel_event: CancellationFlag,
        callbacks: NoiseJobCallbacks,
    ) -> JobSummary: ...
