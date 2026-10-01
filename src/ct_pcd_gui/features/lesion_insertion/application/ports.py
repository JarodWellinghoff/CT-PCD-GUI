from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from threading import Event
from typing import Protocol

from ..domain.models import (
    DicomSeriesCandidate,
    DicomVolume,
    LesionLibraryItem,
    LesionSession,
    PreviewResult,
    ProcessingResult,
    RawDatasetInfo,
    ValidationIssue,
)

ProgressCallback = Callable[[int, int, str], None]
LogCallback = Callable[[str], None]


class DicomService(Protocol):
    def discover_reconstruction_series(
        self, source: str | Path, cancel_event: Event | None = None
    ) -> list[DicomSeriesCandidate]: ...

    def load_reconstruction(
        self, source: str | Path, series_uid: str, cancel_event: Event | None = None
    ) -> DicomVolume: ...

    def inspect_raw(self, source: str | Path) -> RawDatasetInfo: ...

    def validate_association(
        self, volume: DicomVolume, raw: RawDatasetInfo
    ) -> list[ValidationIssue]: ...


class LesionLibrary(Protocol):
    def scan(
        self, source: str | Path, cancel_event: Event | None = None
    ) -> Iterable[LesionLibraryItem]: ...

    def inspect(self, source: str | Path) -> LesionLibraryItem: ...


class PreviewGenerator(Protocol):
    def generate(
        self,
        volume: DicomVolume,
        session: LesionSession,
        slice_index: int,
        generation: int,
        cancel_event: Event | None = None,
    ) -> PreviewResult: ...


class InsertionRunner(Protocol):
    def run(
        self,
        session: LesionSession,
        volume: DicomVolume,
        raw_info: RawDatasetInfo,
        *,
        cancel_event: Event,
        progress: ProgressCallback,
        log: LogCallback,
    ) -> ProcessingResult: ...
