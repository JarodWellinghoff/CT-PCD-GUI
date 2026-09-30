from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event

from .ports import LesionFileLoader, LesionSource, SurfaceBuilder
from ..domain.models import LesionFileData, LesionFileRecord
from ..domain.surface_models import SurfaceBuildRequest, SurfaceBuildResult


class ScanLesionLibrary:
    """Enumerate a lesion source while periodically publishing small batches."""

    def execute(
        self,
        source: LesionSource,
        *,
        cancel_event: Event,
        on_batch: Callable[[list[LesionFileRecord]], None],
        batch_size: int = 100,
    ) -> list[LesionFileRecord]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        records: list[LesionFileRecord] = []
        batch: list[LesionFileRecord] = []
        for record in source.iter_records(cancel_event=cancel_event):
            if cancel_event.is_set():
                break
            records.append(record)
            batch.append(record)
            if len(batch) >= batch_size:
                on_batch(batch.copy())
                batch.clear()

        if batch and not cancel_event.is_set():
            on_batch(batch.copy())

        records.sort(key=lambda item: item.relative_path.casefold())
        return records


class LoadLesionFile:
    def __init__(self, loader: LesionFileLoader) -> None:
        self._loader = loader

    def execute(self, path: str | Path) -> LesionFileData:
        return self._loader.load(Path(path))


class BuildLesionSurface:
    def __init__(self, builder: SurfaceBuilder) -> None:
        self._builder = builder

    def execute(self, request: SurfaceBuildRequest) -> SurfaceBuildResult:
        return self._builder.build(request)
