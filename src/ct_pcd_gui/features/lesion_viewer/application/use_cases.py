from typing import Callable
from pathlib import Path
from .ports import LesionSource, LesionFileLoader, SurfaceBuilder
from ..domain.models import LesionFileRecord, LesionFileData
from ..domain.surface_models import SurfaceBuildRequest, SurfaceBuildResult


class ScanLesionLibrary:
    def execute(
        self,
        source: LesionSource,
        *,
        cancel_event,
        on_batch: Callable[[list[LesionFileRecord]], None],
        batch_size: int = 100,
    ) -> list[LesionFileRecord]: ...


class LoadLesionFile:
    def __init__(self, loader: LesionFileLoader) -> None:
        self._loader = loader

    def execute(self, path: Path) -> LesionFileData:
        return self._loader.load(path)


class BuildLesionSurface:
    def __init__(self, builder: SurfaceBuilder) -> None:
        self._builder = builder

    def execute(self, request: SurfaceBuildRequest) -> SurfaceBuildResult:
        return self._builder.build(request)
