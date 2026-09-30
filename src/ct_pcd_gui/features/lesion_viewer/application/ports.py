from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from ..domain.models import LesionFileData, LesionFileRecord
from ..domain.surface_models import SurfaceBuildRequest, SurfaceBuildResult


class LesionSource(Protocol):
    @property
    def display_name(self) -> str: ...

    def iter_records(
        self,
        cancel_event=None,
    ) -> Iterable[LesionFileRecord]: ...


class LesionFileLoader(Protocol):
    def load(self, path: Path) -> LesionFileData: ...


class SurfaceBuilder(Protocol):
    def build(self, request: SurfaceBuildRequest) -> SurfaceBuildResult: ...


class MeshExporter(Protocol):
    def export(self, polydata: object, path: Path) -> Path: ...
