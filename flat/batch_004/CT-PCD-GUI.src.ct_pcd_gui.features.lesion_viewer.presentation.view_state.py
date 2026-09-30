from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class LesionViewerPhase(Enum):
    IDLE = auto()
    SCANNING = auto()
    LOADING = auto()
    BUILDING_SURFACE = auto()
    READY = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class AppearanceState:
    color_rgb: tuple[float, float, float]
    opacity: float
    representation: str
    show_edges: bool
    parallel_projection: bool


@dataclass(frozen=True, slots=True)
class LesionViewerState:
    phase: LesionViewerPhase = LesionViewerPhase.IDLE
    status: str = "Ready"
    current_directory: Path | None = None
    current_file: Path | None = None
    controls_enabled: bool = False
    file_count: int = 0
    filtered_count: int = 0
