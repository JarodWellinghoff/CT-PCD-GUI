from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceBuildRequest:
    volume: np.ndarray
    source_min: float
    source_max: float
    axis_order: str
    spacing_xyz: tuple[float, float, float]
    threshold: float
    foreground_below: bool
    downsample: int
    smoothing_iterations: int
    reduction_percent: int
    largest_component_only: bool
    pad_border: bool


@dataclass(frozen=True, slots=True)
class SurfaceMetrics:
    foreground_count: int
    foreground_estimated: bool
    mask_volume_mm3: float
    surface_area_mm2: float
    mesh_volume_mm3: float
    points: int
    cells: int
    threshold: float
    spacing_xyz: tuple[float, float, float]
    axis_order: str
    downsample: int
    elapsed_ms: float
    foreground_below: bool


@dataclass(slots=True)
class SurfaceBuildResult:
    polydata: Any
    metrics: SurfaceMetrics
