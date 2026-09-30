from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class NoiseJobConfig:
    input_path: str
    output_dir: str
    input_mode: str = "auto"
    mas_factor: float = 0.25
    electronic_noise: float = 0.0
    seed: int | None = 42
    file_suffix: str = "_noise"
    update_tube_current: bool = True
    overwrite_existing: bool = False
    recursive: bool = False
    continue_on_error: bool = True
    preview_interval: int = 100
    parallel_mode: str = "auto"
    max_workers: int = 0


@dataclass(frozen=True, slots=True)
class DicomWorkItem:
    input_path: Path
    output_path: Path
    frame_count: int
    is_multiframe: bool

    @property
    def work_units(self) -> int:
        return self.frame_count + 1


@dataclass(frozen=True, slots=True)
class PreviewPayload:
    title: str
    iteration: int
    total_iterations: int
    before_u8: np.ndarray
    after_u8: np.ndarray
    window_low: float
    window_high: float
    before_mean: float
    before_std: float
    after_mean: float
    after_std: float


@dataclass(slots=True)
class JobSummary:
    total_files: int = 0
    total_frames: int = 0
    written_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    clipped_pixels: int = 0
    output_paths: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    parallel_mode: str = "sequential"
    max_workers: int = 1


@dataclass(frozen=True, slots=True)
class JobStarted:
    total_units: int
    file_count: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class JobProgress:
    completed_units: int
    total_units: int
    status: str


@dataclass(frozen=True, slots=True)
class JobLog:
    message: str
