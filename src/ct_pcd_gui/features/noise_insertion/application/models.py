from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class MasFactor(float):
    """Float-compatible v2 dose factor carried through worker-process APIs.

    The established executor passes the nominal mAs factor directly to the
    numerical model and DICOM writer.  This value object preserves that numeric
    interface while carrying the v2 fine-tune calibration and the job-wide
    output series description through spawned process workers.
    """

    fine_tune_factor: float
    series_description: str | None

    def __new__(
        cls,
        value: float,
        fine_tune_factor: float = 1.0,
        series_description: str | None = None,
    ) -> MasFactor:
        instance = super().__new__(cls, value)
        instance.fine_tune_factor = float(fine_tune_factor)
        instance.series_description = series_description
        return instance

    def __reduce__(self):
        return (
            type(self),
            (float(self), self.fine_tune_factor, self.series_description),
        )


@dataclass(frozen=True, slots=True)
class NoiseJobConfig:
    input_path: str
    output_dir: str
    input_mode: str = "auto"
    mas_factor: float = 0.25
    electronic_noise: float = 0.0
    seed: int | None = 42
    file_suffix: str = "_noise"
    overwrite_existing: bool = False
    recursive: bool = False
    continue_on_error: bool = True
    preview_interval: int = 100
    parallel_mode: str = "auto"
    max_workers: int = 0
    fine_tune_factor: float = 1.0

    def __post_init__(self) -> None:
        inherited_description = getattr(
            self.mas_factor,
            "series_description",
            None,
        )
        object.__setattr__(
            self,
            "mas_factor",
            MasFactor(
                float(self.mas_factor),
                self.fine_tune_factor,
                inherited_description,
            ),
        )


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
