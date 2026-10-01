from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


class LesionExtractionError(RuntimeError):
    """Base error for lesion-extraction failures that can be shown to users."""


class DependencyUnavailableError(LesionExtractionError):
    """Raised when an optional image-processing dependency is unavailable."""


class InputValidationError(LesionExtractionError):
    """Raised when selected inputs cannot form an extraction job."""


@dataclass(frozen=True, slots=True)
class DicomSeriesSpec:
    source_directory: Path
    series_uid: str
    files: tuple[Path, ...]
    display_name: str
    patient_id: str = ""

    @property
    def first_file(self) -> Path:
        if not self.files:
            raise InputValidationError(f"DICOM series {self.display_name!r} has no files.")
        return self.files[0]


@dataclass(frozen=True, slots=True)
class SegmentDefinition:
    key: str
    name: str
    label_value: int
    layer: int = 0
    color_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class SegmentationInfo:
    path: Path
    size_xyz: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    segments: tuple[SegmentDefinition, ...]
    is_vector: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LesionCandidate:
    candidate_id: str
    segment_key: str
    segment_name: str
    component_label: int
    component_index: int
    component_count: int
    output_name: str
    voxel_count: int
    physical_size_mm3: float
    centroid_xyz_mm: tuple[float, float, float]

    def renamed(self, output_name: str) -> LesionCandidate:
        return replace(self, output_name=output_name)


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    candidate_id: str
    included: bool
    output_name: str


@dataclass(frozen=True, slots=True)
class OverlayEntry:
    label_id: int
    entity_id: str
    display_name: str
    color_rgb: tuple[int, int, int]
    center_index_xyz: tuple[int, int, int] | None = None


@dataclass(slots=True)
class SeriesPreview:
    series: DicomSeriesSpec
    volume_hu_zyx: np.ndarray
    overlay_labels_zyx: np.ndarray
    overlay_entries: tuple[OverlayEntry, ...]
    window_center: float
    window_width: float

    def __post_init__(self) -> None:
        if self.volume_hu_zyx.ndim != 3:
            raise ValueError("Preview volume must be three-dimensional.")
        if self.overlay_labels_zyx.shape != self.volume_hu_zyx.shape:
            raise ValueError("Preview overlay must match the DICOM volume shape.")


@dataclass(frozen=True, slots=True)
class InspectionResult:
    series: tuple[DicomSeriesSpec, ...]
    segmentation: SegmentationInfo
    preview: SeriesPreview


@dataclass(frozen=True, slots=True)
class PreparedExtraction:
    series: tuple[DicomSeriesSpec, ...]
    segmentation: SegmentationInfo
    candidates: tuple[LesionCandidate, ...]
    preview: SeriesPreview


@dataclass(frozen=True, slots=True)
class ExtractionProgress:
    completed: int
    total: int
    message: str


@dataclass(frozen=True, slots=True)
class ExportedLesion:
    candidate_id: str
    output_path: Path
    series_names: tuple[str, ...]

    @property
    def series_name(self) -> str:
        """Human-readable summary retained for the existing GUI log."""

        return ", ".join(self.series_names)


@dataclass(frozen=True, slots=True)
class ExportSummary:
    exported: tuple[ExportedLesion, ...]
    warnings: tuple[str, ...]


@dataclass(slots=True)
class LesionNpzRecord:
    patient_name: str
    lesion_number: int
    org_case_path: str
    org_mask: str
    org_dcm_path: str
    dicom_header: Mapping[str, Any]
    lesion_mask: np.ndarray
    voi: np.ndarray
    org_slice_rng: np.ndarray
    org_col_rng: np.ndarray
    lesion_center: np.ndarray
    org_row_rng: np.ndarray
    lesion_diameter: np.ndarray
    lesion_mean_hu: float
    lesion_roundness: float
    lesion_max_hu: float
    lesion_min_hu: float
    lesion_median_hu: float
    lesion_sigma: float
    lesion_variance: float
    lesion_voxel_count: int
    lesion_physical_size: float
    series_names: tuple[str, ...] = ()
    series_uids: tuple[str, ...] = ()
    series_source_paths: tuple[str, ...] = ()
    dicom_headers: tuple[Mapping[str, Any], ...] = ()
    reference_series_index: int = 0


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def sanitize_output_name(value: str, *, fallback: str = "lesion") -> str:
    name = _INVALID_FILENAME.sub("-", str(value)).strip().strip(".")
    name = _WHITESPACE.sub(" ", name).rstrip(" .")
    return (name or fallback)[:120].rstrip(" .") or fallback


def make_unique_output_names(
    candidates: tuple[LesionCandidate, ...],
) -> tuple[LesionCandidate, ...]:
    counts: dict[str, int] = {}
    output: list[LesionCandidate] = []
    for candidate in candidates:
        base = sanitize_output_name(candidate.output_name)
        key = base.casefold()
        count = counts.get(key, 0) + 1
        counts[key] = count
        name = base if count == 1 else sanitize_output_name(f"{base}-{count:02d}")
        output.append(candidate.renamed(name))
    return tuple(output)


def overlay_color(index: int) -> tuple[int, int, int]:
    palette = (
        (255, 92, 92),
        (74, 214, 255),
        (255, 202, 70),
        (128, 237, 153),
        (201, 139, 255),
        (255, 133, 205),
        (103, 232, 212),
        (255, 154, 74),
        (152, 185, 255),
        (217, 239, 92),
        (255, 110, 147),
        (116, 255, 116),
    )
    return palette[index % len(palette)]
