from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .components import _component_images
from .dicom_metadata import read_dicom_header
from .models import (
    CandidateSelection,
    DicomSeriesSpec,
    ExportedLesion,
    ExportSummary,
    ExtractionProgress,
    InputValidationError,
    LesionCandidate,
    LesionNpzRecord,
    SegmentationInfo,
    sanitize_output_name,
)
from .preview import _resample_labels
from .readers import (
    _fallback_segments,
    _read_segmentation,
    load_dicom_image,
    parse_segment_definitions,
    require_sitk,
)
from .writers import write_lesion_npz


ProgressCallback = Callable[[ExtractionProgress], None]
CancelCallback = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class _LoadedSeries:
    spec: DicomSeriesSpec
    image: Any
    volume_zyx: np.ndarray
    dicom_header: Mapping[str, Any]


def legacy_padded_bbox(bbox_xyz: np.ndarray) -> np.ndarray:
    bbox = np.asarray(bbox_xyz, dtype=np.int64)
    if bbox.shape != (3, 2):
        raise ValueError("bbox_xyz must have shape (3, 2).")
    differences = bbox[:, 1] - bbox[:, 0]
    if np.any(differences < 0):
        raise ValueError("Bounding-box maxima must not be below minima.")
    padding = np.maximum(1, np.ceil(differences / 2.0).astype(np.int64))
    return bbox + np.column_stack((-padding, padding))


def crop_volume_with_padding(
    volume_zyx: np.ndarray,
    mask_zyx: np.ndarray,
    padded_bbox_xyz: np.ndarray,
    *,
    fill_value: int = -1024,
) -> tuple[np.ndarray, np.ndarray]:
    volume = np.asarray(volume_zyx)
    mask = np.asarray(mask_zyx, dtype=bool)
    bbox = np.asarray(padded_bbox_xyz, dtype=np.int64)
    if volume.ndim != 3 or mask.shape != volume.shape or bbox.shape != (3, 2):
        raise ValueError("Volume, mask, or bounding-box dimensions are invalid.")

    x_min, x_max = (int(value) for value in bbox[0])
    y_min, y_max = (int(value) for value in bbox[1])
    z_min, z_max = (int(value) for value in bbox[2])
    shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
    if any(length <= 0 for length in shape):
        raise ValueError("Padded bounding box is empty.")
    output_volume = np.full(shape, fill_value, dtype=volume.dtype)
    output_mask = np.zeros(shape, dtype=bool)

    source: list[slice] = []
    destination: list[slice] = []
    for requested_min, requested_max, length in (
        (z_min, z_max, volume.shape[0]),
        (y_min, y_max, volume.shape[1]),
        (x_min, x_max, volume.shape[2]),
    ):
        start = max(0, requested_min)
        stop = min(length - 1, requested_max)
        if stop < start:
            return (
                np.transpose(output_volume, (1, 2, 0)),
                np.transpose(output_mask, (1, 2, 0)),
            )
        destination_start = start - requested_min
        source.append(slice(start, stop + 1))
        destination.append(slice(destination_start, destination_start + stop - start + 1))
    output_volume[tuple(destination)] = volume[tuple(source)]
    output_mask[tuple(destination)] = mask[tuple(source)]
    return (
        np.transpose(output_volume, (1, 2, 0)),
        np.transpose(output_mask, (1, 2, 0)),
    )


def _bbox_from_mask(mask_zyx: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask_zyx)
    if not coordinates.size:
        raise InputValidationError("The lesion is empty after resampling.")
    z_min, y_min, x_min = coordinates.min(axis=0)
    z_max, y_max, x_max = coordinates.max(axis=0)
    return np.asarray([[x_min, x_max], [y_min, y_max], [z_min, z_max]])


def _safe_stat(func: Callable[[int], Any], label: int, fallback: float) -> float:
    try:
        value = float(func(label))
    except Exception:
        return float(fallback)
    return value if math.isfinite(value) else float(fallback)


def _geometry_text(values: Sequence[float | int]) -> str:
    return "(" + ", ".join(f"{float(value):.8g}" for value in values) + ")"


def validate_aligned_series_geometry(
    reference_image: Any,
    candidate_image: Any,
    *,
    reference_name: str,
    candidate_name: str,
) -> None:
    """Require voxel-for-voxel alignment before series are stored as channels."""

    mismatches: list[str] = []
    reference_size = tuple(int(value) for value in reference_image.GetSize())
    candidate_size = tuple(int(value) for value in candidate_image.GetSize())
    if reference_size != candidate_size:
        mismatches.append(f"size {candidate_size} versus {reference_size}")

    comparisons = (
        (
            "spacing",
            reference_image.GetSpacing(),
            candidate_image.GetSpacing(),
            1e-4,
        ),
        (
            "origin",
            reference_image.GetOrigin(),
            candidate_image.GetOrigin(),
            1e-3,
        ),
        (
            "direction",
            reference_image.GetDirection(),
            candidate_image.GetDirection(),
            1e-6,
        ),
    )
    for name, reference_values, candidate_values, tolerance in comparisons:
        reference_array = np.asarray(reference_values, dtype=np.float64)
        candidate_array = np.asarray(candidate_values, dtype=np.float64)
        if reference_array.shape != candidate_array.shape or not np.allclose(
            reference_array,
            candidate_array,
            rtol=1e-6,
            atol=tolerance,
        ):
            mismatches.append(
                f"{name} {_geometry_text(candidate_array)} versus "
                f"{_geometry_text(reference_array)}"
            )

    if mismatches:
        details = "; ".join(mismatches)
        raise InputValidationError(
            f"DICOM series {candidate_name!r} is not aligned with reference series "
            f"{reference_name!r}: {details}. Multi-series lesion files require "
            "identical size, spacing, origin, and direction so every channel "
            "represents the same voxels. Register or resample the series before export."
        )


def _pooled_statistics(values_by_channel: Sequence[np.ndarray]) -> tuple[float, ...]:
    values = np.concatenate(
        [np.asarray(channel, dtype=np.float64).reshape(-1) for channel in values_by_channel]
    )
    sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return (
        float(np.mean(values)),
        float(np.max(values)),
        float(np.min(values)),
        float(np.median(values)),
        sigma,
        sigma**2,
    )


def _record(
    *,
    loaded_series: Sequence[_LoadedSeries],
    segmentation_path: Path,
    mask_image: Any,
    output_name: str,
    lesion_number: int,
) -> LesionNpzRecord:
    sitk = require_sitk()
    reference = loaded_series[0]
    mask_zyx = np.asarray(sitk.GetArrayFromImage(mask_image), dtype=bool)
    bbox = _bbox_from_mask(mask_zyx)
    padded_bbox = legacy_padded_bbox(bbox)

    vois: list[np.ndarray] = []
    channel_values: list[np.ndarray] = []
    cropped_mask: np.ndarray | None = None
    for loaded in loaded_series:
        voi, candidate_mask = crop_volume_with_padding(
            loaded.volume_zyx,
            mask_zyx,
            padded_bbox,
        )
        vois.append(voi)
        channel_values.append(loaded.volume_zyx[mask_zyx])
        if cropped_mask is None:
            cropped_mask = candidate_mask
    assert cropped_mask is not None

    if len(vois) == 1:
        voi_output = vois[0]
        mask_output = cropped_mask
    else:
        voi_output = np.stack(vois, axis=-1)
        mask_output = np.repeat(
            cropped_mask[..., np.newaxis], len(vois), axis=3
        )

    mean, maximum, minimum, median, sigma, variance = _pooled_statistics(
        channel_values
    )
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(mask_image)
    label = 1
    try:
        center = np.asarray(
            reference.image.TransformPhysicalPointToIndex(
                shape_stats.GetCentroid(label)
            ),
            dtype=np.int64,
        )
    except Exception:
        center = np.rint(np.mean(bbox, axis=1)).astype(np.int64)
    try:
        diameter = np.asarray(
            shape_stats.GetEquivalentEllipsoidDiameter(label), dtype=np.float64
        )
        if diameter.size != 3:
            raise ValueError
    except Exception:
        diameter = (bbox[:, 1] - bbox[:, 0] + 1) * np.asarray(
            reference.image.GetSpacing(), dtype=np.float64
        )
    voxel_count = int(mask_zyx.sum())
    fallback_size = voxel_count * float(np.prod(reference.image.GetSpacing()))

    series_specs = tuple(item.spec for item in loaded_series)
    headers = tuple(item.dicom_header for item in loaded_series)
    reference_path = str(reference.spec.source_directory)
    return LesionNpzRecord(
        patient_name=output_name,
        lesion_number=lesion_number,
        org_case_path=reference_path,
        org_mask=str(segmentation_path),
        org_dcm_path=reference_path,
        dicom_header=reference.dicom_header,
        lesion_mask=mask_output,
        voi=voi_output,
        org_slice_rng=bbox[2],
        org_col_rng=bbox[0],
        lesion_center=center,
        org_row_rng=bbox[1],
        lesion_diameter=diameter,
        lesion_mean_hu=mean,
        lesion_roundness=_safe_stat(shape_stats.GetRoundness, label, np.nan),
        lesion_max_hu=maximum,
        lesion_min_hu=minimum,
        lesion_median_hu=median,
        lesion_sigma=sigma,
        lesion_variance=variance,
        lesion_voxel_count=voxel_count,
        lesion_physical_size=_safe_stat(
            shape_stats.GetPhysicalSize, label, fallback_size
        ),
        series_names=tuple(item.display_name for item in series_specs),
        series_uids=tuple(item.series_uid for item in series_specs),
        series_source_paths=tuple(
            str(item.source_directory) for item in series_specs
        ),
        dicom_headers=headers,
        reference_series_index=0,
    )


def _resolve_selections(
    candidates: Sequence[LesionCandidate],
    selections: Sequence[CandidateSelection],
) -> list[tuple[LesionCandidate, str]]:
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[tuple[LesionCandidate, str]] = []
    used_names: set[str] = set()
    for selection in selections:
        if not selection.included:
            continue
        candidate = candidates_by_id.get(selection.candidate_id)
        if candidate is None:
            raise InputValidationError(
                f"Unknown lesion candidate: {selection.candidate_id}"
            )
        name = sanitize_output_name(selection.output_name)
        if name.casefold() in used_names:
            raise InputValidationError(f"Duplicate included output name: {name}")
        used_names.add(name.casefold())
        selected.append((candidate, name))
    if not selected:
        raise InputValidationError("Include at least one lesion before exporting.")
    return selected


def _load_aligned_series(
    series: Sequence[DicomSeriesSpec],
    *,
    progress: ProgressCallback | None,
    cancelled: CancelCallback | None,
    total: int,
) -> tuple[_LoadedSeries, ...]:
    sitk = require_sitk()
    loaded: list[_LoadedSeries] = []
    for index, series_item in enumerate(series, start=1):
        if cancelled and cancelled():
            raise InputValidationError("Lesion extraction was cancelled.")
        if progress:
            progress(
                ExtractionProgress(
                    0,
                    total,
                    f"Loading series {index}/{len(series)}: "
                    f"{series_item.display_name}",
                )
            )
        image = load_dicom_image(series_item)
        if loaded:
            validate_aligned_series_geometry(
                loaded[0].image,
                image,
                reference_name=loaded[0].spec.display_name,
                candidate_name=series_item.display_name,
            )
        loaded.append(
            _LoadedSeries(
                spec=series_item,
                image=image,
                volume_zyx=np.asarray(
                    sitk.GetArrayFromImage(image), dtype=np.int16
                ),
                dicom_header=read_dicom_header(series_item.first_file),
            )
        )
    return tuple(loaded)


def export_lesions(
    *,
    series: Sequence[DicomSeriesSpec],
    segmentation: SegmentationInfo,
    candidates: Sequence[LesionCandidate],
    selections: Sequence[CandidateSelection],
    output_folder: str | Path,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> ExportSummary:
    if not series:
        raise InputValidationError("Add at least one DICOM series.")
    selected = _resolve_selections(candidates, selections)

    sitk = require_sitk()
    segmentation_image, header = _read_segmentation(segmentation.path)
    definitions = parse_segment_definitions(header) or _fallback_segments(
        segmentation_image
    )
    by_key = {segment.key: segment for segment in definitions}
    components = _component_images(
        segmentation_image, by_key, [candidate for candidate, _ in selected]
    )
    output_root = Path(output_folder).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    total = len(series) * len(selected)
    completed = 0
    exported: list[ExportedLesion] = []
    warnings: list[str] = []

    loaded_series = _load_aligned_series(
        series,
        progress=progress,
        cancelled=cancelled,
        total=total,
    )
    reference_image = loaded_series[0].image
    series_names = tuple(item.spec.display_name for item in loaded_series)

    for lesion_number, (candidate, output_name) in enumerate(selected):
        if cancelled and cancelled():
            raise InputValidationError("Lesion extraction was cancelled.")
        component = components[candidate.segment_key]
        mask = sitk.BinaryThreshold(
            component,
            candidate.component_label,
            candidate.component_label,
            1,
            0,
        )
        mask = _resample_labels(mask, reference_image)
        mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
        if not np.any(sitk.GetArrayViewFromImage(mask)):
            warnings.append(
                f"Skipped {output_name!r}: the component is outside the reference "
                f"series geometry ({series_names[0]!r})."
            )
        else:
            record = _record(
                loaded_series=loaded_series,
                segmentation_path=segmentation.path,
                mask_image=mask,
                output_name=output_name,
                lesion_number=lesion_number,
            )
            path = write_lesion_npz(record, output_root / f"{output_name}.npz")
            exported.append(
                ExportedLesion(
                    candidate_id=candidate.candidate_id,
                    output_path=path,
                    series_names=series_names,
                )
            )
        completed += len(series)
        if progress:
            progress(
                ExtractionProgress(
                    completed,
                    total,
                    f"Combined {len(series)} series into {output_name}.npz",
                )
            )
    if not exported:
        raise InputValidationError(
            "No lesion files were written. Verify that DICOM and NRRD geometry align."
        )
    return ExportSummary(tuple(exported), tuple(warnings))
