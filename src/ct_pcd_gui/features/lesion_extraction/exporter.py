from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
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


def _record(
    *,
    series: DicomSeriesSpec,
    segmentation_path: Path,
    dicom_image: Any,
    volume_zyx: np.ndarray,
    mask_image: Any,
    output_name: str,
    lesion_number: int,
    dicom_header: Mapping[str, Any],
) -> LesionNpzRecord:
    sitk = require_sitk()
    mask_zyx = np.asarray(sitk.GetArrayFromImage(mask_image), dtype=bool)
    bbox = _bbox_from_mask(mask_zyx)
    voi, lesion_mask = crop_volume_with_padding(
        volume_zyx, mask_zyx, legacy_padded_bbox(bbox)
    )
    values = volume_zyx[mask_zyx].astype(np.float64)
    mean = float(np.mean(values))
    maximum = float(np.max(values))
    minimum = float(np.min(values))
    median = float(np.median(values))
    sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0

    label_stats = sitk.LabelStatisticsImageFilter()
    label_stats.Execute(dicom_image, mask_image)
    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(mask_image)
    label = 1
    try:
        center = np.asarray(
            dicom_image.TransformPhysicalPointToIndex(shape_stats.GetCentroid(label)),
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
            dicom_image.GetSpacing(), dtype=np.float64
        )
    voxel_count = int(mask_zyx.sum())
    fallback_size = voxel_count * float(np.prod(dicom_image.GetSpacing()))

    return LesionNpzRecord(
        patient_name=output_name,
        lesion_number=lesion_number,
        org_case_path=str(series.source_directory),
        org_mask=str(segmentation_path),
        org_dcm_path=str(series.source_directory),
        dicom_header=dicom_header,
        lesion_mask=lesion_mask,
        voi=voi,
        org_slice_rng=bbox[2],
        org_col_rng=bbox[0],
        lesion_center=center,
        org_row_rng=bbox[1],
        lesion_diameter=diameter,
        lesion_mean_hu=_safe_stat(label_stats.GetMean, label, mean),
        lesion_roundness=_safe_stat(shape_stats.GetRoundness, label, np.nan),
        lesion_max_hu=_safe_stat(label_stats.GetMaximum, label, maximum),
        lesion_min_hu=_safe_stat(label_stats.GetMinimum, label, minimum),
        lesion_median_hu=_safe_stat(label_stats.GetMedian, label, median),
        lesion_sigma=_safe_stat(label_stats.GetSigma, label, sigma),
        lesion_variance=_safe_stat(label_stats.GetVariance, label, sigma**2),
        lesion_voxel_count=voxel_count,
        lesion_physical_size=_safe_stat(
            shape_stats.GetPhysicalSize, label, fallback_size
        ),
    )


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
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    selected: list[tuple[LesionCandidate, str]] = []
    used_names: set[str] = set()
    for selection in selections:
        if not selection.included:
            continue
        candidate = candidates_by_id.get(selection.candidate_id)
        if candidate is None:
            raise InputValidationError(f"Unknown lesion candidate: {selection.candidate_id}")
        name = sanitize_output_name(selection.output_name)
        if name.casefold() in used_names:
            raise InputValidationError(f"Duplicate included output name: {name}")
        used_names.add(name.casefold())
        selected.append((candidate, name))
    if not selected:
        raise InputValidationError("Include at least one lesion before exporting.")

    sitk = require_sitk()
    segmentation_image, header = _read_segmentation(segmentation.path)
    definitions = parse_segment_definitions(header) or _fallback_segments(segmentation_image)
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

    series_folders: list[Path] = []
    for index, item in enumerate(series, start=1):
        folder = output_root
        if len(series) > 1:
            folder = output_root / sanitize_output_name(
                item.display_name, fallback=f"series-{index:02d}"
            )
        series_folders.append(folder)

    for series_index, (series_item, destination_folder) in enumerate(
        zip(series, series_folders, strict=True), start=1
    ):
        if cancelled and cancelled():
            raise InputValidationError("Lesion extraction was cancelled.")
        if progress:
            progress(
                ExtractionProgress(
                    completed,
                    total,
                    f"Loading series {series_index}/{len(series)}: "
                    f"{series_item.display_name}",
                )
            )
        dicom_image = load_dicom_image(series_item)
        volume = np.asarray(sitk.GetArrayFromImage(dicom_image), dtype=np.int16)
        dicom_header = read_dicom_header(series_item.first_file)
        destination_folder.mkdir(parents=True, exist_ok=True)
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
            mask = _resample_labels(mask, dicom_image)
            mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
            if not np.any(sitk.GetArrayViewFromImage(mask)):
                warnings.append(
                    f"Skipped {output_name!r} for {series_item.display_name!r}: "
                    "the component is outside the series geometry."
                )
            else:
                record = _record(
                    series=series_item,
                    segmentation_path=segmentation.path,
                    dicom_image=dicom_image,
                    volume_zyx=volume,
                    mask_image=mask,
                    output_name=output_name,
                    lesion_number=lesion_number,
                    dicom_header=dicom_header,
                )
                path = write_lesion_npz(
                    record, destination_folder / f"{output_name}.npz"
                )
                exported.append(
                    ExportedLesion(series_item.display_name, candidate.candidate_id, path)
                )
            completed += 1
            if progress:
                progress(
                    ExtractionProgress(
                        completed,
                        total,
                        f"{series_item.display_name}: {output_name}",
                    )
                )
    if not exported:
        raise InputValidationError(
            "No lesion files were written. Verify that DICOM and NRRD geometry align."
        )
    return ExportSummary(tuple(exported), tuple(warnings))
