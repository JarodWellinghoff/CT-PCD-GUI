from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .models import InputValidationError, LesionNpzRecord


# The first 22 members deliberately retain the exact single-series reference
# schema and order used by L005-1-Lesion.npz.
NPZ_FIELDS = (
    "PatientName",
    "LesionNumber",
    "Org_case_path",
    "Org_mask",
    "Org_dcm_path",
    "DicomHeaderJSON",
    "LesionMask",
    "VOI",
    "org_slice_rng",
    "org_col_rng",
    "LesionCenter",
    "org_row_rng",
    "LesionDiameter",
    "LesionMeanHU",
    "LesionRoundness",
    "LesionMaxHU",
    "LesionMinHU",
    "LesionMedianHU",
    "LesionSigma",
    "LesionVariance",
    "LesionVoxelCount",
    "LesionPhysicalSize",
)

# These members are appended only when a lesion file contains more than one
# DICOM series. The last VOI axis and last LesionMask axis use this series order.
MULTI_SERIES_NPZ_FIELDS = (
    "SeriesCount",
    "ReferenceSeriesIndex",
    "SeriesNames",
    "SeriesUIDs",
    "SeriesSourcePaths",
    "DicomHeadersJSON",
    "LesionMeanHUByChannel",
    "LesionMaxHUByChannel",
    "LesionMinHUByChannel",
    "LesionMedianHUByChannel",
    "LesionSigmaByChannel",
    "LesionVarianceByChannel",
)


def _float64(value: float | int | None) -> np.ndarray:
    return np.asarray(np.nan if value is None else value, dtype=np.float64)


def _normalise_volumes(
    lesion_mask: np.ndarray,
    voi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    output_voi = np.asarray(voi, dtype=np.int16)
    output_mask = np.asarray(lesion_mask, dtype=bool)

    if output_voi.ndim == 4 and output_voi.shape[3] == 1:
        output_voi = output_voi[..., 0]
    if output_mask.ndim == 4 and output_mask.shape[3] == 1:
        output_mask = output_mask[..., 0]

    if output_voi.ndim == 3:
        if output_mask.ndim != 3 or output_mask.shape != output_voi.shape:
            raise InputValidationError(
                "LesionMask and VOI must be matching 3-D arrays for one series."
            )
        channel_count = 1
    elif output_voi.ndim == 4:
        channel_count = int(output_voi.shape[3])
        if channel_count < 2:
            raise InputValidationError("A 4-D VOI must contain at least two channels.")
        if output_mask.ndim == 3:
            if output_mask.shape != output_voi.shape[:3]:
                raise InputValidationError(
                    "The 3-D LesionMask must match the VOI spatial dimensions."
                )
            output_mask = np.repeat(output_mask[..., np.newaxis], channel_count, axis=3)
        elif output_mask.ndim == 4 and output_mask.shape[3] == 1:
            if output_mask.shape[:3] != output_voi.shape[:3]:
                raise InputValidationError(
                    "LesionMask and VOI spatial dimensions do not match."
                )
            output_mask = np.repeat(output_mask, channel_count, axis=3)
        elif output_mask.ndim != 4 or output_mask.shape != output_voi.shape:
            raise InputValidationError(
                "A multi-series LesionMask must match the 4-D VOI shape."
            )
    else:
        raise InputValidationError("VOI must be a 3-D or 4-D array.")

    mask_channels = (
        (output_mask,)
        if channel_count == 1
        else tuple(output_mask[..., index] for index in range(channel_count))
    )
    for channel, channel_mask in enumerate(mask_channels):
        if not np.any(channel_mask):
            raise InputValidationError(f"The lesion mask is empty in channel {channel}.")
        if np.all(channel_mask):
            raise InputValidationError(
                "The lesion mask fills the entire VOI; background voxels are required."
            )
    return output_mask, output_voi, channel_count


def _required_text_values(
    values: Sequence[str],
    count: int,
    name: str,
) -> np.ndarray:
    if len(values) != count:
        raise InputValidationError(
            f"{name} contains {len(values)} value(s); expected {count}."
        )
    return np.asarray([str(value) for value in values])


def _required_headers(
    headers: Sequence[Mapping[str, Any]],
    count: int,
) -> np.ndarray:
    if len(headers) != count:
        raise InputValidationError(
            f"dicom_headers contains {len(headers)} value(s); expected {count}."
        )
    return np.asarray(
        [json.dumps(header, ensure_ascii=False) for header in headers]
    )


def _channel_statistics(
    voi: np.ndarray,
    mask: np.ndarray,
    channel_count: int,
) -> dict[str, np.ndarray]:
    if channel_count == 1:
        volume_channels = (voi,)
        mask_channels = (mask,)
    else:
        volume_channels = tuple(voi[..., index] for index in range(channel_count))
        mask_channels = tuple(mask[..., index] for index in range(channel_count))

    means: list[float] = []
    maxima: list[float] = []
    minima: list[float] = []
    medians: list[float] = []
    sigmas: list[float] = []
    variances: list[float] = []
    for volume, channel_mask in zip(volume_channels, mask_channels, strict=True):
        values = np.asarray(volume[channel_mask], dtype=np.float64)
        means.append(float(np.mean(values)))
        maxima.append(float(np.max(values)))
        minima.append(float(np.min(values)))
        medians.append(float(np.median(values)))
        sigma = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        sigmas.append(sigma)
        variances.append(sigma**2)
    return {
        "LesionMeanHUByChannel": np.asarray(means, dtype=np.float64),
        "LesionMaxHUByChannel": np.asarray(maxima, dtype=np.float64),
        "LesionMinHUByChannel": np.asarray(minima, dtype=np.float64),
        "LesionMedianHUByChannel": np.asarray(medians, dtype=np.float64),
        "LesionSigmaByChannel": np.asarray(sigmas, dtype=np.float64),
        "LesionVarianceByChannel": np.asarray(variances, dtype=np.float64),
    }


def record_arrays(record: LesionNpzRecord) -> dict[str, np.ndarray]:
    mask, voi, channel_count = _normalise_volumes(record.lesion_mask, record.voi)

    arrays = {
        "PatientName": np.asarray(record.patient_name),
        "LesionNumber": np.asarray(record.lesion_number, dtype=np.int64),
        "Org_case_path": np.asarray(record.org_case_path),
        "Org_mask": np.asarray(record.org_mask),
        "Org_dcm_path": np.asarray(record.org_dcm_path),
        "DicomHeaderJSON": np.asarray(
            json.dumps(record.dicom_header, ensure_ascii=False)
        ),
        "LesionMask": mask,
        "VOI": voi,
        "org_slice_rng": np.asarray(record.org_slice_rng, dtype=np.int64),
        "org_col_rng": np.asarray(record.org_col_rng, dtype=np.int64),
        "LesionCenter": np.asarray(record.lesion_center, dtype=np.int64),
        "org_row_rng": np.asarray(record.org_row_rng, dtype=np.int64),
        "LesionDiameter": np.asarray(record.lesion_diameter, dtype=np.float64),
        "LesionMeanHU": _float64(record.lesion_mean_hu),
        "LesionRoundness": _float64(record.lesion_roundness),
        "LesionMaxHU": _float64(record.lesion_max_hu),
        "LesionMinHU": _float64(record.lesion_min_hu),
        "LesionMedianHU": _float64(record.lesion_median_hu),
        "LesionSigma": _float64(record.lesion_sigma),
        "LesionVariance": _float64(record.lesion_variance),
        "LesionVoxelCount": np.asarray(record.lesion_voxel_count, dtype=np.int64),
        "LesionPhysicalSize": _float64(record.lesion_physical_size),
    }
    for name, size in (
        ("org_slice_rng", 2),
        ("org_col_rng", 2),
        ("org_row_rng", 2),
        ("LesionCenter", 3),
        ("LesionDiameter", 3),
    ):
        if arrays[name].size != size:
            raise InputValidationError(f"{name} must contain {size} values.")

    if channel_count > 1:
        if not 0 <= int(record.reference_series_index) < channel_count:
            raise InputValidationError(
                "reference_series_index is outside the available series channels."
            )
        arrays.update(
            {
                "SeriesCount": np.asarray(channel_count, dtype=np.int64),
                "ReferenceSeriesIndex": np.asarray(
                    record.reference_series_index, dtype=np.int64
                ),
                "SeriesNames": _required_text_values(
                    record.series_names, channel_count, "series_names"
                ),
                "SeriesUIDs": _required_text_values(
                    record.series_uids, channel_count, "series_uids"
                ),
                "SeriesSourcePaths": _required_text_values(
                    record.series_source_paths,
                    channel_count,
                    "series_source_paths",
                ),
                "DicomHeadersJSON": _required_headers(
                    record.dicom_headers, channel_count
                ),
            }
        )
        arrays.update(_channel_statistics(voi, mask, channel_count))

    return arrays


def write_lesion_npz(record: LesionNpzRecord, output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = record_arrays(record)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".npz",
        dir=str(destination.parent),
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
