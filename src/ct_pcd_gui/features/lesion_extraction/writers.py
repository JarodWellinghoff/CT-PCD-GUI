from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .models import InputValidationError, LesionNpzRecord


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


def _float64(value: float | int | None) -> np.ndarray:
    return np.asarray(np.nan if value is None else value, dtype=np.float64)


def record_arrays(record: LesionNpzRecord) -> dict[str, np.ndarray]:
    mask = np.asarray(record.lesion_mask, dtype=bool)
    voi = np.asarray(record.voi, dtype=np.int16)
    if mask.ndim != 3 or voi.ndim != 3 or mask.shape != voi.shape:
        raise InputValidationError("LesionMask and VOI must be matching 3-D arrays.")
    if not np.any(mask):
        raise InputValidationError("The lesion mask is empty.")
    if np.all(mask):
        raise InputValidationError(
            "The lesion mask fills the entire VOI; background voxels are required."
        )

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
