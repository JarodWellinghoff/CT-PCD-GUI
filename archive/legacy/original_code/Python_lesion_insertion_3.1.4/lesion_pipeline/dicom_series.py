from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset


DICOM_SUFFIXES = {".dcm", ".ima", ""}


@dataclass
class ReconSeries:
    paths: list[Path]
    headers: list[Dataset]
    hu: np.ndarray  # (slice, row, column)
    slice_locations: np.ndarray
    row_spacing_mm: float
    column_spacing_mm: float
    slice_spacing_mm: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.hu.shape)

    @property
    def first_header(self) -> Dataset:
        return self.headers[0]


def iter_candidate_files(root: str | Path) -> Iterable[Path]:
    source = Path(root)
    if source.is_file():
        yield source
        return
    for path in sorted(source.rglob("*")):
        if path.is_file() and path.suffix.lower() in DICOM_SUFFIXES:
            yield path


def _slice_coordinate(ds: Dataset) -> float:
    if "ImagePositionPatient" in ds and "ImageOrientationPatient" in ds:
        orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orientation[:3], orientation[3:])
        return float(np.dot(np.asarray(ds.ImagePositionPatient, dtype=float), normal))
    if "SliceLocation" in ds:
        return float(ds.SliceLocation)
    return float(getattr(ds, "InstanceNumber", 0))


def load_recon_series(directory: str | Path) -> ReconSeries:
    """Load and physically sort one reconstructed CT series."""

    records: list[tuple[float, int, Path, Dataset]] = []
    for path in iter_candidate_files(directory):
        try:
            ds = pydicom.dcmread(path)
        except Exception:
            continue
        if "PixelData" not in ds or not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        records.append(
            (
                _slice_coordinate(ds),
                int(getattr(ds, "InstanceNumber", 0)),
                path,
                ds,
            )
        )
    if not records:
        raise ValueError(f"No readable reconstructed DICOM images found in {directory}")

    series_uids = {
        str(getattr(item[3], "SeriesInstanceUID", "")) for item in records
    }
    if len(series_uids) > 1:
        raise ValueError(
            f"Reconstructed input {directory} contains {len(series_uids)} DICOM series; "
            "select a directory containing only one series."
        )

    records.sort(key=lambda item: (item[0], item[1], str(item[2])))
    paths = [item[2] for item in records]
    headers = [item[3] for item in records]
    coordinates = np.asarray([item[0] for item in records], dtype=float)

    first = headers[0]
    expected = (int(first.Rows), int(first.Columns))
    spacing = np.asarray(getattr(first, "PixelSpacing", [1.0, 1.0]), dtype=float)
    planes: list[np.ndarray] = []
    for path, ds in zip(paths, headers, strict=True):
        if (int(ds.Rows), int(ds.Columns)) != expected:
            raise ValueError(f"Inconsistent image matrix in reconstructed series: {path}")
        if not np.allclose(
            np.asarray(getattr(ds, "PixelSpacing", spacing), dtype=float),
            spacing,
            atol=1e-6,
        ):
            raise ValueError(f"Inconsistent PixelSpacing in reconstructed series: {path}")
        if "ImageOrientationPatient" in first and not np.allclose(
            np.asarray(ds.ImageOrientationPatient, dtype=float),
            np.asarray(first.ImageOrientationPatient, dtype=float),
            atol=1e-6,
        ):
            raise ValueError(
                f"Inconsistent ImageOrientationPatient in reconstructed series: {path}"
            )
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        planes.append(ds.pixel_array.astype(np.float32) * slope + intercept)

    if len(coordinates) > 1:
        differences = np.abs(np.diff(coordinates))
        differences = differences[differences > 1e-6]
        slice_spacing = float(np.median(differences)) if differences.size else float(
            getattr(first, "SliceThickness", 1.0)
        )
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))

    return ReconSeries(
        paths=paths,
        headers=headers,
        hu=np.stack(planes, axis=0),
        slice_locations=coordinates,
        row_spacing_mm=float(spacing[0]),
        column_spacing_mm=float(spacing[1]),
        slice_spacing_mm=slice_spacing,
    )


def validate_aligned_series(first: ReconSeries, second: ReconSeries) -> None:
    if first.shape != second.shape:
        raise ValueError(f"T1/T2 shapes differ: {first.shape} versus {second.shape}")
    if not np.allclose(first.slice_locations, second.slice_locations, atol=1e-3):
        raise ValueError("T1/T2 slice positions do not match.")
    if not np.allclose(
        [first.row_spacing_mm, first.column_spacing_mm],
        [second.row_spacing_mm, second.column_spacing_mm],
        atol=1e-6,
    ):
        raise ValueError("T1/T2 pixel spacing does not match.")
    first_patient = str(getattr(first.first_header, "PatientID", ""))
    second_patient = str(getattr(second.first_header, "PatientID", ""))
    if first_patient and second_patient and first_patient != second_patient:
        raise ValueError("T1/T2 PatientID values do not match.")
    if "ImageOrientationPatient" in first.first_header and not np.allclose(
        np.asarray(first.first_header.ImageOrientationPatient, dtype=float),
        np.asarray(second.first_header.ImageOrientationPatient, dtype=float),
        atol=1e-6,
    ):
        raise ValueError("T1/T2 image orientations do not match.")
    for first_header, second_header in zip(
        first.headers, second.headers, strict=True
    ):
        if "ImagePositionPatient" in first_header and not np.allclose(
            np.asarray(first_header.ImagePositionPatient, dtype=float),
            np.asarray(second_header.ImagePositionPatient, dtype=float),
            atol=1e-3,
        ):
            raise ValueError("T1/T2 ImagePositionPatient values do not match.")
