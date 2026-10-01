from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import DependencyUnavailableError, InputValidationError


def require_pydicom():
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise DependencyUnavailableError(
            "Lesion extraction requires pydicom. Reinstall the application dependencies."
        ) from exc
    return pydicom


def read_dataset(
    path: str | Path,
    *,
    stop_before_pixels: bool = True,
    specific_tags: list[str] | None = None,
):
    pydicom = require_pydicom()
    try:
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=stop_before_pixels,
            force=True,
            specific_tags=specific_tags,
        )
    except Exception as exc:
        raise InputValidationError(f"Could not read DICOM file {path}: {exc}") from exc


def read_series_summary(path: str | Path) -> dict[str, str]:
    dataset = read_dataset(
        path,
        specific_tags=[
            "PatientID",
            "StudyDescription",
            "SeriesDescription",
            "SeriesNumber",
            "SeriesInstanceUID",
        ],
    )

    def text(name: str) -> str:
        value = getattr(dataset, name, "")
        return "" if value is None else str(value).strip()

    return {
        "patient_id": text("PatientID"),
        "study_description": text("StudyDescription"),
        "series_description": text("SeriesDescription"),
        "series_number": text("SeriesNumber"),
        "series_uid": text("SeriesInstanceUID"),
    }


def _json_value(value: Any) -> Any:
    type_name = type(value).__name__
    if "DSfloat" in type_name or "DSdecimal" in type_name:
        return float(value)
    if "MultiValue" in type_name:
        result: list[Any] = []
        for item in value:
            item_type = type(item).__name__
            if "DSfloat" in item_type or "DSdecimal" in item_type:
                result.append(float(item))
            else:
                result.append(str(item))
        return result
    return str(value)


def read_dicom_header(path: str | Path) -> dict[str, Any]:
    """Serialize the non-private header using the legacy NPZ field convention."""

    dataset = read_dataset(path, stop_before_pixels=True)
    header: dict[str, Any] = {}
    for element in dataset:
        name = str(element.name)
        if element.tag.is_private or "Private" in name:
            continue
        if name == "Pixel Data" or "CSA Image" in name:
            continue
        field = "".join(name[:31].split())
        if not field:
            continue
        try:
            header[field] = _json_value(element.value)
        except Exception:
            header[field] = str(element.value)
    return header


def numeric_first(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, bytes)):
        try:
            iterator = iter(value)
        except TypeError:
            pass
        else:
            try:
                value = next(iterator)
            except StopIteration:
                return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
