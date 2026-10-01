from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from ..domain.models import (
    DicomSeriesCandidate,
    DicomVolume,
    RawDatasetInfo,
    SeriesGeometry,
    ValidationError,
    ValidationIssue,
)


def _pydicom():
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required to load medical images.") from exc
    return pydicom


def _candidate_paths(source: str | Path) -> Iterable[Path]:
    root = Path(source).expanduser()
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        raise FileNotFoundError(f"DICOM input does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _text(dataset: Any, name: str) -> str:
    value = getattr(dataset, name, "")
    return str(value).strip() if value is not None else ""


def _float_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"DICOM {name} is not numeric.") from exc
    if vector.size != length or not np.all(np.isfinite(vector)):
        raise ValidationError(f"DICOM {name} must contain {length} finite values.")
    return vector


def _patient_digest(dataset: Any) -> str:
    patient_id = _text(dataset, "PatientID")
    if not patient_id:
        return ""
    return hashlib.sha256(patient_id.encode("utf-8", errors="replace")).hexdigest()


def _series_coordinate(dataset: Any, normal: np.ndarray) -> float:
    if hasattr(dataset, "ImagePositionPatient"):
        position = _float_vector(
            dataset.ImagePositionPatient, length=3, name="ImagePositionPatient"
        )
        return float(np.dot(position, normal))
    if hasattr(dataset, "SliceLocation"):
        return float(dataset.SliceLocation)
    return float(getattr(dataset, "InstanceNumber", 0))


def _orientation(dataset: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(dataset, "ImageOrientationPatient"):
        raise ValidationError(
            "A reconstruction slice is missing ImageOrientationPatient."
        )
    orientation = _float_vector(
        dataset.ImageOrientationPatient, length=6, name="ImageOrientationPatient"
    )
    column_axis = orientation[:3]
    row_axis = orientation[3:]
    normal = np.cross(column_axis, row_axis)
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-8:
        raise ValidationError("ImageOrientationPatient contains degenerate axes.")
    return column_axis, row_axis, normal / normal_length


def _warning_for_spacing(coordinates: np.ndarray) -> tuple[str, ...]:
    if coordinates.size < 2:
        return ("The reconstruction contains only one slice.",)
    distances = np.abs(np.diff(coordinates))
    if np.any(distances <= 1e-5):
        raise ValidationError("The reconstruction contains duplicate slice planes.")
    median = float(np.median(distances))
    warnings: list[str] = []
    if median <= 0:
        raise ValidationError("The reconstruction has invalid slice spacing.")
    relative = np.abs(distances - median) / median
    if np.any(relative > 0.02):
        warnings.append(
            "Slice spacing is non-uniform. Patient-coordinate placement uses the actual "
            "ImagePositionPatient values; final Alpha mapping requires validation."
        )
    if np.any(distances > median * 1.5):
        warnings.append("One or more likely missing-slice gaps were detected.")
    return tuple(warnings)


_SURROGATE_SERIES_PREFIX = "missing-series-uid:"
_METADATA_GROUPING_METHOD = "metadata_fingerprint_v1"
_RECONSTRUCTION_DISCOVERY_TAGS = (
    "SeriesInstanceUID",
    "StudyInstanceUID",
    "FrameOfReferenceUID",
    "SeriesDescription",
    "SeriesNumber",
    "AcquisitionNumber",
    "TemporalPositionIdentifier",
    "ImageType",
    "ConvolutionKernel",
    "PatientPosition",
    "Modality",
    "SOPClassUID",
    "Rows",
    "Columns",
    "SOPInstanceUID",
    "NumberOfFrames",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "PixelSpacing",
)


def _input_root(source: str | Path) -> Path:
    resolved = Path(source).expanduser().resolve()
    return resolved.parent if resolved.is_file() else resolved


def _fingerprint_vector(value: Any) -> str:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return str(value).strip()
    if not np.all(np.isfinite(vector)):
        return str(value).strip()
    return ",".join(f"{float(item):.8g}" for item in vector)


def _relative_parent(path: Path, root: Path) -> str:
    parent = path.expanduser().resolve().parent
    try:
        return parent.relative_to(root).as_posix() or "."
    except ValueError:
        return parent.as_posix()


def _fallback_series_key(dataset: Any, path: Path, root: Path) -> str:
    "Build a stable application key without fabricating a DICOM UID."

    components = (
        _text(dataset, "StudyInstanceUID"),
        _text(dataset, "FrameOfReferenceUID"),
        _text(dataset, "Modality"),
        _text(dataset, "SOPClassUID"),
        _text(dataset, "SeriesNumber"),
        _text(dataset, "AcquisitionNumber"),
        _text(dataset, "TemporalPositionIdentifier"),
        _text(dataset, "ImageType"),
        _text(dataset, "ConvolutionKernel"),
        _text(dataset, "PatientPosition").upper(),
        str(getattr(dataset, "Rows", "")),
        str(getattr(dataset, "Columns", "")),
        _fingerprint_vector(getattr(dataset, "PixelSpacing", ())),
        _fingerprint_vector(getattr(dataset, "ImageOrientationPatient", ())),
        _relative_parent(path, root),
    )
    digest = hashlib.sha256(
        "\x1f".join(components).encode("utf-8", errors="replace")
    ).hexdigest()[:20]
    return f"{_SURROGATE_SERIES_PREFIX}{digest}"


def _series_grouping_method(selection_key: str) -> str:
    if selection_key.startswith(_SURROGATE_SERIES_PREFIX):
        return _METADATA_GROUPING_METHOD
    return "series_instance_uid"


def _is_reconstruction_header(dataset: Any) -> bool:
    return (
        hasattr(dataset, "Rows")
        and hasattr(dataset, "Columns")
        and hasattr(dataset, "ImagePositionPatient")
        and hasattr(dataset, "ImageOrientationPatient")
        and hasattr(dataset, "PixelSpacing")
    )


def _group_reconstruction_headers(
    source: str | Path,
    cancel_event: Event | None,
) -> tuple[
    dict[str, list[tuple[Path, Any]]],
    dict[str, int],
    int,
]:
    pydicom = _pydicom()
    root = _input_root(source)
    groups: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    real_keys_by_fallback: dict[str, set[str]] = defaultdict(set)
    errors = 0

    for path in _candidate_paths(source):
        if cancel_event and cancel_event.is_set():
            break
        try:
            dataset = pydicom.dcmread(
                path,
                stop_before_pixels=True,
                specific_tags=list(_RECONSTRUCTION_DISCOVERY_TAGS),
            )
        except Exception:
            errors += 1
            continue
        if not _is_reconstruction_header(dataset):
            continue
        source_uid = _text(dataset, "SeriesInstanceUID")
        fallback_key = _fallback_series_key(dataset, path, root)
        selection_key = source_uid or fallback_key
        groups[selection_key].append((path, dataset))
        if source_uid:
            real_keys_by_fallback[fallback_key].add(source_uid)

    attached_missing_counts: dict[str, int] = defaultdict(int)
    surrogate_keys = [key for key in groups if key.startswith(_SURROGATE_SERIES_PREFIX)]
    for surrogate_key in surrogate_keys:
        compatible = real_keys_by_fallback.get(surrogate_key, set())
        if len(compatible) != 1:
            continue
        target_key = next(iter(compatible))
        missing_records = groups.pop(surrogate_key)
        groups[target_key].extend(missing_records)
        attached_missing_counts[target_key] += len(missing_records)

    for records in groups.values():
        records.sort(key=lambda item: str(item[0]))
    return dict(groups), dict(attached_missing_counts), errors


class PydicomLesionDicomService:
    """Discover, validate, load, and associate reconstruction and CTPD data."""

    def discover_reconstruction_series(
        self,
        source: str | Path,
        cancel_event: Event | None = None,
    ) -> list[DicomSeriesCandidate]:
        groups, attached_missing_counts, errors = _group_reconstruction_headers(
            source, cancel_event
        )
        candidates: list[DicomSeriesCandidate] = []
        for selection_key, records in groups.items():
            first = records[0][1]
            source_uid = (
                ""
                if selection_key.startswith(_SURROGATE_SERIES_PREFIX)
                else selection_key
            )
            rows = int(first.Rows)
            columns = int(first.Columns)
            warnings: list[str] = []
            missing_uid_count = sum(
                not bool(_text(item, "SeriesInstanceUID")) for _, item in records
            )
            if selection_key.startswith(_SURROGATE_SERIES_PREFIX):
                warnings.append(
                    "SeriesInstanceUID (0020,000E) is missing. This "
                    "input was grouped using directory and "
                    "non-identifying image metadata; source files "
                    "will not be modified."
                )
            elif attached_missing_counts.get(selection_key, 0):
                count = attached_missing_counts[selection_key]
                warnings.append(
                    f"{count} image(s) without SeriesInstanceUID were "
                    "attached to this UID-bearing series because "
                    "exactly one metadata-compatible series was found."
                )
            sop_uids = [_text(item, "SOPInstanceUID") for _, item in records]
            populated = [value for value in sop_uids if value]
            if len(populated) != len(set(populated)):
                warnings.append("Duplicate SOP Instance UIDs were found.")
            if any(
                int(getattr(item, "NumberOfFrames", 1) or 1) > 1 for _, item in records
            ):
                warnings.append(
                    "Enhanced/multi-frame reconstructed images are not "
                    "currently supported."
                )
            if any(
                int(item.Rows) != rows or int(item.Columns) != columns
                for _, item in records
            ):
                warnings.append("Image matrix dimensions vary within the series.")
            candidates.append(
                DicomSeriesCandidate(
                    series_instance_uid=source_uid,
                    study_instance_uid=_text(first, "StudyInstanceUID"),
                    frame_of_reference_uid=_text(first, "FrameOfReferenceUID"),
                    description=_text(first, "SeriesDescription"),
                    modality=_text(first, "Modality"),
                    instance_count=len(records),
                    rows=rows,
                    columns=columns,
                    source_root=str(Path(source).expanduser().resolve()),
                    warnings=tuple(warnings),
                    selection_key=selection_key,
                    uses_surrogate_key=selection_key.startswith(
                        _SURROGATE_SERIES_PREFIX
                    ),
                    source_paths=tuple(
                        str(path.expanduser().resolve()) for path, _item in records
                    ),
                    missing_uid_instance_count=missing_uid_count,
                )
            )
        candidates.sort(
            key=lambda item: (
                item.modality != "CT",
                item.description.casefold(),
                item.effective_selection_key,
            )
        )
        if not candidates:
            detail = f" ({errors} unreadable files skipped)" if errors else ""
            raise ValidationError(
                f"No reconstructed DICOM image series were found" f"{detail}."
            )
        return candidates

    def _records_for_series(
        self,
        source: str | Path,
        series_key: str,
        cancel_event: Event | None,
    ) -> list[tuple[Path, Any]]:
        pydicom = _pydicom()
        groups, _attached, _errors = _group_reconstruction_headers(source, cancel_event)
        header_records = groups.get(series_key, [])
        records: list[tuple[Path, Any]] = []
        for path, _header in header_records:
            if cancel_event and cancel_event.is_set():
                break
            try:
                dataset = pydicom.dcmread(path)
            except Exception:
                continue
            if int(getattr(dataset, "NumberOfFrames", 1) or 1) != 1:
                raise ValidationError(
                    "Multi-frame reconstructed DICOM is not supported "
                    "by this module."
                )
            records.append((path, dataset))
        if not records:
            raise ValidationError(
                f"Series selection {series_key!r} was not found in "
                "the selected input."
            )
        return records

    def load_reconstruction(
        self,
        source: str | Path,
        series_key: str,
        cancel_event: Event | None = None,
    ) -> DicomVolume:
        records = self._records_for_series(source, series_key, cancel_event)
        first = records[0][1]
        column_axis, row_axis, normal = _orientation(first)
        records.sort(
            key=lambda item: (
                _series_coordinate(item[1], normal),
                int(getattr(item[1], "InstanceNumber", 0)),
                str(item[0]),
            )
        )

        rows = int(first.Rows)
        columns = int(first.Columns)
        pixel_spacing = _float_vector(
            getattr(first, "PixelSpacing", None), length=2, name="PixelSpacing"
        )
        expected_orientation = np.concatenate([column_axis, row_axis])
        seen_sop_uids: set[str] = set()
        positions: list[tuple[float, float, float]] = []
        planes: list[np.ndarray] = []
        paths: list[str] = []
        warnings: list[str] = []
        source_series_uids = {
            value
            for _path, dataset in records
            if (value := _text(dataset, "SeriesInstanceUID"))
        }
        if len(source_series_uids) > 1:
            raise ValidationError(
                "The selected reconstruction group contains conflicting "
                "SeriesInstanceUID values."
            )
        source_series_uid = next(iter(source_series_uids), "")
        missing_series_uid_count = sum(
            not bool(_text(dataset, "SeriesInstanceUID")) for _path, dataset in records
        )
        if missing_series_uid_count:
            if source_series_uid:
                warnings.append(
                    f"{missing_series_uid_count} image(s) are missing "
                    "SeriesInstanceUID and were attached to the uniquely "
                    "compatible UID-bearing series."
                )
            else:
                warnings.append(
                    "SeriesInstanceUID (0020,000E) is missing. The "
                    "reconstruction was selected using a deterministic "
                    "metadata fingerprint; source files were not modified."
                )

        for path, dataset in records:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Reconstruction loading was cancelled.")
            sop_uid = _text(dataset, "SOPInstanceUID")
            if sop_uid and sop_uid in seen_sop_uids:
                raise ValidationError(
                    "Duplicate SOP Instance UIDs were found in the series."
                )
            seen_sop_uids.add(sop_uid)
            if int(dataset.Rows) != rows or int(dataset.Columns) != columns:
                raise ValidationError(
                    f"Image matrix dimensions vary within the series: {path.name}"
                )
            spacing = _float_vector(
                getattr(dataset, "PixelSpacing", None), length=2, name="PixelSpacing"
            )
            if not np.allclose(spacing, pixel_spacing, atol=1e-6):
                raise ValidationError(
                    f"PixelSpacing varies within the series: {path.name}"
                )
            orientation = _float_vector(
                getattr(dataset, "ImageOrientationPatient", None),
                length=6,
                name="ImageOrientationPatient",
            )
            if not np.allclose(orientation, expected_orientation, atol=1e-5):
                raise ValidationError(
                    f"ImageOrientationPatient varies within the series: {path.name}"
                )
            position = _float_vector(
                getattr(dataset, "ImagePositionPatient", None),
                length=3,
                name="ImagePositionPatient",
            )
            transfer_syntax = getattr(
                getattr(dataset, "file_meta", None),
                "TransferSyntaxUID",
                None,
            )
            if transfer_syntax is not None and getattr(
                transfer_syntax, "is_compressed", False
            ):
                warnings.append(
                    "The reconstruction uses compressed Pixel Data. Decoding depends on an "
                    "installed pydicom pixel-data handler."
                )
            try:
                stored = np.asarray(dataset.pixel_array)
            except Exception as exc:
                raise ValidationError(
                    f"Could not decode reconstruction Pixel Data in {path.name}: {exc}"
                ) from exc
            if stored.shape != (rows, columns):
                raise ValidationError(
                    f"Decoded image shape {stored.shape} does not match "
                    f"{(rows, columns)}: {path.name}"
                )
            slope = float(getattr(dataset, "RescaleSlope", 1.0))
            intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
            planes.append(stored.astype(np.float32) * slope + intercept)
            positions.append(tuple(float(value) for value in position))
            paths.append(str(path.resolve()))

        coordinates = np.asarray(
            [np.dot(np.asarray(position), normal) for position in positions],
            dtype=float,
        )
        warnings.extend(_warning_for_spacing(coordinates))
        if len(set(paths)) != len(paths):
            raise ValidationError(
                "The same reconstruction file was included more than once."
            )

        if not hasattr(first, "ReconstructionDiameter"):
            warnings.append(
                "ReconstructionDiameter is missing; the legacy Alpha conversion will use "
                "Columns × column spacing, matching pipeline 3.1.4."
            )
        diameter = float(
            getattr(first, "ReconstructionDiameter", columns * float(pixel_spacing[1]))
        )
        if not hasattr(first, "ReconstructionTargetCenterPatient"):
            warnings.append(
                "ReconstructionTargetCenterPatient is missing; the legacy Alpha conversion "
                "will use (0, 0, 0), matching pipeline 3.1.4."
            )
        if not hasattr(first, "DataCollectionCenterPatient"):
            warnings.append(
                "DataCollectionCenterPatient is missing; the legacy Alpha conversion will "
                "use (0, 0, 0), matching pipeline 3.1.4."
            )
        recon_center = tuple(
            float(value)
            for value in _float_vector(
                getattr(first, "ReconstructionTargetCenterPatient", [0.0, 0.0, 0.0]),
                length=3,
                name="ReconstructionTargetCenterPatient",
            )
        )
        data_center = tuple(
            float(value)
            for value in _float_vector(
                getattr(first, "DataCollectionCenterPatient", [0.0, 0.0, 0.0]),
                length=3,
                name="DataCollectionCenterPatient",
            )
        )
        geometry = SeriesGeometry(
            series_instance_uid=source_series_uid,
            study_instance_uid=_text(first, "StudyInstanceUID"),
            frame_of_reference_uid=_text(first, "FrameOfReferenceUID"),
            rows=rows,
            columns=columns,
            image_positions_lps_mm=tuple(positions),
            column_axis_lps=tuple(float(value) for value in column_axis),
            row_axis_lps=tuple(float(value) for value in row_axis),
            row_spacing_mm=float(pixel_spacing[0]),
            column_spacing_mm=float(pixel_spacing[1]),
            patient_position=_text(first, "PatientPosition").upper(),
            reconstruction_diameter_mm=diameter,
            reconstruction_target_center_lps_mm=recon_center,
            data_collection_center_lps_mm=data_center,
            warnings=tuple(dict.fromkeys(warnings)),
            patient_identity_digest=_patient_digest(first),
            series_selection_key=series_key,
            series_grouping_method=_series_grouping_method(series_key),
            series_missing_uid_instance_count=(missing_series_uid_count),
        )
        volume = DicomVolume(
            geometry=geometry,
            hu=np.stack(planes, axis=0),
            source_paths=tuple(paths),
            headers=tuple(dataset for _, dataset in records),
        )
        volume.validate()
        return volume

    def inspect_raw(self, source: str | Path) -> RawDatasetInfo:
        try:
            from .._vendor.lesion_pipeline.ctpd import discover_projection_records
        except ImportError as exc:
            raise RuntimeError("The bundled lesion pipeline is unavailable.") from exc

        pydicom = _pydicom()
        records = discover_projection_records(source)
        missing_series_uid_count = sum(
            not str(record.series_instance_uid).strip() for record in records
        )
        if missing_series_uid_count:
            raise ValidationError(
                "The DICOM-CT-PD input is missing SeriesInstanceUID "
                f"(0020,000E) in {missing_series_uid_count} projection "
                "file(s). Final processing would preserve non-conformant "
                "output metadata; repair the source metadata before "
                "lesion insertion."
            )
        study_uids: set[str] = set()
        series_uids: set[str] = set()
        frame_uids: set[str] = set()
        positions: set[str] = set()
        spectra: set[int] = set()
        sources: set[int] = set()
        patient_digests: set[str] = set()
        warnings: list[str] = []
        visited: set[Path] = set()
        for record in records:
            spectra.update(record.spectrum_indices)
            sources.update(record.source_indices)
            series_uids.add(record.series_instance_uid)
            if record.path in visited:
                continue
            visited.add(record.path)
            try:
                dataset = pydicom.dcmread(record.path, stop_before_pixels=True)
            except Exception as exc:
                raise ValidationError(
                    f"Could not reread DICOM-CT-PD header {record.relative_path}: {exc}"
                ) from exc
            if value := _text(dataset, "StudyInstanceUID"):
                study_uids.add(value)
            if value := _text(dataset, "FrameOfReferenceUID"):
                frame_uids.add(value)
            if value := _text(dataset, "PatientPosition").upper():
                positions.add(value)
            if digest := _patient_digest(dataset):
                patient_digests.add(digest)
        if len(patient_digests) > 1:
            raise ValidationError(
                "The selected DICOM-CT-PD input contains records from multiple patients."
            )
        if len(positions) > 1:
            raise ValidationError(
                "The selected DICOM-CT-PD input contains multiple PatientPosition values."
            )
        if len(frame_uids) > 1:
            raise ValidationError(
                "The selected DICOM-CT-PD input contains multiple FrameOfReferenceUID values."
            )
        if len(study_uids) > 1:
            warnings.append("Multiple StudyInstanceUID values occur in the CTPD input.")
        return RawDatasetInfo(
            source=str(Path(source).expanduser().resolve()),
            study_instance_uids=tuple(sorted(study_uids)),
            series_instance_uids=tuple(sorted(series_uids)),
            frame_of_reference_uids=tuple(sorted(frame_uids)),
            patient_positions=tuple(sorted(positions)),
            spectrum_indices=tuple(sorted(spectra)),
            source_indices=tuple(sorted(sources)),
            projection_file_count=len(records),
            projection_frame_count=sum(record.frame_count for record in records),
            warnings=tuple(warnings),
            patient_identity_digest=next(iter(patient_digests), ""),
        )

    def validate_association(
        self,
        volume: DicomVolume,
        raw: RawDatasetInfo,
    ) -> list[ValidationIssue]:
        geometry = volume.geometry
        issues: list[ValidationIssue] = []
        matched_identifier = False
        if geometry.patient_identity_digest and raw.patient_identity_digest:
            if geometry.patient_identity_digest != raw.patient_identity_digest:
                issues.append(
                    ValidationIssue(
                        "error",
                        "patient-mismatch",
                        "The reconstruction and DICOM-CT-PD PatientID values do not match.",
                    )
                )
            else:
                matched_identifier = True
        if geometry.frame_of_reference_uid and raw.frame_of_reference_uids:
            if geometry.frame_of_reference_uid not in raw.frame_of_reference_uids:
                issues.append(
                    ValidationIssue(
                        "error",
                        "frame-of-reference-mismatch",
                        "The reconstruction FrameOfReferenceUID does not match the CTPD data.",
                    )
                )
            else:
                matched_identifier = True
        if geometry.study_instance_uid and raw.study_instance_uids:
            if geometry.study_instance_uid not in raw.study_instance_uids:
                issues.append(
                    ValidationIssue(
                        "warning",
                        "study-mismatch",
                        "StudyInstanceUID differs between reconstruction and CTPD data. "
                        "Derived reconstructions sometimes receive a new Study UID; "
                        "verify the case manually.",
                    )
                )
            else:
                matched_identifier = True
        if (
            geometry.patient_position
            and raw.patient_positions
            and geometry.patient_position not in raw.patient_positions
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "patient-position-mismatch",
                    "PatientPosition differs between reconstruction and CTPD data.",
                )
            )
        if not matched_identifier and not any(
            issue.severity == "error" for issue in issues
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "association-unconfirmed",
                    "No shared patient, study, or frame-of-reference identifier was available to "
                    "associate the reconstruction with the CTPD data; PatientPosition alone "
                    "is not sufficient to establish case identity.",
                )
            )
        for warning in (*geometry.warnings, *raw.warnings):
            issues.append(ValidationIssue("warning", "dicom-warning", warning))
        return issues


def local_background_hu(
    volume: DicomVolume,
    center_crs: tuple[float, float, float],
    radius_pixels: int = 3,
) -> float:
    column, row, slice_index = center_crs
    z = int(round(slice_index))
    y = int(round(row))
    x = int(round(column))
    z = max(0, min(z, volume.hu.shape[0] - 1))
    y0, y1 = max(0, y - radius_pixels), min(volume.hu.shape[1], y + radius_pixels + 1)
    x0, x1 = max(0, x - radius_pixels), min(volume.hu.shape[2], x + radius_pixels + 1)
    values = volume.hu[z, y0:y1, x0:x1]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValidationError(
            "The selected background region contains no finite pixels."
        )
    return float(np.mean(finite))
