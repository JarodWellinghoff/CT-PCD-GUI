from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .dicom_metadata import read_series_summary
from .models import (
    DependencyUnavailableError,
    DicomSeriesSpec,
    InputValidationError,
    SegmentDefinition,
    SegmentationInfo,
)


_SEGMENT_KEY = re.compile(r"^(Segment\d+)_(.+)$", re.IGNORECASE)


def require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise DependencyUnavailableError(
            "Lesion extraction requires SimpleITK. Install the optional dependency "
            "with: pip install -e \".[lesion-extraction]\""
        ) from exc
    return sitk


def require_nrrd():
    try:
        import nrrd
    except ImportError as exc:  # pragma: no cover - installation dependent
        raise DependencyUnavailableError(
            "Lesion extraction requires pynrrd. Reinstall the application dependencies."
        ) from exc
    return nrrd


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(float(_text(value).strip()))
    except (TypeError, ValueError):
        return default


def _parse_color(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    try:
        values = [float(item) for item in _text(value).replace(",", " ").split()]
    except ValueError:
        return None
    if len(values) < 3:
        return None
    rgb = values[:3]
    if max(rgb) <= 1.0:
        rgb = [item * 255.0 for item in rgb]
    return tuple(int(np.clip(round(item), 0, 255)) for item in rgb)  # type: ignore[return-value]


def parse_segment_definitions(header: Mapping[str, Any]) -> tuple[SegmentDefinition, ...]:
    records: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    for raw_key, value in header.items():
        match = _SEGMENT_KEY.match(_text(raw_key))
        if not match:
            continue
        prefix = match.group(1)
        normalized = prefix.casefold()
        names.setdefault(normalized, prefix)
        records.setdefault(normalized, {})[match.group(2).casefold()] = value

    def sort_key(key: str) -> tuple[int, str]:
        match = re.search(r"(\d+)$", names[key])
        return (int(match.group(1)) if match else 2**31 - 1, names[key].casefold())

    segments: list[SegmentDefinition] = []
    for normalized in sorted(records, key=sort_key):
        fields = records[normalized]
        if "labelvalue" not in fields:
            continue
        label_value = _integer(fields["labelvalue"])
        if label_value <= 0:
            continue
        key = names[normalized]
        segments.append(
            SegmentDefinition(
                key=key,
                name=_text(fields.get("name", key)).strip() or key,
                label_value=label_value,
                layer=max(0, _integer(fields.get("layer", 0))),
                color_rgb=_parse_color(fields.get("color")),
            )
        )
    return tuple(segments)


def _read_segmentation(path: str | Path) -> tuple[Any, Mapping[str, Any]]:
    sitk = require_sitk()
    nrrd = require_nrrd()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise InputValidationError(f"NRRD segmentation does not exist: {source}")
    try:
        image = sitk.ReadImage(str(source))
    except Exception as exc:
        raise InputValidationError(f"Could not read NRRD segmentation {source}: {exc}") from exc
    if image.GetDimension() != 3:
        raise InputValidationError(
            f"NRRD segmentation is {image.GetDimension()}-D; expected a 3-D label map."
        )
    try:
        header = nrrd.read_header(str(source))
    except Exception as exc:
        raise InputValidationError(f"Could not read NRRD metadata {source}: {exc}") from exc
    return image, header


def _fallback_segments(image: Any) -> tuple[SegmentDefinition, ...]:
    sitk = require_sitk()
    components = int(image.GetNumberOfComponentsPerPixel())
    result: list[SegmentDefinition] = []
    if components > 1:
        for layer in range(components):
            scalar = sitk.VectorIndexSelectionCast(image, layer, sitk.sitkUInt16)
            for value in np.unique(sitk.GetArrayViewFromImage(scalar)):
                label = int(value)
                if label:
                    result.append(
                        SegmentDefinition(
                            key=f"Layer{layer}Label{label}",
                            name=f"Layer {layer + 1}, label {label}",
                            label_value=label,
                            layer=layer,
                        )
                    )
    else:
        for value in np.unique(sitk.GetArrayViewFromImage(image)):
            label = int(value)
            if label:
                result.append(
                    SegmentDefinition(
                        key=f"Label{label}",
                        name=f"Label {label}",
                        label_value=label,
                    )
                )
    if len(result) > 4096:
        raise InputValidationError(
            "The NRRD contains thousands of values and does not look like a label map."
        )
    return tuple(result)


def inspect_segmentation(path: str | Path) -> SegmentationInfo:
    image, header = _read_segmentation(path)
    segments = parse_segment_definitions(header)
    warnings: list[str] = []
    if not segments:
        segments = _fallback_segments(image)
        warnings.append(
            "No SegmentN metadata was found; labels were named from numeric values."
        )
    if not segments:
        raise InputValidationError("The NRRD segmentation contains no non-zero labels.")

    component_count = int(image.GetNumberOfComponentsPerPixel())
    valid: list[SegmentDefinition] = []
    for segment in segments:
        if component_count > 1 and segment.layer >= component_count:
            warnings.append(
                f"Ignored {segment.name!r}: layer {segment.layer} is not present."
            )
        else:
            valid.append(segment)
    if not valid:
        raise InputValidationError("No segment definition refers to data in the NRRD.")
    return SegmentationInfo(
        path=Path(path).expanduser().resolve(),
        size_xyz=tuple(int(value) for value in image.GetSize()),
        spacing_xyz=tuple(float(value) for value in image.GetSpacing()),
        segments=tuple(valid),
        is_vector=component_count > 1,
        warnings=tuple(warnings),
    )


def discover_dicom_series(folders: Sequence[str | Path]) -> tuple[DicomSeriesSpec, ...]:
    sitk = require_sitk()
    discovered: list[DicomSeriesSpec] = []
    for raw_folder in folders:
        folder = Path(raw_folder).expanduser().resolve()
        if not folder.is_dir():
            raise InputValidationError(f"DICOM folder does not exist: {folder}")
        try:
            series_ids = tuple(sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or ())
        except Exception as exc:
            raise InputValidationError(f"Could not inspect DICOM folder {folder}: {exc}") from exc
        if not series_ids:
            raise InputValidationError(f"No readable DICOM image series found in {folder}")
        for uid in series_ids:
            files = tuple(
                Path(name)
                for name in sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(folder), uid
                )
            )
            if not files:
                continue
            summary = read_series_summary(files[0])
            description = (
                summary["series_description"]
                or summary["study_description"]
                or folder.name
            )
            number = summary["series_number"]
            name = " · ".join(part for part in (number, description) if part)
            if len(series_ids) > 1:
                name = f"{name} · {str(uid)[-8:]}"
            discovered.append(
                DicomSeriesSpec(
                    source_directory=folder,
                    series_uid=str(uid),
                    files=files,
                    display_name=name,
                    patient_id=summary["patient_id"],
                )
            )
    if not discovered:
        raise InputValidationError("Add at least one DICOM series folder.")

    counts: dict[str, int] = {}
    output: list[DicomSeriesSpec] = []
    for series in discovered:
        key = series.display_name.casefold()
        count = counts.get(key, 0) + 1
        counts[key] = count
        display = series.display_name if count == 1 else f"{series.display_name} ({count})"
        output.append(
            DicomSeriesSpec(
                source_directory=series.source_directory,
                series_uid=series.series_uid,
                files=series.files,
                display_name=display,
                patient_id=series.patient_id,
            )
        )
    return tuple(output)


def load_dicom_image(series: DicomSeriesSpec):
    sitk = require_sitk()
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(path) for path in series.files])
    try:
        image = reader.Execute()
    except Exception as exc:
        raise InputValidationError(
            f"Could not load DICOM series {series.display_name!r}: {exc}"
        ) from exc
    if image.GetDimension() != 3:
        raise InputValidationError(
            f"DICOM series {series.display_name!r} is not a 3-D volume."
        )
    return sitk.Cast(image, sitk.sitkInt16)


def _segment_mask(image: Any, segment: SegmentDefinition):
    sitk = require_sitk()
    scalar = image
    if int(image.GetNumberOfComponentsPerPixel()) > 1:
        scalar = sitk.VectorIndexSelectionCast(image, segment.layer, sitk.sitkUInt16)
    return sitk.BinaryThreshold(
        scalar,
        float(segment.label_value),
        float(segment.label_value),
        1,
        0,
    )
