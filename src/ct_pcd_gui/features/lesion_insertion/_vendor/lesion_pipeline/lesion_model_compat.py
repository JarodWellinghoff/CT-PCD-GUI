from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .lesion_models import (
    INVERSE_MASK_BACKGROUND_METHOD,
    LesionModel,
    load_lesion_model as _load_pipeline_lesion_model,
)

_NEW_NPZ_ARRAY_FIELDS = frozenset({"VOI", "LesionMask"})


def _npz_scalar(value: Any, name: str) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must contain exactly one value.")
    return array.reshape(-1)[0]


def _finite_scalar(value: Any, name: str, *, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is missing.")
        return float(default)
    try:
        result = float(_npz_scalar(value, name))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _positive_scalar(value: Any, name: str, *, default: float | None = None) -> float:
    result = _finite_scalar(value, name, default=default)
    if result <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return result


def _json_text(value: Any, name: str) -> str:
    item = _npz_scalar(value, name)
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def _dicom_json_value(value: Any) -> Any:
    """Unwrap either plain JSON values or DICOM JSON-model Value objects."""

    if isinstance(value, Mapping) and "Value" in value:
        items = value["Value"]
        if isinstance(items, list) and len(items) == 1:
            return items[0]
        return items
    return value


def _header_value(header: Mapping[str, Any], name: str, default: Any = None) -> Any:
    return _dicom_json_value(header.get(name, default))


def _numeric_vector(value: Any, name: str) -> np.ndarray:
    value = _dicom_json_value(value)
    if isinstance(value, str) and "\\" in value:
        value = value.split("\\")
    try:
        result = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values.") from exc
    if not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite numeric values.")
    return result


def _read_dicom_header(data: np.lib.npyio.NpzFile, source: Path) -> Mapping[str, Any]:
    if "DicomHeaderJSON" not in data:
        raise ValueError(
            f"Lesion model {source} uses the new VOI/LesionMask format but does not "
            "contain DicomHeaderJSON."
        )
    try:
        header = json.loads(_json_text(data["DicomHeaderJSON"], "DicomHeaderJSON"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"DicomHeaderJSON in {source} is not valid JSON.") from exc
    if not isinstance(header, Mapping):
        raise ValueError(f"DicomHeaderJSON in {source} must decode to a JSON object.")
    return header


def _channel_volume(value: Any, name: str, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 3:
        return array[..., np.newaxis]
    if array.ndim == 4:
        return array
    raise ValueError(
        f"{name} must be a 3-D volume or a 4-D volume with channels; got {array.shape}."
    )


def _load_new_npz_model(data: np.lib.npyio.NpzFile, source: Path) -> LesionModel:
    header = _read_dicom_header(data, source)
    voi = _channel_volume(data["VOI"], "VOI", np.float32)
    mask = _channel_volume(data["LesionMask"], "LesionMask", bool)

    if voi.shape[:3] != mask.shape[:3]:
        raise ValueError(
            f"VOI and LesionMask spatial shapes differ in {source}: "
            f"{voi.shape[:3]} versus {mask.shape[:3]}."
        )
    if mask.shape[3] == 1 and voi.shape[3] > 1:
        mask = np.repeat(mask, voi.shape[3], axis=3)
    if voi.shape != mask.shape:
        raise ValueError(
            f"VOI and LesionMask channel counts differ in {source}: "
            f"{voi.shape[3]} versus {mask.shape[3]}."
        )

    lesion_means: list[float] = []
    background_means: list[float] = []
    background_counts: list[int] = []
    for channel in range(voi.shape[3]):
        channel_voi = voi[..., channel]
        channel_mask = mask[..., channel]
        lesion_values = channel_voi[channel_mask]
        background_values = channel_voi[~channel_mask]
        if not lesion_values.size:
            raise ValueError(f"LesionMask channel {channel} is empty in {source}.")
        if not background_values.size:
            raise ValueError(
                f"LesionMask channel {channel} fills the entire VOI in {source}; "
                "the original lesion background cannot be estimated."
            )
        if not np.all(np.isfinite(lesion_values)):
            raise ValueError(f"VOI contains non-finite lesion values in channel {channel}.")
        if not np.all(np.isfinite(background_values)):
            raise ValueError(f"VOI contains non-finite background values in channel {channel}.")
        lesion_means.append(float(np.mean(lesion_values, dtype=np.float64)))
        background_means.append(float(np.mean(background_values, dtype=np.float64)))
        background_counts.append(int(background_values.size))

    pixel_spacing = _numeric_vector(
        _header_value(header, "PixelSpacing"), "DicomHeaderJSON.PixelSpacing"
    )
    row_spacing = _positive_scalar(pixel_spacing[0], "row pixel spacing")
    column_spacing = _positive_scalar(
        pixel_spacing[1] if pixel_spacing.size > 1 else pixel_spacing[0],
        "column pixel spacing",
    )
    slice_spacing_value = _header_value(header, "SpacingBetweenSlices")
    if slice_spacing_value is None:
        slice_spacing_value = _header_value(header, "SliceThickness")
    slice_spacing = _positive_scalar(
        slice_spacing_value, "DicomHeaderJSON slice spacing"
    )

    columns = _positive_scalar(
        _header_value(header, "Columns"),
        "DicomHeaderJSON.Columns",
        default=float(voi.shape[1]),
    )
    reconstruction_diameter = _positive_scalar(
        _header_value(header, "ReconstructionDiameter"),
        "DicomHeaderJSON.ReconstructionDiameter",
        default=columns * column_spacing,
    )
    kvp = _finite_scalar(_header_value(header, "KVP"), "DicomHeaderJSON.KVP", default=0.0)

    lesion_number = int(
        _finite_scalar(
            data["LesionNumber"] if "LesionNumber" in data else None,
            "LesionNumber",
            default=0.0,
        )
    )
    lesion_mean = _finite_scalar(
        data["LesionMeanHU"] if "LesionMeanHU" in data else None,
        "LesionMeanHU",
        default=float(np.mean(lesion_means, dtype=np.float64)),
    )

    return LesionModel(
        lesion_number=lesion_number,
        voi_hu=voi,
        mask=mask,
        row_spacing_mm=row_spacing,
        column_spacing_mm=column_spacing,
        slice_spacing_mm=slice_spacing,
        reconstruction_diameter_mm=reconstruction_diameter,
        kvp=kvp,
        lesion_mean_hu=lesion_mean,
        lesion_mean_hu_by_channel=np.asarray(lesion_means, dtype=np.float32),
        old_background_hu=np.asarray(background_means, dtype=np.float32),
        old_background_method=INVERSE_MASK_BACKGROUND_METHOD,
        old_background_voxel_count=background_counts[0],
        source_path=str(source),
    )


def load_lesion_model(path: str | Path) -> LesionModel:
    """Load pipeline models plus the newer VOI/LesionMask NPZ export format.

    The newer export omits the pipeline's old-background fields. They are
    reconstructed as the per-channel mean of VOI voxels outside LesionMask,
    which is the same inverse-mask definition used by pipeline version 3.1.4.
    """

    source = Path(path)
    if source.suffix.lower() != ".npz":
        return _load_pipeline_lesion_model(source)

    with np.load(source, allow_pickle=False) as data:
        if _NEW_NPZ_ARRAY_FIELDS.issubset(data.files):
            return _load_new_npz_model(data, source)

    return _load_pipeline_lesion_model(source)
