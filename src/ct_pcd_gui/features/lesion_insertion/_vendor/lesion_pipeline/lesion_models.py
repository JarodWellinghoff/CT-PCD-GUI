from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import nrrd
import numpy as np
import scipy.io
from scipy.ndimage import affine_transform, label

from .dicom_series import ReconSeries, load_recon_series, validate_aligned_series


INVERSE_MASK_BACKGROUND_METHOD = (
    "inverse_segmentation_within_lesion_bounding_box"
)


@dataclass
class LesionModel:
    lesion_number: int
    voi_hu: np.ndarray  # (row, column, slice, channel)
    mask: np.ndarray  # same shape as voi_hu
    row_spacing_mm: float
    column_spacing_mm: float
    slice_spacing_mm: float
    reconstruction_diameter_mm: float
    kvp: float
    lesion_mean_hu: float
    lesion_mean_hu_by_channel: np.ndarray
    old_background_hu: np.ndarray
    old_background_method: str
    old_background_voxel_count: int
    source_path: str = ""

    @property
    def channel_count(self) -> int:
        return int(self.voi_hu.shape[3])


def _unwrap_mat_object(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _field(value: Any, name: str, default: Any = None) -> Any:
    value = _unwrap_mat_object(value)
    if hasattr(value, name):
        return getattr(value, name)
    if isinstance(value, dict):
        return value.get(name, default)
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return _unwrap_mat_object(value[name])
    return default


def _scalar(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else float(default)


def _channel_vector(value: Any, channel_count: int, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == 1 and channel_count > 1:
        values = np.repeat(values, channel_count)
    if values.size != channel_count:
        raise ValueError(
            f"{name} contains {values.size} value(s); expected {channel_count}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains a non-finite value.")
    return values.astype(np.float32)


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    value = _unwrap_mat_object(value)
    array = np.asarray(value)
    if array.size == 0:
        return default
    if array.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in array.reshape(-1)).strip()
    return str(array.reshape(-1)[0]).strip()


def load_lesion_model(path: str | Path) -> LesionModel:
    """Load either the legacy MATLAB model or the pipeline's NPZ model."""

    source = Path(path)
    if source.suffix.lower() == ".npz":
        with np.load(source, allow_pickle=False) as data:
            if "old_background_hu" not in data:
                raise ValueError(
                    f"Lesion model {source} does not contain old_background_hu. "
                    "Regenerate it with pipeline version 3.1.4 or newer."
                )
            if "old_background_method" not in data:
                raise ValueError(
                    f"Lesion model {source} does not identify how its old background "
                    "was measured. Regenerate it with pipeline version 3.1.4 or newer "
                    "to use inverse segmentation within the lesion bounding box."
                )
            voi = np.asarray(data["voi_hu"], dtype=np.float32)
            mask = np.asarray(data["mask"], dtype=bool)
            channel_count = int(voi.shape[3])
            channel_means = (
                _channel_vector(
                    data["lesion_mean_hu_by_channel"],
                    channel_count,
                    "lesion_mean_hu_by_channel",
                )
                if "lesion_mean_hu_by_channel" in data
                else np.asarray(
                    [np.mean(voi[..., index][mask[..., index]]) for index in range(channel_count)],
                    dtype=np.float32,
                )
            )
            return LesionModel(
                lesion_number=int(data["lesion_number"]),
                voi_hu=voi,
                mask=mask,
                row_spacing_mm=float(data["row_spacing_mm"]),
                column_spacing_mm=float(data["column_spacing_mm"]),
                slice_spacing_mm=float(data["slice_spacing_mm"]),
                reconstruction_diameter_mm=float(data["reconstruction_diameter_mm"]),
                kvp=float(data["kvp"]),
                lesion_mean_hu=float(data["lesion_mean_hu"]),
                lesion_mean_hu_by_channel=channel_means,
                old_background_hu=_channel_vector(
                    data["old_background_hu"], channel_count, "old_background_hu"
                ),
                old_background_method=_text(data["old_background_method"]),
                old_background_voxel_count=int(
                    data["old_background_voxel_count"]
                    if "old_background_voxel_count" in data
                    else -1
                ),
                source_path=str(source),
            )

    # Do not squeeze: singleton spatial dimensions are scientifically meaningful.
    data = scipy.io.loadmat(source, struct_as_record=False, squeeze_me=False)
    patient = _unwrap_mat_object(data["Patient"])
    # Some older dual-source models store Patient as a cell array. Tube A is first.
    if isinstance(patient, np.ndarray) and patient.dtype == object:
        patient = _unwrap_mat_object(patient.reshape(-1)[0])
    lesion = _field(patient, "Lesion")
    header = _field(patient, "DicomHeader")
    voi = np.asarray(_field(lesion, "VOI"), dtype=np.float32)
    mask = np.asarray(_field(lesion, "LesionMask"), dtype=bool)
    if voi.ndim == 3:
        voi = voi[..., np.newaxis]
    if mask.ndim == 3:
        mask = mask[..., np.newaxis]
    if voi.shape != mask.shape:
        raise ValueError(f"VOI and lesion mask shapes differ in {source}")
    channel_count = int(voi.shape[3])
    old_background = _field(
        lesion,
        "OldBackgroundHU",
        _field(lesion, "LesionBackground", None),
    )
    if old_background is None:
        raise ValueError(
            f"Legacy lesion model {source} does not contain OldBackgroundHU or "
            "LesionBackground. Regenerate it with pipeline version 3.1.4 or newer."
        )
    old_background_method = _field(lesion, "OldBackgroundMethod", None)
    if old_background_method is None:
        raise ValueError(
            f"Lesion model {source} does not identify how its old background was "
            "measured. Regenerate it with pipeline version 3.1.4 or newer to use "
            "inverse segmentation within the lesion bounding box."
        )
    channel_means_value = _field(lesion, "LesionMeanHUByChannel", None)
    if channel_means_value is None:
        channel_means_value = [
            np.mean(voi[..., index][mask[..., index]])
            for index in range(channel_count)
        ]

    pixel_spacing = np.asarray(_field(header, "PixelSpacing", [1.0, 1.0]), dtype=float).reshape(-1)
    rows = int(_scalar(_field(header, "Rows", 512), 512))
    cols = int(_scalar(_field(header, "Columns", 512), 512))
    diameter = _scalar(
        _field(
            header,
            "ReconstructionDiameter",
            cols * float(pixel_spacing[1] if pixel_spacing.size > 1 else pixel_spacing[0]),
        ),
        cols * float(pixel_spacing[1] if pixel_spacing.size > 1 else pixel_spacing[0]),
    )
    return LesionModel(
        lesion_number=int(_scalar(_field(patient, "LesionNumber", 1), 1)),
        voi_hu=voi,
        mask=mask,
        row_spacing_mm=float(pixel_spacing[0]),
        column_spacing_mm=float(pixel_spacing[1] if pixel_spacing.size > 1 else pixel_spacing[0]),
        slice_spacing_mm=_scalar(_field(header, "SliceThickness", 1.0), 1.0),
        reconstruction_diameter_mm=diameter,
        kvp=_scalar(_field(header, "KVP", 0.0), 0.0),
        lesion_mean_hu=_scalar(
            _field(lesion, "LesionMeanHU", np.mean(voi[mask])),
            float(np.mean(voi[mask])),
        ),
        lesion_mean_hu_by_channel=_channel_vector(
            channel_means_value, channel_count, "LesionMeanHUByChannel"
        ),
        old_background_hu=_channel_vector(
            old_background, channel_count, "OldBackgroundHU"
        ),
        old_background_method=_text(old_background_method),
        old_background_voxel_count=int(
            _scalar(_field(lesion, "OldBackgroundVoxelCount", -1), -1)
        ),
        source_path=str(source),
    )


def save_lesion_model_npz(model: LesionModel, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        lesion_number=np.asarray(model.lesion_number),
        voi_hu=model.voi_hu.astype(np.float32),
        mask=model.mask.astype(np.uint8),
        row_spacing_mm=np.asarray(model.row_spacing_mm),
        column_spacing_mm=np.asarray(model.column_spacing_mm),
        slice_spacing_mm=np.asarray(model.slice_spacing_mm),
        reconstruction_diameter_mm=np.asarray(model.reconstruction_diameter_mm),
        kvp=np.asarray(model.kvp),
        lesion_mean_hu=np.asarray(model.lesion_mean_hu),
        lesion_mean_hu_by_channel=np.asarray(
            model.lesion_mean_hu_by_channel, dtype=np.float32
        ),
        old_background_hu=np.asarray(model.old_background_hu, dtype=np.float32),
        old_background_method=np.asarray(model.old_background_method),
        old_background_voxel_count=np.asarray(model.old_background_voxel_count),
    )
    return target


def save_lesion_model_mat(model: LesionModel, path: str | Path, source_series: ReconSeries) -> Path:
    """Save a MATLAB-compatible model for interoperability with prior work."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    first = source_series.first_header
    header = {
        "Rows": int(first.Rows),
        "Columns": int(first.Columns),
        "PixelSpacing": np.asarray(first.PixelSpacing, dtype=float),
        "SliceThickness": float(getattr(first, "SliceThickness", model.slice_spacing_mm)),
        "ReconstructionDiameter": model.reconstruction_diameter_mm,
        "RescaleIntercept": float(getattr(first, "RescaleIntercept", 0.0)),
        "RescaleSlope": float(getattr(first, "RescaleSlope", 1.0)),
        "KVP": model.kvp,
    }
    patient = {
        "PatientName": str(getattr(first, "PatientID", "")),
        "LesionNumber": model.lesion_number,
        "DicomHeader": header,
        "Lesion": {
            "LesionMask": model.mask.astype(np.uint8),
            "VOI": model.voi_hu.astype(np.float32),
            "LesionMeanHU": model.lesion_mean_hu,
            "LesionMeanHUByChannel": np.asarray(
                model.lesion_mean_hu_by_channel, dtype=np.float32
            ),
            "OldBackgroundHU": np.asarray(model.old_background_hu, dtype=np.float32),
            "LesionBackground": np.asarray(model.old_background_hu, dtype=np.float32),
            "OldBackgroundMethod": model.old_background_method,
            "OldBackgroundVoxelCount": model.old_background_voxel_count,
        },
    }
    scipy.io.savemat(target, {"Patient": patient}, do_compression=True)
    return target


def _space_to_lps(space: str) -> np.ndarray:
    normalized = space.strip().lower().replace("_", "-")
    mapping = {
        "left-posterior-superior": np.diag([1.0, 1.0, 1.0]),
        "lps": np.diag([1.0, 1.0, 1.0]),
        "right-anterior-superior": np.diag([-1.0, -1.0, 1.0]),
        "ras": np.diag([-1.0, -1.0, 1.0]),
        "left-anterior-superior": np.diag([1.0, -1.0, 1.0]),
        "las": np.diag([1.0, -1.0, 1.0]),
        "right-posterior-superior": np.diag([-1.0, 1.0, 1.0]),
        "rps": np.diag([-1.0, 1.0, 1.0]),
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported NRRD coordinate space: {space!r}")
    return mapping[normalized]


def _resample_mask_to_dicom(
    mask: np.ndarray,
    header: dict[str, Any],
    series: ReconSeries,
) -> np.ndarray:
    directions = np.asarray(header.get("space directions"), dtype=float)
    origin = np.asarray(header.get("space origin"), dtype=float)
    if directions.shape != (3, 3) or origin.shape != (3,):
        raise ValueError("NRRD space directions/origin are missing or not three-dimensional.")
    conversion = _space_to_lps(str(header.get("space", "left-posterior-superior")))
    input_matrix = conversion @ directions.T
    input_origin = conversion @ origin

    first = series.first_header
    image_position = np.asarray(first.ImagePositionPatient, dtype=float)
    orientation = np.asarray(first.ImageOrientationPatient, dtype=float)
    column_direction = orientation[:3]
    row_direction = orientation[3:]
    if len(series.headers) > 1:
        slice_vector = (
            np.asarray(series.headers[1].ImagePositionPatient, dtype=float)
            - image_position
        )
    else:
        slice_vector = np.cross(column_direction, row_direction) * series.slice_spacing_mm
    output_matrix = np.column_stack(
        (
            slice_vector,
            row_direction * series.row_spacing_mm,
            column_direction * series.column_spacing_mm,
        )
    )
    inverse = np.linalg.inv(input_matrix)
    matrix = inverse @ output_matrix
    offset = inverse @ (image_position - input_origin)
    return affine_transform(
        mask.astype(np.uint8),
        matrix=matrix,
        offset=offset,
        output_shape=series.hu.shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)


def _legacy_align_mask(mask: np.ndarray, series: ReconSeries) -> np.ndarray:
    aligned = np.rot90(np.flip(mask, axis=(2, 1)), axes=(0, 1))
    aligned = np.transpose(aligned, (2, 0, 1))
    if aligned.shape != series.hu.shape:
        raise ValueError(
            f"Legacy-aligned mask shape {aligned.shape} does not match DICOM shape {series.hu.shape}."
        )
    return aligned.astype(bool)


def _lesion_label_value(header: dict[str, Any]) -> int:
    for key, value in header.items():
        if key.endswith("_Name") and str(value).strip().lower() == "lesion":
            prefix = key[: -len("_Name")]
            return int(header[f"{prefix}_LabelValue"])
    values = [
        int(value)
        for key, value in header.items()
        if key.endswith("_LabelValue")
    ]
    if len(values) == 1:
        return values[0]
    raise ValueError(
        "Could not identify the lesion label. Name one NRRD segment 'Lesion' or retain a single segment."
    )


def generate_lesion_models(
    t1_series_dir: str | Path,
    t2_series_dir: str | Path,
    segmentation_nrrd: str | Path,
    output_dir: str | Path,
    *,
    mask_alignment: str = "physical",
    boundary_margin_pixels: int = 0,
    write_mat: bool = True,
) -> tuple[list[Path], ReconSeries, ReconSeries]:
    """Create one model per connected lesion and return the saved NPZ paths."""

    t1 = load_recon_series(t1_series_dir)
    t2 = load_recon_series(t2_series_dir)
    validate_aligned_series(t1, t2)
    raw_mask, nrrd_header = nrrd.read(segmentation_nrrd, index_order="F")
    label_value = _lesion_label_value(nrrd_header)
    selected = np.asarray(raw_mask == label_value)
    if mask_alignment == "physical":
        aligned = _resample_mask_to_dicom(selected, nrrd_header, t1)
    elif mask_alignment == "legacy":
        aligned = _legacy_align_mask(selected, t1)
    else:
        raise ValueError("mask_alignment must be 'physical' or 'legacy'.")
    if not np.any(aligned):
        raise ValueError("The aligned NRRD lesion mask is empty.")

    components, count = label(aligned)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    first = t1.first_header
    diameter = float(
        getattr(first, "ReconstructionDiameter", t1.shape[2] * t1.column_spacing_mm)
    )
    kvp = float(getattr(first, "KVP", 0.0))
    saved: list[Path] = []

    for number in range(1, count + 1):
        points = np.argwhere(components == number)
        low = np.maximum(points.min(axis=0) - boundary_margin_pixels, 0)
        high = np.minimum(points.max(axis=0) + boundary_margin_pixels + 1, t1.hu.shape)
        z0, y0, x0 = (int(value) for value in low)
        z1, y1, x1 = (int(value) for value in high)
        model_slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
        # The original background is every unsegmented voxel inside this
        # lesion's local bounding-box VOI. Excluding the combined mask avoids
        # treating another segmented lesion as background if boxes overlap.
        background_mask = ~aligned[model_slices]
        background_voxel_count = int(np.count_nonzero(background_mask))
        if background_voxel_count == 0:
            raise ValueError(
                f"Lesion {number}'s bounding-box VOI contains no unsegmented "
                "background voxels. The inverse-mask background cannot be measured."
            )
        old_background_hu = np.asarray(
            [
                np.mean(t1.hu[model_slices][background_mask]),
                np.mean(t2.hu[model_slices][background_mask]),
            ],
            dtype=np.float32,
        )
        component_mask = components[z0:z1, y0:y1, x0:x1] == number
        voi_zyxc = np.stack(
            (t1.hu[z0:z1, y0:y1, x0:x1], t2.hu[z0:z1, y0:y1, x0:x1]),
            axis=-1,
        )
        mask_zyxc = np.repeat(component_mask[..., np.newaxis], 2, axis=3)
        # Preserve the legacy model layout: row, column, slice, channel.
        voi = np.transpose(voi_zyxc, (1, 2, 0, 3))
        model_mask = np.transpose(mask_zyxc, (1, 2, 0, 3))
        lesion_mean_hu_by_channel = np.asarray(
            [
                np.mean(voi[..., channel][model_mask[..., channel]])
                for channel in range(voi.shape[3])
            ],
            dtype=np.float32,
        )
        mean_hu = float(np.mean(lesion_mean_hu_by_channel))
        model = LesionModel(
            lesion_number=number,
            voi_hu=voi.astype(np.float32),
            mask=model_mask,
            row_spacing_mm=t1.row_spacing_mm,
            column_spacing_mm=t1.column_spacing_mm,
            slice_spacing_mm=t1.slice_spacing_mm,
            reconstruction_diameter_mm=diameter,
            kvp=kvp,
            lesion_mean_hu=mean_hu,
            lesion_mean_hu_by_channel=lesion_mean_hu_by_channel,
            old_background_hu=old_background_hu,
            old_background_method=INVERSE_MASK_BACKGROUND_METHOD,
            old_background_voxel_count=background_voxel_count,
        )
        npz_path = save_lesion_model_npz(model, destination / f"Lesion{number:02d}.npz")
        saved.append(npz_path)
        if write_mat:
            save_lesion_model_mat(model, destination / f"Lesion{number:02d}.mat", t1)

    return saved, t1, t2


def model_summary(model: LesionModel) -> SimpleNamespace:
    return SimpleNamespace(
        lesion_number=model.lesion_number,
        shape=model.voi_hu.shape,
        channel_count=model.channel_count,
        mean_hu=model.lesion_mean_hu,
        mean_hu_by_channel=model.lesion_mean_hu_by_channel.copy(),
        old_background_hu=model.old_background_hu.copy(),
        old_background_method=model.old_background_method,
        old_background_voxel_count=model.old_background_voxel_count,
    )
