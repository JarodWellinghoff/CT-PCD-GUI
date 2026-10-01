from __future__ import annotations

import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from pydicom.uid import UID


NUMBER_OF_DETECTOR_ROWS = Tag(0x7029, 0x1010)
NUMBER_OF_DETECTOR_COLUMNS = Tag(0x7029, 0x1011)
DETECTOR_TRANSVERSE_SPACING = Tag(0x7029, 0x1002)
DETECTOR_AXIAL_SPACING = Tag(0x7029, 0x1006)
DETECTOR_SHAPE = Tag(0x7029, 0x100B)
DETECTOR_ANGULAR_POSITION = Tag(0x7031, 0x1001)
DETECTOR_AXIAL_POSITION = Tag(0x7031, 0x1002)
DETECTOR_RADIAL_DISTANCE = Tag(0x7031, 0x1003)
CONSTANT_RADIAL_DISTANCE = Tag(0x7031, 0x1031)
DETECTOR_CENTRAL_ELEMENT = Tag(0x7031, 0x1033)
SOURCE_ANGULAR_SHIFT = Tag(0x7033, 0x100B)
SOURCE_AXIAL_SHIFT = Tag(0x7033, 0x100C)
SOURCE_RADIAL_SHIFT = Tag(0x7033, 0x100D)
FLYING_FOCAL_SPOT_MODE = Tag(0x7033, 0x100E)
NUMBER_OF_SOURCES = Tag(0x7033, 0x105B)
SOURCE_INDEX = Tag(0x7033, 0x105D)
NUMBER_OF_SPECTRA = Tag(0x7033, 0x1061)
SPECTRUM_INDEX = Tag(0x7033, 0x1063)
TYPE_OF_PROJECTION_DATA = Tag(0x7037, 0x1009)
TYPE_OF_PROJECTION_GEOMETRY = Tag(0x7037, 0x100A)
WATER_ATTENUATION_COEFFICIENT = Tag(0x7041, 0x1001)
LEGACY_HU_CALIBRATION_FACTOR = Tag(0x0018, 0x0061)
RESCALE_INTERCEPT = Tag(0x0028, 0x1052)
RESCALE_SLOPE = Tag(0x0028, 0x1053)
SHARED_FUNCTIONAL_GROUPS_SEQUENCE = Tag(0x5200, 0x9229)
PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE = Tag(0x5200, 0x9230)

# Match the original MATLAB Alpha forward projector. MATLAB used 0.1917 cm^-1
# and converted Siddon path lengths from millimeters to centimeters. Python
# retains millimeter path lengths, so the equivalent coefficient is 0.01917
# mm^-1. DICOM tags (7041,1001) and (0018,0061) are intentionally not used.
MATLAB_WATER_ATTENUATION_CM_INVERSE = 0.1917
MATLAB_WATER_ATTENUATION_MM_INVERSE = (
    MATLAB_WATER_ATTENUATION_CM_INVERSE / 10.0
)


@dataclass(frozen=True)
class ProjectionGeometry:
    frame_index: int
    frame_count: int
    rows: int
    columns: int
    stored_rows: int
    stored_columns: int
    pixel_data_transposed: bool
    detector_shape: str
    detector_transverse_spacing_mm: float
    detector_axial_spacing_mm: float
    focal_center_phi_rad: float
    focal_center_z_mm: float
    focal_center_rho_mm: float
    focal_center_to_detector_mm: float
    central_column: float
    central_row: float
    source_phi_shift_rad: float
    source_z_shift_mm: float
    source_rho_shift_mm: float
    spectrum_index: int
    source_index: int
    water_attenuation_mm_inverse: float
    kvp: float
    instance_number: int
    series_instance_uid: str


@dataclass(frozen=True)
class ProjectionRecord:
    path: Path
    relative_path: Path
    spectrum_index: int
    source_index: int
    instance_number: int
    series_instance_uid: str
    frame_count: int
    spectrum_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    frame_group_counts: tuple[tuple[int, int, int], ...]


def _endianness(ds: Dataset) -> str:
    transfer_syntax = UID(str(ds.file_meta.TransferSyntaxUID))
    return ">" if transfer_syntax.is_little_endian is False else "<"


def _decode_bytes(value: bytes, vr: str, ds: Dataset) -> Any:
    endian = _endianness(ds)
    if vr == "US":
        count = len(value) // 2
        result = struct.unpack(endian + "H" * count, value)
    elif vr == "SS":
        count = len(value) // 2
        result = struct.unpack(endian + "h" * count, value)
    elif vr == "UL":
        count = len(value) // 4
        result = struct.unpack(endian + "I" * count, value)
    elif vr == "FL":
        count = len(value) // 4
        result = struct.unpack(endian + "f" * count, value)
    elif vr == "FD":
        count = len(value) // 8
        result = struct.unpack(endian + "d" * count, value)
    elif vr in {"DS", "IS", "CS", "LO", "SH"}:
        text = value.rstrip(b"\x00 ").decode("ascii")
        parts = text.split("\\") if text else []
        if vr in {"DS", "IS"}:
            result = tuple(float(item) for item in parts)
        else:
            result = tuple(parts)
    else:
        return value
    if len(result) == 1:
        return result[0]
    return result


def tag_value(
    ds: Dataset,
    tag: Tag,
    vr: str,
    *,
    default: Any = None,
    required: bool = False,
    encoding_ds: Dataset | None = None,
) -> Any:
    element = ds.get(tag)
    if (
        element is None
        or element.value is None
        or (isinstance(element.value, str) and element.value == "")
    ):
        if required:
            raise KeyError(f"Required DICOM-CT-PD tag {tag} is missing.")
        return default
    cache = ds.__dict__.setdefault("_ctpd_decoded_value_cache", {})
    cache_key = (int(tag), vr)
    if cache_key in cache:
        return cache[cache_key]
    value = element.value
    if isinstance(value, bytes):
        # Nested Functional Group items do not own file_meta; decode their
        # implicit-VR private bytes using the parent FileDataset's syntax.
        value = _decode_bytes(value, vr, encoding_ds or ds)
    cache[cache_key] = value
    return value


def projection_frame_count(ds: Dataset) -> int:
    frame_count = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frame_count < 1:
        raise ValueError("NumberOfFrames must be at least 1.")
    return frame_count


def _recursive_tag_container(ds: Dataset, tag: Tag) -> Dataset | None:
    if ds.get(tag) is not None:
        return ds
    for element in ds:
        if element.VR != "SQ":
            continue
        for item in element.value:
            container = _recursive_tag_container(item, tag)
            if container is not None:
                return container
    return None


def _functional_group_value(
    ds: Dataset,
    sequence_tag: Tag,
    item_index: int,
    tag: Tag,
    vr: str,
) -> Any:
    sequence_element = ds.get(sequence_tag)
    if sequence_element is None or not sequence_element.value:
        return None
    sequence = sequence_element.value
    if item_index >= len(sequence):
        raise ValueError(
            f"Functional group {sequence_tag} contains {len(sequence)} item(s), "
            f"but frame {item_index + 1} was requested."
        )
    return _frame_sequence_value(
        ds,
        sequence,
        sequence_tag,
        item_index,
        tag,
        vr,
    )


def _sequence_item_value(
    ds: Dataset,
    sequence: Any,
    item_index: int,
    tag: Tag,
    vr: str,
) -> Any:
    container = _recursive_tag_container(sequence[item_index], tag)
    if container is None:
        return None
    return tag_value(container, tag, vr, encoding_ds=ds)


def _numeric_scalar(value: Any) -> float | None:
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size != 1 or not np.isfinite(values[0]):
        return None
    return float(values[0])


def _recover_sparse_frame_value(
    ds: Dataset,
    sequence: Any,
    sequence_tag: Tag,
    frame_index: int,
    tag: Tag,
    vr: str,
) -> Any:
    """Recover one absent value in an otherwise frame-aligned sequence.

    Some DICOM-CT-PD generator files contain an empty first private-acquisition
    item in Per-frame Functional Groups while all remaining items are valid.
    Continuous scalar values are linearly interpolated/extrapolated by frame
    index; categorical and non-scalar values use the nearest populated item.
    Recovery is attempted only for a sequence aligned with NumberOfFrames.
    """

    if len(sequence) != projection_frame_count(ds):
        return None

    series_cache = ds.__dict__.setdefault("_ctpd_sparse_frame_series_cache", {})
    series_key = (id(sequence), int(tag), vr)
    if series_key not in series_cache:
        series_cache[series_key] = tuple(
            (index, value)
            for index in range(len(sequence))
            if (
                value := _sequence_item_value(ds, sequence, index, tag, vr)
            )
            is not None
        )
    populated = series_cache[series_key]
    if not populated:
        return None

    result_cache = ds.__dict__.setdefault("_ctpd_sparse_frame_value_cache", {})
    result_key = series_key + (frame_index,)
    if result_key in result_cache:
        return result_cache[result_key]

    numeric = tuple(
        (index, scalar)
        for index, value in populated
        if (scalar := _numeric_scalar(value)) is not None
    )
    continuous_vr = vr in {"FL", "FD", "DS"}
    if continuous_vr and len(numeric) >= 2 and len(numeric) == len(populated):
        indices = np.asarray([item[0] for item in numeric], dtype=int)
        insertion = int(np.searchsorted(indices, frame_index))
        if insertion == 0:
            left, right = numeric[0], numeric[1]
            method = "linear extrapolation"
        elif insertion == len(numeric):
            left, right = numeric[-2], numeric[-1]
            method = "linear extrapolation"
        else:
            left, right = numeric[insertion - 1], numeric[insertion]
            method = "linear interpolation"
        fraction = (frame_index - left[0]) / (right[0] - left[0])
        delta = right[1] - left[1]
        if tag == DETECTOR_ANGULAR_POSITION:
            # Follow the shortest angular step across a +/-pi or 0/2pi wrap.
            delta = float(np.arctan2(np.sin(delta), np.cos(delta)))
        result = left[1] + fraction * delta
        source_frames = (left[0] + 1, right[0] + 1)
    else:
        nearest_index, result = min(
            populated,
            key=lambda item: (abs(item[0] - frame_index), item[0] > frame_index),
        )
        method = "nearest populated frame"
        source_frames = (nearest_index + 1,)

    result_cache[result_key] = result
    source_text = ", ".join(str(index) for index in source_frames)
    warnings.warn(
        f"DICOM-CT-PD frame {frame_index + 1} is missing tag {tag} in "
        f"sequence {sequence_tag}; recovered it by {method} from frame(s) "
        f"{source_text}.",
        RuntimeWarning,
        stacklevel=3,
    )
    return result


def _frame_sequence_value(
    ds: Dataset,
    sequence: Any,
    sequence_tag: Tag,
    frame_index: int,
    tag: Tag,
    vr: str,
) -> Any:
    value = _sequence_item_value(ds, sequence, frame_index, tag, vr)
    if value is not None:
        return value
    return _recover_sparse_frame_value(
        ds,
        sequence,
        sequence_tag,
        frame_index,
        tag,
        vr,
    )


def _find_aligned_frame_sequence(
    container: Dataset,
    tag: Tag,
    frame_count: int,
) -> tuple[Any, Tag] | None:
    """Find a sequence whose items align one-to-one with projection frames."""

    for element in container:
        if element.VR != "SQ" or not element.value:
            continue
        sequence = element.value
        if len(sequence) == frame_count:
            if any(
                _recursive_tag_container(item, tag) is not None for item in sequence
            ):
                return sequence, element.tag
        # Frame containers may themselves be nested in a one-item module
        # sequence. Inspect the first item to identify that structure without
        # traversing every frame during each lookup.
        nested = _find_aligned_frame_sequence(sequence[0], tag, frame_count)
        if nested is not None:
            return nested
    return None


def _aligned_frame_sequence_value(
    ds: Dataset,
    tag: Tag,
    vr: str,
    frame_index: int,
) -> Any:
    """Read a frame value from any NumberOfFrames-aligned sequence.

    This is a compatibility fallback for valid generator layouts that place
    per-view acquisition data in a sequence other than the standard
    Per-frame Functional Groups Sequence.
    """

    cache = ds.__dict__.setdefault("_ctpd_frame_sequence_cache", {})
    cache_key = int(tag)
    if cache_key not in cache:
        cache[cache_key] = _find_aligned_frame_sequence(
            ds, tag, projection_frame_count(ds)
        )
    located = cache[cache_key]
    if located is None:
        return None
    sequence, _sequence_tag = located
    return _frame_sequence_value(
        ds,
        sequence,
        _sequence_tag,
        frame_index,
        tag,
        vr,
    )


def frame_tag_value(
    ds: Dataset,
    tag: Tag,
    vr: str,
    frame_index: int,
    *,
    default: Any = None,
    required: bool = False,
    select_top_level_frame: bool = True,
) -> Any:
    """Read a private value from per-frame, shared, or top-level storage.

    A multi-frame generator may place changing geometry inside Per-frame
    Functional Groups or store one top-level value per frame. Both layouts are
    supported; Per-frame Functional Groups take precedence.
    """

    frame_count = projection_frame_count(ds)
    if frame_index < 0 or frame_index >= frame_count:
        raise IndexError(
            f"Frame index {frame_index} is outside 0..{frame_count - 1}."
        )

    value = _functional_group_value(
        ds, PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE, frame_index, tag, vr
    )
    if value is None:
        value = _aligned_frame_sequence_value(ds, tag, vr, frame_index)
    if value is None:
        value = _functional_group_value(
            ds, SHARED_FUNCTIONAL_GROUPS_SEQUENCE, 0, tag, vr
        )
    if value is None:
        value = tag_value(ds, tag, vr, default=None)
        if value is not None and select_top_level_frame and frame_count > 1:
            values = np.asarray(value).reshape(-1)
            if values.size == frame_count:
                value = values[frame_index]
            elif values.size != 1:
                raise ValueError(
                    f"Frame-varying tag {tag} contains {values.size} values; "
                    f"expected 1 or NumberOfFrames={frame_count}."
                )
            else:
                value = values[0]
    if value is None:
        if required:
            raise KeyError(
                f"Required DICOM-CT-PD tag {tag} is missing for frame "
                f"{frame_index + 1}."
            )
        return default
    return value


def _pair(value: Any, name: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError(f"{name} must contain column and row indices.")
    return float(values[0]), float(values[1])


def read_geometry(ds: Dataset, frame_index: int = 0) -> ProjectionGeometry:
    # The physical detector dimensions are defined by the DICOM-CT-PD private
    # tags.  Some generators store the projection transposed in the standard
    # DICOM pixel matrix, e.g. detector (144, 1376) versus stored
    # Rows/Columns (1376, 144).  Standard Rows/Columns therefore describe only
    # the byte layout, not the physical detector axes used for ray tracing.
    frame_count = projection_frame_count(ds)
    rows = int(
        frame_tag_value(
            ds,
            NUMBER_OF_DETECTOR_ROWS,
            "US",
            frame_index,
            required=True,
            select_top_level_frame=False,
        )
    )
    columns = int(
        frame_tag_value(
            ds,
            NUMBER_OF_DETECTOR_COLUMNS,
            "US",
            frame_index,
            required=True,
            select_top_level_frame=False,
        )
    )
    stored_rows = int(ds.Rows)
    stored_columns = int(ds.Columns)
    if (stored_rows, stored_columns) == (rows, columns):
        pixel_data_transposed = False
    elif (stored_rows, stored_columns) == (columns, rows):
        pixel_data_transposed = True
    else:
        raise ValueError(
            f"DICOM-CT-PD private detector size ({rows}, {columns}) is neither "
            f"the direct nor transposed form of stored Rows/Columns "
            f"({stored_rows}, {stored_columns})."
        )
    central_column, central_row = _pair(
        frame_tag_value(
            ds,
            DETECTOR_CENTRAL_ELEMENT,
            "FL",
            frame_index,
            required=True,
            select_top_level_frame=False,
        ),
        "DetectorCentralElement",
    )
    return ProjectionGeometry(
        frame_index=frame_index,
        frame_count=frame_count,
        rows=rows,
        columns=columns,
        stored_rows=stored_rows,
        stored_columns=stored_columns,
        pixel_data_transposed=pixel_data_transposed,
        detector_shape=str(
            frame_tag_value(
                ds,
                DETECTOR_SHAPE,
                "CS",
                frame_index,
                required=True,
                select_top_level_frame=False,
            )
        ).strip().upper(),
        detector_transverse_spacing_mm=float(
            frame_tag_value(
                ds,
                DETECTOR_TRANSVERSE_SPACING,
                "FL",
                frame_index,
                required=True,
                select_top_level_frame=False,
            )
        ),
        detector_axial_spacing_mm=float(
            frame_tag_value(
                ds,
                DETECTOR_AXIAL_SPACING,
                "FL",
                frame_index,
                required=True,
                select_top_level_frame=False,
            )
        ),
        focal_center_phi_rad=float(
            frame_tag_value(
                ds, DETECTOR_ANGULAR_POSITION, "FL", frame_index, required=True
            )
        ),
        focal_center_z_mm=float(
            frame_tag_value(
                ds, DETECTOR_AXIAL_POSITION, "FL", frame_index, required=True
            )
        ),
        focal_center_rho_mm=float(
            frame_tag_value(
                ds, DETECTOR_RADIAL_DISTANCE, "FL", frame_index, required=True
            )
        ),
        focal_center_to_detector_mm=float(
            frame_tag_value(
                ds,
                CONSTANT_RADIAL_DISTANCE,
                "FL",
                frame_index,
                required=True,
                select_top_level_frame=False,
            )
        ),
        central_column=central_column,
        central_row=central_row,
        source_phi_shift_rad=float(
            frame_tag_value(ds, SOURCE_ANGULAR_SHIFT, "FL", frame_index, default=0.0)
        ),
        source_z_shift_mm=float(
            frame_tag_value(ds, SOURCE_AXIAL_SHIFT, "FL", frame_index, default=0.0)
        ),
        source_rho_shift_mm=float(
            frame_tag_value(ds, SOURCE_RADIAL_SHIFT, "FL", frame_index, default=0.0)
        ),
        spectrum_index=int(
            frame_tag_value(ds, SPECTRUM_INDEX, "US", frame_index, default=1)
        ),
        source_index=int(
            frame_tag_value(ds, SOURCE_INDEX, "US", frame_index, default=1)
        ),
        water_attenuation_mm_inverse=MATLAB_WATER_ATTENUATION_MM_INVERSE,
        kvp=float(getattr(ds, "KVP", 0.0)),
        instance_number=int(getattr(ds, "InstanceNumber", 0)),
        series_instance_uid=str(getattr(ds, "SeriesInstanceUID", "")),
    )


def _cylindrical_to_cartesian(rho: float, phi: float, z_value: float) -> np.ndarray:
    # DICOM-CT-PD uses x=-rho*sin(phi), y=rho*cos(phi), z=z.
    return np.asarray([-rho * np.sin(phi), rho * np.cos(phi), z_value], dtype=float)


def source_and_detector_positions(geometry: ProjectionGeometry) -> tuple[np.ndarray, np.ndarray]:
    """Return the actual focal spot and detector element centers.

    Detector output has shape ``(rows, columns, 3)`` and follows DICOM image
    row-major ordering.
    """

    focal_center = _cylindrical_to_cartesian(
        geometry.focal_center_rho_mm,
        geometry.focal_center_phi_rad,
        geometry.focal_center_z_mm,
    )
    source = _cylindrical_to_cartesian(
        geometry.focal_center_rho_mm + geometry.source_rho_shift_mm,
        geometry.focal_center_phi_rad + geometry.source_phi_shift_rad,
        geometry.focal_center_z_mm + geometry.source_z_shift_mm,
    )
    radial = focal_center[:2] / np.linalg.norm(focal_center[:2])
    central_direction = -radial
    columns = np.arange(1, geometry.columns + 1, dtype=float)
    rows = np.arange(1, geometry.rows + 1, dtype=float)
    column_offset = columns - geometry.central_column
    row_offset = rows - geometry.central_row
    distance = geometry.focal_center_to_detector_mm
    shape = geometry.detector_shape

    if shape == "CYLINDRICAL":
        gamma = column_offset * geometry.detector_transverse_spacing_mm / distance
        cosine = np.cos(gamma)
        sine = np.sin(gamma)
        # Positive columns rotate the central ray counter-clockwise in x/y.
        directions = np.column_stack(
            (
                central_direction[0] * cosine - central_direction[1] * sine,
                central_direction[0] * sine + central_direction[1] * cosine,
            )
        )
        detector_xy = focal_center[:2] + distance * directions
        detector_z = geometry.focal_center_z_mm - row_offset * geometry.detector_axial_spacing_mm
        detector = np.empty((geometry.rows, geometry.columns, 3), dtype=float)
        detector[:, :, :2] = detector_xy[np.newaxis, :, :]
        detector[:, :, 2] = detector_z[:, np.newaxis]
    elif shape == "FLAT":
        column_basis = np.asarray([-central_direction[1], central_direction[0]])
        central_xy = focal_center[:2] + distance * central_direction
        detector_xy = central_xy + (
            column_offset * geometry.detector_transverse_spacing_mm
        )[:, np.newaxis] * column_basis
        detector_z = geometry.focal_center_z_mm - row_offset * geometry.detector_axial_spacing_mm
        detector = np.empty((geometry.rows, geometry.columns, 3), dtype=float)
        detector[:, :, :2] = detector_xy[np.newaxis, :, :]
        detector[:, :, 2] = detector_z[:, np.newaxis]
    elif shape == "SPHERICAL":
        gamma = column_offset * geometry.detector_transverse_spacing_mm / distance
        eta = row_offset * geometry.detector_axial_spacing_mm / distance
        cosine = np.cos(gamma)
        sine = np.sin(gamma)
        horizontal = np.column_stack(
            (
                central_direction[0] * cosine - central_direction[1] * sine,
                central_direction[0] * sine + central_direction[1] * cosine,
            )
        )
        detector = np.empty((geometry.rows, geometry.columns, 3), dtype=float)
        for row_index, angle in enumerate(eta):
            detector[row_index, :, :2] = (
                focal_center[:2] + distance * np.cos(angle) * horizontal
            )
            detector[row_index, :, 2] = geometry.focal_center_z_mm - distance * np.sin(angle)
    else:
        raise ValueError(
            f"Unsupported detector shape {shape!r}; expected CYLINDRICAL, FLAT, or SPHERICAL."
        )
    return source, detector


def _transfer_syntax(ds: Dataset) -> UID:
    if not hasattr(ds, "file_meta") or "TransferSyntaxUID" not in ds.file_meta:
        raise ValueError("DICOM file is missing TransferSyntaxUID.")
    transfer_syntax = UID(str(ds.file_meta.TransferSyntaxUID))
    if transfer_syntax.is_compressed:
        raise ValueError(
            f"Compressed Pixel Data is not supported ({transfer_syntax.name}); "
            "decompress the DICOM-CT-PD series first."
        )
    return transfer_syntax


def projection_pixel_dtype(ds: Dataset) -> np.dtype:
    transfer_syntax = _transfer_syntax(ds)
    return np.dtype("<u2" if transfer_syntax.is_little_endian is not False else ">u2")


def _validate_pixel_module(ds: Dataset) -> None:
    _transfer_syntax(ds)
    if int(getattr(ds, "BitsAllocated", 0)) != 16:
        raise ValueError("DICOM-CT-PD Pixel Data must use 16 allocated bits.")
    if int(getattr(ds, "PixelRepresentation", 0)) != 0:
        raise ValueError("DICOM-CT-PD Pixel Data must be unsigned.")
    if int(getattr(ds, "SamplesPerPixel", 1)) != 1:
        raise ValueError("DICOM-CT-PD projections must have SamplesPerPixel=1.")


def stored_projection_shape(
    ds: Dataset,
    geometry: ProjectionGeometry | None = None,
) -> tuple[int, int, int]:
    geometry = geometry or read_geometry(ds, 0)
    return (
        projection_frame_count(ds),
        geometry.stored_rows,
        geometry.stored_columns,
    )


def _frame_rescale(ds: Dataset, frame_index: int) -> tuple[float, float]:
    slope = float(frame_tag_value(ds, RESCALE_SLOPE, "DS", frame_index, default=1.0))
    intercept = float(
        frame_tag_value(ds, RESCALE_INTERCEPT, "DS", frame_index, default=0.0)
    )
    if slope == 0:
        raise ValueError(f"RescaleSlope must not be zero for frame {frame_index + 1}.")
    return slope, intercept


def decode_projection_frame(
    ds: Dataset,
    stored_frames: np.ndarray,
    frame_index: int,
    geometry: ProjectionGeometry | None = None,
) -> np.ndarray:
    """Decode one frame to physical detector row/column order."""

    _validate_pixel_module(ds)
    geometry = geometry or read_geometry(ds, frame_index)
    expected_shape = stored_projection_shape(ds, geometry)
    if stored_frames.shape != expected_shape:
        raise ValueError(
            f"Stored projection array shape {stored_frames.shape} does not match "
            f"{expected_shape}."
        )
    stored_matrix = stored_frames[frame_index]
    detector_matrix = stored_matrix.T if geometry.pixel_data_transposed else stored_matrix
    slope, intercept = _frame_rescale(ds, frame_index)
    return detector_matrix.astype(np.float64) * slope + intercept


def encode_projection_frame(
    ds: Dataset,
    physical_values: np.ndarray,
    frame_index: int,
    geometry: ProjectionGeometry | None = None,
) -> tuple[np.ndarray, int]:
    """Encode one detector-order frame in the source DICOM storage order."""

    _validate_pixel_module(ds)
    geometry = geometry or read_geometry(ds, frame_index)
    values = np.asarray(physical_values, dtype=np.float64)
    expected_shape = (geometry.rows, geometry.columns)
    if values.shape != expected_shape:
        raise ValueError(f"Projection shape {values.shape} does not match {expected_shape}.")
    slope, intercept = _frame_rescale(ds, frame_index)
    stored_values = values.T if geometry.pixel_data_transposed else values
    unrounded = (stored_values - intercept) / slope
    rounded = np.rint(unrounded)
    clipped = np.clip(rounded, 0, 65535)
    clipped_count = int(np.count_nonzero(clipped != rounded))
    return clipped.astype(projection_pixel_dtype(ds), copy=False), clipped_count


def decode_projection(
    ds: Dataset,
    geometry: ProjectionGeometry | None = None,
) -> np.ndarray:
    """Decode all projection frames in private detector row/column order.

    A single-frame dataset returns ``(rows, columns)`` for backward
    compatibility. A multi-frame dataset returns ``(frames, rows, columns)``.
    """

    _validate_pixel_module(ds)
    geometry = geometry or read_geometry(ds, 0)
    shape = stored_projection_shape(ds, geometry)
    stored = np.frombuffer(ds.PixelData, dtype=projection_pixel_dtype(ds))
    expected = int(np.prod(shape))
    if stored.size != expected:
        raise ValueError(
            f"Pixel Data contains {stored.size} samples; expected {expected}."
        )
    stored_frames = stored.reshape(shape)
    decoded = [
        decode_projection_frame(ds, stored_frames, index, read_geometry(ds, index))
        for index in range(shape[0])
    ]
    return decoded[0] if shape[0] == 1 else np.stack(decoded, axis=0)


def encode_projection(
    ds: Dataset,
    physical_values: np.ndarray,
    geometry: ProjectionGeometry | None = None,
) -> tuple[bytes, int]:
    """Encode one or all detector-order frames in the source stored layout."""

    _validate_pixel_module(ds)
    geometry = geometry or read_geometry(ds, 0)
    frame_count = projection_frame_count(ds)
    values = np.asarray(physical_values, dtype=np.float64)
    expected_shape = (
        (geometry.rows, geometry.columns)
        if frame_count == 1
        else (frame_count, geometry.rows, geometry.columns)
    )
    if values.shape != expected_shape:
        raise ValueError(f"Projection shape {values.shape} does not match {expected_shape}.")
    encoded_frames: list[np.ndarray] = []
    clipped_count = 0
    for frame_index in range(frame_count):
        frame_values = values if frame_count == 1 else values[frame_index]
        encoded, clipped = encode_projection_frame(
            ds,
            frame_values,
            frame_index,
            read_geometry(ds, frame_index),
        )
        encoded_frames.append(encoded)
        clipped_count += clipped
    return np.stack(encoded_frames, axis=0).tobytes(order="C"), clipped_count


def discover_projection_records(root: str | Path) -> list[ProjectionRecord]:
    source = Path(root)
    candidates: Iterable[Path] = [source] if source.is_file() else source.rglob("*")
    records: list[ProjectionRecord] = []
    errors: list[tuple[Path, str]] = []
    base = source.parent if source.is_file() else source
    for path in sorted(candidates):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            geometry = read_geometry(ds, 0)
            frame_count = projection_frame_count(ds)
            frame_groups: dict[tuple[int, int], int] = {}
            for frame_index in range(frame_count):
                spectrum = int(
                    frame_tag_value(
                        ds, SPECTRUM_INDEX, "US", frame_index, default=1
                    )
                )
                source_index = int(
                    frame_tag_value(ds, SOURCE_INDEX, "US", frame_index, default=1)
                )
                key = (spectrum, source_index)
                frame_groups[key] = frame_groups.get(key, 0) + 1
        except Exception as error:
            errors.append((path, str(error)))
            continue
        records.append(
            ProjectionRecord(
                path=path,
                relative_path=path.relative_to(base),
                spectrum_index=geometry.spectrum_index,
                source_index=geometry.source_index,
                instance_number=geometry.instance_number,
                series_instance_uid=geometry.series_instance_uid,
                frame_count=frame_count,
                spectrum_indices=tuple(sorted({key[0] for key in frame_groups})),
                source_indices=tuple(sorted({key[1] for key in frame_groups})),
                frame_group_counts=tuple(
                    (key[0], key[1], count)
                    for key, count in sorted(frame_groups.items())
                ),
            )
        )
    if not records:
        details = ""
        if errors:
            examples = "; ".join(
                f"{path.name}: {message}" for path, message in errors[:3]
            )
            details = f" First validation error(s): {examples}"
        raise ValueError(f"No valid DICOM-CT-PD projection files found in {root}.{details}")
    records.sort(
        key=lambda record: (
            record.spectrum_index,
            record.source_index,
            record.series_instance_uid,
            record.instance_number,
            str(record.relative_path),
        )
    )
    return records


def inspect_projection_series(root: str | Path) -> dict[str, Any]:
    records = discover_projection_records(root)
    groups: dict[tuple[int, int, str], dict[str, int]] = {}
    for record in records:
        for spectrum, source_index, frame_count in record.frame_group_counts:
            key = (spectrum, source_index, record.series_instance_uid)
            counts = groups.setdefault(key, {"file_count": 0, "frame_count": 0})
            counts["file_count"] += 1
            counts["frame_count"] += frame_count
    first_ds = pydicom.dcmread(records[0].path, stop_before_pixels=True)
    geometry = read_geometry(first_ds)
    return {
        "projection_files": len(records),
        "projection_frames": sum(record.frame_count for record in records),
        "groups": [
            {
                "spectrum_index": key[0],
                "source_index": key[1],
                "series_instance_uid": key[2],
                "dicom_file_count": counts["file_count"],
                "projection_count": counts["frame_count"],
                "frame_count": counts["frame_count"],
            }
            for key, counts in sorted(groups.items())
        ],
        "detector": {
            "shape": geometry.detector_shape,
            "rows": geometry.rows,
            "columns": geometry.columns,
            "stored_rows": geometry.stored_rows,
            "stored_columns": geometry.stored_columns,
            "pixel_data_transposed": geometry.pixel_data_transposed,
            "transverse_spacing_mm": geometry.detector_transverse_spacing_mm,
            "axial_spacing_mm": geometry.detector_axial_spacing_mm,
        },
        "spectrum_indices": sorted(
            {value for record in records for value in record.spectrum_indices}
        ),
        "source_indices": sorted(
            {value for record in records for value in record.source_indices}
        ),
    }
