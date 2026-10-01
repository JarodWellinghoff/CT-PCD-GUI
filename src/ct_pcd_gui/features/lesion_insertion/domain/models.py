from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

Vec3 = tuple[float, float, float]
Severity = Literal["error", "warning", "info"]
PreviewMode = Literal["before", "overlay", "after"]


class LesionInsertionError(RuntimeError):
    """Base error for the lesion-insertion feature."""


class ValidationError(LesionInsertionError):
    """Raised when a session or parameter value is unsafe or incomplete."""


class UnsupportedSpatialGeometry(ValidationError):
    """Raised when the legacy Alpha projector cannot map a reconstruction safely."""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class LesionParameters:
    """Per-instance transformations applied before the validated projector."""

    contrast_scale: float = 1.0
    scale_xyz: Vec3 = (1.0, 1.0, 1.0)
    rotation_deg_xyz: Vec3 = (0.0, 0.0, 0.0)
    preview_opacity: float = 0.65

    def validate(self) -> None:
        if not math.isfinite(self.contrast_scale) or not 0.0 <= self.contrast_scale <= 5.0:
            raise ValidationError("Contrast scale must be between 0 and 5.")
        if len(self.scale_xyz) != 3 or any(
            not math.isfinite(value) or not 0.25 <= value <= 4.0
            for value in self.scale_xyz
        ):
            raise ValidationError("Lesion X/Y/Z scale values must be between 0.25 and 4.")
        if len(self.rotation_deg_xyz) != 3 or any(
            not math.isfinite(value) or not -180.0 <= value <= 180.0
            for value in self.rotation_deg_xyz
        ):
            raise ValidationError("Lesion rotations must be between -180 and 180 degrees.")
        if not math.isfinite(self.preview_opacity) or not 0.0 <= self.preview_opacity <= 1.0:
            raise ValidationError("Preview opacity must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class LesionLibraryItem:
    path: str
    lesion_id: str
    display_name: str
    relative_path: str
    compatible: bool = True
    channel_count: int = 0
    dimensions_rcs: tuple[int, int, int] = (0, 0, 0)
    spacing_rcs_mm: Vec3 = (1.0, 1.0, 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)
    warning: str = ""


@dataclass(frozen=True, slots=True)
class LesionInstance:
    instance_id: str
    lesion_path: str
    lesion_id: str
    label: str
    center_voxel_crs: Vec3
    center_patient_lps_mm: Vec3
    center_ctpd_mm: Vec3
    background_hu: tuple[float, ...]
    parameters: LesionParameters = field(default_factory=LesionParameters)
    enabled: bool = True
    visible: bool = True

    @classmethod
    def create(
        cls,
        *,
        lesion_path: str,
        lesion_id: str,
        label: str,
        center_voxel_crs: Vec3,
        center_patient_lps_mm: Vec3,
        center_ctpd_mm: Vec3,
        background_hu: tuple[float, ...],
    ) -> LesionInstance:
        return cls(
            instance_id=str(uuid.uuid4()),
            lesion_path=lesion_path,
            lesion_id=lesion_id,
            label=label,
            center_voxel_crs=center_voxel_crs,
            center_patient_lps_mm=center_patient_lps_mm,
            center_ctpd_mm=center_ctpd_mm,
            background_hu=background_hu,
        )

    def validate(self) -> None:
        if not self.instance_id:
            raise ValidationError("A lesion instance is missing its ID.")
        if not Path(self.lesion_path).is_file():
            raise ValidationError(f"Lesion model is unavailable: {self.lesion_path}")
        if len(self.center_voxel_crs) != 3:
            raise ValidationError(f"{self.label}: voxel position must contain three values.")
        if len(self.center_patient_lps_mm) != 3:
            raise ValidationError(f"{self.label}: patient position must contain three values.")
        if len(self.center_ctpd_mm) != 3:
            raise ValidationError(f"{self.label}: CTPD position must contain three values.")
        if any(not math.isfinite(value) for value in self.center_voxel_crs):
            raise ValidationError(f"{self.label}: voxel position contains a non-finite value.")
        if any(not math.isfinite(value) for value in self.center_patient_lps_mm):
            raise ValidationError(f"{self.label}: patient position contains a non-finite value.")
        if any(not math.isfinite(value) for value in self.center_ctpd_mm):
            raise ValidationError(f"{self.label}: CTPD position contains a non-finite value.")
        if not self.background_hu or any(not math.isfinite(value) for value in self.background_hu):
            raise ValidationError(f"{self.label}: background HU values are missing or invalid.")
        self.parameters.validate()

    def duplicate(self, suffix: str = " copy") -> LesionInstance:
        return replace(self, instance_id=str(uuid.uuid4()), label=f"{self.label}{suffix}")


@dataclass(frozen=True, slots=True)
class SeriesGeometry:
    series_instance_uid: str
    study_instance_uid: str
    frame_of_reference_uid: str
    rows: int
    columns: int
    image_positions_lps_mm: tuple[Vec3, ...]
    column_axis_lps: Vec3
    row_axis_lps: Vec3
    row_spacing_mm: float
    column_spacing_mm: float
    patient_position: str
    reconstruction_diameter_mm: float
    reconstruction_target_center_lps_mm: Vec3 = (0.0, 0.0, 0.0)
    data_collection_center_lps_mm: Vec3 = (0.0, 0.0, 0.0)
    warnings: tuple[str, ...] = ()
    patient_identity_digest: str = field(default="", repr=False, compare=False)
    series_selection_key: str = ""
    series_grouping_method: str = "series_instance_uid"
    series_missing_uid_instance_count: int = 0

    @property
    def effective_series_key(self) -> str:
        return self.series_selection_key or self.series_instance_uid

    @property
    def source_has_series_instance_uid(self) -> bool:
        return bool(self.series_instance_uid)

    @property
    def source_was_nonconformant(self) -> bool:
        return self.series_missing_uid_instance_count > 0

    @property
    def slice_count(self) -> int:
        return len(self.image_positions_lps_mm)

    @property
    def normal_axis_lps(self) -> np.ndarray:
        column = np.asarray(self.column_axis_lps, dtype=float)
        row = np.asarray(self.row_axis_lps, dtype=float)
        normal = np.cross(column, row)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            raise ValidationError("Image orientation row/column axes are degenerate.")
        return normal / norm

    @property
    def slice_coordinates_mm(self) -> np.ndarray:
        normal = self.normal_axis_lps
        return np.asarray(
            [
                np.dot(np.asarray(position, dtype=float), normal)
                for position in self.image_positions_lps_mm
            ],
            dtype=float,
        )

    @property
    def median_slice_spacing_mm(self) -> float:
        coordinates = self.slice_coordinates_mm
        if len(coordinates) < 2:
            return 1.0
        distances = np.abs(np.diff(coordinates))
        distances = distances[distances > 1e-6]
        return float(np.median(distances)) if distances.size else 1.0

    def validate(self) -> None:
        if self.rows < 1 or self.columns < 1 or self.slice_count < 1:
            raise ValidationError("Reconstruction dimensions must be positive.")
        if self.row_spacing_mm <= 0 or self.column_spacing_mm <= 0:
            raise ValidationError("DICOM PixelSpacing must contain positive values.")
        column = np.asarray(self.column_axis_lps, dtype=float)
        row = np.asarray(self.row_axis_lps, dtype=float)
        if not np.isclose(np.linalg.norm(column), 1.0, atol=1e-4):
            raise ValidationError("The DICOM column direction cosine is not normalized.")
        if not np.isclose(np.linalg.norm(row), 1.0, atol=1e-4):
            raise ValidationError("The DICOM row direction cosine is not normalized.")
        if not np.isclose(np.dot(column, row), 0.0, atol=1e-4):
            raise ValidationError("The DICOM row and column axes are not orthogonal.")

    def voxel_to_patient(self, center_crs: Vec3) -> np.ndarray:
        """Convert (column, row, slice) to DICOM patient LPS millimetres."""

        column_index, row_index, slice_index = (float(value) for value in center_crs)
        if not 0.0 <= column_index <= self.columns - 1:
            raise ValidationError("Column position is outside the reconstruction.")
        if not 0.0 <= row_index <= self.rows - 1:
            raise ValidationError("Row position is outside the reconstruction.")
        if not 0.0 <= slice_index <= self.slice_count - 1:
            raise ValidationError("Slice position is outside the reconstruction.")

        lower_index = int(math.floor(slice_index))
        upper_index = min(lower_index + 1, self.slice_count - 1)
        fraction = slice_index - lower_index
        lower = np.asarray(self.image_positions_lps_mm[lower_index], dtype=float)
        upper = np.asarray(self.image_positions_lps_mm[upper_index], dtype=float)
        origin = lower + fraction * (upper - lower)
        return (
            origin
            + column_index * self.column_spacing_mm * np.asarray(self.column_axis_lps)
            + row_index * self.row_spacing_mm * np.asarray(self.row_axis_lps)
        )

    def patient_to_voxel(self, patient_lps_mm: Vec3) -> np.ndarray:
        """Convert patient LPS to nearest-slice fractional (column, row, slice)."""

        point = np.asarray(patient_lps_mm, dtype=float)
        normal = self.normal_axis_lps
        slice_coordinates = self.slice_coordinates_mm
        coordinate = float(np.dot(point, normal))
        nearest = int(np.argmin(np.abs(slice_coordinates - coordinate)))
        origin = np.asarray(self.image_positions_lps_mm[nearest], dtype=float)
        difference = point - origin
        column = float(
            np.dot(difference, np.asarray(self.column_axis_lps)) / self.column_spacing_mm
        )
        row = float(np.dot(difference, np.asarray(self.row_axis_lps)) / self.row_spacing_mm)

        slice_index = float(nearest)
        if self.slice_count > 1:
            if coordinate >= slice_coordinates[nearest] and nearest < self.slice_count - 1:
                other = nearest + 1
            elif coordinate < slice_coordinates[nearest] and nearest > 0:
                other = nearest - 1
            else:
                other = nearest
            denominator = slice_coordinates[other] - slice_coordinates[nearest]
            if other != nearest and abs(denominator) > 1e-8:
                slice_index = nearest + (coordinate - slice_coordinates[nearest]) / denominator * (
                    other - nearest
                )
        return np.asarray([column, row, slice_index], dtype=float)

    def assert_legacy_alpha_compatible(self, atol: float = 1e-3) -> None:
        """Require the orientation supported by the coworker's Alpha mapping."""

        position = self.patient_position.upper()
        if position not in {"HFS", "FFS"}:
            raise UnsupportedSpatialGeometry(
                f"PatientPosition {position or '<missing>'!r} is not supported by the "
                "validated lesion projector; use HFS or FFS data."
            )
        column = np.asarray(self.column_axis_lps, dtype=float)
        row = np.asarray(self.row_axis_lps, dtype=float)
        normal = self.normal_axis_lps
        if not (
            np.allclose(column, [1.0, 0.0, 0.0], atol=atol)
            and np.allclose(row, [0.0, 1.0, 0.0], atol=atol)
            and np.allclose(normal, [0.0, 0.0, 1.0], atol=atol)
        ):
            raise UnsupportedSpatialGeometry(
                "The reconstruction is oblique or non-axial. It can be viewed and placed "
                "in patient coordinates, but final CTPD insertion is blocked because the "
                "coworker's validated Alpha conversion supports only axis-aligned HFS/FFS data."
            )
        coordinates = self.slice_coordinates_mm
        if coordinates.size > 2:
            distances = np.abs(np.diff(coordinates))
            median = float(np.median(distances))
            if median <= 0 or np.any(
                np.abs(distances - median) > max(0.02 * median, 1e-3)
            ):
                raise UnsupportedSpatialGeometry(
                    "The reconstruction has non-uniform or missing-slice spacing. "
                    "Patient-coordinate placement remains available, but the coworker's "
                    "validated Alpha conversion assumes a uniform slice grid, so final "
                    "CTPD insertion is blocked."
                )

    def voxel_to_legacy_ctpd(self, center_crs: Vec3) -> np.ndarray:
        """Apply the validated Alpha HFS/FFS conversion used by pipeline 3.1.4."""

        self.assert_legacy_alpha_compatible()
        column, row, slice_index = (float(value) for value in center_crs)
        _ = self.voxel_to_patient(center_crs)
        selected_mm = np.asarray(
            [
                column * self.column_spacing_mm,
                row * self.row_spacing_mm,
                slice_index * self.median_slice_spacing_mm,
            ],
            dtype=float,
        )
        corrected = (
            selected_mm
            - np.asarray(self.reconstruction_target_center_lps_mm, dtype=float)
            + np.asarray(self.data_collection_center_lps_mm, dtype=float)
        )
        half_diameter = self.reconstruction_diameter_mm / 2.0
        if self.patient_position.upper() == "HFS":
            x_value = corrected[0] - half_diameter
        else:
            x_value = -(corrected[0] - half_diameter)
        y_value = -(corrected[1] - half_diameter)

        mapped_index = int(round(corrected[2] / self.median_slice_spacing_mm))
        mapped_index = int(np.clip(mapped_index, 0, self.slice_count - 1))
        z_value = float(self.slice_coordinates_mm[mapped_index])
        if self.patient_position.upper() == "FFS":
            z_value = -z_value
        return np.asarray([x_value, y_value, z_value], dtype=float)


@dataclass(slots=True)
class DicomVolume:
    geometry: SeriesGeometry
    hu: np.ndarray
    source_paths: tuple[str, ...]
    headers: tuple[Any, ...] = ()

    def validate(self) -> None:
        self.geometry.validate()
        expected = (
            self.geometry.slice_count,
            self.geometry.rows,
            self.geometry.columns,
        )
        if tuple(int(value) for value in self.hu.shape) != expected:
            raise ValidationError(
                f"Loaded volume shape {self.hu.shape} does not match DICOM geometry {expected}."
            )


@dataclass(frozen=True, slots=True)
class DicomSeriesCandidate:
    series_instance_uid: str
    study_instance_uid: str
    frame_of_reference_uid: str
    description: str
    modality: str
    instance_count: int
    rows: int
    columns: int
    source_root: str
    warnings: tuple[str, ...] = ()
    selection_key: str = ""
    uses_surrogate_key: bool = False
    source_paths: tuple[str, ...] = ()
    missing_uid_instance_count: int = 0

    @property
    def effective_selection_key(self) -> str:
        return self.selection_key or self.series_instance_uid

    @property
    def display_name(self) -> str:
        description = self.description.strip() or "Unnamed series"
        name = (
            f"{description} — {self.instance_count} images "
            f"({self.rows}×{self.columns})"
        )
        if self.uses_surrogate_key:
            return (
                f"{name} [metadata-grouped: missing SeriesInstanceUID]"
            )
        if self.missing_uid_instance_count:
            return (
                f"{name} [{self.missing_uid_instance_count} image(s) "
                "missing SeriesInstanceUID]"
            )
        return name


@dataclass(frozen=True, slots=True)
class RawDatasetInfo:
    source: str
    study_instance_uids: tuple[str, ...]
    series_instance_uids: tuple[str, ...]
    frame_of_reference_uids: tuple[str, ...]
    patient_positions: tuple[str, ...]
    spectrum_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    projection_file_count: int
    projection_frame_count: int
    warnings: tuple[str, ...] = ()
    patient_identity_digest: str = field(default="", repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class LesionSession:
    version: int = 1
    reconstruction_source: str = ""
    reconstruction_series_uid: str = ""
    reconstruction_series_selection_key: str = ""
    raw_source: str = ""
    lesion_library_source: str = ""
    output_directory: str = ""
    lesions: tuple[LesionInstance, ...] = ()
    spectrum_channel_map: tuple[tuple[int, int], ...] = ()
    workers: int = 1
    preview_enabled: bool = True
    preview_mode: PreviewMode = "overlay"
    reconstruction_command: str = ""
    reconstruction_output_directory: str = ""

    @property
    def reconstruction_series_key(self) -> str:
        "Return the application key used to reopen the selected series."

        return (
            self.reconstruction_series_selection_key
            or self.reconstruction_series_uid
        )

    @property
    def spectrum_map(self) -> dict[int, int]:
        return {int(spectrum): int(channel) for spectrum, channel in self.spectrum_channel_map}

    def with_spectrum_map(self, mapping: dict[int, int]) -> LesionSession:
        return replace(
            self,
            spectrum_channel_map=tuple(
                sorted((int(key), int(value)) for key, value in mapping.items())
            ),
        )

    def validate_basic(self) -> None:
        if self.version != 1:
            raise ValidationError(f"Unsupported lesion session version: {self.version}")
        if not 1 <= self.workers <= 64:
            raise ValidationError("Projection workers must be between 1 and 64.")
        if self.preview_mode not in {"before", "overlay", "after"}:
            raise ValidationError(f"Unsupported preview mode: {self.preview_mode}")
        ids = [lesion.instance_id for lesion in self.lesions]
        if len(ids) != len(set(ids)):
            raise ValidationError("The session contains duplicate lesion instance IDs.")
        spectra = [int(spectrum) for spectrum, _channel in self.spectrum_channel_map]
        if len(spectra) != len(set(spectra)):
            raise ValidationError("The spectrum mapping contains duplicate spectrum indices.")
        if any(
            int(spectrum) < 1 or int(channel) < 0
            for spectrum, channel in self.spectrum_channel_map
        ):
            raise ValidationError(
                "Spectrum indices are 1-based and lesion channels are 0-based."
            )
        for lesion in self.lesions:
            lesion.parameters.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lesions"] = [asdict(item) for item in self.lesions]
        data["spectrum_channel_map"] = {
            str(key): value for key, value in self.spectrum_channel_map
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LesionSession:
        lesions: list[LesionInstance] = []
        for raw_item in data.get("lesions", []):
            item = dict(raw_item)
            parameter_data = dict(item.pop("parameters", {}))
            parameter_data["scale_xyz"] = tuple(
                parameter_data.get("scale_xyz", (1.0, 1.0, 1.0))
            )
            parameter_data["rotation_deg_xyz"] = tuple(
                parameter_data.get("rotation_deg_xyz", (0.0, 0.0, 0.0))
            )
            parameters = LesionParameters(**parameter_data)
            item["center_voxel_crs"] = tuple(item["center_voxel_crs"])
            item["center_patient_lps_mm"] = tuple(item["center_patient_lps_mm"])
            item["center_ctpd_mm"] = tuple(item["center_ctpd_mm"])
            item["background_hu"] = tuple(item["background_hu"])
            lesions.append(LesionInstance(parameters=parameters, **item))
        raw_mapping = data.get("spectrum_channel_map", {})
        if isinstance(raw_mapping, dict):
            mapping = tuple(sorted((int(key), int(value)) for key, value in raw_mapping.items()))
        else:
            mapping = tuple((int(pair[0]), int(pair[1])) for pair in raw_mapping)
        preview_mode_value = str(data.get("preview_mode", "overlay"))
        if preview_mode_value not in {"before", "overlay", "after"}:
            raise ValidationError(f"Unsupported preview mode: {preview_mode_value}")
        preview_mode = cast(PreviewMode, preview_mode_value)
        session = cls(
            version=int(data.get("version", 1)),
            reconstruction_source=str(data.get("reconstruction_source", "")),
            reconstruction_series_uid=str(data.get("reconstruction_series_uid", "")),
            reconstruction_series_selection_key=str(
                data.get(
                    "reconstruction_series_selection_key",
                    data.get("reconstruction_series_uid", ""),
                )
            ),
            raw_source=str(data.get("raw_source", "")),
            lesion_library_source=str(data.get("lesion_library_source", "")),
            output_directory=str(data.get("output_directory", "")),
            lesions=tuple(lesions),
            spectrum_channel_map=mapping,
            workers=int(data.get("workers", 1)),
            preview_enabled=bool(data.get("preview_enabled", True)),
            preview_mode=preview_mode,
            reconstruction_command=str(data.get("reconstruction_command", "")),
            reconstruction_output_directory=str(
                data.get("reconstruction_output_directory", "")
            ),
        )
        session.validate_basic()
        return session

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> LesionSession:
        source = Path(path).expanduser()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValidationError("A lesion session must contain a JSON object.")
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PreviewResult:
    slice_index: int
    before_hu: np.ndarray
    approximate_after_hu: np.ndarray
    overlay_alpha: np.ndarray
    generation: int
    cache_hits: int = 0


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    output_directory: str
    modified_ctpd_directory: str
    manifest_path: str
    summary_path: str
    reconstruction_output_directory: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
