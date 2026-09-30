"""Local MAT/NPZ lesion-library browser and interactive 3-D viewer.

This module is intentionally self-contained for the flat CT-PCD-GUI project layout.
It recursively scans a user-selected directory for ``.mat`` and ``.npz`` files,
loads lesion volumes off the GUI thread, and renders an adjustable VTK surface in
an embedded PySide6 widget.
"""

from __future__ import annotations

import json
import os

# VTK's Qt bridge checks QT_API while importing. Do not override an explicit
# application-level selection, but default the standalone module to PySide6.
os.environ.setdefault("QT_API", "pyside6")

# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class LesionFileRecord:
    """A lesion file exposed by a source.

    The UI only relies on this lightweight record. A future TCIA-backed source can
    produce the same record type after materializing a remote object locally.
    """

    path: Path
    relative_path: str
    size_bytes: int
    modified_time: float

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(slots=True)
class VolumeCandidate:
    """One renderable 3-D array found inside a MAT/NPZ file."""

    key: str
    array: np.ndarray
    score: float
    min_value: float
    max_value: float
    suggested_threshold: float
    finite_fraction: float
    nonzero_fraction: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        shape = tuple(int(v) for v in self.array.shape)
        if len(shape) != 3:
            raise ValueError(f"Candidate {self.key!r} is not 3-D: {shape}")
        return shape  # type: ignore[return-value]

    @property
    def dtype_name(self) -> str:
        return str(self.array.dtype)

    @property
    def voxel_count(self) -> int:
        return int(self.array.size)

    @property
    def display_name(self) -> str:
        dimensions = " x ".join(str(value) for value in self.shape)
        return f"{self.key}  [{dimensions}, {self.dtype_name}]"


@dataclass(slots=True)
class LesionFileData:
    """All useful data discovered in one lesion file."""

    path: Path
    candidates: list[VolumeCandidate]
    detected_spacing_xyz: tuple[float, float, float] | None = None
    file_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def recommended_index(self) -> int:
        if not self.candidates:
            return -1
        return max(range(len(self.candidates)), key=lambda i: self.candidates[i].score)


class LesionLoadError(RuntimeError):
    """Raised when a lesion file cannot be interpreted safely."""


# -----------------------------------------------------------------------------
# Local source abstraction
# -----------------------------------------------------------------------------

import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Protocol

SUPPORTED_EXTENSIONS = frozenset({".mat", ".npz"})


class LesionSource(Protocol):
    """Minimal source contract used by the viewer.

    A TCIA implementation can later expose downloaded/cached local files through
    this same interface without changing the viewer widget.
    """

    @property
    def display_name(self) -> str: ...

    def iter_records(
        self,
        cancel_event: threading.Event | None = None,
    ) -> Iterable[LesionFileRecord]: ...


class LocalDirectoryLesionSource:
    """Recursively enumerates local MAT/NPZ lesion files."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        include_hidden: bool = False,
        follow_symlinks: bool = False,
        extensions: frozenset[str] = SUPPORTED_EXTENSIONS,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.include_hidden = include_hidden
        self.follow_symlinks = follow_symlinks
        self.extensions = frozenset(ext.lower() for ext in extensions)

    @property
    def display_name(self) -> str:
        return str(self.root)

    def iter_records(
        self,
        cancel_event: threading.Event | None = None,
    ) -> Iterable[LesionFileRecord]:
        if not self.root.is_dir():
            raise NotADirectoryError(
                f"Lesion library directory does not exist: {self.root}"
            )

        root_str = str(self.root)
        for current_root, dirnames, filenames in os.walk(
            root_str,
            followlinks=self.follow_symlinks,
        ):
            if cancel_event and cancel_event.is_set():
                return

            if not self.include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            current = Path(current_root)
            for filename in filenames:
                if cancel_event and cancel_event.is_set():
                    return
                if not self.include_hidden and filename.startswith("."):
                    continue

                path = current / filename
                if path.suffix.lower() not in self.extensions:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                try:
                    relative = str(path.relative_to(self.root))
                except ValueError:
                    relative = path.name
                yield LesionFileRecord(
                    path=path,
                    relative_path=relative,
                    size_bytes=int(stat.st_size),
                    modified_time=float(stat.st_mtime),
                )

    def scan(
        self,
        cancel_event: threading.Event | None = None,
        batch_callback: Callable[[list[LesionFileRecord]], None] | None = None,
        *,
        batch_size: int = 100,
    ) -> list[LesionFileRecord]:
        records: list[LesionFileRecord] = []
        batch: list[LesionFileRecord] = []
        for record in self.iter_records(cancel_event):
            records.append(record)
            if batch_callback is not None:
                batch.append(record)
                if len(batch) >= batch_size:
                    batch_callback(batch)
                    batch = []
        if batch_callback is not None and batch:
            batch_callback(batch)
        records.sort(key=lambda item: item.relative_path.casefold())
        return records


# -----------------------------------------------------------------------------
# MAT/NPZ loading
# -----------------------------------------------------------------------------

import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

try:  # Optional: needed only for MATLAB v7.3/HDF5 files.
    import h5py  # type: ignore
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    h5py = None


_NAME_BONUS = {
    "lesion": 45.0,
    "mask": 40.0,
    "seg": 34.0,
    "segmentation": 36.0,
    "roi": 30.0,
    "label": 24.0,
    "volume": 15.0,
    "voxel": 12.0,
    "image": 8.0,
    "data": 3.0,
}
_NAME_PENALTY = {
    "spacing": 55.0,
    "affine": 45.0,
    "transform": 40.0,
    "origin": 35.0,
    "direction": 35.0,
    "metadata": 25.0,
    "header": 25.0,
}
_SPACING_KEYS = {
    "spacing",
    "voxelspacing",
    "voxel_spacing",
    "voxelsize",
    "voxel_size",
    "resolution",
    "pixelspacing",
    "pixel_spacing",
    "pixdim",
}


def load_lesion_file(path: str | Path) -> LesionFileData:
    """Load all plausible 3-D volumes from a ``.mat`` or ``.npz`` file.

    NPZ object arrays are intentionally not unpickled. MATLAB v7.3 files are
    supported when ``h5py`` is installed.
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise LesionLoadError(f"File does not exist: {source_path}")

    suffix = source_path.suffix.lower()
    warnings: list[str] = []
    file_metadata: dict[str, Any] = {"format": suffix.lstrip(".")}

    if suffix == ".npz":
        root, npz_warnings = _load_npz(source_path)
        warnings.extend(npz_warnings)
        arrays = list(_walk_arrays(root))
        spacing = _find_spacing(root)
        file_metadata["suggested_axis_order"] = _infer_axis_order(root)
    elif suffix == ".mat":
        try:
            root = scipy.io.loadmat(
                source_path,
                simplify_cells=True,
                chars_as_strings=True,
            )
            arrays = list(_walk_arrays(root))
            spacing = _find_spacing(root)
            file_metadata["suggested_axis_order"] = _infer_axis_order(root)
            file_metadata["mat_backend"] = "scipy"
        except (NotImplementedError, ValueError, OSError) as exc:
            if not _looks_like_hdf5_mat_error(exc):
                raise LesionLoadError(f"Could not read MATLAB file: {exc}") from exc
            if h5py is None:
                raise LesionLoadError(
                    "This appears to be a MATLAB v7.3 (HDF5) file. Install h5py "
                    "to open it."
                ) from exc
            arrays, spacing, hdf5_warnings = _load_hdf5_mat(source_path)
            warnings.extend(hdf5_warnings)
            file_metadata["suggested_axis_order"] = "ZYX"
            file_metadata["mat_backend"] = "h5py"
    else:
        raise LesionLoadError(
            f"Unsupported lesion format {suffix!r}. Supported formats: .mat, .npz"
        )

    file_metadata.setdefault("suggested_axis_order", "ZYX")

    candidates: list[VolumeCandidate] = []
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    for key, raw in arrays:
        for expanded_key, array in _expand_to_3d_candidates(key, raw):
            identity = (expanded_key, id(array), tuple(int(v) for v in array.shape))
            if identity in seen:
                continue
            seen.add(identity)
            candidate = _make_candidate(expanded_key, array)
            if candidate is not None:
                candidates.append(candidate)

    candidates.sort(key=lambda item: (-item.score, item.key.casefold()))
    if not candidates:
        details = ""
        if warnings:
            details = "\n\n" + "\n".join(warnings)
        raise LesionLoadError(
            "No numeric 3-D arrays were found in this file. Arrays must have at "
            f"least two voxels along every axis.{details}"
        )

    return LesionFileData(
        path=source_path,
        candidates=candidates,
        detected_spacing_xyz=spacing,
        file_metadata=file_metadata,
        warnings=warnings,
    )


def _load_npz(path: Path) -> tuple[dict[str, Any], list[str]]:
    root: dict[str, Any] = {}
    warnings: list[str] = []
    try:
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                try:
                    root[key] = archive[key]
                except ValueError as exc:
                    warnings.append(
                        f"Skipped NPZ member {key!r}: object/pickled arrays are not "
                        f"loaded for safety ({exc})."
                    )
    except (OSError, ValueError, EOFError) as exc:
        raise LesionLoadError(f"Could not read NPZ file: {exc}") from exc
    return root, warnings


def _load_hdf5_mat(
    path: Path,
) -> tuple[list[tuple[str, np.ndarray]], tuple[float, float, float] | None, list[str]]:
    assert h5py is not None
    arrays: list[tuple[str, np.ndarray]] = []
    spacing: tuple[float, float, float] | None = None
    warnings: list[str] = []

    try:
        with h5py.File(path, "r") as handle:
            spacing = _find_hdf5_spacing(handle)

            def visitor(name: str, obj: Any) -> None:
                if not isinstance(obj, h5py.Dataset):
                    return
                if obj.dtype.kind not in "buif":
                    return
                if obj.size == 0:
                    return
                squeezed_shape = tuple(int(v) for v in obj.shape if int(v) != 1)
                if len(squeezed_shape) not in (3, 4):
                    return
                try:
                    arrays.append((name or "dataset", np.asarray(obj)))
                except Exception as exc:  # pragma: no cover - corrupt HDF5 edge case.
                    warnings.append(f"Skipped HDF5 dataset {name!r}: {exc}")

            handle.visititems(visitor)
    except (OSError, ValueError) as exc:
        raise LesionLoadError(f"Could not read MATLAB v7.3 file: {exc}") from exc

    if not arrays:
        warnings.append("No numeric 3-D/4-D HDF5 datasets were found.")
    return arrays, spacing, warnings


def _walk_arrays(
    value: Any,
    path: str = "",
    *,
    _visited: set[int] | None = None,
    _depth: int = 0,
) -> Iterable[tuple[str, np.ndarray]]:
    """Walk MATLAB structs/cells and Python containers without exploding arrays."""

    if _depth > 18:
        return
    if _visited is None:
        _visited = set()

    track_identity = isinstance(value, (Mapping, list, tuple, np.ndarray))
    if track_identity:
        identity = id(value)
        if identity in _visited:
            return
        _visited.add(identity)

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field in value.dtype.names:
                yield from _walk_arrays(
                    value[field],
                    _join_key(path, field),
                    _visited=_visited,
                    _depth=_depth + 1,
                )
            return

        if value.dtype == object:
            # MATLAB cells and structs are usually small object arrays. Cap the
            # traversal so a malformed file cannot create millions of Python calls.
            max_items = min(int(value.size), 10_000)
            for flat_index, item in enumerate(value.flat):
                if flat_index >= max_items:
                    break
                index = np.unravel_index(flat_index, value.shape)
                index_text = ",".join(str(v) for v in index)
                yield from _walk_arrays(
                    item,
                    f"{path}[{index_text}]" if path else f"[{index_text}]",
                    _visited=_visited,
                    _depth=_depth + 1,
                )
            return

        if value.dtype.kind in "buif" and value.size > 0:
            yield path or "array", value
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.startswith("__"):
                continue
            yield from _walk_arrays(
                item,
                _join_key(path, key_text),
                _visited=_visited,
                _depth=_depth + 1,
            )
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value[:10_000]):
            yield from _walk_arrays(
                item,
                f"{path}[{index}]" if path else f"[{index}]",
                _visited=_visited,
                _depth=_depth + 1,
            )
        return

    field_names = getattr(value, "_fieldnames", None)
    if field_names:
        for field in field_names:
            try:
                item = getattr(value, field)
            except Exception:
                continue
            yield from _walk_arrays(
                item,
                _join_key(path, str(field)),
                _visited=_visited,
                _depth=_depth + 1,
            )


def _expand_to_3d_candidates(
    key: str,
    raw: np.ndarray,
) -> Iterable[tuple[str, np.ndarray]]:
    array = np.asarray(raw)
    if array.dtype.kind not in "buif" or np.iscomplexobj(array):
        return

    squeezed = np.squeeze(array)
    if squeezed.ndim == 3:
        yield key, squeezed
        return

    # Helpful for archives that store a small number of lesions/timepoints in a
    # 4-D tensor. Split only a small axis to avoid turning a large medical volume
    # into hundreds of accidental candidates.
    if squeezed.ndim == 4:
        eligible_axes = [axis for axis, size in enumerate(squeezed.shape) if size <= 32]
        if not eligible_axes:
            return
        axis = min(
            eligible_axes, key=lambda candidate_axis: squeezed.shape[candidate_axis]
        )
        for index in range(int(squeezed.shape[axis])):
            candidate = np.take(squeezed, index, axis=axis)
            if candidate.ndim == 3:
                yield f"{key}[axis{axis}={index}]", candidate


def _make_candidate(key: str, raw: np.ndarray) -> VolumeCandidate | None:
    array = np.asarray(raw)
    if array.ndim != 3 or any(int(size) < 2 for size in array.shape):
        return None
    if array.dtype.kind not in "buif" or np.iscomplexobj(array):
        return None

    # Keep the original integer/bool array when possible; float arrays are reduced
    # to float32 to avoid doubling the memory footprint of large MAT files.
    if array.dtype.kind == "f" and array.dtype.itemsize > 4:
        array = array.astype(np.float32, copy=False)
    array = np.ascontiguousarray(array)

    sample = _sample_values(array)
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return None

    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    finite_fraction = float(finite.size / max(sample.size, 1))
    nonzero_fraction = float(np.count_nonzero(finite) / max(finite.size, 1))
    threshold = _suggest_threshold(finite, min_value, max_value)
    score = _candidate_score(
        key=key,
        array=array,
        sample=finite,
        min_value=min_value,
        max_value=max_value,
        nonzero_fraction=nonzero_fraction,
    )

    return VolumeCandidate(
        key=key,
        array=array,
        score=score,
        min_value=min_value,
        max_value=max_value,
        suggested_threshold=threshold,
        finite_fraction=finite_fraction,
        nonzero_fraction=nonzero_fraction,
    )


def _sample_values(array: np.ndarray, limit: int = 200_000) -> np.ndarray:
    flat = array.reshape(-1)
    if flat.size <= limit:
        return flat.astype(np.float64, copy=False)
    step = max(1, flat.size // limit)
    return flat[::step][:limit].astype(np.float64, copy=False)


def _candidate_score(
    *,
    key: str,
    array: np.ndarray,
    sample: np.ndarray,
    min_value: float,
    max_value: float,
    nonzero_fraction: float,
) -> float:
    normalized_name = re.sub(r"[^a-z0-9]+", " ", key.casefold())
    score = 0.0
    for token, bonus in _NAME_BONUS.items():
        if token in normalized_name:
            score += bonus
    for token, penalty in _NAME_PENALTY.items():
        if token in normalized_name:
            score -= penalty

    if array.dtype == np.bool_:
        score += 40.0
    elif array.dtype.kind in "ui":
        score += 12.0

    if min_value < max_value:
        score += 10.0
    else:
        score -= 60.0

    # Binary/label masks are especially likely lesion candidates.
    unique = np.unique(sample[:100_000])
    if unique.size <= 2:
        score += 38.0
    elif unique.size <= 8:
        score += 24.0
    elif unique.size <= 32:
        score += 8.0

    # A segmentation is usually sparse, but do not reject dense synthetic lesions.
    if 0.00001 < nonzero_fraction < 0.5:
        score += 15.0
    elif nonzero_fraction in (0.0, 1.0):
        score -= 15.0

    min_dim = min(int(v) for v in array.shape)
    score += min(12.0, math.log2(max(min_dim, 2)))
    return score


def _suggest_threshold(values: np.ndarray, min_value: float, max_value: float) -> float:
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return 0.5
    if min_value == max_value:
        return min_value

    unique = np.unique(values[:100_000])
    if unique.size <= 32:
        positive = unique[unique > min_value]
        if min_value <= 0.0 and positive.size:
            return float((min_value + positive[0]) / 2.0)
        if unique.size >= 2:
            return float((unique[0] + unique[1]) / 2.0)
    return float((min_value + max_value) / 2.0)


def _find_spacing(value: Any) -> tuple[float, float, float] | None:
    """Find XYZ voxel spacing from common metadata conventions.

    In addition to explicit three-value spacing fields, this understands DICOM
    ``PixelSpacing`` plus a through-plane value. For lesion-library files that
    carry ``LesionPhysicalSize`` and ``LesionVoxelCount``, the through-plane
    spacing is derived from those values; this preserves the volume represented
    by the source metadata and is often more accurate than SliceThickness.
    """

    explicit: list[tuple[int, tuple[float, float, float]]] = []
    pixel_pairs: list[tuple[float, float]] = []  # DICOM order: row (Y), column (X)
    through_plane: list[tuple[int, float]] = []
    physical_sizes: list[float] = []
    voxel_counts: list[float] = []
    visited: set[int] = set()

    def scalar(node: Any) -> float | None:
        try:
            array = np.asarray(node, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        finite = array[np.isfinite(array)]
        if finite.size != 1:
            return None
        result = float(finite[0])
        return result if result > 0.0 else None

    def pair(node: Any) -> tuple[float, float] | None:
        try:
            array = np.asarray(node, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        array = array[np.isfinite(array)]
        if array.size < 2:
            return None
        row, column = float(array[0]), float(array[1])
        if not (0.0 < row <= 1000.0 and 0.0 < column <= 1000.0):
            return None
        return row, column

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 14:
            return

        if isinstance(node, (Mapping, list, tuple, np.ndarray)):
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)

        if isinstance(node, Mapping):
            for key, item in node.items():
                key_text = str(key)
                compact = re.sub(r"[^a-z0-9]", "", key_text.casefold())
                underscored = re.sub(r"[^a-z0-9_]", "", key_text.casefold())

                if underscored in _SPACING_KEYS:
                    spacing = _coerce_spacing(item)
                    if spacing:
                        priority = 0 if "voxel" in compact else 2
                        explicit.append((priority, spacing))

                if compact in {"pixelspacing", "imagerpixelspacing"}:
                    candidate_pair = pair(item)
                    if candidate_pair:
                        pixel_pairs.append(candidate_pair)
                elif compact in {
                    "spacingbetweenslices",
                    "slicespacing",
                    "zspacing",
                    "spacingz",
                }:
                    candidate = scalar(item)
                    if candidate and candidate <= 1000.0:
                        through_plane.append((0, candidate))
                elif compact == "slicethickness":
                    candidate = scalar(item)
                    if candidate and candidate <= 1000.0:
                        through_plane.append((2, candidate))
                elif compact in {"lesionphysicalsize", "physicalsize"}:
                    candidate = scalar(item)
                    if candidate:
                        physical_sizes.append(candidate)
                elif compact in {"lesionvoxelcount", "voxelcount"}:
                    candidate = scalar(item)
                    if candidate:
                        voxel_counts.append(candidate)
                elif compact in {"dicomheaderjson", "headerjson"}:
                    try:
                        raw = np.asarray(item).reshape(-1)
                        if raw.size == 1:
                            parsed = json.loads(str(raw[0]))
                            if isinstance(parsed, Mapping):
                                walk(parsed, depth + 1)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass

                if isinstance(item, (Mapping, list, tuple)) or (
                    isinstance(item, np.ndarray) and item.dtype == object
                ):
                    walk(item, depth + 1)
            return

        if isinstance(node, (list, tuple)):
            for item in node[:200]:
                walk(item, depth + 1)
            return

        if isinstance(node, np.ndarray) and node.dtype == object:
            for item in list(node.flat[:200]):
                walk(item, depth + 1)

    walk(value)

    if explicit:
        explicit.sort(key=lambda item: item[0])
        return explicit[0][1]

    if not pixel_pairs:
        return None

    # DICOM PixelSpacing is [row_spacing, column_spacing] == [Y, X].
    row_spacing, column_spacing = pixel_pairs[0]
    x_spacing = column_spacing
    y_spacing = row_spacing

    # The source lesion library stores physical size in mm^3. Derive Z from it
    # when possible, before falling back to thickness metadata.
    if physical_sizes and voxel_counts:
        z_spacing = physical_sizes[0] / (voxel_counts[0] * x_spacing * y_spacing)
        if math.isfinite(z_spacing) and 0.0001 <= z_spacing <= 1000.0:
            return (x_spacing, y_spacing, float(z_spacing))

    if through_plane:
        through_plane.sort(key=lambda item: item[0])
        return (x_spacing, y_spacing, through_plane[0][1])

    return None


def _infer_axis_order(value: Any) -> str:
    """Infer the stored dimension labels for recognized lesion-library schemas."""

    keys: set[str] = set()
    visited: set[int] = set()

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(node, (Mapping, list, tuple, np.ndarray)):
            identity = id(node)
            if identity in visited:
                return
            visited.add(identity)

        if isinstance(node, Mapping):
            for key, item in node.items():
                keys.add(re.sub(r"[^a-z0-9]", "", str(key).casefold()))
                if isinstance(item, (Mapping, list, tuple)) or (
                    isinstance(item, np.ndarray) and item.dtype == object
                ):
                    walk(item, depth + 1)
        elif isinstance(node, (list, tuple)):
            for item in node[:100]:
                walk(item, depth + 1)
        elif isinstance(node, np.ndarray) and node.dtype == object:
            for item in list(node.flat[:100]):
                walk(item, depth + 1)

    walk(value)
    source_schema = {"orgrowrng", "orgcolrng", "orgslicerng"}
    if source_schema.issubset(keys):
        # These lesion-library masks are stored as [row, column, slice].
        return "YXZ"
    return "ZYX"


def _find_hdf5_spacing(handle: Any) -> tuple[float, float, float] | None:
    assert h5py is not None
    found: list[tuple[int, tuple[float, float, float]]] = []

    def consider(name: str, value: Any) -> None:
        normalized = re.sub(r"[^a-z0-9_]", "", name.casefold().split("/")[-1])
        if normalized not in _SPACING_KEYS:
            return
        spacing = _coerce_spacing(value)
        if spacing:
            priority = 0 if "voxel" in normalized else 1
            found.append((priority, spacing))

    for attr_name, attr_value in handle.attrs.items():
        consider(str(attr_name), attr_value)

    def visitor(name: str, obj: Any) -> None:
        for attr_name, attr_value in getattr(obj, "attrs", {}).items():
            consider(str(attr_name), attr_value)
        if isinstance(obj, h5py.Dataset) and obj.size <= 16:
            try:
                consider(name, np.asarray(obj))
            except Exception:
                pass

    handle.visititems(visitor)
    if not found:
        return None
    found.sort(key=lambda item: item[0])
    return found[0][1]


def _coerce_spacing(value: Any) -> tuple[float, float, float] | None:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)]
    if array.size >= 4 and str(getattr(value, "name", "")).casefold().endswith(
        "pixdim"
    ):
        array = array[1:4]
    if array.size < 3:
        return None
    result = tuple(float(v) for v in array[:3])
    if any(v <= 0.0 or v > 1000.0 for v in result):
        return None
    return result  # type: ignore[return-value]


def _looks_like_hdf5_mat_error(exc: BaseException) -> bool:
    text = str(exc).casefold()
    return any(
        marker in text
        for marker in (
            "matlab 7.3",
            "hdf reader",
            "hdf5",
            "unknown mat file type",
            "please use hdf reader",
        )
    )


def _join_key(parent: str, child: str) -> str:
    return f"{parent}.{child}" if parent else child


# -----------------------------------------------------------------------------
# PySide6 / VTK viewer
# -----------------------------------------------------------------------------

import datetime as _dt
import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import vtkmodules.all as vtk
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    QSignalBlocker,
    QThreadPool,
    QTimer,
    Qt,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk


class _ScanSignals(QObject):
    batch = Signal(int, object)
    finished = Signal(int, int)
    failed = Signal(int, str)


class _ScanWorker(QRunnable):
    def __init__(
        self,
        generation: int,
        source: LocalDirectoryLesionSource,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self.generation = generation
        self.source = source
        self.cancel_event = cancel_event
        self.signals = _ScanSignals()

    @Slot()
    def run(self) -> None:
        try:
            records = self.source.scan(
                cancel_event=self.cancel_event,
                batch_callback=lambda batch: self.signals.batch.emit(
                    self.generation, batch
                ),
                batch_size=100,
            )
            if not self.cancel_event.is_set():
                self.signals.finished.emit(self.generation, len(records))
        except Exception as exc:
            if not self.cancel_event.is_set():
                self.signals.failed.emit(self.generation, str(exc))


class _LoadSignals(QObject):
    loaded = Signal(int, object)
    failed = Signal(int, str)


class _LoadWorker(QRunnable):
    def __init__(self, generation: int, path: Path) -> None:
        super().__init__()
        self.generation = generation
        self.path = path
        self.signals = _LoadSignals()

    @Slot()
    def run(self) -> None:
        try:
            data = load_lesion_file(self.path)
            self.signals.loaded.emit(self.generation, data)
        except Exception as exc:
            self.signals.failed.emit(self.generation, str(exc))


class LesionViewerModule(QWidget):
    """Recursive local lesion library browser with an embedded VTK 3-D viewer.

    The widget is intentionally self-contained so it can be mounted as a central
    workspace/module in the existing PySide6 application. File discovery and MAT/NPZ loading
    run in the shared Qt thread pool; rendering remains on the GUI thread as required
    by the Qt/VTK render window.
    """

    directory_changed = Signal(str)
    file_loaded = Signal(str)
    status_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        initial_directory: str | Path | None = None,
        restore_last_directory: bool = True,
        settings_prefix: str = "lesion_viewer",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("LesionViewerModule")
        self.setAcceptDrops(True)

        self._settings = QSettings()
        self._settings_prefix = settings_prefix.rstrip("/")
        self._thread_pool = QThreadPool.globalInstance()
        self._scan_generation = 0
        self._scan_cancel = threading.Event()
        self._load_generation = 0
        self._records_by_path: dict[str, LesionFileRecord] = {}
        self._file_data: LesionFileData | None = None
        self._candidate: VolumeCandidate | None = None
        self._current_file: Path | None = None
        self._surface_polydata: vtk.vtkPolyData | None = None
        self._last_render_metrics: dict[str, Any] = {}
        self._threshold_bounds = (0.0, 1.0)
        self._surface_color = QColor(230, 128, 76)
        self._needs_camera_reset = True
        self._cleaned_up = False
        self._pending_selection_path: Path | None = None

        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(180)
        self._selection_timer.timeout.connect(self._load_pending_selection)

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(140)
        self._render_timer.timeout.connect(self._rebuild_surface)

        self._build_ui()
        self._build_vtk_scene()
        self._connect_controls()
        self._install_shortcuts()
        self._apply_actor_style()
        self._set_controls_enabled(False)

        directory_to_open: str | Path | None = initial_directory
        if directory_to_open is None and restore_last_directory:
            directory_to_open = self._settings.value(
                f"{self._settings_prefix}/last_directory", "", type=str
            )
        if directory_to_open and Path(directory_to_open).is_dir():
            QTimer.singleShot(0, lambda: self.set_directory(directory_to_open))

    # ------------------------------------------------------------------
    # Public integration API
    # ------------------------------------------------------------------

    @property
    def current_directory(self) -> Path | None:
        text = self.directory_field.text().strip()
        return Path(text) if text else None

    @property
    def current_file(self) -> Path | None:
        return self._current_file

    def set_directory(self, directory: str | Path) -> None:
        path = Path(directory).expanduser().resolve()
        if not path.is_dir():
            QMessageBox.warning(
                self,
                "Invalid lesion library",
                f"The selected directory does not exist:\n{path}",
            )
            return

        self.directory_field.setText(str(path))
        self.directory_field.setToolTip(str(path))
        self._settings.setValue(f"{self._settings_prefix}/last_directory", str(path))
        self.directory_changed.emit(str(path))
        self.refresh()

    def load_file(self, path: str | Path) -> None:
        file_path = Path(path).expanduser().resolve()
        if file_path.suffix.lower() not in {".mat", ".npz"}:
            QMessageBox.warning(
                self,
                "Unsupported file",
                "Only .mat and .npz lesion files are supported.",
            )
            return
        if not file_path.is_file():
            QMessageBox.warning(self, "Missing file", f"File not found:\n{file_path}")
            return
        self._request_load(file_path)

    def refresh(self) -> None:
        directory = self.current_directory
        if directory is None:
            return

        self._scan_cancel.set()
        self._scan_cancel = threading.Event()
        self._scan_generation += 1
        generation = self._scan_generation

        self.file_tree.setSortingEnabled(False)
        self.file_tree.clear()
        self._records_by_path.clear()
        self.count_label.setText("Scanning recursively…")
        self._set_busy("Scanning lesion library…", indeterminate=True)

        source = LocalDirectoryLesionSource(directory)
        worker = _ScanWorker(generation, source, self._scan_cancel)
        worker.signals.batch.connect(self._on_scan_batch)
        worker.signals.finished.connect(self._on_scan_finished)
        worker.signals.failed.connect(self._on_scan_failed)
        self._thread_pool.start(worker)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        self._scan_cancel.set()
        self._selection_timer.stop()
        self._render_timer.stop()
        try:
            self.axes_widget.EnabledOff()
        except Exception:
            pass
        try:
            render_window = self.vtk_widget.GetRenderWindow()
            if render_window is not None:
                render_window.Finalize()
        except Exception:
            pass
        self.vtk_widget.close()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        # Browser panel -------------------------------------------------
        browser = QWidget(splitter)
        browser_layout = QVBoxLayout(browser)
        browser_layout.setContentsMargins(0, 0, 4, 0)
        browser_layout.setSpacing(6)

        heading = QLabel("Local Lesion Library")
        heading.setStyleSheet("font-weight: 600; font-size: 15px;")
        browser_layout.addWidget(heading)

        directory_row = QHBoxLayout()
        self.directory_field = QLineEdit()
        self.directory_field.setReadOnly(True)
        self.directory_field.setPlaceholderText(
            "Select a directory containing lesion files"
        )
        directory_row.addWidget(self.directory_field, 1)
        self.browse_button = QPushButton("Browse…")
        directory_row.addWidget(self.browse_button)
        browser_layout.addLayout(directory_row)

        self.filter_field = QLineEdit()
        self.filter_field.setPlaceholderText("Filter by filename or relative folder…")
        self.filter_field.setClearButtonEnabled(True)
        browser_layout.addWidget(self.filter_field)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["Name", "Folder", "Type", "Size"])
        self.file_tree.setRootIsDecorated(False)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.file_tree.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_tree.setSortingEnabled(True)
        self.file_tree.setColumnWidth(0, 175)
        self.file_tree.setColumnWidth(1, 180)
        self.file_tree.setColumnWidth(2, 55)
        browser_layout.addWidget(self.file_tree, 1)

        browser_buttons = QHBoxLayout()
        self.load_button = QPushButton("Load Selected")
        self.load_button.setEnabled(False)
        browser_buttons.addWidget(self.load_button)
        self.refresh_button = QPushButton("Refresh")
        browser_buttons.addWidget(self.refresh_button)
        browser_layout.addLayout(browser_buttons)

        options_row = QHBoxLayout()
        self.auto_load_checkbox = QCheckBox("Auto-load selection")
        self.auto_load_checkbox.setChecked(True)
        options_row.addWidget(self.auto_load_checkbox)
        options_row.addStretch(1)
        self.count_label = QLabel("0 files")
        self.count_label.setStyleSheet("color: palette(mid);")
        options_row.addWidget(self.count_label)
        browser_layout.addLayout(options_row)

        # Viewer panel --------------------------------------------------
        viewer_panel = QWidget(splitter)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(4, 0, 0, 0)
        viewer_layout.setSpacing(6)

        viewer_toolbar = QHBoxLayout()
        viewer_toolbar.addWidget(QLabel("Array:"))
        self.candidate_combo = QComboBox()
        self.candidate_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.candidate_combo.setMinimumContentsLength(28)
        viewer_toolbar.addWidget(self.candidate_combo, 1)

        self.reset_camera_button = QPushButton("Reset Camera")
        viewer_toolbar.addWidget(self.reset_camera_button)
        self.view_x_button = QPushButton("+X")
        self.view_x_button.setToolTip("View along the positive X axis")
        viewer_toolbar.addWidget(self.view_x_button)
        self.view_y_button = QPushButton("+Y")
        self.view_y_button.setToolTip("View along the positive Y axis")
        viewer_toolbar.addWidget(self.view_y_button)
        self.view_z_button = QPushButton("+Z")
        self.view_z_button.setToolTip("View along the positive Z axis")
        viewer_toolbar.addWidget(self.view_z_button)

        self.screenshot_button = QPushButton("Screenshot…")
        viewer_toolbar.addWidget(self.screenshot_button)
        self.export_button = QPushButton("Export Surface…")
        viewer_toolbar.addWidget(self.export_button)
        viewer_layout.addLayout(viewer_toolbar)

        self.viewer_stack = QStackedWidget()
        self.placeholder = QLabel(
            "No lesion model loaded.\n\n"
            "Choose a local library directory, then select a .mat or .npz file.\n"
            "Files are discovered recursively."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet(
            "QLabel { color: palette(mid); border: 1px solid palette(mid); "
            "border-radius: 4px; padding: 24px; }"
        )
        self.placeholder.setAcceptDrops(False)
        self.viewer_stack.addWidget(self.placeholder)

        self.vtk_widget = QVTKRenderWindowInteractor(self.viewer_stack)
        self.viewer_stack.addWidget(self.vtk_widget)
        self.viewer_stack.setCurrentWidget(self.placeholder)
        viewer_layout.addWidget(self.viewer_stack, 1)

        self.controls_tabs = QTabWidget()
        self.controls_tabs.setMaximumHeight(275)
        viewer_layout.addWidget(self.controls_tabs)

        self._build_surface_tab()
        self._build_appearance_tab()
        self._build_info_tab()

        splitter.addWidget(browser)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 1040])

        # Module-local status line -------------------------------------
        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(2, 2, 2, 2)
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 1)
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(190)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        status_layout.addWidget(self.progress_bar)
        root_layout.addWidget(status_frame)

    def _build_surface_tab(self) -> None:
        tab = QWidget()
        grid = QGridLayout(tab)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)

        grid.addWidget(QLabel("Threshold:"), 0, 0)
        threshold_container = QWidget()
        threshold_layout = QHBoxLayout(threshold_container)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        threshold_layout.addWidget(self.threshold_slider, 1)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(5)
        self.threshold_spin.setKeyboardTracking(False)
        self.threshold_spin.setMinimumWidth(100)
        threshold_layout.addWidget(self.threshold_spin)
        self.auto_threshold_button = QPushButton("Auto")
        threshold_layout.addWidget(self.auto_threshold_button)
        grid.addWidget(threshold_container, 0, 1, 1, 5)

        grid.addWidget(QLabel("Spacing X/Y/Z:"), 1, 0)
        self.spacing_x = self._make_spacing_spinbox()
        self.spacing_y = self._make_spacing_spinbox()
        self.spacing_z = self._make_spacing_spinbox()
        grid.addWidget(self.spacing_x, 1, 1)
        grid.addWidget(self.spacing_y, 1, 2)
        grid.addWidget(self.spacing_z, 1, 3)
        self.spacing_source_label = QLabel("manual")
        self.spacing_source_label.setStyleSheet("color: palette(mid);")
        grid.addWidget(self.spacing_source_label, 1, 4, 1, 2)

        grid.addWidget(QLabel("Input axis order:"), 2, 0)
        self.axis_order_combo = QComboBox()
        for order in ("ZYX", "XYZ", "ZXY", "YXZ", "YZX", "XZY"):
            self.axis_order_combo.addItem(order, order)
        self.axis_order_combo.setCurrentText("ZYX")
        self.axis_order_combo.setToolTip(
            "Names the axes represented by dimensions 0, 1, and 2 of the stored array."
        )
        grid.addWidget(self.axis_order_combo, 2, 1)

        grid.addWidget(QLabel("Preview downsample:"), 2, 2)
        self.downsample_combo = QComboBox()
        self.downsample_combo.addItem("1x (full)", 1)
        self.downsample_combo.addItem("2x", 2)
        self.downsample_combo.addItem("4x", 4)
        self.downsample_combo.addItem("8x", 8)
        grid.addWidget(self.downsample_combo, 2, 3)

        grid.addWidget(QLabel("Smoothing:"), 2, 4)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 100)
        self.smoothing_spin.setSingleStep(5)
        self.smoothing_spin.setValue(20)
        self.smoothing_spin.setToolTip("0 disables smoothing.")
        grid.addWidget(self.smoothing_spin, 2, 5)

        self.largest_component_checkbox = QCheckBox("Largest component only")
        self.largest_component_checkbox.setChecked(True)
        grid.addWidget(self.largest_component_checkbox, 3, 0, 1, 2)

        self.pad_border_checkbox = QCheckBox("Pad border for closed surfaces")
        self.pad_border_checkbox.setChecked(True)
        grid.addWidget(self.pad_border_checkbox, 3, 2, 1, 2)

        grid.addWidget(QLabel("Mesh reduction:"), 3, 4)
        self.reduction_spin = QSpinBox()
        self.reduction_spin.setRange(0, 95)
        self.reduction_spin.setSuffix(" %")
        self.reduction_spin.setValue(0)
        self.reduction_spin.setToolTip(
            "Reduces triangle count after surface extraction; 0 keeps the full mesh."
        )
        grid.addWidget(self.reduction_spin, 3, 5)

        self.foreground_below_checkbox = QCheckBox(
            "Treat values below the threshold as the lesion interior"
        )
        self.foreground_below_checkbox.setToolTip(
            "Useful when the lesion is encoded with lower values than the background."
        )
        grid.addWidget(self.foreground_below_checkbox, 4, 0, 1, 4)

        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setColumnStretch(3, 1)
        self.controls_tabs.addTab(tab, "Surface")

    def _build_appearance_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(10, 8, 10, 8)

        self.color_button = QPushButton("Choose…")
        form.addRow("Surface color:", self.color_button)

        opacity_container = QWidget()
        opacity_layout = QHBoxLayout(opacity_container)
        opacity_layout.setContentsMargins(0, 0, 0, 0)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(100)
        opacity_layout.addWidget(self.opacity_slider, 1)
        self.opacity_label = QLabel("100%")
        self.opacity_label.setMinimumWidth(42)
        opacity_layout.addWidget(self.opacity_label)
        form.addRow("Opacity:", opacity_container)

        self.representation_combo = QComboBox()
        self.representation_combo.addItems(["Surface", "Wireframe", "Points"])
        form.addRow("Representation:", self.representation_combo)

        self.edge_checkbox = QCheckBox("Show mesh edges")
        form.addRow("Edges:", self.edge_checkbox)

        self.parallel_projection_checkbox = QCheckBox("Use parallel projection")
        form.addRow("Camera:", self.parallel_projection_checkbox)

        self.controls_tabs.addTab(tab, "Appearance")

    def _build_info_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlaceholderText("File and mesh information will appear here.")
        layout.addWidget(self.info_text)
        self.controls_tabs.addTab(tab, "Information")

    @staticmethod
    def _make_spacing_spinbox() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0001, 1000.0)
        box.setDecimals(5)
        box.setSingleStep(0.1)
        box.setValue(1.0)
        box.setSuffix(" mm")
        box.setKeyboardTracking(False)
        return box

    def _build_vtk_scene(self) -> None:
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.075, 0.085, 0.11)
        self.renderer.SetBackground2(0.16, 0.18, 0.22)
        self.renderer.GradientBackgroundOn()

        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)
        render_window.SetMultiSamples(4)
        self.interactor = render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.VisibilityOff()
        self.renderer.AddActor(self.actor)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.14, 0.14)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        self.vtk_widget.Initialize()

    def _connect_controls(self) -> None:
        self.browse_button.clicked.connect(self._browse_directory)
        self.refresh_button.clicked.connect(self.refresh)
        self.load_button.clicked.connect(self._load_selected_item)
        self.filter_field.textChanged.connect(self._apply_filter)
        self.file_tree.itemSelectionChanged.connect(self._selection_changed)
        self.file_tree.itemDoubleClicked.connect(
            lambda _item, _column: self._load_selected_item()
        )
        self.file_tree.customContextMenuRequested.connect(self._show_file_context_menu)

        self.candidate_combo.currentIndexChanged.connect(self._candidate_changed)
        self.threshold_slider.valueChanged.connect(self._threshold_slider_changed)
        self.threshold_spin.valueChanged.connect(self._threshold_spin_changed)
        self.auto_threshold_button.clicked.connect(self._restore_auto_threshold)

        for control in (
            self.spacing_x,
            self.spacing_y,
            self.spacing_z,
            self.smoothing_spin,
            self.reduction_spin,
        ):
            control.valueChanged.connect(self._schedule_surface_rebuild)
        self.axis_order_combo.currentIndexChanged.connect(
            self._schedule_surface_rebuild
        )
        self.downsample_combo.currentIndexChanged.connect(
            self._schedule_surface_rebuild
        )
        self.largest_component_checkbox.toggled.connect(self._schedule_surface_rebuild)
        self.pad_border_checkbox.toggled.connect(self._schedule_surface_rebuild)
        self.foreground_below_checkbox.toggled.connect(self._schedule_surface_rebuild)

        self.color_button.clicked.connect(self._choose_surface_color)
        self.opacity_slider.valueChanged.connect(self._apply_actor_style)
        self.representation_combo.currentIndexChanged.connect(self._apply_actor_style)
        self.edge_checkbox.toggled.connect(self._apply_actor_style)
        self.parallel_projection_checkbox.toggled.connect(
            self._parallel_projection_changed
        )

        self.reset_camera_button.clicked.connect(self._reset_camera)
        self.view_x_button.clicked.connect(lambda: self._set_axis_view("X"))
        self.view_y_button.clicked.connect(lambda: self._set_axis_view("Y"))
        self.view_z_button.clicked.connect(lambda: self._set_axis_view("Z"))
        self.screenshot_button.clicked.connect(self._save_screenshot)
        self.export_button.clicked.connect(self._export_surface)

    def _install_shortcuts(self) -> None:
        self._shortcuts: list[QShortcut] = []
        for sequence, callback in (
            ("Ctrl+O", self._browse_directory),
            ("Ctrl+R", self.refresh),
            ("Ctrl+F", self._focus_filter),
            ("R", self._reset_camera),
            ("Ctrl+Shift+S", self._save_screenshot),
        ):
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------------
    # Browser behavior
    # ------------------------------------------------------------------

    def _browse_directory(self) -> None:
        start = str(self.current_directory or Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Lesion Library Directory",
            start,
        )
        if directory:
            self.set_directory(directory)

    def _focus_filter(self) -> None:
        self.filter_field.setFocus()
        self.filter_field.selectAll()

    def _on_scan_batch(self, generation: int, records: object) -> None:
        if generation != self._scan_generation:
            return
        for record in records:  # type: ignore[union-attr]
            if not isinstance(record, LesionFileRecord):
                continue
            path_key = str(record.path)
            self._records_by_path[path_key] = record
            relative_parent = str(Path(record.relative_path).parent)
            if relative_parent == ".":
                relative_parent = ""
            item = QTreeWidgetItem(
                [
                    record.name,
                    relative_parent,
                    record.suffix.upper().lstrip("."),
                    _format_size(record.size_bytes),
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, path_key)
            modified = _dt.datetime.fromtimestamp(record.modified_time).strftime(
                "%Y-%m-%d %H:%M"
            )
            item.setToolTip(
                0,
                f"{record.path}\nModified: {modified}\nSize: {record.size_bytes:,} bytes",
            )
            self.file_tree.addTopLevelItem(item)
        self.count_label.setText(f"{len(self._records_by_path):,} found…")
        self._apply_filter(self.filter_field.text())

    def _on_scan_finished(self, generation: int, count: int) -> None:
        if generation != self._scan_generation:
            return
        self.file_tree.setSortingEnabled(True)
        self.file_tree.sortItems(1, Qt.SortOrder.AscendingOrder)
        self.count_label.setText(f"{count:,} file{'s' if count != 1 else ''}")
        self._set_ready(f"Found {count:,} lesion file{'s' if count != 1 else ''}.")
        if count == 0:
            self.placeholder.setText(
                "No .mat or .npz files were found in this directory or its subdirectories."
            )
            self.viewer_stack.setCurrentWidget(self.placeholder)

    def _on_scan_failed(self, generation: int, message: str) -> None:
        if generation != self._scan_generation:
            return
        self.file_tree.setSortingEnabled(True)
        self.count_label.setText("Scan failed")
        self._set_ready("Lesion library scan failed.")
        QMessageBox.critical(self, "Library scan failed", message)

    def _apply_filter(self, text: str) -> None:
        query = text.strip().casefold()
        visible_count = 0
        for index in range(self.file_tree.topLevelItemCount()):
            item = self.file_tree.topLevelItem(index)
            haystack = " ".join(item.text(column) for column in (0, 1, 2)).casefold()
            visible = not query or query in haystack
            item.setHidden(not visible)
            if visible:
                visible_count += 1
        total = len(self._records_by_path)
        if query:
            self.count_label.setText(f"{visible_count:,} / {total:,}")
        elif total:
            self.count_label.setText(f"{total:,} file{'s' if total != 1 else ''}")

    def _selection_changed(self) -> None:
        path = self._selected_path()
        self.load_button.setEnabled(path is not None)
        if path and self.auto_load_checkbox.isChecked():
            self._pending_selection_path = path
            self._selection_timer.start()

    def _load_pending_selection(self) -> None:
        if self._pending_selection_path is not None:
            self._request_load(self._pending_selection_path)

    def _load_selected_item(self) -> None:
        path = self._selected_path()
        if path is not None:
            self._selection_timer.stop()
            self._request_load(path)

    def _selected_path(self) -> Path | None:
        items = self.file_tree.selectedItems()
        if not items:
            return None
        value = items[0].data(0, Qt.ItemDataRole.UserRole)
        return Path(str(value)) if value else None

    def _show_file_context_menu(self, position: Any) -> None:
        item = self.file_tree.itemAt(position)
        if item is None:
            return
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if not value:
            return
        path = Path(str(value))
        menu = QMenu(self)
        load_action = menu.addAction("Load")
        copy_action = menu.addAction("Copy Path")
        reveal_action = menu.addAction("Open Containing Folder")
        chosen = menu.exec(self.file_tree.viewport().mapToGlobal(position))
        if chosen == load_action:
            self._request_load(path)
        elif chosen == copy_action:
            QApplication.clipboard().setText(str(path))
        elif chosen == reveal_action:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    # ------------------------------------------------------------------
    # File loading and candidate selection
    # ------------------------------------------------------------------

    def _request_load(self, path: Path) -> None:
        if self._current_file == path and self._file_data is not None:
            return
        self._load_generation += 1
        generation = self._load_generation
        self._set_busy(f"Loading {path.name}…", indeterminate=True)
        self.candidate_combo.setEnabled(False)
        worker = _LoadWorker(generation, path)
        worker.signals.loaded.connect(self._on_file_loaded)
        worker.signals.failed.connect(self._on_file_load_failed)
        self._thread_pool.start(worker)

    def _on_file_loaded(self, generation: int, payload: object) -> None:
        if generation != self._load_generation or not isinstance(
            payload, LesionFileData
        ):
            return

        self._file_data = payload
        self._current_file = payload.path
        self._surface_polydata = None
        self._needs_camera_reset = True

        with QSignalBlocker(self.candidate_combo):
            self.candidate_combo.clear()
            for candidate in payload.candidates:
                self.candidate_combo.addItem(candidate.display_name)
            self.candidate_combo.setCurrentIndex(payload.recommended_index)
        self.candidate_combo.setEnabled(True)

        detected_spacing = payload.detected_spacing_xyz
        if detected_spacing:
            self._set_spacing(detected_spacing)
            self.spacing_source_label.setText("detected from file")
        else:
            self._set_spacing((1.0, 1.0, 1.0))
            self.spacing_source_label.setText("default / editable")

        suggested_order = str(payload.file_metadata.get("suggested_axis_order", "ZYX"))
        if self.axis_order_combo.findData(suggested_order) >= 0:
            with QSignalBlocker(self.axis_order_combo):
                self.axis_order_combo.setCurrentIndex(
                    self.axis_order_combo.findData(suggested_order)
                )

        self._candidate_changed(payload.recommended_index)
        self.file_loaded.emit(str(payload.path))
        warning_suffix = (
            f" ({len(payload.warnings)} warning(s))" if payload.warnings else ""
        )
        self._set_ready(f"Loaded {payload.path.name}{warning_suffix}")

    def _on_file_load_failed(self, generation: int, message: str) -> None:
        if generation != self._load_generation:
            return
        self.candidate_combo.setEnabled(bool(self._file_data))
        self._set_ready("Lesion file load failed.")
        QMessageBox.critical(self, "Could not load lesion file", message)

    def _candidate_changed(self, index: int) -> None:
        if self._file_data is None or not (
            0 <= index < len(self._file_data.candidates)
        ):
            self._candidate = None
            self._set_controls_enabled(False)
            return
        self._candidate = self._file_data.candidates[index]
        self._configure_threshold_controls(self._candidate)
        self._set_controls_enabled(True)
        self._needs_camera_reset = True
        self._rebuild_surface()

    def _configure_threshold_controls(self, candidate: VolumeCandidate) -> None:
        minimum = candidate.min_value
        maximum = candidate.max_value
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum == maximum
        ):
            padding = max(0.5, abs(minimum) * 0.05)
            low, high = minimum - padding, maximum + padding
        else:
            low, high = minimum, maximum
        self._threshold_bounds = (float(low), float(high))

        with QSignalBlocker(self.threshold_spin), QSignalBlocker(self.threshold_slider):
            self.threshold_spin.setRange(low, high)
            span = max(high - low, 1e-12)
            self.threshold_spin.setSingleStep(span / 200.0)
            decimals = max(3, min(8, int(max(0.0, -math.log10(span))) + 4))
            self.threshold_spin.setDecimals(decimals)
            value = min(max(candidate.suggested_threshold, low), high)
            self.threshold_spin.setValue(value)
            self.threshold_slider.setValue(self._threshold_to_slider(value))

    def _set_spacing(self, spacing: tuple[float, float, float]) -> None:
        for box, value in zip(
            (self.spacing_x, self.spacing_y, self.spacing_z), spacing, strict=True
        ):
            with QSignalBlocker(box):
                box.setValue(float(value))

    # ------------------------------------------------------------------
    # Surface controls and rendering
    # ------------------------------------------------------------------

    def _threshold_slider_changed(self, slider_value: int) -> None:
        low, high = self._threshold_bounds
        value = low + (high - low) * (slider_value / 1000.0)
        with QSignalBlocker(self.threshold_spin):
            self.threshold_spin.setValue(value)
        self._schedule_surface_rebuild()

    def _threshold_spin_changed(self, value: float) -> None:
        with QSignalBlocker(self.threshold_slider):
            self.threshold_slider.setValue(self._threshold_to_slider(value))
        self._schedule_surface_rebuild()

    def _threshold_to_slider(self, value: float) -> int:
        low, high = self._threshold_bounds
        if high <= low:
            return 500
        ratio = (value - low) / (high - low)
        return int(round(min(max(ratio, 0.0), 1.0) * 1000.0))

    def _restore_auto_threshold(self) -> None:
        if self._candidate is not None:
            self.threshold_spin.setValue(self._candidate.suggested_threshold)

    def _schedule_surface_rebuild(self, *_args: Any) -> None:
        if self._candidate is not None:
            self._render_timer.start()

    def _rebuild_surface(self) -> None:
        candidate = self._candidate
        if candidate is None:
            return

        started = time.perf_counter()
        self._set_busy("Building 3-D surface…", indeterminate=True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            order = str(self.axis_order_combo.currentData() or "ZYX")
            permutation = tuple(order.index(axis_name) for axis_name in "ZYX")
            volume_zyx = np.transpose(candidate.array, permutation)

            downsample = int(self.downsample_combo.currentData() or 1)
            if downsample > 1:
                volume_zyx = volume_zyx[::downsample, ::downsample, ::downsample]
            if any(int(size) < 2 for size in volume_zyx.shape):
                raise ValueError(
                    "The selected downsample factor is too large for this volume."
                )

            threshold = float(self.threshold_spin.value())
            foreground_below = self.foreground_below_checkbox.isChecked()
            spacing_xyz = (
                float(self.spacing_x.value()),
                float(self.spacing_y.value()),
                float(self.spacing_z.value()),
            )
            render_spacing = tuple(value * downsample for value in spacing_xyz)

            render_volume = np.asarray(volume_zyx, dtype=np.float32)
            pad = self.pad_border_checkbox.isChecked()
            origin = (0.0, 0.0, 0.0)
            if pad:
                span = max(abs(candidate.max_value - candidate.min_value), 1.0)
                if foreground_below:
                    background = threshold + span * 0.05 + 1e-6
                else:
                    background = threshold - span * 0.05 - 1e-6
                render_volume = np.pad(
                    render_volume,
                    1,
                    mode="constant",
                    constant_values=background,
                )
                origin = tuple(-value for value in render_spacing)
            if not np.isfinite(render_volume).all():
                offset = max(abs(threshold) * 0.01, 1e-6)
                replacement = (
                    threshold + offset if foreground_below else threshold - offset
                )
                render_volume = np.nan_to_num(
                    render_volume,
                    nan=replacement,
                    posinf=replacement,
                    neginf=replacement,
                )
            render_volume = np.ascontiguousarray(render_volume)

            z_size, y_size, x_size = (int(v) for v in render_volume.shape)
            image = vtk.vtkImageData()
            image.SetDimensions(x_size, y_size, z_size)
            image.SetSpacing(*render_spacing)
            image.SetOrigin(*origin)
            scalars = numpy_to_vtk(render_volume.ravel(order="C"), deep=True)
            scalars.SetName("LesionScalars")
            image.GetPointData().SetScalars(scalars)

            extractor = vtk.vtkFlyingEdges3D()
            extractor.SetInputData(image)
            extractor.SetValue(0, threshold)
            extractor.ComputeNormalsOff()
            extractor.ComputeGradientsOff()
            pipeline: Any = extractor

            if self.largest_component_checkbox.isChecked():
                connectivity = vtk.vtkPolyDataConnectivityFilter()
                connectivity.SetInputConnection(pipeline.GetOutputPort())
                connectivity.SetExtractionModeToLargestRegion()
                connectivity.ColorRegionsOff()
                pipeline = connectivity

            smoothing_iterations = int(self.smoothing_spin.value())
            if smoothing_iterations > 0:
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputConnection(pipeline.GetOutputPort())
                smoother.SetNumberOfIterations(smoothing_iterations)
                smoother.BoundarySmoothingOff()
                smoother.FeatureEdgeSmoothingOff()
                smoother.SetFeatureAngle(120.0)
                smoother.SetPassBand(0.01)
                smoother.NonManifoldSmoothingOn()
                smoother.NormalizeCoordinatesOn()
                pipeline = smoother

            reduction = float(self.reduction_spin.value()) / 100.0
            if reduction > 0.0:
                decimator = vtk.vtkDecimatePro()
                decimator.SetInputConnection(pipeline.GetOutputPort())
                decimator.SetTargetReduction(reduction)
                decimator.PreserveTopologyOn()
                decimator.SplittingOff()
                pipeline = decimator

            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(pipeline.GetOutputPort())
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOff()
            normals.SetFeatureAngle(80.0)
            normals.Update()

            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(normals.GetOutput())
            if polydata.GetNumberOfPoints() == 0:
                raise ValueError(
                    "The selected threshold produced an empty surface. Adjust the "
                    "threshold or select a different array."
                )

            self._surface_polydata = polydata
            self.mapper.SetInputData(polydata)
            self.mapper.Update()
            self.actor.VisibilityOn()
            self.viewer_stack.setCurrentWidget(self.vtk_widget)
            self._apply_actor_style(render=False)

            if self._needs_camera_reset:
                self.renderer.ResetCamera()
                self.renderer.ResetCameraClippingRange()
                self._needs_camera_reset = False
            self.vtk_widget.GetRenderWindow().Render()

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_render_metrics = self._calculate_metrics(
                candidate=candidate,
                threshold=threshold,
                spacing_xyz=spacing_xyz,
                axis_order=order,
                downsample=downsample,
                polydata=polydata,
                elapsed_ms=elapsed_ms,
                foreground_below=foreground_below,
            )
            self._update_info_text()
            self._set_ready(
                f"Rendered {polydata.GetNumberOfCells():,} triangles in "
                f"{elapsed_ms / 1000.0:.2f} s."
            )
        except ValueError as exc:
            self.actor.VisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()
            self._set_ready(str(exc))
        except Exception as exc:
            self._set_ready("Surface rendering failed.")
            QMessageBox.warning(self, "Could not render lesion surface", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _calculate_metrics(
        self,
        *,
        candidate: VolumeCandidate,
        threshold: float,
        spacing_xyz: tuple[float, float, float],
        axis_order: str,
        downsample: int,
        polydata: vtk.vtkPolyData,
        elapsed_ms: float,
        foreground_below: bool,
    ) -> dict[str, Any]:
        foreground_count: int | None
        estimated = False
        array = candidate.array
        if array.size <= 50_000_000:
            values = np.asarray(array)
            comparison = values < threshold if foreground_below else values > threshold
            foreground_count = int(np.count_nonzero(comparison))
        else:
            stride = max(1, array.size // 1_000_000)
            sample = np.asarray(array).reshape(-1)[::stride]
            comparison = sample < threshold if foreground_below else sample > threshold
            fraction = float(np.count_nonzero(comparison) / max(sample.size, 1))
            foreground_count = int(round(array.size * fraction))
            estimated = True

        voxel_volume_mm3 = float(np.prod(spacing_xyz))
        mask_volume_mm3 = foreground_count * voxel_volume_mm3

        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        mass.Update()
        surface_area = float(mass.GetSurfaceArea())
        mesh_volume = float(mass.GetVolume())

        return {
            "foreground_count": foreground_count,
            "foreground_estimated": estimated,
            "mask_volume_mm3": mask_volume_mm3,
            "surface_area_mm2": surface_area,
            "mesh_volume_mm3": mesh_volume,
            "points": int(polydata.GetNumberOfPoints()),
            "cells": int(polydata.GetNumberOfCells()),
            "threshold": threshold,
            "spacing_xyz": spacing_xyz,
            "axis_order": axis_order,
            "downsample": downsample,
            "elapsed_ms": elapsed_ms,
            "foreground_below": foreground_below,
        }

    def _update_info_text(self) -> None:
        if self._file_data is None or self._candidate is None:
            self.info_text.clear()
            return
        data = self._file_data
        candidate = self._candidate
        metrics = self._last_render_metrics
        spacing = metrics.get("spacing_xyz", (1.0, 1.0, 1.0))
        estimated_marker = " (estimated)" if metrics.get("foreground_estimated") else ""

        lines = [
            f"File: {data.path}",
            f"Format: {data.path.suffix.upper().lstrip('.')}",
            f"Array: {candidate.key}",
            f"Stored shape: {candidate.shape}",
            f"Dtype: {candidate.dtype_name}",
            f"Value range: {candidate.min_value:g} to {candidate.max_value:g}",
            f"Finite sample: {candidate.finite_fraction * 100.0:.2f}%",
            f"Axis order: {metrics.get('axis_order', self.axis_order_combo.currentText())}",
            f"Spacing (X, Y, Z): {spacing[0]:g}, {spacing[1]:g}, {spacing[2]:g} mm",
            f"Threshold: {metrics.get('threshold', self.threshold_spin.value()):g}",
            "Lesion interior: "
            + (
                "below threshold"
                if metrics.get("foreground_below")
                else "above threshold"
            ),
            f"Downsample: {metrics.get('downsample', 1)}x",
        ]
        if metrics:
            lines.extend(
                [
                    f"Foreground voxels: {metrics['foreground_count']:,}{estimated_marker}",
                    f"Mask volume: {metrics['mask_volume_mm3'] / 1000.0:.4f} cm³",
                    f"Mesh points: {metrics['points']:,}",
                    f"Mesh triangles: {metrics['cells']:,}",
                    f"Mesh surface area: {metrics['surface_area_mm2']:.4f} mm²",
                    f"Closed-mesh volume: {metrics['mesh_volume_mm3'] / 1000.0:.4f} cm³",
                    f"Last render: {metrics['elapsed_ms'] / 1000.0:.3f} s",
                ]
            )
        if data.detected_spacing_xyz:
            lines.append(
                "Detected spacing: "
                + ", ".join(f"{value:g}" for value in data.detected_spacing_xyz)
                + " mm"
            )
        if data.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"• {warning}" for warning in data.warnings)
        self.info_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Appearance and camera
    # ------------------------------------------------------------------

    def _choose_surface_color(self) -> None:
        color = QColorDialog.getColor(
            self._surface_color,
            self,
            "Choose Lesion Surface Color",
        )
        if color.isValid():
            self._surface_color = color
            self._apply_actor_style()

    def _apply_actor_style(self, *_args: Any, render: bool = True) -> None:
        prop = self.actor.GetProperty() if hasattr(self, "actor") else None
        if prop is None:
            return
        prop.SetColor(
            self._surface_color.redF(),
            self._surface_color.greenF(),
            self._surface_color.blueF(),
        )
        prop.SetOpacity(self.opacity_slider.value() / 100.0)
        prop.SetAmbient(0.12)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(28.0)
        prop.SetEdgeVisibility(self.edge_checkbox.isChecked())

        representation = self.representation_combo.currentText()
        if representation == "Wireframe":
            prop.SetRepresentationToWireframe()
        elif representation == "Points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(3.0)
        else:
            prop.SetRepresentationToSurface()

        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self.color_button.setStyleSheet(
            "QPushButton { background-color: "
            f"{self._surface_color.name()}; color: "
            f"{'black' if self._surface_color.lightnessF() > 0.55 else 'white'}; }}"
        )
        if render and hasattr(self, "vtk_widget"):
            self.vtk_widget.GetRenderWindow().Render()

    def _parallel_projection_changed(self, enabled: bool) -> None:
        self.renderer.GetActiveCamera().SetParallelProjection(enabled)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _reset_camera(self) -> None:
        if self._surface_polydata is None:
            return
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _set_axis_view(self, axis: str) -> None:
        if self._surface_polydata is None:
            return
        bounds = self.actor.GetBounds()
        if bounds is None or len(bounds) != 6:
            return
        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        distance = max(span * 2.5, 1.0)
        direction = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }[axis]
        view_up = (0.0, 0.0, 1.0) if axis in {"X", "Y"} else (0.0, 1.0, 0.0)

        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(
            center[0] + direction[0] * distance,
            center[1] + direction[1] * distance,
            center[2] + direction[2] * distance,
        )
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _save_screenshot(self) -> None:
        if self._surface_polydata is None:
            return
        suggested = (
            f"{self._current_file.stem}_lesion.png"
            if self._current_file
            else "lesion.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lesion Viewer Screenshot",
            suggested,
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(self.vtk_widget.GetRenderWindow())
            capture.SetScale(2)
            capture.ReadFrontBufferOff()
            capture.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(path)
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            self._set_ready(f"Saved screenshot: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Screenshot failed", str(exc))

    def _export_surface(self) -> None:
        if self._surface_polydata is None:
            return
        suggested = (
            f"{self._current_file.stem}_surface.stl"
            if self._current_file
            else "lesion_surface.stl"
        )
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Lesion Surface",
            suggested,
            "STL mesh (*.stl);;PLY mesh (*.ply);;VTK PolyData (*.vtp)",
        )
        if not path:
            return

        try:
            suffix = Path(path).suffix.lower()
            if "PLY" in selected_filter or suffix == ".ply":
                if suffix != ".ply":
                    path += ".ply"
                writer = vtk.vtkPLYWriter()
                writer.SetFileTypeToBinary()
            elif "VTK" in selected_filter or suffix == ".vtp":
                if suffix != ".vtp":
                    path += ".vtp"
                writer = vtk.vtkXMLPolyDataWriter()
                writer.SetDataModeToBinary()
            else:
                if suffix != ".stl":
                    path += ".stl"
                writer = vtk.vtkSTLWriter()
                writer.SetFileTypeToBinary()
            writer.SetFileName(path)
            writer.SetInputData(self._surface_polydata)
            if writer.Write() != 1:
                raise OSError("VTK did not confirm that the mesh was written.")
            self._set_ready(f"Exported surface: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Surface export failed", str(exc))

    # ------------------------------------------------------------------
    # Status, state, drag/drop, teardown
    # ------------------------------------------------------------------

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.controls_tabs,
            self.reset_camera_button,
            self.view_x_button,
            self.view_y_button,
            self.view_z_button,
            self.screenshot_button,
            self.export_button,
        ):
            widget.setEnabled(enabled)

    def _set_busy(self, message: str, *, indeterminate: bool) -> None:
        self.status_label.setText(message)
        self.status_changed.emit(message)
        self.progress_bar.setRange(0, 0 if indeterminate else 100)
        self.progress_bar.show()

    def _set_ready(self, message: str = "Ready") -> None:
        self.status_label.setText(message)
        self.status_changed.emit(message)
        self.progress_bar.hide()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

    def dragEnterEvent(self, event: Any) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(
            Path(url.toLocalFile()).is_dir()
            or Path(url.toLocalFile()).suffix.lower() in {".mat", ".npz"}
            for url in urls
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: Any) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.is_dir():
                self.set_directory(path)
                event.acceptProposedAction()
                return
            if path.suffix.lower() in {".mat", ".npz"} and path.is_file():
                self.set_directory(path.parent)
                self.load_file(path)
                event.acceptProposedAction()
                return
        event.ignore()

    def closeEvent(self, event: Any) -> None:
        self.cleanup()
        super().closeEvent(event)


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size_bytes:,} B"


__all__ = [
    "LesionFileData",
    "LesionFileRecord",
    "LesionLoadError",
    "LesionSource",
    "LocalDirectoryLesionSource",
    "LesionViewerModule",
    "VolumeCandidate",
    "load_lesion_file",
]
