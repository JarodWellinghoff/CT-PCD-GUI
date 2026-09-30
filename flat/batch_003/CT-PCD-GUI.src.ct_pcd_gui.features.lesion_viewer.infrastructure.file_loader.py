from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np
import scipy

from ct_pcd_gui.features.lesion_viewer.domain.models import (
    LesionFileData,
    LesionLoadError,
    VolumeCandidate,
)

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


class MatNpzLesionFileLoader:
    def load(self, path: Path) -> LesionFileData:
        return load_lesion_file(path)


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
