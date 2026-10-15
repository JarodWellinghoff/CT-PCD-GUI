"""Lesion reader with lossless HU channels and inspectable file metadata."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

from ..domain.inspection import describe_members, extract_channels
from ..domain.models import LesionFileData, LesionLoadError, VolumeCandidate
from .file_loader import (
    _expand_to_3d_candidates, _find_spacing, _infer_axis_order,
    _load_npz, _make_candidate, load_lesion_file,
)


class InspectableLesionFileLoader:
    """Load metadata in the existing worker, not on the GUI thread."""

    def load(self, path: Path) -> LesionFileData:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise LesionLoadError(f"File does not exist: {source}")
        if source.suffix.lower() == ".npz":
            root, warnings = _load_npz(source)  # allow_pickle=False
            channels = extract_channels(root, warnings)
            candidates = []
            for key, raw in root.items():
                array = np.asarray(raw)
                # VOI and LesionMask use a documented final channel axis. The
                # generic smallest-axis heuristic is not valid for these fields.
                if key.casefold() in {"voi", "lesionmask"} and array.ndim in (3, 4):
                    expanded = ([(key, array)] if array.ndim == 3 else [
                        (f"{key}[axis3={i}]", array[..., i]) for i in range(array.shape[3])
                    ])
                else:
                    expanded = _expand_to_3d_candidates(key, array)
                for name, volume in expanded:
                    candidate = _candidate(name, volume)
                    if candidate is not None:
                        candidates.append(candidate)
            if not candidates:
                detail = "\n".join(warnings)
                raise LesionLoadError(f"No numeric 3-D arrays were found in this file.\n{detail}")
            candidates.sort(key=lambda item: (-item.score, item.key.casefold()))
            result = LesionFileData(
                source, candidates, _find_spacing(root),
                {"format": "npz", "suggested_axis_order": _infer_axis_order(root)}, warnings,
            )
        else:
            # Retain the established MATLAB reader and its geometry conventions.
            result = load_lesion_file(source)
            root = {}
            try:
                if result.file_metadata.get("mat_backend") == "h5py":
                    with h5py.File(source, "r") as handle:
                        def visit(name, obj):
                            if not isinstance(obj, h5py.Dataset):
                                return
                            if obj.dtype.kind in "buifSU":
                                values = np.asarray(obj)
                                # MATLAB v7.3 reverses dimension order in HDF5:
                                # a final MATLAB channel axis becomes the first
                                # HDF5 axis. Retain the base reader's spatial
                                # ordering while normalizing the channel axis.
                                if (values.ndim == 4 and "MATLAB_class" in obj.attrs
                                        and name.casefold() in {"voi", "lesionmask"}):
                                    root.setdefault("HDF5 stored dataset shapes", {})[name] = list(values.shape)
                                    values = np.moveaxis(values, 0, -1)
                                root[name] = values
                            elif h5py.check_string_dtype(obj.dtype) is not None:
                                root[name] = obj.asstr()[()]
                        handle.visititems(visit)
                        root["HDF5 attributes"] = dict(handle.attrs)
                else:
                    root = scipy.io.loadmat(source, simplify_cells=True, chars_as_strings=True)
            except (OSError, ValueError, TypeError) as exc:
                result.warnings.append(f"Additional MATLAB metadata unavailable: {exc}")
            channels = extract_channels(root, result.warnings)
        result.file_metadata["members"] = describe_members(root)
        result.file_metadata["slice_channels"] = channels
        return result


def _candidate(key: str, array: np.ndarray) -> VolumeCandidate | None:
    """Preserve precision and allow single-slice volumes for 2-D inspection."""
    values = np.asarray(array)
    if values.ndim != 3 or not values.size or values.dtype.kind not in "buif":
        return None
    candidate = _make_candidate(key, values)
    if candidate is not None:
        # The surface builder performs its own conversion. Do not reduce the
        # original float64 values retained by the inspection reader.
        candidate.array = values
        return candidate
    if all(size >= 2 for size in values.shape):
        return None  # For example, an all-NaN array rejected by the base reader.
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    low, high = float(finite.min()), float(finite.max())
    return VolumeCandidate(
        key, values, 150.0 if "mask" in key.casefold() else 10.0,
        low, high, (low + high) / 2.0,
        float(finite.size / values.size), float(np.count_nonzero(finite) / finite.size),
    )
