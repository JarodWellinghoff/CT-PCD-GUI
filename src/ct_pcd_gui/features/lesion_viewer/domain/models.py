from __future__ import annotations
import os

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
