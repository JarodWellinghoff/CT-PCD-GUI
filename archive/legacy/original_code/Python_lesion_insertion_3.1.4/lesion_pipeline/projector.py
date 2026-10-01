from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ctpd import MATLAB_WATER_ATTENUATION_MM_INVERSE
from .lesion_models import LesionModel


@dataclass(frozen=True)
class VoxelVolume:
    """Axis-aligned difference in linear attenuation coefficient (mm^-1)."""

    data: np.ndarray  # (x, y, z)
    edge_origin_mm: np.ndarray
    spacing_mm: np.ndarray
    center_mm: np.ndarray
    radius_mm: float

    @property
    def upper_edge_mm(self) -> np.ndarray:
        return self.edge_origin_mm + np.asarray(self.data.shape) * self.spacing_mm


def _legacy_alpha_orientation(array: np.ndarray, patient_position: str) -> np.ndarray:
    transformed = np.flip(array, axis=0)
    transformed = np.transpose(transformed, (1, 0, 2))
    transformed = np.flip(transformed, axis=0)
    if patient_position.upper() == "HFS":
        transformed = np.flip(transformed, axis=0)
    elif patient_position.upper() != "FFS":
        raise ValueError("Only HFS and FFS patient positions are supported.")
    return transformed


def prepare_difference_volume(
    model: LesionModel,
    channel: int,
    center_ctpd_mm: np.ndarray,
    patient_position: str,
) -> VoxelVolume:
    """Convert a model channel to the line-integral difference volume.

    Version 3.1.4 preserves the source lesion-to-background contrast by
    subtracting the original perilesional background stored in the model.
    The water coefficient is the MATLAB-equivalent hard-coded value 0.1917
    cm^-1 = 0.01917 mm^-1.
    """

    if channel < 0 or channel >= model.channel_count:
        raise IndexError(
            f"Lesion channel {channel} is invalid for a {model.channel_count}-channel model."
        )
    hu = _legacy_alpha_orientation(model.voi_hu[..., channel], patient_position)
    mask = _legacy_alpha_orientation(model.mask[..., channel], patient_position).astype(bool)
    old_background_hu = float(model.old_background_hu[channel])
    delta = np.zeros(hu.shape, dtype=np.float32)
    delta[mask] = (
        MATLAB_WATER_ATTENUATION_MM_INVERSE
        * (hu[mask].astype(np.float64) - old_background_hu)
        / 1000.0
    ).astype(np.float32)
    spacing = np.asarray(
        [model.column_spacing_mm, model.row_spacing_mm, model.slice_spacing_mm],
        dtype=float,
    )
    center = np.asarray(center_ctpd_mm, dtype=float)
    physical_size = np.asarray(delta.shape, dtype=float) * spacing
    edge_origin = center - physical_size / 2.0
    radius = float(np.linalg.norm(physical_size) / 2.0)
    return VoxelVolume(
        data=delta,
        edge_origin_mm=edge_origin,
        spacing_mm=spacing,
        center_mm=center,
        radius_mm=radius,
    )


def siddon_integral(source: np.ndarray, detector: np.ndarray, volume: VoxelVolume) -> float:
    """Integrate one ray through an axis-aligned voxel volume using Siddon lengths."""

    source = np.asarray(source, dtype=float)
    detector = np.asarray(detector, dtype=float)
    direction = detector - source
    ray_length = float(np.linalg.norm(direction))
    if ray_length == 0:
        return 0.0

    lower = volume.edge_origin_mm
    upper = volume.upper_edge_mm
    entry = 0.0
    exit_value = 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if source[axis] < lower[axis] or source[axis] > upper[axis]:
                return 0.0
            continue
        first = (lower[axis] - source[axis]) / direction[axis]
        last = (upper[axis] - source[axis]) / direction[axis]
        if first > last:
            first, last = last, first
        entry = max(entry, first)
        exit_value = min(exit_value, last)
        if exit_value <= entry:
            return 0.0

    alphas: list[np.ndarray] = [np.asarray([entry, exit_value])]
    for axis, voxel_count in enumerate(volume.data.shape):
        if abs(direction[axis]) < 1e-12 or voxel_count <= 1:
            continue
        planes = lower[axis] + np.arange(1, voxel_count, dtype=float) * volume.spacing_mm[axis]
        crossings = (planes - source[axis]) / direction[axis]
        crossings = crossings[(crossings > entry) & (crossings < exit_value)]
        if crossings.size:
            alphas.append(crossings)
    alpha = np.unique(np.concatenate(alphas))
    if alpha.size < 2:
        return 0.0
    midpoint_alpha = (alpha[:-1] + alpha[1:]) / 2.0
    points = source[np.newaxis, :] + midpoint_alpha[:, np.newaxis] * direction[np.newaxis, :]
    indices = np.floor((points - lower[np.newaxis, :]) / volume.spacing_mm[np.newaxis, :]).astype(int)
    shape = np.asarray(volume.data.shape)
    indices = np.clip(indices, 0, shape - 1)
    segment_lengths = np.diff(alpha) * ray_length
    samples = volume.data[indices[:, 0], indices[:, 1], indices[:, 2]]
    return float(np.dot(samples.astype(np.float64), segment_lengths))


def candidate_ray_mask(
    source: np.ndarray,
    detector_positions: np.ndarray,
    center_mm: np.ndarray,
    radius_mm: float,
) -> np.ndarray:
    vectors = detector_positions - source[np.newaxis, np.newaxis, :]
    center_vector = center_mm - source
    denominator = np.einsum("...i,...i->...", vectors, vectors)
    parameter = np.einsum("...i,i->...", vectors, center_vector) / denominator
    parameter = np.clip(parameter, 0.0, 1.0)
    closest = source[np.newaxis, np.newaxis, :] + parameter[..., np.newaxis] * vectors
    distance_squared = np.einsum(
        "...i,...i->...",
        closest - center_mm[np.newaxis, np.newaxis, :],
        closest - center_mm[np.newaxis, np.newaxis, :],
    )
    return distance_squared <= radius_mm * radius_mm


def project_volume(
    source: np.ndarray,
    detector_positions: np.ndarray,
    volume: VoxelVolume,
) -> np.ndarray:
    """Forward project only detector rays that can intersect the lesion sphere."""

    output = np.zeros(detector_positions.shape[:2], dtype=np.float64)
    candidates = candidate_ray_mask(
        source,
        detector_positions,
        volume.center_mm,
        volume.radius_mm,
    )
    for row, column in np.argwhere(candidates):
        output[row, column] = siddon_integral(
            source,
            detector_positions[row, column],
            volume,
        )
    return output
