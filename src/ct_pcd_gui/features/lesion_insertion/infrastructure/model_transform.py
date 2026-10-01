from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from typing import Any

import numpy as np
from scipy.ndimage import rotate, zoom

from ..domain.models import LesionParameters, ValidationError


def _rotate_volume(
    array: np.ndarray,
    rotations_xyz: tuple[float, float, float],
    order: int,
) -> np.ndarray:
    result = array
    axis_pairs = ((0, 2), (1, 2), (0, 1))
    for angle, axes in zip(rotations_xyz, axis_pairs, strict=True):
        if abs(angle) > 1e-8:
            result = rotate(
                result,
                angle=float(angle),
                axes=axes,
                reshape=True,
                order=order,
                mode="constant",
                cval=0.0,
                prefilter=order > 1,
            )
    return result


def transform_model(model: Any, parameters: LesionParameters) -> Any:
    """Create a temporary transformed model while retaining projector semantics."""

    parameters.validate()
    voi = np.asarray(model.voi_hu, dtype=np.float32)
    mask = np.asarray(model.mask, dtype=bool)
    if voi.ndim != 4 or mask.shape != voi.shape:
        raise ValidationError("A lesion model must have matching 4-D VOI and mask arrays.")
    channel_count = voi.shape[3]
    old_background = np.asarray(model.old_background_hu, dtype=np.float32).reshape(-1)
    if old_background.size != channel_count:
        raise ValidationError("Lesion old-background values do not match the channel count.")

    scale_rcs = (
        float(parameters.scale_xyz[1]),
        float(parameters.scale_xyz[0]),
        float(parameters.scale_xyz[2]),
    )
    transformed_differences: list[np.ndarray] = []
    transformed_masks: list[np.ndarray] = []
    for channel in range(channel_count):
        source_mask = mask[..., channel]
        difference = np.zeros(source_mask.shape, dtype=np.float32)
        source_contrast = (
            voi[..., channel][source_mask] - old_background[channel]
        ).astype(np.float32)
        difference[source_mask] = source_contrast
        target_mean_contrast = (
            float(np.mean(source_contrast)) * float(parameters.contrast_scale)
            if source_contrast.size
            else 0.0
        )
        scaled_difference = zoom(
            difference,
            scale_rcs,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        scaled_mask = zoom(
            source_mask.astype(np.uint8),
            scale_rcs,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        transformed_difference = _rotate_volume(
            scaled_difference, parameters.rotation_deg_xyz, order=1
        )
        transformed_mask = _rotate_volume(
            scaled_mask.astype(np.uint8), parameters.rotation_deg_xyz, order=0
        ).astype(bool)
        transformed_difference[~transformed_mask] = 0.0
        if not np.any(transformed_mask):
            raise ValidationError(
                f"Lesion channel {channel} became empty after scaling/rotation."
            )
        current_mean = float(np.mean(transformed_difference[transformed_mask]))
        if abs(current_mean) > 1e-8:
            transformed_difference *= target_mean_contrast / current_mean
        elif abs(target_mean_contrast) > 1e-8:
            transformed_difference[transformed_mask] = target_mean_contrast
        transformed_differences.append(transformed_difference.astype(np.float32, copy=False))
        transformed_masks.append(transformed_mask)

    target_shape = np.max(
        np.asarray([array.shape for array in transformed_differences], dtype=int), axis=0
    )

    def centered_pad(array: np.ndarray, shape: np.ndarray, value: float) -> np.ndarray:
        padding: list[tuple[int, int]] = []
        for current, target in zip(array.shape, shape, strict=True):
            total = int(target - current)
            padding.append((total // 2, total - total // 2))
        return np.pad(array, padding, mode="constant", constant_values=value)

    output_voi = np.empty((*target_shape.tolist(), channel_count), dtype=np.float32)
    output_mask = np.empty((*target_shape.tolist(), channel_count), dtype=bool)
    means = np.empty(channel_count, dtype=np.float32)
    for channel in range(channel_count):
        difference = centered_pad(transformed_differences[channel], target_shape, 0.0)
        channel_mask = centered_pad(
            transformed_masks[channel].astype(np.uint8), target_shape, 0.0
        ).astype(bool)
        output_mask[..., channel] = channel_mask
        output_voi[..., channel] = old_background[channel] + difference
        means[channel] = float(np.mean(output_voi[..., channel][channel_mask]))

    updates = {
        "voi_hu": output_voi,
        "mask": output_mask,
        "lesion_mean_hu_by_channel": means,
        "lesion_mean_hu": float(means[0]),
    }
    if is_dataclass(model):
        return replace(model, **updates)
    transformed_model = copy.deepcopy(model)
    for name, value in updates.items():
        setattr(transformed_model, name, value)
    return transformed_model


def model_cache_key(path: str, parameters: LesionParameters) -> tuple:
    return (
        path,
        round(parameters.contrast_scale, 8),
        tuple(round(value, 8) for value in parameters.scale_xyz),
        tuple(round(value, 8) for value in parameters.rotation_deg_xyz),
    )
