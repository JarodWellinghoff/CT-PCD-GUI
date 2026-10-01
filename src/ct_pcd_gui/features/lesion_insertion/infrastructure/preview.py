from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
from scipy.ndimage import zoom

from ..domain.models import DicomVolume, LesionSession, PreviewResult
from .model_transform import model_cache_key, transform_model


class ApproximatePreviewGenerator:
    """Fast image-domain approximation; never used for final raw-data output."""

    def __init__(self, model_loader: Callable[[str], Any]) -> None:
        self._model_loader = model_loader
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def _resampled_difference(
        self,
        path: str,
        parameters,
        target_spacing_rcs: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        source_path = Path(path)
        modified = source_path.stat().st_mtime_ns if source_path.exists() else 0
        key = (
            *model_cache_key(path, parameters),
            modified,
            tuple(round(value, 8) for value in target_spacing_rcs),
        )
        cached = self._cache.get(key)
        if cached is not None:
            return cached[0], cached[1], True
        model = transform_model(self._model_loader(path), parameters)
        mask = np.asarray(model.mask[..., 0], dtype=bool)
        difference = np.zeros(mask.shape, dtype=np.float32)
        old_background = float(np.asarray(model.old_background_hu).reshape(-1)[0])
        difference[mask] = np.asarray(model.voi_hu[..., 0], dtype=np.float32)[mask] - old_background
        factors = (
            float(model.row_spacing_mm) / target_spacing_rcs[0],
            float(model.column_spacing_mm) / target_spacing_rcs[1],
            float(model.slice_spacing_mm) / target_spacing_rcs[2],
        )
        difference = zoom(
            difference,
            factors,
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ).astype(np.float32, copy=False)
        mask = zoom(
            mask.astype(np.uint8),
            factors,
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        ).astype(bool)
        difference[~mask] = 0.0
        self._cache[key] = (difference, mask)
        return difference, mask, False

    def generate(
        self,
        volume: DicomVolume,
        session: LesionSession,
        slice_index: int,
        generation: int,
        cancel_event: Event | None = None,
    ) -> PreviewResult:
        z = max(0, min(int(slice_index), volume.hu.shape[0] - 1))
        before = np.asarray(volume.hu[z], dtype=np.float32).copy()
        added = np.zeros_like(before)
        alpha = np.zeros_like(before, dtype=np.float32)
        spacing = (
            volume.geometry.row_spacing_mm,
            volume.geometry.column_spacing_mm,
            volume.geometry.median_slice_spacing_mm,
        )
        cache_hits = 0
        for lesion in session.lesions:
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("Preview generation was cancelled.")
            if not lesion.enabled or not lesion.visible:
                continue
            difference, mask, hit = self._resampled_difference(
                lesion.lesion_path, lesion.parameters, spacing
            )
            cache_hits += int(hit)
            center_column, center_row, center_slice = lesion.center_voxel_crs
            local_z = int(round(z - center_slice + (difference.shape[2] - 1) / 2.0))
            if local_z < 0 or local_z >= difference.shape[2]:
                continue
            plane = difference[:, :, local_z]
            plane_mask = mask[:, :, local_z]
            center_y = int(round(center_row))
            center_x = int(round(center_column))
            y0 = center_y - plane.shape[0] // 2
            x0 = center_x - plane.shape[1] // 2
            target_y0 = max(0, y0)
            target_x0 = max(0, x0)
            target_y1 = min(before.shape[0], y0 + plane.shape[0])
            target_x1 = min(before.shape[1], x0 + plane.shape[1])
            if target_y0 >= target_y1 or target_x0 >= target_x1:
                continue
            source_y0 = target_y0 - y0
            source_x0 = target_x0 - x0
            source_y1 = source_y0 + (target_y1 - target_y0)
            source_x1 = source_x0 + (target_x1 - target_x0)
            source_plane = plane[source_y0:source_y1, source_x0:source_x1]
            source_mask = plane_mask[source_y0:source_y1, source_x0:source_x1]
            region = added[target_y0:target_y1, target_x0:target_x1]
            region[source_mask] += source_plane[source_mask]
            alpha_region = alpha[target_y0:target_y1, target_x0:target_x1]
            alpha_region[source_mask] = np.maximum(
                alpha_region[source_mask], float(lesion.parameters.preview_opacity)
            )
        return PreviewResult(
            slice_index=z,
            before_hu=before,
            approximate_after_hu=before + added,
            overlay_alpha=np.clip(alpha, 0.0, 1.0),
            generation=generation,
            cache_hits=cache_hits,
        )
