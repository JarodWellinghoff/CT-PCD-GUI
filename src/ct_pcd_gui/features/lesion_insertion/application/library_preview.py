"""Read-only source-model inspection, independent of Qt and the insertion state."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from ct_pcd_gui.features.lesion_viewer.domain.inspection import (
    SliceChannel,
    describe_members,
)
from ct_pcd_gui.features.lesion_viewer.domain.surface_models import SurfaceBuildRequest

if TYPE_CHECKING:
    from ..domain.models import LesionLibraryItem


class SourceModel(Protocol):
    voi_hu: np.ndarray
    mask: np.ndarray
    row_spacing_mm: float
    column_spacing_mm: float
    slice_spacing_mm: float


@dataclass(frozen=True, slots=True)
class LibraryPreviewData:
    path: str
    channels: tuple[SliceChannel, ...]
    spacing_xyz: tuple[float, float, float]
    summary: str
    metadata: str

    def surface_request(self, channel_index: int) -> SurfaceBuildRequest:
        if not 0 <= channel_index < len(self.channels):
            raise ValueError("Select an available source threshold for the 3-D preview.")
        mask = self.channels[channel_index].mask
        if mask is None or not np.any(mask):
            raise ValueError("This source threshold has no segmented voxels to display in 3-D.")
        if any(size < 2 for size in mask.shape):
            raise ValueError(
                "3-D requires at least two voxels on each axis; "
                "the 2-D preview remains available."
            )
        return SurfaceBuildRequest(
            volume=mask, source_min=0.0, source_max=1.0,
            axis_order="YXZ", spacing_xyz=self.spacing_xyz,
            threshold=0.5, foreground_below=False, downsample=1,
            smoothing_iterations=0, reduction_percent=0,
            largest_component_only=False, pad_border=True,
        )


def prepare_library_preview(
    model: SourceModel,
    item: LesionLibraryItem,
    *,
    check_cancelled: Callable[[], None] = lambda: None,
) -> LibraryPreviewData:
    """Adapt the insertion loader's R/C/S/channel arrays without altering HU.

    Statistics are exact for finite voxels inside each channel's stored mask.
    Arrays are read-only views, not copies or in-place normalized display data.
    """
    check_cancelled()
    values = np.asarray(model.voi_hu)
    if values.ndim == 3:
        values = values[..., np.newaxis]
    if values.ndim != 4 or not values.size or values.dtype.kind not in "iuf":
        raise ValueError("Source HU must be a nonempty numeric row/column/slice/channel array.")
    raw_mask = np.asarray(model.mask)
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[..., np.newaxis]
    if raw_mask.dtype.kind not in "buif" or raw_mask.shape[:3] != values.shape[:3]:
        raise ValueError("The source lesion mask does not match the HU array.")
    if raw_mask.ndim != 4 or raw_mask.shape[3] not in (1, values.shape[3]):
        raise ValueError("The source lesion mask has an incompatible threshold count.")
    spacing_rcs = (
        float(model.row_spacing_mm), float(model.column_spacing_mm),
        float(model.slice_spacing_mm),
    )
    if any(not np.isfinite(value) or value <= 0 for value in spacing_rcs):
        raise ValueError("Source voxel spacing must contain three positive finite values.")
    shape = tuple(int(size) for size in values.shape[:3])
    voxel_mm3 = float(np.prod(spacing_rcs))
    extent = tuple(size * spacing for size, spacing in zip(shape, spacing_rcs, strict=True))
    lines = [
        f"Lesion: {item.display_name} ({item.lesion_id})",
        f"File: {item.path}",
        f"Source thresholds: {values.shape[3]}",
        f"VOI shape (row, column, slice): {shape}",
        "Voxel spacing (row, column, slice), mm: " + ", ".join(f"{v:g}" for v in spacing_rcs),
        "VOI extent (row, column, slice), mm: " + " x ".join(f"{v:g}" for v in extent),
        "Values: loaded source-model HU, before placement, scaling or contrast changes.",
    ]
    channels: list[SliceChannel] = []
    statistics: list[dict[str, object]] = []
    for index in range(values.shape[3]):
        check_cancelled()
        image = values[..., index].view()
        image.setflags(write=False)
        source_mask = raw_mask[..., index if raw_mask.shape[3] > 1 else 0]
        mask = np.isfinite(source_mask) & (source_mask != 0)
        mask.setflags(write=False)
        channel = SliceChannel(f"Source threshold {index + 1}", image, mask, "HU")
        channels.append(channel)
        count = int(np.count_nonzero(mask))
        lesion_values = image[mask]
        finite = lesion_values[np.isfinite(lesion_values)]
        record: dict[str, object] = {
            "source_threshold": index + 1, "segmented_voxels": count,
            "mask_volume_mm3": count * voxel_mm3,
            "nonfinite_segmented_voxels": int(lesion_values.size - finite.size),
        }
        lines.extend([
            "", channel.name,
            f"  Segmented voxels: {count:,}",
            f"  Mask volume: {count * voxel_mm3:.3f} mm^3 ({count * voxel_mm3 / 1000:.4f} mL)",
        ])
        if finite.size:
            mean = float(np.mean(finite, dtype=np.float64))
            std = float(np.std(finite, dtype=np.float64))
            low, high = float(np.min(finite)), float(np.max(finite))
            record.update(mean_hu=mean, std_hu=std, min_hu=low, max_hu=high)
            lines.append(f"  Mask HU: mean {mean:.2f}; SD {std:.2f}; range {low:g} to {high:g}")
        else:
            lines.append("  Mask HU: unavailable (no finite segmented voxels).")
        if record["nonfinite_segmented_voxels"]:
            lines.append(
                "  Non-finite segmented voxels excluded from HU statistics: "
                f"{record['nonfinite_segmented_voxels']}"
            )
        for key, label in (("old_background_hu", "Stored background"),
                           ("contrast_hu_by_channel", "Stored lesion/background contrast")):
            raw = item.metadata.get(key)
            if raw is not None:
                vector = np.asarray(raw).reshape(-1)
                if index < vector.size:
                    lines.append(f"  {label}: {vector[index]} HU")
        statistics.append(record)
    for key, label in (("kvp", "Tube potential (kVp; 0 means unspecified)"),
                       ("old_background_method", "Background method"),
                       ("old_background_voxel_count", "Background voxels (-1 means unspecified)")):
        if key in item.metadata:
            lines.append(f"\n{label}: {item.metadata[key]}")
    check_cancelled()
    metadata = describe_members({
        "file": item.path, "lesion_id": item.lesion_id,
        "array_axis_order": "YXZ (row, column, slice)",
        "source_hu_dtype": str(values.dtype),
        "source_hu_shape": list(values.shape),
        "spacing_rcs_mm": spacing_rcs,
        "library_metadata": dict(item.metadata),
        "computed_mask_statistics": statistics,
        "note": (
            "Loaded insertion-model metadata, not a full DICOM-header export. "
            "No insertion parameters are applied."
        ),
    })
    return LibraryPreviewData(
        item.path, tuple(channels), (spacing_rcs[1], spacing_rcs[0], spacing_rcs[2]),
        "\n".join(lines), json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
    )
