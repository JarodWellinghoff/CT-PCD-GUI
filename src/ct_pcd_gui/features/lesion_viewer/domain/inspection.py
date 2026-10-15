"""Lossless slice data and bounded, JSON-safe lesion-file inspection."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SliceChannel:
    name: str
    values: np.ndarray
    mask: np.ndarray | None = None
    units: str = "HU"


def to_zyx(array: np.ndarray, axis_order: str) -> np.ndarray:
    """Reorder spatial axes without resampling or changing stored values."""
    order = axis_order.upper()
    if len(order) != 3 or set(order) != set("XYZ"):
        raise ValueError("Axis order must be a permutation of XYZ.")
    if array.ndim != 3:
        raise ValueError("A slice channel must be three-dimensional.")
    return np.transpose(array, tuple(order.index(axis) for axis in "ZYX"))


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """One-pixel inner boundary, including lesions touching the image edge."""
    source = np.asarray(mask)
    if source.ndim != 2:
        raise ValueError("A lesion outline requires a two-dimensional mask.")
    inside = np.isfinite(source) & (source != 0)
    padded = np.pad(inside, 1, constant_values=False)
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return inside & ~interior


def _text(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _lookup(root: Mapping, name: str) -> Any:
    return next((v for k, v in root.items() if str(k).casefold() == name.casefold()), None)


def extract_channels(root: Mapping, warnings: list[str]) -> list[SliceChannel]:
    """Read the documented VOI[..., channel] schema, never guess a channel axis.

    A mask is segmentation, not an HU image. Only explicitly named HU arrays or
    the lesion-library VOI are automatically labelled as HU.
    """
    voi = _lookup(root, "VOI")
    if voi is None:
        for key in ("image_hu", "volume_hu", "original_hu"):
            voi = _lookup(root, key)
            if voi is not None:
                break
    if voi is None:
        return []
    values = np.asarray(voi)
    if values.dtype.kind not in "iuf" or values.ndim not in (3, 4) or not values.size:
        warnings.append("Original HU unavailable: VOI must be a numeric 3-D or 4-D array.")
        return []
    count = values.shape[3] if values.ndim == 4 else 1
    raw_count = _lookup(root, "SeriesCount")
    if raw_count is not None:
        try:
            if np.asarray(raw_count).size != 1 or int(np.asarray(raw_count).item()) != count:
                warnings.append("SeriesCount does not match VOI; using the stored last-axis channel count.")
        except (TypeError, ValueError, OverflowError):
            warnings.append("Invalid SeriesCount; using the stored last-axis channel count.")
    raw_names = _lookup(root, "SeriesNames")
    names = np.asarray(raw_names).reshape(-1) if raw_names is not None else []
    if len(names) not in (0, count):
        warnings.append("SeriesNames does not match VOI; using numbered channel labels.")
    raw_mask = _lookup(root, "LesionMask")
    mask = np.asarray(raw_mask) if raw_mask is not None else None
    if mask is not None:
        valid = mask.dtype.kind in "buif" and (
            mask.shape == values.shape[:3]
            or mask.shape == (*values.shape[:3], count)
            or mask.shape == (*values.shape[:3], 1)
        )
        if not valid:
            warnings.append("LesionMask does not match VOI; the 2-D outline is unavailable.")
            mask = None
    channels = []
    for index in range(count):
        image = values[..., index] if values.ndim == 4 else values
        channel_mask = mask
        if mask is not None and mask.ndim == 4:
            channel_mask = mask[..., index if mask.shape[3] > 1 else 0]
        label = _text(names[index]) if len(names) == count else f"Channel {index + 1}"
        channels.append(SliceChannel(label, image, channel_mask))
    return channels


def describe_members(value: Any, *, max_items: int = 64, max_nodes: int = 10000) -> Any:
    """Preserve metadata and JSON headers, summarizing arrays rather than dumping voxels.

    This function never enables pickle. Limits apply to nested/cyclic containers
    and oversized text as well as arrays, so the metadata widget stays bounded.
    """
    visited: dict[int, Any] = {}
    remaining = [max_nodes]

    def walk(item: Any, depth: int = 0) -> Any:
        remaining[0] -= 1
        if remaining[0] < 0 or depth > 16:
            return "<inspection limit reached>"
        if isinstance(item, (Mapping, list, tuple, np.ndarray)):
            identity = id(item)
            if identity in visited:
                return "<shared or cyclic reference>"
            visited[identity] = item  # Keep temporary JSON/list objects alive; IDs can be reused.
        if isinstance(item, Mapping):
            output = {}
            for index, (key, member) in enumerate(item.items()):
                if index >= 512 or remaining[0] <= 0:
                    output["<remaining fields>"] = "Inspection limit reached"
                    break
                output[str(key)] = walk(member, depth + 1)
            return output
        if isinstance(item, np.ndarray):
            if item.dtype.names:
                return {name: walk(item[name], depth + 1) for name in item.dtype.names}
            if item.size == 1:
                return walk(item.reshape(-1)[0], depth + 1)
            info: dict[str, Any] = {"shape": list(item.shape), "dtype": str(item.dtype)}
            if item.size <= max_items:
                info["values"] = walk(item.tolist(), depth + 1)
            elif item.dtype.kind in "buif":
                # Bounded deterministic sample; never claim sampled extrema are exact.
                step = max(1, item.size // 100000)
                sample = item.flat[::step][:100000]
                finite = sample[np.isfinite(sample)]
                info["statistics"] = "exact" if item.size <= 100000 else "sampled"
                if finite.size:
                    info["min"] = walk(np.min(finite), depth + 1)
                    info["max"] = walk(np.max(finite), depth + 1)
                info["elements"] = int(item.size)
            else:
                info["preview"] = [walk(x, depth + 1) for x in item.flat[:max_items]]
                info["note"] = "Array preview truncated"
            return info
        if isinstance(item, (list, tuple)):
            values = [walk(x, depth + 1) for x in item[:512]]
            if len(item) > 512:
                values.append("<remaining entries omitted>")
            return values
        if isinstance(item, np.generic):
            return walk(item.item(), depth + 1)
        if isinstance(item, (str, bytes)):
            text = _text(item)
            if len(text) > 250000:
                return text[:250000] + "\n<text truncated>"
            if text.lstrip().startswith(("{", "[")):
                try:
                    return walk(json.loads(text), depth + 1)
                except (ValueError, RecursionError):
                    pass
            return text
        if isinstance(item, float) and not np.isfinite(item):
            return str(item)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return _text(item)

    return walk(value)
