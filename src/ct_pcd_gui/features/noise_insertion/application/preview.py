import numpy as np
import math
from .models import PreviewPayload


def normalize_preview_pair(
    before: np.ndarray,
    after: np.ndarray,
    max_dimension: int = 640,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    before_f = np.asarray(before, dtype=np.float64)
    after_f = np.asarray(after, dtype=np.float64)
    finite = before_f[np.isfinite(before_f)]

    if finite.size:
        low, high = np.percentile(finite, [0.5, 99.5])
    else:
        low, high = 0.0, 1.0
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite)) if finite.size else 0.0
        high = float(np.nanmax(finite)) if finite.size else 1.0
    if high <= low:
        high = low + 1.0

    def convert(image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
        normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
        height, width = normalized.shape
        step = max(1, int(math.ceil(max(height, width) / max_dimension)))
        downsampled = normalized[::step, ::step]
        return np.ascontiguousarray(np.rint(downsampled * 255.0).astype(np.uint8))

    return convert(before_f), convert(after_f), low, high


def make_preview_payload(
    title: str,
    iteration: int,
    total_iterations: int,
    before: np.ndarray,
    after: np.ndarray,
) -> PreviewPayload:
    before_u8, after_u8, low, high = normalize_preview_pair(before, after)
    return PreviewPayload(
        title=title,
        iteration=int(iteration),
        total_iterations=int(total_iterations),
        before_u8=before_u8,
        after_u8=after_u8,
        window_low=low,
        window_high=high,
        before_mean=float(np.nanmean(before)),
        before_std=float(np.nanstd(before)),
        after_mean=float(np.nanmean(after)),
        after_std=float(np.nanstd(after)),
    )


def is_preview_iteration(iteration: int, preview_interval: int) -> bool:
    """Return whether a one-based ordered projection iteration is scheduled."""
    interval = int(preview_interval)
    return interval > 0 and int(iteration) > 0 and int(iteration) % interval == 0


def local_preview_indices(
    global_frame_start: int,
    frame_count: int,
    preview_interval: int,
) -> range:
    """Return local zero-based frame indices matching the global interval."""
    interval = int(preview_interval)
    if interval <= 0 or frame_count <= 0:
        return range(0)

    # global_frame_start is the number of projections before this item. Preview
    # iterations are one-based multiples of the selected interval.
    first_iteration = ((global_frame_start // interval) + 1) * interval
    item_last_iteration = global_frame_start + frame_count
    if first_iteration > item_last_iteration:
        return range(0)

    first_local_index = first_iteration - global_frame_start - 1
    return range(first_local_index, frame_count, interval)
