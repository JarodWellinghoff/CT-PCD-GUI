"""Explicit lesion-source and target-spectrum selection (UI numbers are 1-based)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def threshold_mapping(
    source_channel: int | None,
    target_spectrum_indices: Iterable[int],
    channel_count: int,
    available_spectra: Iterable[int] | None = None,
) -> dict[int, int]:
    """Map every selected target to ONE source; never infer a different source."""
    if source_channel is None:
        raise ValueError("Choose one lesion source threshold.")
    if type(source_channel) is not int or not 0 <= source_channel < channel_count:
        raise ValueError(
            f"The lesion source threshold must be between 1 and {channel_count}."
        )
    targets = tuple(target_spectrum_indices)
    if not targets:
        raise ValueError("Choose at least one target scan threshold.")
    if any(type(target) is not int or target < 1 for target in targets):
        raise ValueError("Target scan threshold IDs must be positive integers.")
    if len(set(targets)) != len(targets):
        raise ValueError("A target scan threshold was selected more than once.")
    if available_spectra is not None:
        unknown = sorted(set(targets) - set(available_spectra))
        if unknown:
            raise ValueError(
                f"Selected target threshold(s) {unknown} are not in the inspected raw scan. "
                "Inspect the intended scan or select an available target."
            )
    return {target: source_channel for target in sorted(targets)}


def threshold_summary(
    source_channel: int | None,
    targets: Iterable[int],
    available_spectra: Iterable[int] | None = None,
) -> str:
    """Describe provenance and destinations without implying spectral conversion."""
    targets = tuple(sorted(targets))
    if source_channel is None:
        return "Choose one lesion source threshold before final insertion."
    if not targets:
        return f"Lesion threshold {source_channel + 1}: choose the target scan threshold(s)."
    destination = ", ".join(str(target) for target in targets)
    text = f"Lesion threshold {source_channel + 1} -> target threshold(s) {destination}."
    if len(targets) > 1:
        text += " The same lesion source threshold is used for every selected target."
    if available_spectra is not None:
        available = set(available_spectra)
        unknown = set(targets) - available
        if unknown:
            text += " WARNING: selected target(s) are not in the inspected raw scan."
        unchanged = sorted(available - set(targets))
        if unchanged:
            text += (
                " This lesion is not inserted into target threshold(s) "
                + ", ".join(map(str, unchanged)) + "."
            )
    return text


def migrate_threshold_fields(
    item: Mapping[str, Any], legacy_mapping: Mapping[int, int]
) -> dict[str, Any]:
    """Preserve unambiguous v1 routing; require review of mixed-source v1 maps."""
    result = dict(item)
    if "source_channel" not in result:
        channels = set(legacy_mapping.values())
        # No source was selected when the old map used different source channels.
        result["source_channel"] = next(iter(channels)) if len(channels) == 1 else None
        if not channels and len(result.get("background_hu", ())) == 1:
            result["source_channel"] = 0
    result["target_spectrum_indices"] = tuple(
        result.get("target_spectrum_indices", sorted(legacy_mapping))
    )
    return result
