from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import (
    InputValidationError,
    LesionCandidate,
    SegmentDefinition,
    make_unique_output_names,
    sanitize_output_name,
)
from .readers import (
    _fallback_segments,
    _read_segmentation,
    _segment_mask,
    parse_segment_definitions,
    require_sitk,
)


def _relabel_components(mask: Any, minimum_voxels: int):
    sitk = require_sitk()
    connected = sitk.ConnectedComponent(mask, True)
    relabel = sitk.RelabelComponentImageFilter()
    relabel.SortByObjectSizeOn()
    relabel.SetMinimumObjectSize(max(1, int(minimum_voxels)))
    return relabel.Execute(connected)


def split_segments(
    segmentation_path: str | Path,
    selected_segment_keys: Sequence[str],
    *,
    minimum_voxels: int = 1,
) -> tuple[LesionCandidate, ...]:
    image, header = _read_segmentation(segmentation_path)
    definitions = parse_segment_definitions(header) or _fallback_segments(image)
    by_key = {segment.key: segment for segment in definitions}
    unknown = [key for key in selected_segment_keys if key not in by_key]
    if unknown:
        raise InputValidationError(f"Unknown segmentation labels: {', '.join(unknown)}")
    if not selected_segment_keys:
        raise InputValidationError("Select at least one segmentation label.")

    sitk = require_sitk()
    candidates: list[LesionCandidate] = []
    for segment_key in selected_segment_keys:
        segment = by_key[segment_key]
        components = _relabel_components(_segment_mask(image, segment), minimum_voxels)
        shape = sitk.LabelShapeStatisticsImageFilter()
        shape.Execute(components)
        labels = tuple(int(label) for label in shape.GetLabels())
        count = len(labels)
        for index, label in enumerate(labels, start=1):
            output_name = (
                segment.name if count == 1 else f"{segment.name}-part-{index:02d}"
            )
            candidates.append(
                LesionCandidate(
                    candidate_id=f"{segment.key}:{label}",
                    segment_key=segment.key,
                    segment_name=segment.name,
                    component_label=label,
                    component_index=index,
                    component_count=count,
                    output_name=sanitize_output_name(output_name),
                    voxel_count=int(shape.GetNumberOfPixels(label)),
                    physical_size_mm3=float(shape.GetPhysicalSize(label)),
                    centroid_xyz_mm=tuple(
                        float(value) for value in shape.GetCentroid(label)
                    ),
                )
            )
    if not candidates:
        raise InputValidationError(
            "The selected labels contain no connected components at this size threshold."
        )
    return make_unique_output_names(tuple(candidates))


def _component_images(
    segmentation_image: Any,
    definitions: Mapping[str, SegmentDefinition],
    candidates: Sequence[LesionCandidate],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in dict.fromkeys(candidate.segment_key for candidate in candidates):
        result[key] = _relabel_components(
            _segment_mask(segmentation_image, definitions[key]), 1
        )
    return result
