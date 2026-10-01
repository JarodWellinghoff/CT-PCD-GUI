from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .components import _component_images
from .dicom_metadata import numeric_first, read_dataset
from .models import (
    DicomSeriesSpec,
    LesionCandidate,
    OverlayEntry,
    SegmentationInfo,
    SeriesPreview,
    overlay_color,
)
from .readers import (
    _fallback_segments,
    _read_segmentation,
    _segment_mask,
    load_dicom_image,
    parse_segment_definitions,
    require_sitk,
)


def _blank_labels(reference: Any):
    sitk = require_sitk()
    labels = sitk.Image(reference.GetSize(), sitk.sitkUInt16)
    labels.CopyInformation(reference)
    return labels


def _resample_labels(labels: Any, reference: Any):
    sitk = require_sitk()
    return sitk.Resample(
        labels,
        reference,
        sitk.Transform(reference.GetDimension(), sitk.sitkIdentity),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt16,
    )


def _default_window(series: DicomSeriesSpec, volume: np.ndarray) -> tuple[float, float]:
    try:
        dataset = read_dataset(
            series.first_file,
            specific_tags=["WindowCenter", "WindowWidth"],
        )
        center = numeric_first(getattr(dataset, "WindowCenter", None))
        width = numeric_first(getattr(dataset, "WindowWidth", None))
        if center is not None and width is not None and width > 0:
            return center, width
    except Exception:
        pass
    sample = volume.reshape(-1)
    if sample.size > 2_000_000:
        sample = sample[:: max(1, sample.size // 2_000_000)]
    low, high = np.percentile(sample.astype(np.float32), (1.0, 99.0))
    width = max(1.0, float(high - low))
    return float((low + high) / 2.0), width


def _preview(
    series: DicomSeriesSpec,
    segmentation: SegmentationInfo,
    candidates: Sequence[LesionCandidate] | None,
) -> SeriesPreview:
    sitk = require_sitk()
    dicom_image = load_dicom_image(series)
    segmentation_image, header = _read_segmentation(segmentation.path)
    definitions = parse_segment_definitions(header) or _fallback_segments(segmentation_image)
    by_key = {segment.key: segment for segment in definitions}
    labels = _blank_labels(segmentation_image)
    entries: list[OverlayEntry] = []

    if candidates is None:
        for label_id, segment in enumerate(segmentation.segments, start=1):
            encoded = sitk.Cast(_segment_mask(segmentation_image, segment), sitk.sitkUInt16)
            labels = sitk.Maximum(labels, encoded * label_id)
            entries.append(
                OverlayEntry(
                    label_id=label_id,
                    entity_id=segment.key,
                    display_name=segment.name,
                    color_rgb=segment.color_rgb or overlay_color(label_id - 1),
                )
            )
    else:
        components = _component_images(segmentation_image, by_key, candidates)
        for label_id, candidate in enumerate(candidates, start=1):
            component = components[candidate.segment_key]
            mask = sitk.BinaryThreshold(
                component,
                candidate.component_label,
                candidate.component_label,
                1,
                0,
            )
            labels = sitk.Maximum(
                labels, sitk.Cast(mask, sitk.sitkUInt16) * label_id
            )
            try:
                center = tuple(
                    int(value)
                    for value in dicom_image.TransformPhysicalPointToIndex(
                        candidate.centroid_xyz_mm
                    )
                )
                size = tuple(int(value) for value in dicom_image.GetSize())
                center = tuple(
                    int(np.clip(center[index], 0, size[index] - 1))
                    for index in range(3)
                )
            except Exception:
                center = None
            entries.append(
                OverlayEntry(
                    label_id=label_id,
                    entity_id=candidate.candidate_id,
                    display_name=candidate.output_name,
                    color_rgb=overlay_color(label_id - 1),
                    center_index_xyz=center,
                )
            )

    labels = _resample_labels(labels, dicom_image)
    volume = np.asarray(sitk.GetArrayFromImage(dicom_image), dtype=np.int16)
    overlay = np.asarray(sitk.GetArrayFromImage(labels), dtype=np.uint16)
    center, width = _default_window(series, volume)
    return SeriesPreview(series, volume, overlay, tuple(entries), center, width)


def load_segment_preview(
    series: DicomSeriesSpec, segmentation: SegmentationInfo
) -> SeriesPreview:
    return _preview(series, segmentation, None)


def load_candidate_preview(
    series: DicomSeriesSpec,
    segmentation: SegmentationInfo,
    candidates: Sequence[LesionCandidate],
) -> SeriesPreview:
    return _preview(series, segmentation, candidates)
