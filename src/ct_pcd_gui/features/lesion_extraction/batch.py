from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .extract_case import (
    discover_dicom_series,
    export_lesions,
    inspect_segmentation,
    load_candidate_preview,
    load_segment_preview,
    split_segments,
)
from .models import (
    CandidateSelection,
    DicomSeriesSpec,
    ExportSummary,
    InspectionResult,
    InputValidationError,
    LesionCandidate,
    PreparedExtraction,
    SegmentationInfo,
    SeriesPreview,
)


class LesionExtractionService:
    """Qt-independent facade shared by the GUI and command-line workflow."""

    def inspect_inputs(
        self,
        dicom_folders: Sequence[str | Path],
        segmentation_path: str | Path,
    ) -> InspectionResult:
        series = discover_dicom_series(dicom_folders)
        segmentation = inspect_segmentation(segmentation_path)
        return InspectionResult(
            series,
            segmentation,
            load_segment_preview(series[0], segmentation),
        )

    def prepare_candidates(
        self,
        series: Sequence[DicomSeriesSpec],
        segmentation: SegmentationInfo,
        selected_segment_keys: Sequence[str],
        *,
        minimum_voxels: int = 1,
        preview_series_index: int = 0,
    ) -> PreparedExtraction:
        if not series:
            raise InputValidationError("At least one DICOM series is required.")
        if not 0 <= preview_series_index < len(series):
            raise InputValidationError("Preview series index is out of range.")
        candidates = split_segments(
            segmentation.path,
            selected_segment_keys,
            minimum_voxels=minimum_voxels,
        )
        preview = load_candidate_preview(
            series[preview_series_index], segmentation, candidates
        )
        return PreparedExtraction(tuple(series), segmentation, candidates, preview)

    def preview_segments(
        self, series: DicomSeriesSpec, segmentation: SegmentationInfo
    ) -> SeriesPreview:
        return load_segment_preview(series, segmentation)

    def preview_candidates(
        self,
        series: DicomSeriesSpec,
        segmentation: SegmentationInfo,
        candidates: Sequence[LesionCandidate],
    ) -> SeriesPreview:
        return load_candidate_preview(series, segmentation, candidates)

    def export(
        self,
        *,
        series: Sequence[DicomSeriesSpec],
        segmentation: SegmentationInfo,
        candidates: Sequence[LesionCandidate],
        selections: Sequence[CandidateSelection],
        output_folder: str | Path,
        progress=None,
        cancelled=None,
    ) -> ExportSummary:
        return export_lesions(
            series=series,
            segmentation=segmentation,
            candidates=candidates,
            selections=selections,
            output_folder=output_folder,
            progress=progress,
            cancelled=cancelled,
        )
