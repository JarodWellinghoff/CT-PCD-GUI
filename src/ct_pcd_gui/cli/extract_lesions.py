from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from ct_pcd_gui.features.lesion_extraction.extract_case import (
    discover_dicom_series,
    export_lesions,
    inspect_segmentation,
    split_segments,
)
from ct_pcd_gui.features.lesion_extraction.models import (
    CandidateSelection,
    LesionExtractionError,
    SegmentationInfo,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract connected lesion components from a NRRD segmentation into "
            "VOI/LesionMask NPZ files compatible with CT-PCD-GUI."
        )
    )
    parser.add_argument(
        "--dicom",
        action="append",
        default=[],
        metavar="FOLDER",
        help="DICOM series folder. Repeat for multiple folders.",
    )
    parser.add_argument("--segmentation", required=True, metavar="FILE")
    parser.add_argument("--output", metavar="FOLDER")
    parser.add_argument(
        "--segment",
        action="append",
        dest="segments",
        metavar="KEY_OR_NAME",
        help="Segment key, exact name, or label value. Repeat as needed.",
    )
    parser.add_argument("--minimum-voxels", type=int, default=1)
    parser.add_argument(
        "--omit",
        action="append",
        default=[],
        metavar="ID_OR_NAME",
        help="Candidate ID or output name to omit.",
    )
    parser.add_argument("--list-labels", action="store_true")
    parser.add_argument("--list-candidates", action="store_true")
    return parser


def _resolve_segment_keys(
    segmentation: SegmentationInfo, selectors: Sequence[str] | None
) -> tuple[str, ...]:
    if not selectors:
        return tuple(segment.key for segment in segmentation.segments)
    keys: list[str] = []
    unknown: list[str] = []
    for selector in selectors:
        query = selector.strip().casefold()
        matches = [
            segment.key
            for segment in segmentation.segments
            if segment.key.casefold() == query
            or segment.name.casefold() == query
            or str(segment.label_value) == query
        ]
        if not matches:
            unknown.append(selector)
        for key in matches:
            if key not in keys:
                keys.append(key)
    if unknown:
        raise LesionExtractionError(
            f"Unknown segment selector(s): {', '.join(unknown)}"
        )
    return tuple(keys)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.minimum_voxels < 1:
        parser.error("--minimum-voxels must be at least 1")
    try:
        segmentation = inspect_segmentation(args.segmentation)
        if args.list_labels:
            print("key\tname\tlabel\tlayer")
            for segment in segmentation.segments:
                print(
                    f"{segment.key}\t{segment.name}\t"
                    f"{segment.label_value}\t{segment.layer}"
                )
            return 0
        keys = _resolve_segment_keys(segmentation, args.segments)
        candidates = split_segments(
            segmentation.path,
            keys,
            minimum_voxels=args.minimum_voxels,
        )
        if args.list_candidates:
            print("candidate_id\toutput_name\tsegment\tpart\tvoxels\tvolume_mm3")
            for candidate in candidates:
                print(
                    f"{candidate.candidate_id}\t{candidate.output_name}\t"
                    f"{candidate.segment_name}\t{candidate.component_index}/"
                    f"{candidate.component_count}\t{candidate.voxel_count}\t"
                    f"{candidate.physical_size_mm3:.6g}"
                )
            return 0
        if not args.dicom:
            parser.error("at least one --dicom folder is required for export")
        if not args.output:
            parser.error("--output is required for export")
        omitted = {str(item).strip().casefold() for item in args.omit}
        selections = tuple(
            CandidateSelection(
                candidate_id=candidate.candidate_id,
                included=(
                    candidate.candidate_id.casefold() not in omitted
                    and candidate.output_name.casefold() not in omitted
                ),
                output_name=candidate.output_name,
            )
            for candidate in candidates
        )
        series = discover_dicom_series(args.dicom)

        def report(event) -> None:
            print(f"[{event.completed}/{event.total}] {event.message}", file=sys.stderr)

        summary = export_lesions(
            series=series,
            segmentation=segmentation,
            candidates=candidates,
            selections=selections,
            output_folder=Path(args.output),
            progress=report,
        )
    except LesionExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for exported in summary.exported:
        print(exported.output_path)
    for warning in summary.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
