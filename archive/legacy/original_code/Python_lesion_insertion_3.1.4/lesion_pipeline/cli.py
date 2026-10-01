from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import InsertionConfig
from .ctpd import inspect_projection_series
from .pipeline import run_insertion


def _progress(done: int, total: int, name: str) -> None:
    print(f"[{done:>6}/{total}] {name}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure-Python DICOM-CT-PD lesion insertion pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Validate and summarize a DICOM-CT-PD input series"
    )
    inspect_parser.add_argument("--input", required=True, help="Projection file or directory")

    run_parser = subparsers.add_parser("run", help="Run insertion from a JSON configuration")
    run_parser.add_argument("--config", required=True, help="Configuration JSON file")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        result = inspect_projection_series(args.input)
        print(json.dumps(result, indent=2))
        return 0

    config = InsertionConfig.load(Path(args.config))
    summary = run_insertion(config, progress=_progress)
    print(
        json.dumps(
            {
                "output": summary["output"],
                "projection_file_count": summary["projection_file_count"],
                "projection_frame_count": summary["projection_frame_count"],
                "changed_projection_files": summary["changed_projection_files"],
                "changed_projection_frames": summary["changed_projection_frames"],
                "changed_pixels": summary["changed_pixels"],
                "clipped_pixels": summary["clipped_pixels"],
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0
