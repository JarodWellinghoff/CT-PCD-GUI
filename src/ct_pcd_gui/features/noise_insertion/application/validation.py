from __future__ import annotations

import os
from pathlib import Path

from .models import NoiseJobConfig


def validate_config(config: NoiseJobConfig) -> None:
    raw_input = str(config.input_path).strip()
    if not raw_input:
        raise ValueError("An input DICOM folder or file is required.")

    input_path = Path(raw_input).expanduser()
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    if not input_path.is_file() and not input_path.is_dir():
        raise ValueError(f"Input path is not a file or directory: {input_path}")

    raw_output = str(config.output_dir).strip()
    if not raw_output:
        raise ValueError("An output directory is required.")

    output_path = Path(raw_output).expanduser()
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"Output path is not a directory: {output_path}")

    if config.input_mode not in {"auto", "single", "multi"}:
        raise ValueError(f"Unknown input mode: {config.input_mode}")

    if not (0.0 < float(config.mas_factor) <= 1.0):
        raise ValueError("mAs factor must be greater than 0 and no greater than 1.")

    if float(config.electronic_noise) < 0.0:
        raise ValueError("Electronic noise (Ne) cannot be negative.")

    if config.seed is not None and int(config.seed) < 0:
        raise ValueError("Random seed cannot be negative.")

    if int(config.preview_interval) < 0:
        raise ValueError("Preview interval cannot be negative.")

    if config.parallel_mode not in {"auto", "processes", "threads", "sequential"}:
        raise ValueError(f"Unknown parallel mode: {config.parallel_mode}")

    if int(config.max_workers) < 0:
        raise ValueError("Parallel worker count cannot be negative.")

    if any(sep in config.file_suffix for sep in (os.sep, os.altsep) if sep):
        raise ValueError("Output suffix cannot contain path separators.")
