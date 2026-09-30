import os
from pathlib import Path
from .models import NoiseJobConfig


def validate_config(config: NoiseJobConfig) -> None:
    input_path = Path(config.input_path).expanduser()
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if not str(config.output_dir).strip():
        raise ValueError("An output directory is required.")

    if config.input_mode not in {"auto", "single", "multi"}:
        raise ValueError(f"Unknown input mode: {config.input_mode}")

    if not (0.0 < float(config.mas_factor) <= 1.0):
        raise ValueError("mAs factor must be greater than 0 and no greater than 1.")

    if float(config.electronic_noise) < 0.0:
        raise ValueError("Electronic noise (Ne) cannot be negative.")

    if int(config.preview_interval) < 0:
        raise ValueError("Preview interval cannot be negative.")

    if config.parallel_mode not in {"auto", "processes", "threads", "sequential"}:
        raise ValueError(f"Unknown parallel mode: {config.parallel_mode}")

    if int(config.max_workers) < 0:
        raise ValueError("Parallel worker count cannot be negative.")

    if any(sep in config.file_suffix for sep in (os.sep, os.altsep) if sep):
        raise ValueError("Output suffix cannot contain path separators.")
