from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .config import bind_paths, dump_config, load_config, validate_config
from .images import normalize_volume, slice_positions
from .models import JobContext, PreparedJob, ReconstructionResult
from .process import resolve_executable, run_process


def _write_status(directory: Path, state: str, **details) -> None:
    temporary = directory / "status.json.part"
    temporary.write_text(json.dumps({"state": state, **details}, indent=2), encoding="utf-8")
    os.replace(temporary, directory / "status.json")


def _unique_directory(parent: Path, prefix: str) -> Path:
    name = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    directory = parent / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def prepare(
    executable: str, source: Path, output: Path, context: JobContext,
    *, spectrum: int = -1, source_index: int = -1,
) -> PreparedJob:
    # Lazy import keeps process/config/volume functionality usable without Qt or
    # pydicom, and avoids changing the rest of the application at import time.
    from .dicom import stage_projections

    engine = resolve_executable(executable)
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.exists():
        raise ValueError("The selected CTPD input does not exist.")
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("Choose an output folder outside the input tree to avoid reading outputs.")
    context.check()
    job = _unique_directory(output, "freect")
    _write_status(job, "preparing")
    try:
        projections = stage_projections(source, job / "projections", context, spectrum, source_index)
        stdout = run_process(
            [str(engine), "--no-telemetry", "generate-base-config",
             f"--input-battelle={job / 'projections'}"],
            job, "generate-config", context, log_stdout=False, timeout=300,
        )
        config = load_config(stdout.read_text(encoding="utf-8-sig"))
        geometry, volume = config["ScanGeometry"], config["ReconVolume"]
        # Main.go zeroes the WFBP override struct before command dispatch. Some
        # v2 builds consequently emit zero kernels from generate-base-config.
        for key in ("KernelA", "KernelC"):
            if config["ReconSpecific"].get(key) == 0:
                config["ReconSpecific"][key] = 0.5
                context.report(f"Using FreeCT's default {key}=0.5 instead of generated zero.")
        start, end = geometry.get("StartTablePosition"), geometry.get("EndTablePosition")
        if start == end and isinstance(start, (int, float)) and math.isfinite(start):
            # GenerateBaseConfig divides by |end-start| for an axial acquisition.
            volume["StartPosition"] = volume["EndPosition"] = float(start)
            context.report("Stationary table: defaulting to one central axial slice; review the range.")
        calibration = geometry.get("HUCalibrationFactor", 0)
        config["ImageOutput"]["RescaleToHU"] = (
            isinstance(calibration, (int, float)) and math.isfinite(calibration) and calibration > 0
        )
        config = bind_paths(config, job / "projections", job, False)
        text = dump_config(config)
        (job / "generated-config.yaml").write_text(text, encoding="utf-8")
        _write_status(job, "prepared", projections=projections, executable=str(engine))
        context.report(f"Prepared {projections} projections. Review geometry and reconstruction settings.", 100)
        return PreparedJob(job, engine, projections, text)
    except Exception as exc:
        _write_status(job, "preparation_failed_or_cancelled", error=str(exc))
        raise


def reconstruct(
    prepared: PreparedJob, config_text: str, native_dicom: bool, context: JobContext,
) -> ReconstructionResult:
    context.check()
    config = load_config(config_text)
    original = load_config(prepared.config_text)
    for key in ("DetectorRows", "DetectorChannels", "NumProjectionsTotal"):
        if config["ScanGeometry"].get(key) != original["ScanGeometry"].get(key):
            raise ValueError(f"{key} cannot differ from the staged acquisition.")
    staging = prepared.directory / "projections"
    staged = list(staging.iterdir())
    if len(staged) != prepared.projections or any(not p.is_file() or p.suffix != ".dcm" for p in staged):
        raise ValueError("Staged projection directory has changed; prepare the input again.")
    # Validate before allocating a reconstruction attempt directory.
    config = bind_paths(config, staging, prepared.directory, native_dicom)
    validate_config(config, prepared.projections)
    attempt = _unique_directory(prepared.directory, "recon")
    config = bind_paths(config, staging, attempt, native_dicom)
    config_file = attempt / "reconstruction.yaml"
    config_file.write_text(dump_config(config), encoding="utf-8")
    if native_dicom:
        (attempt / "dicom_native").mkdir()
    _write_status(attempt, "reconstructing")
    try:
        run_process(
            [str(prepared.executable), "--no-telemetry", "recon", str(config_file)],
            attempt, "recon", context,
        )
        # This FreeCT build logs some HU/writer failures but still exits zero.
        for suffix in ("stdout", "stderr"):
            with (attempt / f"recon.{suffix}.log").open(encoding="utf-8", errors="replace") as log:
                for line in log:
                    if any(term in line.lower() for term in ("level=error", "level=fatal", "panic:",
                                                            "could not rescale to hu")):
                        raise RuntimeError(f"FreeCT reported an error despite exiting zero: {line.strip()}")
        image_path = attempt / "reconstruction.nii"
        header = normalize_volume(
            attempt / "freect_native.nii.gz", image_path, config["ReconVolume"], context,
        )
        if native_dicom:
            files = list((attempt / "dicom_native").glob("*.dcm"))
            if len(files) != header.shape[0] or any(p.stat().st_size == 0 for p in files):
                raise ValueError("FreeCT did not write the expected native DICOM files.")
        positions = slice_positions(config["ReconVolume"])
        pitch = float(config["ReconVolume"]["SlicePitch"])
        if positions[-1] < positions[0]:
            pitch = -pitch
        rescaled = bool(config["ImageOutput"]["RescaleToHU"])
        metadata = {
            "image": image_path.name, "shape_zyx": list(header.shape),
            "intensity_units": "HU" if rescaled else "attenuation (FreeCT native units)",
            "slice_positions_mm": positions, "patient_orientation": "unknown",
            "coordinate_system": "FreeCT reconstruction/scanner grid, NOT patient LPS/RAS",
            "spacing_zyx_mm": [abs(pitch), config["ReconVolume"]["FieldOfView"] / header.shape[1],
                               config["ReconVolume"]["FieldOfView"] / header.shape[2]],
            "slice_thickness_mm": config["ReconVolume"]["SliceThickness"],
            "native_output": "freect_native.nii.gz", "native_dicom_experimental": native_dicom,
            "executable": str(prepared.executable), "config": config_file.name,
            "research_only": True,
        }
        (attempt / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        context.check()
        _write_status(attempt, "completed", **metadata)
        context.report(f"Reconstruction completed: {image_path}", 100)
        return ReconstructionResult(attempt, image_path, header.shape, rescaled, positions[0], pitch)
    except Exception as exc:
        _write_status(attempt, "failed_or_cancelled", error=str(exc))
        raise
