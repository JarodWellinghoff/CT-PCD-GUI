from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import Any

from ct_pcd_gui import __version__ as application_version

from ..domain.models import (
    DicomVolume,
    LesionSession,
    ProcessingResult,
    RawDatasetInfo,
    ValidationError,
)
from .model_transform import transform_model
from .output_paths import validate_output_location
from .reconstruction import ExternalReconstructionRunner, ReconstructionCancelled


class InsertionCancelled(RuntimeError):
    pass


def _legacy_api():
    try:
        from .._vendor.lesion_pipeline import InsertionConfig, InsertionSpec, run_insertion
        from .._vendor.lesion_pipeline import __version__ as pipeline_version
        from .._vendor.lesion_pipeline.lesion_models import (
            load_lesion_model,
            save_lesion_model_npz,
        )
    except ImportError as exc:
        raise RuntimeError("The bundled lesion insertion pipeline is unavailable.") from exc
    return (
        InsertionConfig,
        InsertionSpec,
        run_insertion,
        pipeline_version,
        load_lesion_model,
        save_lesion_model_npz,
    )


def _safe_remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _relocate_paths(value: Any, source_root: Path, target_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _relocate_paths(item, source_root, target_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relocate_paths(item, source_root, target_root) for item in value]
    if isinstance(value, tuple):
        return tuple(_relocate_paths(item, source_root, target_root) for item in value)
    if isinstance(value, str):
        source_text = str(source_root.resolve())
        if value == source_text or value.startswith(source_text + os.sep):
            return str(target_root.resolve()) + value[len(source_text) :]
    return value


class LegacyLesionInsertionAdapter:
    """Clean application adapter around the coworker's pipeline 3.1.4."""

    def __init__(
        self,
        reconstruction_runner: ExternalReconstructionRunner | None = None,
    ) -> None:
        self._reconstruction_runner = reconstruction_runner or ExternalReconstructionRunner()

    def run(
        self,
        session: LesionSession,
        volume: DicomVolume,
        raw_info: RawDatasetInfo,
        *,
        cancel_event: Event,
        progress: Callable[[int, int, str], None],
        log: Callable[[str], None],
    ) -> ProcessingResult:
        session.validate_basic()
        output = validate_output_location(
            session.output_directory,
            input_locations=(
                session.reconstruction_source,
                session.raw_source,
                session.lesion_library_source,
                *(lesion.lesion_path for lesion in session.lesions),
            ),
        )
        if output.exists() and any(output.iterdir()):
            raise ValidationError(
                "The output directory is not empty; choose a new or empty directory."
            )
        if cancel_event.is_set():
            raise InsertionCancelled("Lesion insertion was cancelled.")
        enabled = [lesion for lesion in session.lesions if lesion.enabled]
        if not enabled:
            raise ValidationError("At least one lesion must be enabled.")
        volume.geometry.assert_legacy_alpha_compatible()

        (
            InsertionConfig,
            InsertionSpec,
            run_insertion,
            pipeline_version,
            load_lesion_model,
            save_lesion_model_npz,
        ) = _legacy_api()

        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=str(output.parent))
        )
        transformed_directory = staging / "lesion_models_used"
        modified_directory = staging / "dicom_ctpd_modified"
        config_path = staging / "lesion_insertion_config.json"
        session_path = staging / "lesion_insertion_session.json"
        manifest_path = staging / "lesion_insertion_manifest.json"
        insertion_specs = []
        transformed_records: list[dict[str, Any]] = []

        try:
            log("Preparing independent transformed lesion models…")
            transformed_directory.mkdir(parents=True, exist_ok=True)
            for index, lesion in enumerate(enabled, start=1):
                if cancel_event.is_set():
                    raise InsertionCancelled("Lesion insertion was cancelled.")
                source_model = load_lesion_model(lesion.lesion_path)
                transformed = transform_model(source_model, lesion.parameters)
                model_name = f"{index:03d}_{lesion.instance_id}.npz"
                model_path = save_lesion_model_npz(
                    transformed, transformed_directory / model_name
                )
                insertion_specs.append(
                    InsertionSpec(
                        lesion_model=str(model_path),
                        background_hu=[float(value) for value in lesion.background_hu],
                        center_ctpd_mm=[float(value) for value in lesion.center_ctpd_mm],
                        label=lesion.label,
                    )
                )
                transformed_records.append(
                    {
                        "instance_id": lesion.instance_id,
                        "lesion_id": lesion.lesion_id,
                        "source_model": str(Path(lesion.lesion_path).resolve()),
                        "transformed_model": str(Path("lesion_models_used") / model_name),
                        "enabled": lesion.enabled,
                        "visible": lesion.visible,
                        "center_voxel_crs": list(lesion.center_voxel_crs),
                        "center_patient_lps_mm": list(lesion.center_patient_lps_mm),
                        "center_ctpd_mm": list(lesion.center_ctpd_mm),
                        "background_hu": list(lesion.background_hu),
                        "parameters": {
                            "contrast_scale": lesion.parameters.contrast_scale,
                            "scale_xyz": list(lesion.parameters.scale_xyz),
                            "rotation_deg_xyz": list(lesion.parameters.rotation_deg_xyz),
                            "preview_opacity": lesion.parameters.preview_opacity,
                        },
                    }
                )

            config = InsertionConfig(
                ctpd_input=session.raw_source,
                output_dir=str(modified_directory),
                insertions=insertion_specs,
                t1_series_dir=None,
                spectrum_channel_map=session.spectrum_map,
                patient_position=volume.geometry.patient_position,
                workers=session.workers,
                overwrite=False,
            )
            config.save(config_path)
            session.save(session_path)

            def legacy_progress(completed: int, total: int, status: str) -> None:
                progress(completed, total, status)
                if cancel_event.is_set():
                    raise InsertionCancelled(
                        "Lesion insertion was cancelled after the current projection file."
                    )

            log(
                f"Running coworker pipeline {pipeline_version} on "
                f"{raw_info.projection_frame_count:,} projection frame(s)…"
            )
            summary = run_insertion(config, progress=legacy_progress)
            if cancel_event.is_set():
                raise InsertionCancelled("Lesion insertion was cancelled.")
            summary = _relocate_paths(summary, staging, output)
            _write_json(modified_directory / "lesion_insertion_summary.json", summary)
            _write_json(config_path, _relocate_paths(config.to_dict(), staging, output))

            manifest: dict[str, Any] = {
                "manifest_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "application": {
                    "name": "CT-PCD-GUI",
                    "version": application_version,
                    "module": "lesion-insertion",
                    "module_version": "0.1.0",
                },
                "algorithm": {
                    "name": "DICOM-CT-PD Lesion Insertion Pipeline",
                    "version": pipeline_version,
                    "preview": "image-domain approximation only",
                    "final_processing": "raw DICOM-CT-PD forward projection",
                },
                "inputs": {
                    "reconstruction_source": str(
                        Path(session.reconstruction_source).expanduser().resolve()
                    ),
                    "reconstruction_series_instance_uid": volume.geometry.series_instance_uid,
                    "reconstruction_study_instance_uid": volume.geometry.study_instance_uid,
                    "reconstruction_frame_of_reference_uid": (
                        volume.geometry.frame_of_reference_uid
                    ),
                    "ctpd_source": str(Path(session.raw_source).expanduser().resolve()),
                    "ctpd_series_instance_uids": list(raw_info.series_instance_uids),
                    "ctpd_study_instance_uids": list(raw_info.study_instance_uids),
                    "ctpd_frame_of_reference_uids": list(
                        raw_info.frame_of_reference_uids
                    ),
                    "projection_file_count": raw_info.projection_file_count,
                    "projection_frame_count": raw_info.projection_frame_count,
                },
                "spatial_conventions": {
                    "viewer_voxel": "(column, row, zero-based slice)",
                    "patient": "DICOM LPS millimetres",
                    "projector": "validated Alpha HFS/FFS CTPD millimetres",
                    "patient_position": volume.geometry.patient_position,
                    "row_spacing_mm": volume.geometry.row_spacing_mm,
                    "column_spacing_mm": volume.geometry.column_spacing_mm,
                    "slice_positions_lps_mm": [
                        list(value) for value in volume.geometry.image_positions_lps_mm
                    ],
                },
                "spectrum_channel_map": {
                    str(key): value for key, value in session.spectrum_map.items()
                },
                "workers": session.workers,
                "lesions": transformed_records,
                "outputs": {
                    "root": str(output.resolve()),
                    "modified_ctpd": "dicom_ctpd_modified",
                    "summary": "dicom_ctpd_modified/lesion_insertion_summary.json",
                    "configuration": "lesion_insertion_config.json",
                    "session": "lesion_insertion_session.json",
                    "manifest": "lesion_insertion_manifest.json",
                },
                "reconstruction": {
                    "configured": bool(session.reconstruction_command.strip()),
                    "status": (
                        "pending" if session.reconstruction_command.strip() else "not-configured"
                    ),
                    "command_template": session.reconstruction_command,
                    "output_directory": session.reconstruction_output_directory,
                },
                "pipeline_summary": summary,
            }
            _write_json(manifest_path, manifest)

            if output.exists():
                output.rmdir()
            os.replace(staging, output)
            final_modified = output / "dicom_ctpd_modified"
            final_session = output / "lesion_insertion_session.json"
            reconstruction_output = ""
            if session.reconstruction_command.strip():
                reconstruction_target = validate_output_location(
                    session.reconstruction_output_directory,
                    input_locations=(
                        session.reconstruction_source,
                        session.raw_source,
                        session.lesion_library_source,
                        output,
                        *(lesion.lesion_path for lesion in session.lesions),
                    ),
                )
                try:
                    result = self._reconstruction_runner.run(
                        session.reconstruction_command,
                        input_directory=final_modified,
                        output_directory=reconstruction_target,
                        session_path=final_session,
                        cancel_event=cancel_event,
                        log=log,
                    )
                except ReconstructionCancelled as exc:
                    manifest["reconstruction"]["status"] = "cancelled"
                    manifest["reconstruction"]["message"] = str(exc)
                    _write_json(output / "lesion_insertion_manifest.json", manifest)
                    raise InsertionCancelled(str(exc)) from exc
                except Exception as exc:
                    manifest["reconstruction"]["status"] = "failed"
                    manifest["reconstruction"]["message"] = str(exc)
                    _write_json(output / "lesion_insertion_manifest.json", manifest)
                    raise
                reconstruction_output = str(result)
                manifest["reconstruction"]["status"] = "completed"
                manifest["outputs"]["reconstruction"] = reconstruction_output
                _write_json(output / "lesion_insertion_manifest.json", manifest)

            return ProcessingResult(
                output_directory=str(output),
                modified_ctpd_directory=str(final_modified),
                manifest_path=str(output / "lesion_insertion_manifest.json"),
                summary_path=str(final_modified / "lesion_insertion_summary.json"),
                reconstruction_output_directory=reconstruction_output,
                summary=summary,
            )
        except Exception:
            if staging.exists():
                _safe_remove_tree(staging)
            raise
