from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ct_pcd_gui.features.lesion_insertion.application.session import LesionSessionState
from ct_pcd_gui.features.lesion_insertion.domain.models import (
    LesionInstance,
    LesionParameters,
    LesionSession,
    SeriesGeometry,
    UnsupportedSpatialGeometry,
    ValidationError,
)


def _geometry(*, oblique: bool = False) -> SeriesGeometry:
    column_axis = (0.70710678, 0.70710678, 0.0) if oblique else (1.0, 0.0, 0.0)
    row_axis = (-0.70710678, 0.70710678, 0.0) if oblique else (0.0, 1.0, 0.0)
    return SeriesGeometry(
        series_instance_uid="series",
        study_instance_uid="study",
        frame_of_reference_uid="frame",
        rows=10,
        columns=20,
        image_positions_lps_mm=((1.0, 2.0, 3.0), (1.0, 2.0, 5.0)),
        column_axis_lps=column_axis,
        row_axis_lps=row_axis,
        row_spacing_mm=2.0,
        column_spacing_mm=0.5,
        patient_position="HFS",
        reconstruction_diameter_mm=10.0,
    )


def _lesion(path: Path, label: str = "L1") -> LesionInstance:
    path.write_bytes(b"model")
    return LesionInstance.create(
        lesion_path=str(path),
        lesion_id="Lesion01",
        label=label,
        center_voxel_crs=(2.0, 3.0, 1.0),
        center_patient_lps_mm=(2.0, 3.0, 4.0),
        center_ctpd_mm=(1.0, 2.0, 3.0),
        background_hu=(55.0, 60.0),
    )


def test_parameters_enforce_ranges() -> None:
    LesionParameters().validate()
    with pytest.raises(ValidationError, match="Contrast"):
        LesionParameters(contrast_scale=6.0).validate()
    with pytest.raises(ValidationError, match="scale"):
        LesionParameters(scale_xyz=(0.1, 1.0, 1.0)).validate()
    with pytest.raises(ValidationError, match="rotations"):
        LesionParameters(rotation_deg_xyz=(181.0, 0.0, 0.0)).validate()


def test_voxel_patient_round_trip_handles_non_square_spacing() -> None:
    geometry = _geometry()
    patient = geometry.voxel_to_patient((4.0, 3.0, 1.0))
    assert patient == pytest.approx([3.0, 8.0, 5.0])
    voxel = geometry.patient_to_voxel(tuple(patient))
    assert voxel == pytest.approx([4.0, 3.0, 1.0])


def test_oblique_patient_coordinates_supported_but_final_mapping_blocked() -> None:
    geometry = _geometry(oblique=True)
    patient = geometry.voxel_to_patient((4.0, 3.0, 0.0))
    assert np.isfinite(patient).all()
    with pytest.raises(UnsupportedSpatialGeometry, match="oblique"):
        geometry.voxel_to_legacy_ctpd((4.0, 3.0, 0.0))


def test_multiple_lesions_are_independent_and_reorderable(tmp_path: Path) -> None:
    first = _lesion(tmp_path / "first.npz", "first")
    second = _lesion(tmp_path / "second.npz", "second")
    state = LesionSessionState()
    state.add(first)
    state.add(second)
    updated = replace(first, parameters=LesionParameters(contrast_scale=2.0))
    state.update(updated)

    assert state.session.lesions[0].parameters.contrast_scale == 2.0
    assert state.session.lesions[1].parameters.contrast_scale == 1.0

    copy = state.duplicate(second.instance_id)
    assert copy.instance_id != second.instance_id
    state.move(copy.instance_id, -2)
    assert state.session.lesions[0].instance_id == copy.instance_id
    state.remove(first.instance_id)
    assert all(item.instance_id != first.instance_id for item in state.session.lesions)


def test_nonuniform_spacing_blocks_final_alpha_mapping() -> None:
    geometry = replace(
        _geometry(),
        image_positions_lps_mm=(
            (1.0, 2.0, 3.0),
            (1.0, 2.0, 5.0),
            (1.0, 2.0, 9.0),
        ),
    )
    with pytest.raises(UnsupportedSpatialGeometry, match="non-uniform"):
        geometry.voxel_to_legacy_ctpd((4.0, 3.0, 1.0))


def test_session_serialization_round_trip(tmp_path: Path) -> None:
    lesion = _lesion(tmp_path / "model.npz")
    session = LesionSession(
        reconstruction_source="recon",
        reconstruction_series_uid="series",
        raw_source="raw",
        lesion_library_source="library",
        output_directory="output",
        lesions=(lesion,),
        spectrum_channel_map=((1, 0), (2, 1)),
        workers=3,
        reconstruction_command="reconstruct {input} {output}",
    )
    path = session.save(tmp_path / "session.json")
    restored = LesionSession.load(path)
    assert restored == session
    assert restored.lesions[0].parameters == lesion.parameters
    assert restored.spectrum_map == {1: 0, 2: 1}


def test_association_validation_rejects_patient_and_frame_mismatch() -> None:
    from ct_pcd_gui.features.lesion_insertion.domain.models import DicomVolume, RawDatasetInfo
    from ct_pcd_gui.features.lesion_insertion.infrastructure.dicom_service import (
        PydicomLesionDicomService,
    )

    geometry = replace(
        _geometry(),
        patient_identity_digest="recon-patient",
        frame_of_reference_uid="recon-frame",
    )
    volume = DicomVolume(
        geometry=geometry,
        hu=np.zeros((2, 10, 20), dtype=np.float32),
        source_paths=(),
    )
    raw = RawDatasetInfo(
        source="raw",
        study_instance_uids=(geometry.study_instance_uid,),
        series_instance_uids=("raw-series",),
        frame_of_reference_uids=("other-frame",),
        patient_positions=("HFS",),
        spectrum_indices=(1,),
        source_indices=(1,),
        projection_file_count=1,
        projection_frame_count=1,
        patient_identity_digest="other-patient",
    )
    issues = PydicomLesionDicomService().validate_association(volume, raw)
    error_codes = {issue.code for issue in issues if issue.severity == "error"}
    assert "patient-mismatch" in error_codes
    assert "frame-of-reference-mismatch" in error_codes


def test_flipped_axial_orientation_is_not_silently_approximated() -> None:
    geometry = replace(
        _geometry(),
        column_axis_lps=(-1.0, 0.0, 0.0),
        row_axis_lps=(0.0, -1.0, 0.0),
    )
    with pytest.raises(UnsupportedSpatialGeometry, match="oblique or non-axial"):
        geometry.voxel_to_legacy_ctpd((4.0, 3.0, 0.0))


def test_patient_position_alone_does_not_confirm_case_association() -> None:
    from ct_pcd_gui.features.lesion_insertion.domain.models import DicomVolume, RawDatasetInfo
    from ct_pcd_gui.features.lesion_insertion.infrastructure.dicom_service import (
        PydicomLesionDicomService,
    )

    geometry = replace(
        _geometry(),
        study_instance_uid="",
        frame_of_reference_uid="",
        patient_identity_digest="",
    )
    volume = DicomVolume(
        geometry=geometry,
        hu=np.zeros((2, 10, 20), dtype=np.float32),
        source_paths=(),
    )
    raw = RawDatasetInfo(
        source="raw",
        study_instance_uids=(),
        series_instance_uids=("raw-series",),
        frame_of_reference_uids=(),
        patient_positions=("HFS",),
        spectrum_indices=(1,),
        source_indices=(1,),
        projection_file_count=1,
        projection_frame_count=1,
    )
    issues = PydicomLesionDicomService().validate_association(volume, raw)
    assert any(issue.code == "association-unconfirmed" for issue in issues)
