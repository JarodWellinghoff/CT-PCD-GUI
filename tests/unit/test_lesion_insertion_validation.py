from __future__ import annotations

from pathlib import Path

import numpy as np

from ct_pcd_gui.features.lesion_insertion.application.validation import validate_session
from ct_pcd_gui.features.lesion_insertion.domain.models import (
    DicomVolume,
    LesionInstance,
    LesionSession,
    RawDatasetInfo,
    SeriesGeometry,
)


def _volume() -> DicomVolume:
    geometry = SeriesGeometry(
        series_instance_uid="series",
        study_instance_uid="study",
        frame_of_reference_uid="frame",
        rows=4,
        columns=4,
        image_positions_lps_mm=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        column_axis_lps=(1.0, 0.0, 0.0),
        row_axis_lps=(0.0, 1.0, 0.0),
        row_spacing_mm=1.0,
        column_spacing_mm=1.0,
        patient_position="HFS",
        reconstruction_diameter_mm=4.0,
    )
    return DicomVolume(geometry, np.zeros((2, 4, 4), dtype=np.float32), ())


def test_validation_rejects_channel_mapping_outside_model(tmp_path: Path) -> None:
    recon = tmp_path / "recon"
    raw = tmp_path / "raw"
    library = tmp_path / "library"
    for path in (recon, raw, library):
        path.mkdir()
    model = library / "model.npz"
    model.write_bytes(b"x")
    lesion = LesionInstance.create(
        lesion_path=str(model),
        lesion_id="one-channel",
        label="one-channel",
        center_voxel_crs=(1.0, 1.0, 0.0),
        center_patient_lps_mm=(1.0, 1.0, 0.0),
        center_ctpd_mm=(-1.0, 1.0, 0.0),
        background_hu=(30.0,),
    )
    session = LesionSession(
        reconstruction_source=str(recon),
        reconstruction_series_uid="series",
        raw_source=str(raw),
        lesion_library_source=str(library),
        output_directory=str(tmp_path / "output"),
        lesions=(lesion,),
        spectrum_channel_map=((1, 1),),
    )
    raw_info = RawDatasetInfo(
        source=str(raw),
        study_instance_uids=("study",),
        series_instance_uids=("raw-series",),
        frame_of_reference_uids=("frame",),
        patient_positions=("HFS",),
        spectrum_indices=(1,),
        source_indices=(1,),
        projection_file_count=1,
        projection_frame_count=1,
    )

    issues = validate_session(session, _volume(), raw_info)

    assert any(issue.code == "lesion-channel-mismatch" for issue in issues)
