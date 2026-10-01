from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ct_pcd_gui.features.lesion_extraction.extract_case import (
    crop_volume_with_padding,
    legacy_padded_bbox,
    parse_segment_definitions,
)
from ct_pcd_gui.features.lesion_extraction.models import (
    InputValidationError,
    LesionCandidate,
    LesionNpzRecord,
    make_unique_output_names,
    sanitize_output_name,
)
from ct_pcd_gui.features.lesion_extraction.writers import (
    NPZ_FIELDS,
    record_arrays,
    write_lesion_npz,
)


REFERENCE_FIELDS = (
    "PatientName",
    "LesionNumber",
    "Org_case_path",
    "Org_mask",
    "Org_dcm_path",
    "DicomHeaderJSON",
    "LesionMask",
    "VOI",
    "org_slice_rng",
    "org_col_rng",
    "LesionCenter",
    "org_row_rng",
    "LesionDiameter",
    "LesionMeanHU",
    "LesionRoundness",
    "LesionMaxHU",
    "LesionMinHU",
    "LesionMedianHU",
    "LesionSigma",
    "LesionVariance",
    "LesionVoxelCount",
    "LesionPhysicalSize",
)


def _record() -> LesionNpzRecord:
    voi = np.arange(5 * 7 * 3, dtype=np.int16).reshape(5, 7, 3)
    mask = np.zeros_like(voi, dtype=bool)
    mask[1:4, 2:5, 1] = True
    return LesionNpzRecord(
        patient_name="L005-1-Lesion",
        lesion_number=0,
        org_case_path="C:/case",
        org_mask="C:/case/segments.seg.nrrd",
        org_dcm_path="C:/case",
        dicom_header={
            "Rows": "512",
            "Columns": "512",
            "PixelSpacing": ["0.7", "0.7"],
            "SliceThickness": "1.0",
            "ReconstructionDiameter": "358.4",
            "KVP": "120",
        },
        lesion_mask=mask,
        voi=voi,
        org_slice_rng=np.asarray([20, 22]),
        org_col_rng=np.asarray([100, 106]),
        lesion_center=np.asarray([103, 202, 21]),
        org_row_rng=np.asarray([200, 204]),
        lesion_diameter=np.asarray([4.1, 5.2, 6.3]),
        lesion_mean_hu=42.5,
        lesion_roundness=0.81,
        lesion_max_hu=80.0,
        lesion_min_hu=-5.0,
        lesion_median_hu=40.0,
        lesion_sigma=8.0,
        lesion_variance=64.0,
        lesion_voxel_count=int(mask.sum()),
        lesion_physical_size=18.0,
    )


def _candidate(candidate_id: str, output_name: str) -> LesionCandidate:
    return LesionCandidate(
        candidate_id=candidate_id,
        segment_key="Segment0",
        segment_name="Lesion",
        component_label=1,
        component_index=1,
        component_count=1,
        output_name=output_name,
        voxel_count=10,
        physical_size_mm3=10.0,
        centroid_xyz_mm=(0.0, 0.0, 0.0),
    )


def test_npz_field_order_matches_reference_format() -> None:
    assert NPZ_FIELDS == REFERENCE_FIELDS
    assert tuple(record_arrays(_record())) == REFERENCE_FIELDS


def test_writer_emits_reference_dtypes_without_pickle(tmp_path: Path) -> None:
    output = write_lesion_npz(_record(), tmp_path / "lesion.npz")
    with np.load(output, allow_pickle=False) as data:
        assert tuple(data.files) == REFERENCE_FIELDS
        assert data["PatientName"].shape == ()
        assert data["PatientName"].dtype.kind == "U"
        assert data["LesionNumber"].dtype == np.dtype(np.int64)
        assert data["LesionMask"].dtype == np.dtype(bool)
        assert data["VOI"].dtype == np.dtype(np.int16)
        assert data["LesionMask"].shape == data["VOI"].shape == (5, 7, 3)
        assert data["LesionCenter"].shape == (3,)
        assert data["LesionDiameter"].dtype == np.dtype(np.float64)
        assert json.loads(str(data["DicomHeaderJSON"]))["Columns"] == "512"


def test_reference_padding_reproduces_attached_volume_shape() -> None:
    original_bbox = np.asarray(
        [[182, 198], [286, 300], [188, 194]], dtype=np.int64
    )
    padded = legacy_padded_bbox(original_bbox)
    dimensions_xyz = padded[:, 1] - padded[:, 0] + 1
    assert padded.tolist() == [[174, 206], [279, 307], [185, 197]]
    assert (dimensions_xyz[1], dimensions_xyz[0], dimensions_xyz[2]) == (
        29,
        33,
        13,
    )


def test_crop_returns_row_column_slice_order_with_boundary_padding() -> None:
    volume = np.fromfunction(
        lambda z, y, x: 100 * z + 10 * y + x,
        (2, 3, 4),
        dtype=int,
    ).astype(np.int16)
    mask = np.zeros_like(volume, dtype=bool)
    mask[1, 2, 3] = True
    bbox = np.asarray([[-1, 3], [0, 2], [0, 2]], dtype=np.int64)
    voi, lesion_mask = crop_volume_with_padding(volume, mask, bbox)
    assert voi.shape == lesion_mask.shape == (3, 5, 3)
    assert voi[2, 4, 1] == volume[1, 2, 3]
    assert lesion_mask[2, 4, 1]
    assert voi[2, 0, 1] == -1024
    assert voi[2, 4, 2] == -1024


def test_parse_slicer_segment_metadata() -> None:
    header = {
        "Segment0_Name": "Liver lesion",
        "Segment0_LabelValue": "1",
        "Segment0_Layer": "0",
        "Segment0_Color": "0.2 0.4 0.6",
        "Segment1_Name": "Kidney",
        "Segment1_LabelValue": 3,
        "Segment1_Layer": 2,
    }
    segments = parse_segment_definitions(header)
    assert [segment.key for segment in segments] == ["Segment0", "Segment1"]
    assert segments[0].name == "Liver lesion"
    assert segments[0].color_rgb == (51, 102, 153)
    assert segments[1].layer == 2


def test_output_names_are_windows_safe_and_unique_case_insensitively() -> None:
    candidates = (
        _candidate("a", "Lesion: 1"),
        _candidate("b", "lesion- 1"),
        _candidate("c", "Lesion: 1"),
    )
    unique = make_unique_output_names(candidates)
    assert sanitize_output_name(" bad/name?. ") == "bad-name-"
    assert unique[0].output_name == "Lesion- 1"
    assert unique[1].output_name == "lesion- 1-02"
    assert unique[2].output_name == "Lesion- 1-03"


def test_writer_rejects_mask_without_background() -> None:
    record = _record()
    record.lesion_mask[:] = True
    with pytest.raises(InputValidationError, match="fills the entire VOI"):
        record_arrays(record)
