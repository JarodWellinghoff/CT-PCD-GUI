from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ct_pcd_gui.cli import extract_lesions as cli
from ct_pcd_gui.features.lesion_extraction import exporter
from ct_pcd_gui.features.lesion_extraction.extract_case import (
    crop_volume_with_padding,
    legacy_padded_bbox,
    parse_segment_definitions,
    validate_aligned_series_geometry,
)
from ct_pcd_gui.features.lesion_extraction.models import (
    LESION_TYPES,
    LESION_TYPE_KEY,
    CandidateSelection,
    DicomSeriesSpec,
    ExportSummary,
    InputValidationError,
    LesionCandidate,
    LesionNpzRecord,
    SegmentDefinition,
    SegmentationInfo,
    make_unique_output_names,
    sanitize_output_name,
)
from ct_pcd_gui.features.lesion_extraction.writers import (
    MULTI_SERIES_NPZ_FIELDS,
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


class _ImageGeometry:
    def __init__(
        self,
        *,
        size=(16, 12, 8),
        spacing=(0.7, 0.7, 1.0),
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
    ) -> None:
        self._size = size
        self._spacing = spacing
        self._origin = origin
        self._direction = direction

    def GetSize(self):
        return self._size

    def GetSpacing(self):
        return self._spacing

    def GetOrigin(self):
        return self._origin

    def GetDirection(self):
        return self._direction


def _record(channel_count: int = 1) -> LesionNpzRecord:
    base_voi = np.arange(5 * 7 * 3, dtype=np.int16).reshape(5, 7, 3)
    base_mask = np.zeros_like(base_voi, dtype=bool)
    base_mask[1:4, 2:5, 1] = True
    if channel_count == 1:
        voi = base_voi
        mask = base_mask
    else:
        voi = np.stack(
            [base_voi + 100 * index for index in range(channel_count)], axis=-1
        )
        mask = np.repeat(base_mask[..., np.newaxis], channel_count, axis=3)
    headers = tuple(
        {
            "Rows": "512",
            "Columns": "512",
            "PixelSpacing": ["0.7", "0.7"],
            "SliceThickness": "1.0",
            "ReconstructionDiameter": "358.4",
            "KVP": str(80 + 20 * index),
        }
        for index in range(channel_count)
    )
    return LesionNpzRecord(
        patient_name="L005-1-Lesion",
        lesion_type="BENIGN",
        lesion_number=0,
        org_case_path="C:/case/series-1",
        org_mask="C:/case/segments.seg.nrrd",
        org_dcm_path="C:/case/series-1",
        dicom_header=headers[0],
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
        lesion_voxel_count=int(base_mask.sum()),
        lesion_physical_size=18.0,
        series_names=tuple(
            f"Series {index + 1}" for index in range(channel_count)
        ),
        series_uids=tuple(
            f"1.2.840.10008.{index + 1}" for index in range(channel_count)
        ),
        series_source_paths=tuple(
            f"C:/case/series-{index + 1}" for index in range(channel_count)
        ),
        dicom_headers=headers,
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
    assert NPZ_FIELDS == REFERENCE_FIELDS + (LESION_TYPE_KEY,)
    assert tuple(record_arrays(_record())) == NPZ_FIELDS


def test_writer_emits_reference_dtypes_without_pickle(tmp_path: Path) -> None:
    output = write_lesion_npz(_record(), tmp_path / "lesion.npz")
    with np.load(output, allow_pickle=False) as data:
        assert tuple(data.files) == NPZ_FIELDS
        assert data[LESION_TYPE_KEY].item() == "BENIGN"
        assert data[LESION_TYPE_KEY].shape == ()
        assert data[LESION_TYPE_KEY].dtype.kind == "U"
        assert data["PatientName"].shape == ()
        assert data["PatientName"].dtype.kind == "U"
        assert data["LesionNumber"].dtype == np.dtype(np.int64)
        assert data["LesionMask"].dtype == np.dtype(bool)
        assert data["VOI"].dtype == np.dtype(np.int16)
        assert data["LesionMask"].shape == data["VOI"].shape == (5, 7, 3)
        assert data["LesionCenter"].shape == (3,)
        assert data["LesionDiameter"].dtype == np.dtype(np.float64)
        assert json.loads(str(data["DicomHeaderJSON"]))["Columns"] == "512"


def test_multi_series_writer_stores_all_series_as_channels(tmp_path: Path) -> None:
    output = write_lesion_npz(_record(2), tmp_path / "lesion.npz")
    with np.load(output, allow_pickle=False) as data:
        assert tuple(data.files) == NPZ_FIELDS + MULTI_SERIES_NPZ_FIELDS
        assert data["VOI"].shape == data["LesionMask"].shape == (5, 7, 3, 2)
        assert data["VOI"].dtype == np.dtype(np.int16)
        assert data["LesionMask"].dtype == np.dtype(bool)
        assert int(data["SeriesCount"]) == 2
        assert int(data["ReferenceSeriesIndex"]) == 0
        assert data["SeriesNames"].tolist() == ["Series 1", "Series 2"]
        assert data["SeriesUIDs"].tolist() == [
            "1.2.840.10008.1",
            "1.2.840.10008.2",
        ]
        assert len(data["DicomHeadersJSON"]) == 2
        assert json.loads(str(data["DicomHeadersJSON"][1]))["KVP"] == "100"
        expected_means = [
            np.mean(data["VOI"][..., channel][data["LesionMask"][..., channel]])
            for channel in range(2)
        ]
        assert np.allclose(data["LesionMeanHUByChannel"], expected_means)


def test_multi_series_writer_repeats_shared_3d_mask() -> None:
    record = _record(2)
    record.lesion_mask = record.lesion_mask[..., 0]
    arrays = record_arrays(record)
    assert arrays["LesionMask"].shape == arrays["VOI"].shape
    assert np.array_equal(
        arrays["LesionMask"][..., 0], arrays["LesionMask"][..., 1]
    )


def test_multi_series_writer_requires_metadata_for_each_channel() -> None:
    record = _record(2)
    record.series_names = ("only one",)
    with pytest.raises(InputValidationError, match="series_names"):
        record_arrays(record)


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


def test_multi_series_geometry_must_be_voxel_aligned() -> None:
    reference = _ImageGeometry()
    validate_aligned_series_geometry(
        reference,
        _ImageGeometry(origin=(0.0005, 0.0, 0.0)),
        reference_name="T1",
        candidate_name="T2",
    )
    with pytest.raises(InputValidationError, match="not aligned"):
        validate_aligned_series_geometry(
            reference,
            _ImageGeometry(spacing=(0.8, 0.7, 1.0)),
            reference_name="T1",
            candidate_name="T2",
        )


@pytest.mark.parametrize("lesion_type", ("METASTATIC", "BENIGN", "PRIMARY"))
@pytest.mark.parametrize("channel_count", (1, 2))
def test_diagnosis_round_trips_under_exact_key(
    tmp_path: Path, lesion_type: str, channel_count: int
) -> None:
    record = _record(channel_count)
    record.lesion_type = lesion_type
    output = write_lesion_npz(record, tmp_path / "lesion.npz")
    with np.load(output, allow_pickle=False) as data:
        assert data["Patient.Lesion.LesionType"].item() == lesion_type
        assert data["Patient.Lesion.LesionType"].shape == ()
        assert data["Patient.Lesion.LesionType"].dtype.kind == "U"
        np.testing.assert_array_equal(data["VOI"], record.voi)
        np.testing.assert_array_equal(data["LesionMask"], record.lesion_mask)


@pytest.mark.parametrize("lesion_type", ("", " ", "UNKNOWN", "benign", None))
def test_writer_rejects_missing_or_invalid_diagnosis_without_creating_output(
    tmp_path: Path, lesion_type
) -> None:
    record = _record()
    record.lesion_type = lesion_type
    output = tmp_path / "not-created" / "lesion.npz"
    with pytest.raises(InputValidationError, match="diagnosis"):
        write_lesion_npz(record, output)
    assert not output.parent.exists()


def test_selection_has_no_implicit_diagnosis() -> None:
    assert CandidateSelection("a", True, "Lesion").lesion_type == ""
    assert LESION_TYPES == ("METASTATIC", "BENIGN", "PRIMARY")


def test_omitted_lesions_do_not_require_a_diagnosis() -> None:
    first = _candidate("a", "Lesion A")
    second = _candidate("b", "Lesion B")
    selected = exporter._resolve_selections(
        (first, second),
        (
            CandidateSelection("a", True, "Renamed: A", "METASTATIC"),
            CandidateSelection("b", False, "Lesion B"),
        ),
    )
    assert selected == [(first, "Renamed- A", "METASTATIC")]


@pytest.mark.parametrize("lesion_type", ("", " ", "UNKNOWN", "benign", None))
def test_export_validates_all_diagnoses_before_image_io(
    monkeypatch, tmp_path: Path, lesion_type
) -> None:
    def unexpected_io(*args, **kwargs):
        pytest.fail("Invalid diagnoses must be rejected before image loading")

    monkeypatch.setattr(exporter, "require_sitk", unexpected_io)
    output = tmp_path / "not-created"
    with pytest.raises(InputValidationError, match="Lesion B"):
        exporter.export_lesions(
            series=(DicomSeriesSpec(tmp_path, "uid", (), "Series"),),
            segmentation=SegmentationInfo(
                tmp_path / "unused.nrrd", (3, 3, 3), (1.0, 1.0, 1.0), (), False
            ),
            candidates=(_candidate("a", "Lesion A"), _candidate("b", "Lesion B")),
            selections=(
                CandidateSelection("a", True, "Lesion A", "PRIMARY"),
                CandidateSelection("b", True, "Lesion B", lesion_type),
            ),
            output_folder=output,
        )
    assert not output.exists()


@pytest.mark.parametrize("channel_count", (1, 2))
def test_export_preserves_each_lesion_diagnosis(monkeypatch, tmp_path: Path, channel_count):
    """Exercise export, record construction and NPZ writing with image I/O mocked."""
    component_labels = np.zeros((3, 3, 3), dtype=np.uint16)
    for label, location in enumerate(((1, 1, 1), (1, 1, 2), (1, 2, 1)), start=1):
        component_labels[location] = label
    shape_stats = SimpleNamespace(
        Execute=lambda mask: None,
        GetCentroid=lambda label: (1.0, 1.0, 1.0),
        GetEquivalentEllipsoidDiameter=lambda label: (1.0, 1.0, 1.0),
        GetRoundness=lambda label: 1.0,
        GetPhysicalSize=lambda label: 1.0,
    )
    sitk = SimpleNamespace(
        sitkUInt8=np.uint8,
        BinaryThreshold=lambda image, lower, upper, inside, outside: np.where(
            (image >= lower) & (image <= upper), inside, outside
        ),
        Cast=lambda image, dtype: image.astype(dtype),
        GetArrayViewFromImage=np.asarray,
        GetArrayFromImage=np.asarray,
        LabelShapeStatisticsImageFilter=lambda: shape_stats,
    )
    image = SimpleNamespace(
        GetSpacing=lambda: (1.0, 1.0, 1.0),
        TransformPhysicalPointToIndex=lambda point: tuple(int(value) for value in point),
    )
    series = tuple(
        DicomSeriesSpec(tmp_path, f"uid-{index}", (), f"Series {index}")
        for index in range(channel_count)
    )
    loaded = tuple(
        exporter._LoadedSeries(
            spec, image, np.full((3, 3, 3), 40 + index, dtype=np.int16), {}
        )
        for index, spec in enumerate(series)
    )
    segmentation = SegmentationInfo(
        tmp_path / "unused.nrrd", (3, 3, 3), (1.0, 1.0, 1.0),
        (SegmentDefinition("Segment0", "Lesion", 1),), False,
    )
    candidates = tuple(
        LesionCandidate(
            str(index), "Segment0", "Lesion", index, index, 3,
            f"Lesion {index}", 1, 1.0, (1.0, 1.0, 1.0),
        )
        for index in range(1, 4)
    )
    selections = tuple(
        CandidateSelection(candidate.candidate_id, True, candidate.output_name, diagnosis)
        for candidate, diagnosis in zip(candidates, LESION_TYPES, strict=True)
    )
    monkeypatch.setattr(exporter, "require_sitk", lambda: sitk)
    monkeypatch.setattr(exporter, "_read_segmentation", lambda path: (component_labels, {}))
    monkeypatch.setattr(exporter, "parse_segment_definitions", lambda header: segmentation.segments)
    monkeypatch.setattr(exporter, "_component_images", lambda *args: {"Segment0": component_labels})
    monkeypatch.setattr(exporter, "_resample_labels", lambda mask, reference: mask)
    monkeypatch.setattr(exporter, "_load_aligned_series", lambda *args, **kwargs: loaded)
    summary = exporter.export_lesions(
        series=series, segmentation=segmentation, candidates=candidates,
        selections=selections, output_folder=tmp_path / "output",
    )
    assert len(summary.exported) == 3
    for exported, expected in zip(summary.exported, LESION_TYPES, strict=True):
        with np.load(exported.output_path, allow_pickle=False) as data:
            assert data["Patient.Lesion.LesionType"].item() == expected
            assert data["VOI"].ndim == (3 if channel_count == 1 else 4)


def test_cli_requires_diagnosis_before_input_io(monkeypatch, capsys) -> None:
    def unexpected_io(*args, **kwargs):
        pytest.fail("A missing diagnosis must fail before reading inputs")

    monkeypatch.setattr(cli, "inspect_segmentation", unexpected_io)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--segmentation", "unused.nrrd"])
    assert exc.value.code == 2
    assert "--lesion-type is required for export" in capsys.readouterr().err


def test_cli_rejects_unsupported_diagnosis(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--segmentation", "unused.nrrd", "--lesion-type", "UNKNOWN"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ("--list-labels", "--list-candidates"))
def test_cli_listing_does_not_require_diagnosis(monkeypatch, tmp_path: Path, flag) -> None:
    segmentation = SegmentationInfo(
        tmp_path / "unused.nrrd", (3, 3, 3), (1.0, 1.0, 1.0),
        (SegmentDefinition("Segment0", "Lesion", 1),), False,
    )
    monkeypatch.setattr(cli, "inspect_segmentation", lambda path: segmentation)
    monkeypatch.setattr(cli, "split_segments", lambda *args, **kwargs: (_candidate("a", "A"),))
    assert cli.main(["--segmentation", "unused.nrrd", flag]) == 0


@pytest.mark.parametrize("lesion_type", ("METASTATIC", "BENIGN", "PRIMARY"))
def test_cli_passes_diagnosis_to_export(monkeypatch, tmp_path: Path, lesion_type) -> None:
    segmentation = SegmentationInfo(
        tmp_path / "unused.nrrd", (3, 3, 3), (1.0, 1.0, 1.0),
        (SegmentDefinition("Segment0", "Lesion", 1),), False,
    )
    captured = {}

    def export(**kwargs):
        captured.update(kwargs)
        return ExportSummary((), ())

    monkeypatch.setattr(cli, "inspect_segmentation", lambda path: segmentation)
    monkeypatch.setattr(cli, "split_segments", lambda *args, **kwargs: (_candidate("a", "A"),))
    monkeypatch.setattr(cli, "discover_dicom_series", lambda folders: ())
    monkeypatch.setattr(cli, "export_lesions", export)
    assert cli.main([
        "--segmentation", "unused.nrrd", "--dicom", str(tmp_path),
        "--output", str(tmp_path / "output"), "--lesion-type", lesion_type,
    ]) == 0
    assert captured["selections"][0].lesion_type == lesion_type
