from __future__ import annotations

import numpy as np
import pydicom
import pytest

from lesion_pipeline.ctpd import (
    decode_projection,
    discover_projection_records,
    encode_projection,
    inspect_projection_series,
    projection_frame_count,
    read_geometry,
    source_and_detector_positions,
)

from conftest import make_projection_file


def test_geometry_center_ray(tmp_path):
    path = make_projection_file(tmp_path / "projection.dcm")
    geometry = read_geometry(pydicom.dcmread(path, stop_before_pixels=True))
    source, detector = source_and_detector_positions(geometry)
    assert np.allclose(source, [0.0, 600.0, 0.0], atol=1e-6)
    assert np.allclose(detector[1, 2], [0.0, -400.0, 0.0], atol=1e-6)
    assert detector[1, 3, 0] > 0.0
    assert detector[0, 2, 2] > detector[2, 2, 2]


def test_private_detector_dimensions_control_transposed_pixel_storage(tmp_path):
    path = make_projection_file(
        tmp_path / "transposed.dcm",
        rows=3,
        columns=5,
        stored_transposed=True,
    )
    ds = pydicom.dcmread(path)
    geometry = read_geometry(ds)

    assert (geometry.rows, geometry.columns) == (3, 5)
    assert (geometry.stored_rows, geometry.stored_columns) == (5, 3)
    assert geometry.pixel_data_transposed is True

    physical = decode_projection(ds, geometry)
    assert physical.shape == (3, 5)
    physical[1, 2] += 0.2
    encoded, clipped = encode_projection(ds, physical, geometry)
    assert clipped == 0

    ds.PixelData = encoded
    # Detector [row 1, column 2] is stored at DICOM pixel [row 2, column 1].
    assert np.isclose(ds.pixel_array[2, 1] * 0.001, 1.2)
    assert np.isclose(decode_projection(ds, geometry)[1, 2], 1.2)


def test_implicit_vr_private_tags_are_decoded(tmp_path):
    path = make_projection_file(tmp_path / "implicit.dcm", implicit_vr=True)
    geometry = read_geometry(pydicom.dcmread(path, stop_before_pixels=True))
    assert geometry.rows == 3
    assert geometry.columns == 5
    assert geometry.detector_shape == "CYLINDRICAL"
    assert geometry.water_attenuation_mm_inverse == 0.01917


def test_projection_scaling_round_trip(tmp_path):
    path = make_projection_file(tmp_path / "projection.dcm")
    ds = pydicom.dcmread(path)
    physical = decode_projection(ds)
    assert np.allclose(physical, 1.0)
    physical[1, 2] += 0.2
    encoded, clipped = encode_projection(ds, physical)
    assert clipped == 0
    ds.PixelData = encoded
    decoded = decode_projection(ds)
    assert np.isclose(decoded[1, 2], 1.2)
    assert np.count_nonzero(decoded != 1.0) == 1


@pytest.mark.parametrize(
    "geometry_storage",
    ["top_level", "functional_groups", "alternate_sequence"],
)
def test_multiframe_geometry_and_pixel_round_trip(tmp_path, geometry_storage):
    path = make_projection_file(
        tmp_path / f"multiframe_{geometry_storage}.dcm",
        implicit_vr=geometry_storage != "alternate_sequence",
        stored_transposed=True,
        frame_count=3,
        geometry_storage=geometry_storage,
    )
    ds = pydicom.dcmread(path)

    assert projection_frame_count(ds) == 3
    assert np.isclose(read_geometry(ds, 0).focal_center_phi_rad, 0.0)
    assert np.isclose(read_geometry(ds, 1).focal_center_phi_rad, 0.1)
    assert np.isclose(read_geometry(ds, 2).focal_center_phi_rad, 0.2)

    physical = decode_projection(ds)
    assert physical.shape == (3, 3, 5)
    physical[:, 1, 2] += [0.1, 0.2, 0.3]
    encoded, clipped = encode_projection(ds, physical)
    assert clipped == 0
    ds.PixelData = encoded
    decoded = decode_projection(ds)
    assert np.allclose(decoded[:, 1, 2], [1.1, 1.2, 1.3])

    records = discover_projection_records(path)
    assert len(records) == 1
    assert records[0].frame_count == 3
    summary = inspect_projection_series(path)
    assert summary["projection_files"] == 1
    assert summary["projection_frames"] == 3
    assert summary["groups"][0]["frame_count"] == 3


def test_multiframe_missing_first_private_geometry_is_recovered(tmp_path):
    path = make_projection_file(
        tmp_path / "multiframe_missing_first.dcm",
        stored_transposed=True,
        frame_count=3,
        geometry_storage="functional_groups_missing_first",
    )
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    with pytest.warns(RuntimeWarning) as recovered_warnings:
        geometry = read_geometry(ds, 0)

    assert any(
        "frame 1" in str(item.message)
        and "linear extrapolation" in str(item.message)
        for item in recovered_warnings
    )

    assert np.isclose(geometry.focal_center_phi_rad, 0.0)
    assert np.isclose(geometry.focal_center_z_mm, 0.0)
    assert np.isclose(geometry.focal_center_rho_mm, 600.0)
    assert geometry.spectrum_index == 1
    assert geometry.source_index == 1

    # Regression for discovery rejecting an otherwise valid multi-frame file
    # solely because its first per-frame item is the generator's placeholder.
    with pytest.warns(RuntimeWarning):
        records = discover_projection_records(path)
    assert len(records) == 1
    assert records[0].frame_count == 3
