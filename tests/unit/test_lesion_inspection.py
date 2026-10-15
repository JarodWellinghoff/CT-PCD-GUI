"""Synthetic fixtures: no patient images or identifying metadata are required."""
import itertools
import json

import numpy as np
import pytest
import scipy.io

from ct_pcd_gui.features.lesion_viewer.domain.inspection import (
    describe_members, extract_channels, mask_outline, to_zyx,
)
from ct_pcd_gui.features.lesion_viewer.domain.models import LesionLoadError
from ct_pcd_gui.features.lesion_viewer.infrastructure.inspection_loader import InspectableLesionFileLoader


@pytest.mark.parametrize("order", ["".join(x) for x in itertools.permutations("XYZ")])
def test_axis_permutations_preserve_values_and_memory(order):
    zyx = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5) + 0.123456789
    source = np.transpose(zyx, tuple("ZYX".index(x) for x in order))
    result = to_zyx(source, order)
    np.testing.assert_array_equal(result, zyx)
    assert np.shares_memory(source, result)
    assert result.dtype == source.dtype


@pytest.mark.parametrize("order", ["", "XXZ", "ABC", "XYZC", "XY"])
def test_bad_axis_order_rejected(order):
    with pytest.raises(ValueError):
        to_zyx(np.zeros((2, 3, 4)), order)


def test_outline_includes_edges_and_excludes_interior():
    mask = np.ones((5, 5), dtype=bool)
    before = mask.copy()
    expected = np.ones_like(mask)
    expected[1:-1, 1:-1] = False
    np.testing.assert_array_equal(mask_outline(mask), expected)
    np.testing.assert_array_equal(mask, before)


def test_empty_single_pixel_and_nonfinite_masks():
    assert not mask_outline(np.zeros((3, 3))).any()
    mask = np.zeros((3, 3)); mask[1, 1] = 2; mask[0, 0] = np.nan
    result = mask_outline(mask)
    assert result[1, 1] and result.sum() == 1
    with pytest.raises(ValueError):
        mask_outline(np.zeros((2, 2, 2)))


@pytest.mark.parametrize("mask_style", ["shared", "singleton_channel", "per_channel", "missing"])
def test_dual_channels_last_axis_and_mask_pairing(mask_style):
    # A spatial axis is shorter than the number of channels. It is NOT a channel axis.
    voi = np.arange(1 * 5 * 4 * 2, dtype=np.int16).reshape(1, 5, 4, 2) - 1000
    root = {"VOI": voi, "SeriesNames": np.array(["Threshold low", "Threshold high"])}
    mask = np.zeros(voi.shape[:3], dtype=bool); mask[:, 1:3, 1:3] = True
    if mask_style == "shared": root["LesionMask"] = mask
    elif mask_style == "singleton_channel": root["LesionMask"] = mask[..., None]
    elif mask_style == "per_channel": root["LesionMask"] = np.stack([mask, ~mask], axis=-1)
    warnings = []
    channels = extract_channels(root, warnings)
    assert not warnings and len(channels) == 2
    assert channels[1].name == "Threshold high"
    for i, channel in enumerate(channels):
        np.testing.assert_array_equal(channel.values, voi[..., i])
        assert np.shares_memory(channel.values, voi)
        if mask_style == "missing": assert channel.mask is None
        else:
            np.testing.assert_array_equal(channel.mask, ~mask if i == 1 and mask_style == "per_channel" else mask)


def test_mismatched_mask_and_metadata_are_not_silently_used():
    warnings = []
    channels = extract_channels({"VOI": np.ones((3, 4, 5, 2)),
                                 "LesionMask": np.ones((9, 8, 7)),
                                 "SeriesCount": 8, "SeriesNames": ["wrong"]}, warnings)
    assert len(channels) == 2 and all(x.mask is None for x in channels)
    assert len(warnings) == 3
    assert channels[0].name == "Channel 1"


@pytest.mark.parametrize("voi", [np.zeros((3, 4)), np.ones((3, 4, 5), dtype=bool),
                                  np.zeros((3, 4, 5), dtype=complex), np.empty((3, 0, 5))])
def test_invalid_voi_does_not_become_hu(voi):
    warnings = []
    assert not extract_channels({"VOI": voi}, warnings)
    assert warnings


def test_mask_is_never_mislabeled_as_hu():
    assert extract_channels({"LesionMask": np.ones((3, 4, 5))}, []) == []


def test_member_inspection_preserves_scalar_statistics_vectors_and_json():
    root = {"LesionMeanHU": np.array(-127.25), "LesionCenter": np.array([1, 2, 3]),
            "DicomHeaderJSON": np.array('{"PixelSpacing": [0.7, 0.8], "SeriesDescription": "Test"}'),
            "DicomHeadersJSON": np.array(['{"KVP": 100}', '{"KVP": 120}']),
            "nan": np.array(np.nan), "VOI": np.zeros((20, 30, 40), dtype=np.int16)}
    info = describe_members(root)
    assert info["LesionMeanHU"] == -127.25
    assert info["LesionCenter"]["values"] == [1, 2, 3]
    assert info["DicomHeaderJSON"]["PixelSpacing"] == [0.7, 0.8]
    assert info["DicomHeadersJSON"]["values"][1]["KVP"] == 120
    assert info["VOI"]["shape"] == [20, 30, 40]
    assert "values" not in info["VOI"]
    assert len(json.dumps(info, allow_nan=False)) < 2000


def test_metadata_bounds_and_cycles():
    loop = {}; loop["self"] = loop
    assert "cyclic" in describe_members(loop)["self"]
    info = describe_members({"large": np.zeros((101, 100, 100)), "text": "x" * 300000})
    assert info["large"]["statistics"] == "sampled"
    assert info["text"].endswith("<text truncated>")
    nested = {"x": {"y": {"z": 3}}}
    assert "limit" in json.dumps(describe_members(nested, max_nodes=2))


def write_npz(tmp_path, *, shape=(4, 5, 3, 2), **extra):
    voi = np.arange(np.prod(shape), dtype=np.int16).reshape(shape) - 900
    mask = np.zeros(shape, dtype=bool)
    mask[1:3, 1:4, 1:2, ...] = True
    fields = dict(VOI=voi, LesionMask=mask, org_row_rng=np.array([0, shape[0]]),
                  org_col_rng=np.array([0, shape[1]]), org_slice_rng=np.array([0, shape[2]]),
                  DicomHeaderJSON=np.array('{"PixelSpacing": [0.6, 0.8], "SliceThickness": 1.2}'))
    fields.update(extra)
    path = tmp_path / "synthetic.npz"
    np.savez_compressed(path, **fields)
    return path, fields


@pytest.mark.parametrize("shape", [(4, 5, 3), (4, 5, 3, 1), (4, 5, 3, 2), (1, 5, 3, 2), (4, 5, 1, 2)])
def test_npz_loader_geometry_channels_and_single_slice(tmp_path, shape):
    path, fields = write_npz(tmp_path, shape=shape)
    result = InspectableLesionFileLoader().load(path)
    channels = result.file_metadata["slice_channels"]
    count = shape[3] if len(shape) == 4 else 1
    assert len(channels) == count
    assert result.file_metadata["suggested_axis_order"] == "YXZ"
    assert result.detected_spacing_xyz == (0.8, 0.6, 1.2)
    assert all(c.array.shape == shape[:3] for c in result.candidates)
    assert len(result.candidates) == count * 2
    assert result.file_metadata["members"]["DicomHeaderJSON"]["SliceThickness"] == 1.2
    for i, channel in enumerate(channels):
        expected = fields["VOI"][..., i] if len(shape) == 4 else fields["VOI"]
        np.testing.assert_array_equal(channel.values, expected)


def test_loader_never_unpickles_npz_objects(tmp_path):
    path, _ = write_npz(tmp_path, unsafe=np.array([{"a": 1}], dtype=object))
    result = InspectableLesionFileLoader().load(path)
    assert any("unsafe" in text and "pickled" in text for text in result.warnings)
    assert "unsafe" not in result.file_metadata["members"]


def test_float64_original_values_survive_loading(tmp_path):
    values = np.full((3, 4, 5), -123.1234567890123, dtype=np.float64)
    path, _ = write_npz(tmp_path, shape=(3, 4, 5), VOI=values)
    result = InspectableLesionFileLoader().load(path)
    np.testing.assert_array_equal(result.file_metadata["slice_channels"][0].values, values)
    candidate = next(c for c in result.candidates if c.key == "VOI")
    assert candidate.array.dtype == np.float64
    np.testing.assert_array_equal(candidate.array, values)


def test_generic_legacy_npz_keeps_candidates_without_claiming_hu(tmp_path):
    path = tmp_path / "mask.npz"
    mask = np.zeros((3, 4, 5), dtype=bool); mask[1, 1, 1] = True
    np.savez(path, lesion=mask)
    result = InspectableLesionFileLoader().load(path)
    assert result.candidates[0].key == "lesion"
    assert result.file_metadata["slice_channels"] == []


def test_mat_metadata_and_hu_are_available(tmp_path):
    path = tmp_path / "lesion.mat"
    voi = np.arange(60, dtype=np.float64).reshape(3, 4, 5)
    scipy.io.savemat(path, {"VOI": voi, "LesionMask": voi > 30, "LesionMeanHU": 34.5})
    result = InspectableLesionFileLoader().load(path)
    assert result.file_metadata["mat_backend"] == "scipy"
    assert result.file_metadata["members"]["LesionMeanHU"] == 34.5
    np.testing.assert_array_equal(result.file_metadata["slice_channels"][0].values, voi)


def test_missing_corrupt_and_nonvolume_files_fail_explicitly(tmp_path):
    with pytest.raises(LesionLoadError):
        InspectableLesionFileLoader().load(tmp_path / "missing.npz")
    path = tmp_path / "bad.npz"; path.write_bytes(b"not an archive")
    with pytest.raises(LesionLoadError):
        InspectableLesionFileLoader().load(path)
    path = tmp_path / "metadata.npz"; np.savez(path, PatientName="synthetic")
    with pytest.raises(LesionLoadError):
        InspectableLesionFileLoader().load(path)


def test_hdf5_mat_preserves_spatial_order_and_normalizes_matlab_channel_axis(tmp_path):
    import h5py
    path = tmp_path / "matlab73.mat"
    raw = np.arange(2 * 3 * 4 * 5, dtype=np.float64).reshape(2, 3, 4, 5)
    with h5py.File(path, "w") as handle:
        for key, values in (("VOI", raw), ("LesionMask", raw > 30)):
            dataset = handle.create_dataset(key, data=values)
            dataset.attrs["MATLAB_class"] = np.bytes_("double")
        handle.create_dataset("LesionMeanHU", data=np.array([[24.5]]))
    result = InspectableLesionFileLoader().load(path)
    assert result.file_metadata["mat_backend"] == "h5py"
    channels = result.file_metadata["slice_channels"]
    assert len(channels) == 2
    np.testing.assert_array_equal(channels[1].values, raw[1])
    np.testing.assert_array_equal(channels[1].mask, raw[1] > 30)
    assert result.file_metadata["members"]["LesionMeanHU"] == 24.5
