from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest

from lesion_pipeline.config import InsertionConfig, InsertionSpec
from lesion_pipeline.ctpd import decode_projection
from lesion_pipeline.lesion_models import LesionModel, save_lesion_model_npz
from lesion_pipeline.pipeline import run_insertion

from conftest import make_projection_file


def _header_json(ds):
    result = ds.to_json_dict()
    result.pop("7FE00010", None)
    return result, ds.file_meta.to_json_dict()


@pytest.mark.parametrize(
    ("workers", "implicit_vr", "stored_transposed"),
    [(1, False, False), (1, True, False), (2, False, False), (1, True, True)],
)
def test_end_to_end_changes_only_pixel_data(
    tmp_path, workers, implicit_vr, stored_transposed
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_path = make_projection_file(
        input_dir / "p0001.dcm",
        implicit_vr=implicit_vr,
        stored_transposed=stored_transposed,
    )
    model = LesionModel(
        lesion_number=1,
        voi_hu=np.asarray([[[[1000.0]]]], dtype=np.float32),
        mask=np.asarray([[[[True]]]]),
        row_spacing_mm=10.0,
        column_spacing_mm=10.0,
        slice_spacing_mm=10.0,
        reconstruction_diameter_mm=400.0,
        kvp=120.0,
        lesion_mean_hu=1000.0,
        lesion_mean_hu_by_channel=np.asarray([1000.0], dtype=np.float32),
        old_background_hu=np.asarray([0.0], dtype=np.float32),
        old_background_method="inverse_segmentation_within_lesion_bounding_box",
        old_background_voxel_count=8,
    )
    model_path = save_lesion_model_npz(model, tmp_path / "Lesion01.npz")
    config = InsertionConfig(
        ctpd_input=str(input_dir),
        output_dir=str(output_dir),
        insertions=[
            InsertionSpec(
                lesion_model=str(model_path),
                center_ctpd_mm=[0.0, 0.0, 0.0],
                background_hu=[0.0],
            )
        ],
        spectrum_channel_map={1: 0},
        workers=workers,
    )
    summary = run_insertion(config)
    output_path = output_dir / "p0001.dcm"
    before = pydicom.dcmread(source_path)
    after = pydicom.dcmread(output_path)
    assert _header_json(before) == _header_json(after)
    before_raw = pydicom.dcmread(source_path)
    after_raw = pydicom.dcmread(output_path)
    before_element = before_raw.get_item(0x7FE00010)
    after_element = after_raw.get_item(0x7FE00010)
    assert before_element.value_tell == after_element.value_tell
    offset = before_element.value_tell
    length = before_element.length
    before_bytes = source_path.read_bytes()
    after_bytes = output_path.read_bytes()
    assert before_bytes[:offset] == after_bytes[:offset]
    assert before_bytes[offset + length :] == after_bytes[offset + length :]
    values = decode_projection(after)
    assert np.isclose(values[1, 2], 1.192, atol=0.001)
    assert np.count_nonzero(values != 1.0) == 1
    assert summary["changed_projection_files"] == 1
    assert summary["changed_projection_frames"] == 1
    assert summary["projection_frame_count"] == 1
    assert summary["clipped_pixels"] == 0
    assert summary["lesions"][0]["old_background_hu"] == [0.0]
    assert summary["lesions"][0]["new_background_hu"] == [0.0]
    assert summary["lesions"][0]["original_contrast_hu_by_channel"] == [1000.0]
    assert summary["lesions"][0]["old_background_method"] == (
        "inverse_segmentation_within_lesion_bounding_box"
    )
    assert summary["lesions"][0]["old_background_voxel_count"] == 8
    assert (output_dir / "lesion_insertion_summary.json").is_file()


@pytest.mark.parametrize(
    ("geometry_storage", "frame_count"),
    [("top_level", 40), ("functional_groups", 3)],
)
def test_multiframe_end_to_end_preserves_container(
    tmp_path, geometry_storage, frame_count
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source_path = make_projection_file(
        input_dir / "multi.dcm",
        implicit_vr=True,
        stored_transposed=True,
        frame_count=frame_count,
        geometry_storage=geometry_storage,
    )
    model = LesionModel(
        lesion_number=1,
        voi_hu=np.asarray([[[[1000.0]]]], dtype=np.float32),
        mask=np.asarray([[[[True]]]]),
        row_spacing_mm=10.0,
        column_spacing_mm=10.0,
        slice_spacing_mm=10.0,
        reconstruction_diameter_mm=400.0,
        kvp=120.0,
        lesion_mean_hu=1000.0,
        lesion_mean_hu_by_channel=np.asarray([1000.0], dtype=np.float32),
        old_background_hu=np.asarray([0.0], dtype=np.float32),
        old_background_method="inverse_segmentation_within_lesion_bounding_box",
        old_background_voxel_count=8,
    )
    model_path = save_lesion_model_npz(model, tmp_path / "Lesion01.npz")
    config = InsertionConfig(
        ctpd_input=str(input_dir),
        output_dir=str(output_dir),
        insertions=[
            InsertionSpec(
                lesion_model=str(model_path),
                center_ctpd_mm=[0.0, 0.0, 0.0],
                background_hu=[0.0],
            )
        ],
        spectrum_channel_map={1: 0},
        workers=1,
    )
    summary = run_insertion(config)
    output_path = output_dir / "multi.dcm"
    before = pydicom.dcmread(source_path)
    after = pydicom.dcmread(output_path)

    assert _header_json(before) == _header_json(after)
    assert int(after.NumberOfFrames) == frame_count
    values = decode_projection(after)
    assert values.shape == (frame_count, 3, 5)
    # The central ray remains inside the lesion for every view; its path
    # length varies slightly as the gantry angle changes.
    assert np.all(values[:, 1, 2] > 1.19)
    assert np.count_nonzero(values != 1.0) == frame_count
    assert summary["projection_file_count"] == 1
    assert summary["projection_frame_count"] == frame_count
    assert summary["changed_projection_files"] == 1
    assert summary["changed_projection_frames"] == frame_count

    before_element = pydicom.dcmread(source_path).get_item(0x7FE00010)
    offset = before_element.value_tell
    length = before_element.length
    before_bytes = source_path.read_bytes()
    after_bytes = output_path.read_bytes()
    assert before_bytes[:offset] == after_bytes[:offset]
    assert before_bytes[offset + length :] == after_bytes[offset + length :]
