from __future__ import annotations

import nrrd
import numpy as np
import pytest

from lesion_pipeline.lesion_models import generate_lesion_models, load_lesion_model

from conftest import make_recon_series


def _write_segmentation(path, segmentation):
    nrrd.write(
        str(path),
        segmentation,
        header={
            "space": "left-posterior-superior",
            "space directions": np.asarray(
                [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 4.0]]
            ),
            "space origin": np.asarray([0.0, 0.0, 0.0]),
            "Segment0_Name": "Lesion",
            "Segment0_LabelValue": "1",
        },
        index_order="F",
    )


def test_physical_nrrd_alignment_and_model_generation(tmp_path):
    t1_dir = make_recon_series(tmp_path / "t1", 100)
    t2_dir = make_recon_series(tmp_path / "t2", 50)
    segmentation = np.zeros((5, 4, 3), dtype=np.uint8)
    # An L-shaped connected lesion leaves one unsegmented voxel in its tight
    # 2x2x1 bounding box. That voxel is the original-background sample.
    segmentation[2, 1, 1] = 1
    segmentation[3, 1, 1] = 1
    segmentation[2, 2, 1] = 1
    nrrd_path = tmp_path / "lesion.seg.nrrd"
    _write_segmentation(nrrd_path, segmentation)
    paths, _, _ = generate_lesion_models(
        t1_dir,
        t2_dir,
        nrrd_path,
        tmp_path / "models",
        mask_alignment="physical",
        write_mat=True,
    )
    assert len(paths) == 1
    model = load_lesion_model(paths[0])
    assert model.voi_hu.shape == (2, 2, 1, 2)
    assert np.allclose(
        model.lesion_mean_hu_by_channel,
        [113.6666667, 63.6666667],
    )
    assert np.allclose(model.old_background_hu, [115.0, 65.0])
    assert model.old_background_method == (
        "inverse_segmentation_within_lesion_bounding_box"
    )
    assert model.old_background_voxel_count == 1
    legacy = load_lesion_model(tmp_path / "models" / "Lesion01.mat")
    assert np.array_equal(legacy.mask, model.mask)
    assert np.allclose(legacy.voi_hu, model.voi_hu)
    assert np.allclose(legacy.lesion_mean_hu_by_channel, model.lesion_mean_hu_by_channel)
    assert np.allclose(legacy.old_background_hu, model.old_background_hu)
    assert legacy.old_background_method == model.old_background_method
    assert legacy.old_background_voxel_count == model.old_background_voxel_count


def test_inverse_mask_background_rejects_fully_filled_bounding_box(tmp_path):
    t1_dir = make_recon_series(tmp_path / "t1", 100)
    t2_dir = make_recon_series(tmp_path / "t2", 50)
    segmentation = np.zeros((5, 4, 3), dtype=np.uint8)
    segmentation[2, 1, 1] = 1
    nrrd_path = tmp_path / "solid.seg.nrrd"
    _write_segmentation(nrrd_path, segmentation)

    with pytest.raises(ValueError, match="no unsegmented background voxels"):
        generate_lesion_models(
            t1_dir,
            t2_dir,
            nrrd_path,
            tmp_path / "models",
            mask_alignment="physical",
            write_mat=False,
        )


def test_shell_based_npz_model_requires_regeneration(tmp_path):
    path = tmp_path / "Lesion01.npz"
    np.savez_compressed(
        path,
        lesion_number=np.asarray(1),
        voi_hu=np.ones((1, 1, 1, 1), dtype=np.float32),
        mask=np.ones((1, 1, 1, 1), dtype=np.uint8),
        row_spacing_mm=np.asarray(1.0),
        column_spacing_mm=np.asarray(1.0),
        slice_spacing_mm=np.asarray(1.0),
        reconstruction_diameter_mm=np.asarray(400.0),
        kvp=np.asarray(120.0),
        lesion_mean_hu=np.asarray(1.0),
        lesion_mean_hu_by_channel=np.asarray([1.0]),
        old_background_hu=np.asarray([0.0]),
        background_shell_width_mm=np.asarray(5.0),
    )

    with pytest.raises(ValueError, match="Regenerate it with pipeline version 3.1.4"):
        load_lesion_model(path)
