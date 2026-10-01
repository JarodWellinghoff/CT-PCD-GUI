from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ct_pcd_gui.features.lesion_insertion._vendor.lesion_pipeline import (
    lesion_models as lesion_models_module,
    load_lesion_model,
    pipeline as pipeline_module,
)
from ct_pcd_gui.features.lesion_insertion._vendor.lesion_pipeline.lesion_models import (
    INVERSE_MASK_BACKGROUND_METHOD,
)


def test_package_routes_all_model_loading_through_compatibility_loader() -> None:
    assert lesion_models_module.load_lesion_model is load_lesion_model
    assert pipeline_module.load_lesion_model is load_lesion_model


def test_loads_new_voi_lesion_mask_npz_format(tmp_path: Path) -> None:
    path = tmp_path / "new-lesion.npz"
    voi = np.asarray(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
        ],
        dtype=np.int16,
    )
    mask = np.zeros(voi.shape, dtype=bool)
    mask[0, 1, 1] = True
    mask[1, 1, 1] = True
    header = {
        "Rows": 512,
        "Columns": 512,
        "PixelSpacing": [0.7, 0.8],
        "SliceThickness": 2.5,
        "ReconstructionDiameter": 409.6,
        "KVP": 140.0,
    }
    np.savez_compressed(
        path,
        PatientName=np.asarray("synthetic-lesion"),
        LesionNumber=np.asarray(5),
        DicomHeaderJSON=np.asarray(json.dumps(header)),
        LesionMask=mask,
        VOI=voi,
        LesionMeanHU=np.asarray(60.0),
    )

    model = load_lesion_model(path)

    assert model.lesion_number == 5
    assert model.channel_count == 1
    assert model.voi_hu.shape == (2, 2, 2, 1)
    assert model.mask.shape == model.voi_hu.shape
    assert model.voi_hu.dtype == np.float32
    assert model.mask.dtype == np.bool_
    assert model.row_spacing_mm == 0.7
    assert model.column_spacing_mm == 0.8
    assert model.slice_spacing_mm == 2.5
    assert model.reconstruction_diameter_mm == 409.6
    assert model.kvp == 140.0
    assert model.lesion_mean_hu == 60.0
    assert np.allclose(model.lesion_mean_hu_by_channel, [60.0])
    assert np.allclose(model.old_background_hu, [40.0])
    assert model.old_background_method == INVERSE_MASK_BACKGROUND_METHOD
    assert model.old_background_voxel_count == 6
    assert model.source_path == str(path)


def test_existing_pipeline_npz_format_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "legacy-pipeline-lesion.npz"
    voi = np.asarray([[[[25.0]]]], dtype=np.float32)
    mask = np.ones_like(voi, dtype=np.uint8)
    np.savez_compressed(
        path,
        lesion_number=np.asarray(3),
        voi_hu=voi,
        mask=mask,
        row_spacing_mm=np.asarray(0.5),
        column_spacing_mm=np.asarray(0.6),
        slice_spacing_mm=np.asarray(1.5),
        reconstruction_diameter_mm=np.asarray(300.0),
        kvp=np.asarray(120.0),
        lesion_mean_hu=np.asarray(25.0),
        lesion_mean_hu_by_channel=np.asarray([25.0], dtype=np.float32),
        old_background_hu=np.asarray([5.0], dtype=np.float32),
        old_background_method=np.asarray(INVERSE_MASK_BACKGROUND_METHOD),
        old_background_voxel_count=np.asarray(10),
    )

    model = load_lesion_model(path)

    assert model.lesion_number == 3
    assert model.channel_count == 1
    assert np.allclose(model.old_background_hu, [5.0])
    assert model.old_background_voxel_count == 10
