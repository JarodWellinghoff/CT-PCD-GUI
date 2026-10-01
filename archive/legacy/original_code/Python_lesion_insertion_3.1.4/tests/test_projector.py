from __future__ import annotations

import numpy as np

from lesion_pipeline.lesion_models import LesionModel
from lesion_pipeline.projector import (
    VoxelVolume,
    prepare_difference_volume,
    siddon_integral,
)


def test_siddon_integrates_constant_cube():
    volume = VoxelVolume(
        data=np.full((2, 2, 2), 2.0, dtype=np.float32),
        edge_origin_mm=np.asarray([-1.0, -1.0, -1.0]),
        spacing_mm=np.asarray([1.0, 1.0, 1.0]),
        center_mm=np.zeros(3),
        radius_mm=np.sqrt(3.0),
    )
    result = siddon_integral(
        np.asarray([-2.0, 0.0, 0.0]),
        np.asarray([2.0, 0.0, 0.0]),
        volume,
    )
    assert np.isclose(result, 4.0)


def test_siddon_returns_zero_for_missed_ray():
    volume = VoxelVolume(
        data=np.ones((2, 2, 2), dtype=np.float32),
        edge_origin_mm=np.asarray([-1.0, -1.0, -1.0]),
        spacing_mm=np.ones(3),
        center_mm=np.zeros(3),
        radius_mm=np.sqrt(3.0),
    )
    assert siddon_integral(
        np.asarray([-2.0, 3.0, 0.0]),
        np.asarray([2.0, 3.0, 0.0]),
        volume,
    ) == 0.0


def test_difference_volume_uses_old_background_and_matlab_water_coefficient():
    model = LesionModel(
        lesion_number=1,
        voi_hu=np.asarray([[[[80.0]]]], dtype=np.float32),
        mask=np.asarray([[[[True]]]]),
        row_spacing_mm=1.0,
        column_spacing_mm=1.0,
        slice_spacing_mm=1.0,
        reconstruction_diameter_mm=400.0,
        kvp=120.0,
        lesion_mean_hu=80.0,
        lesion_mean_hu_by_channel=np.asarray([80.0], dtype=np.float32),
        old_background_hu=np.asarray([120.0], dtype=np.float32),
        old_background_method="inverse_segmentation_within_lesion_bounding_box",
        old_background_voxel_count=20,
    )

    volume = prepare_difference_volume(
        model,
        channel=0,
        center_ctpd_mm=np.zeros(3),
        patient_position="HFS",
    )

    expected = 0.01917 * (80.0 - 120.0) / 1000.0
    assert np.isclose(volume.data.item(), expected)
