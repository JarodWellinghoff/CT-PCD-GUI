from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from ct_pcd_gui.features.lesion_insertion.domain.models import (
    DicomVolume,
    LesionInstance,
    LesionParameters,
    LesionSession,
    SeriesGeometry,
)
from ct_pcd_gui.features.lesion_insertion.infrastructure.model_transform import transform_model
from ct_pcd_gui.features.lesion_insertion.infrastructure.preview import (
    ApproximatePreviewGenerator,
)


@dataclass
class FakeModel:
    voi_hu: np.ndarray
    mask: np.ndarray
    old_background_hu: np.ndarray
    lesion_mean_hu_by_channel: np.ndarray
    lesion_mean_hu: float
    row_spacing_mm: float = 1.0
    column_spacing_mm: float = 1.0
    slice_spacing_mm: float = 1.0


def _model() -> FakeModel:
    mask = np.zeros((3, 3, 3, 1), dtype=bool)
    mask[1, 1, 1, 0] = True
    voi = np.full(mask.shape, 10.0, dtype=np.float32)
    voi[1, 1, 1, 0] = 30.0
    return FakeModel(
        voi_hu=voi,
        mask=mask,
        old_background_hu=np.asarray([10.0], dtype=np.float32),
        lesion_mean_hu_by_channel=np.asarray([30.0], dtype=np.float32),
        lesion_mean_hu=30.0,
    )


def _volume() -> DicomVolume:
    geometry = SeriesGeometry(
        series_instance_uid="series",
        study_instance_uid="study",
        frame_of_reference_uid="frame",
        rows=9,
        columns=9,
        image_positions_lps_mm=tuple((0.0, 0.0, float(index)) for index in range(5)),
        column_axis_lps=(1.0, 0.0, 0.0),
        row_axis_lps=(0.0, 1.0, 0.0),
        row_spacing_mm=1.0,
        column_spacing_mm=1.0,
        patient_position="HFS",
        reconstruction_diameter_mm=9.0,
    )
    return DicomVolume(
        geometry=geometry,
        hu=np.zeros((5, 9, 9), dtype=np.float32),
        source_paths=(),
    )


def test_transform_applies_contrast_and_scale_without_mutating_source() -> None:
    source = _model()
    transformed = transform_model(
        source,
        LesionParameters(contrast_scale=2.0, scale_xyz=(2.0, 1.0, 1.0)),
    )
    assert source.voi_hu.shape == (3, 3, 3, 1)
    assert transformed.voi_hu.shape[1] == 6
    values = transformed.voi_hu[..., 0][transformed.mask[..., 0]]
    assert np.max(values) == pytest.approx(50.0)


def test_preview_is_cached_and_only_changes_approximate_result(tmp_path: Path) -> None:
    model_path = tmp_path / "model.npz"
    model_path.write_bytes(b"x")
    model = _model()
    calls = 0

    def loader(_path: str) -> FakeModel:
        nonlocal calls
        calls += 1
        return replace(model)

    lesion = LesionInstance.create(
        lesion_path=str(model_path),
        lesion_id="Lesion01",
        label="test",
        center_voxel_crs=(4.0, 4.0, 2.0),
        center_patient_lps_mm=(4.0, 4.0, 2.0),
        center_ctpd_mm=(0.0, 0.0, 2.0),
        background_hu=(0.0,),
    )
    generator = ApproximatePreviewGenerator(loader)
    volume = _volume()
    session = LesionSession(lesions=(lesion,))

    first = generator.generate(volume, session, 2, 1)
    second = generator.generate(volume, session, 2, 2)

    assert calls == 1
    assert first.cache_hits == 0
    assert second.cache_hits == 1
    assert np.max(first.approximate_after_hu) == pytest.approx(20.0)
    assert np.array_equal(volume.hu, np.zeros_like(volume.hu))


class MutableModel:
    pass


def test_transform_does_not_mutate_non_dataclass_model() -> None:
    source = MutableModel()
    template = _model()
    source.voi_hu = template.voi_hu.copy()
    source.mask = template.mask.copy()
    source.old_background_hu = template.old_background_hu.copy()
    source.lesion_mean_hu_by_channel = template.lesion_mean_hu_by_channel.copy()
    source.lesion_mean_hu = template.lesion_mean_hu
    source.row_spacing_mm = 1.0
    source.column_spacing_mm = 1.0
    source.slice_spacing_mm = 1.0

    transformed = transform_model(source, LesionParameters(contrast_scale=2.0))

    assert transformed is not source
    assert np.max(source.voi_hu) == pytest.approx(30.0)
    assert np.max(transformed.voi_hu) == pytest.approx(50.0)
