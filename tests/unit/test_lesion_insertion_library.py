from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ct_pcd_gui.features.lesion_insertion.infrastructure import lesion_library
from ct_pcd_gui.features.lesion_insertion.infrastructure.lesion_library import (
    LocalLesionModelLibrary,
)


@dataclass
class FakeModel:
    lesion_number: int
    voi_hu: np.ndarray
    mask: np.ndarray
    row_spacing_mm: float
    column_spacing_mm: float
    slice_spacing_mm: float
    lesion_mean_hu_by_channel: np.ndarray
    old_background_hu: np.ndarray
    old_background_method: str
    old_background_voxel_count: int
    kvp: float

    @property
    def channel_count(self) -> int:
        return self.voi_hu.shape[3]


def test_library_marks_malformed_records_without_aborting_scan(
    monkeypatch, tmp_path: Path
) -> None:
    valid = tmp_path / "valid.npz"
    malformed = tmp_path / "malformed.mat"
    valid.write_bytes(b"valid")
    malformed.write_bytes(b"bad")
    model = FakeModel(
        lesion_number=7,
        voi_hu=np.ones((3, 4, 5, 2), dtype=np.float32),
        mask=np.ones((3, 4, 5, 2), dtype=bool),
        row_spacing_mm=1.0,
        column_spacing_mm=0.5,
        slice_spacing_mm=2.0,
        lesion_mean_hu_by_channel=np.asarray([90.0, 100.0]),
        old_background_hu=np.asarray([40.0, 45.0]),
        old_background_method="perilesional",
        old_background_voxel_count=42,
        kvp=120.0,
    )

    def load(path):
        if Path(path).name == "malformed.mat":
            raise ValueError("missing version 3.1.4 metadata")
        return model

    monkeypatch.setattr(lesion_library, "_loader", lambda: load)
    items = list(LocalLesionModelLibrary().scan(tmp_path))

    assert len(items) == 2
    by_name = {Path(item.path).name: item for item in items}
    assert by_name["valid.npz"].compatible is True
    assert by_name["valid.npz"].lesion_id == "Lesion07"
    assert by_name["valid.npz"].channel_count == 2
    assert by_name["malformed.mat"].compatible is False
    assert "version 3.1.4" in by_name["malformed.mat"].warning
