from pathlib import Path
import threading

import numpy as np
import pytest

from ct_pcd_gui.features.lesion_viewer.application.use_cases import ScanLesionLibrary
from ct_pcd_gui.features.lesion_viewer.domain.surface_models import SurfaceBuildRequest
from ct_pcd_gui.features.lesion_viewer.infrastructure.local_source import LocalDirectoryLesionSource
from ct_pcd_gui.features.lesion_viewer.infrastructure.mesh_exporter import VtkMeshExporter
from ct_pcd_gui.features.lesion_viewer.infrastructure.vtk_surface_builder import VtkSurfaceBuilder


def request(volume: np.ndarray, **overrides) -> SurfaceBuildRequest:
    values = dict(
        volume=volume,
        source_min=float(np.min(volume)),
        source_max=float(np.max(volume)),
        axis_order="ZYX",
        spacing_xyz=(0.5, 1.0, 2.0),
        threshold=0.5,
        foreground_below=False,
        downsample=1,
        smoothing_iterations=0,
        reduction_percent=0,
        largest_component_only=True,
        pad_border=True,
    )
    values.update(overrides)
    return SurfaceBuildRequest(**values)


def test_builds_surface_and_physical_metrics(tmp_path: Path) -> None:
    volume = np.zeros((12, 10, 8), dtype=np.uint8)
    volume[3:9, 2:8, 2:6] = 1

    result = VtkSurfaceBuilder().build(request(volume))

    assert result.polydata.GetNumberOfPoints() > 0
    assert result.polydata.GetNumberOfCells() > 0
    assert result.metrics.foreground_count == 6 * 6 * 4
    assert result.metrics.mask_volume_mm3 == pytest.approx(144.0)
    assert result.metrics.surface_area_mm2 > 0.0
    assert result.metrics.mesh_volume_mm3 > 0.0

    output = VtkMeshExporter().export(result.polydata, tmp_path / "lesion.stl")
    assert output.is_file()
    assert output.stat().st_size > 84


def test_below_threshold_surface() -> None:
    volume = np.ones((10, 10, 10), dtype=np.float32)
    volume[2:8, 3:7, 4:6] = 0.0

    result = VtkSurfaceBuilder().build(
        request(volume, foreground_below=True)
    )

    assert result.metrics.foreground_count == 6 * 4 * 2
    assert result.metrics.foreground_below is True
    assert result.polydata.GetNumberOfCells() > 0


def test_axis_order_is_respected() -> None:
    volume = np.zeros((8, 6, 10), dtype=np.uint8)
    volume[2:6, 1:5, 3:8] = 1
    result = VtkSurfaceBuilder().build(
        request(volume, axis_order="YXZ", spacing_xyz=(2.0, 1.0, 0.5))
    )
    bounds = result.polydata.GetBounds()
    assert bounds[1] - bounds[0] > bounds[3] - bounds[2]


def test_rejects_threshold_outside_source_range() -> None:
    volume = np.zeros((4, 4, 4), dtype=np.uint8)
    volume[1:3, 1:3, 1:3] = 1
    with pytest.raises(ValueError, match="inside"):
        VtkSurfaceBuilder().build(request(volume, threshold=2.0))


def test_scan_use_case_batches_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "z.npz").write_bytes(b"z")
    nested = tmp_path / "a"
    nested.mkdir()
    (nested / "b.mat").write_bytes(b"b")
    (nested / "ignore.txt").write_text("x")

    batches: list[list[str]] = []
    records = ScanLesionLibrary().execute(
        LocalDirectoryLesionSource(tmp_path),
        cancel_event=threading.Event(),
        on_batch=lambda batch: batches.append([item.relative_path for item in batch]),
        batch_size=1,
    )

    assert [item.relative_path for item in records] == ["a/b.mat", "z.npz"]
    assert len(batches) == 2


def test_scan_use_case_validates_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        ScanLesionLibrary().execute(
            LocalDirectoryLesionSource(tmp_path),
            cancel_event=threading.Event(),
            on_batch=lambda _batch: None,
            batch_size=0,
        )
