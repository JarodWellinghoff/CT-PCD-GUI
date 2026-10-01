from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from ct_pcd_gui.features.lesion_insertion.domain.models import (
    DicomVolume,
    LesionInstance,
    LesionSession,
    RawDatasetInfo,
    SeriesGeometry,
)
from ct_pcd_gui.features.lesion_insertion.infrastructure import legacy_adapter
from ct_pcd_gui.features.lesion_insertion.infrastructure.legacy_adapter import (
    InsertionCancelled,
    LegacyLesionInsertionAdapter,
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


@dataclass
class FakeSpec:
    lesion_model: str
    background_hu: list[float]
    center_pixel: list[float] | None = None
    center_ctpd_mm: list[float] | None = None
    label: str = ""


@dataclass
class FakeConfig:
    ctpd_input: str
    output_dir: str
    insertions: list[FakeSpec]
    t1_series_dir: str | None
    spectrum_channel_map: dict[int, int]
    patient_position: str
    workers: int
    overwrite: bool

    def to_dict(self):
        return {
            **asdict(self),
            "spectrum_channel_map": {
                str(key): value for key, value in self.spectrum_channel_map.items()
            },
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")


def _model() -> FakeModel:
    mask = np.zeros((3, 3, 3, 1), dtype=bool)
    mask[1, 1, 1, 0] = True
    voi = np.zeros(mask.shape, dtype=np.float32)
    voi[1, 1, 1, 0] = 100.0
    return FakeModel(
        voi_hu=voi,
        mask=mask,
        old_background_hu=np.asarray([0.0], dtype=np.float32),
        lesion_mean_hu_by_channel=np.asarray([100.0], dtype=np.float32),
        lesion_mean_hu=100.0,
    )


def _volume() -> DicomVolume:
    geometry = SeriesGeometry(
        series_instance_uid="1.2.3.4",
        study_instance_uid="1.2.3",
        frame_of_reference_uid="1.2.3.5",
        rows=8,
        columns=8,
        image_positions_lps_mm=tuple((0.0, 0.0, float(index)) for index in range(3)),
        column_axis_lps=(1.0, 0.0, 0.0),
        row_axis_lps=(0.0, 1.0, 0.0),
        row_spacing_mm=1.0,
        column_spacing_mm=1.0,
        patient_position="HFS",
        reconstruction_diameter_mm=8.0,
    )
    return DicomVolume(
        geometry=geometry,
        hu=np.zeros((3, 8, 8), dtype=np.float32),
        source_paths=(),
    )


def _raw(source: Path) -> RawDatasetInfo:
    return RawDatasetInfo(
        source=str(source),
        study_instance_uids=("1.2.3",),
        series_instance_uids=("9.8.7",),
        frame_of_reference_uids=("1.2.3.5",),
        patient_positions=("HFS",),
        spectrum_indices=(1,),
        source_indices=(1,),
        projection_file_count=1,
        projection_frame_count=2,
    )


def test_adapter_stages_real_algorithm_contract_and_relocates_manifest(
    monkeypatch, tmp_path: Path
) -> None:
    recon = tmp_path / "recon"
    raw = tmp_path / "raw"
    recon.mkdir()
    raw.mkdir()
    (raw / "input.dcm").write_bytes(b"unchanged")
    model_path = tmp_path / "library" / "model.npz"
    model_path.parent.mkdir()
    model_path.write_bytes(b"source-model")
    output = tmp_path / "output"
    lesion = LesionInstance.create(
        lesion_path=str(model_path),
        lesion_id="Lesion01",
        label="test",
        center_voxel_crs=(4.0, 4.0, 1.0),
        center_patient_lps_mm=(4.0, 4.0, 1.0),
        center_ctpd_mm=(0.0, 0.0, 1.0),
        background_hu=(0.0,),
    )
    session = LesionSession(
        reconstruction_source=str(recon),
        reconstruction_series_uid="1.2.3.4",
        raw_source=str(raw),
        lesion_library_source=str(model_path.parent),
        output_directory=str(output),
        lesions=(lesion,),
        spectrum_channel_map=((1, 0),),
    )

    def save_model(model, path):
        path = Path(path)
        np.savez_compressed(path, voi_hu=model.voi_hu, mask=model.mask)
        return path

    def run(config, progress):
        destination = Path(config.output_dir)
        destination.mkdir(parents=True)
        (destination / "projection.dcm").write_bytes(b"modified")
        summary = {
            "output": str(destination),
            "lesions": [config.insertions[0].lesion_model],
        }
        (destination / "lesion_insertion_summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
        progress(2, 2, "projection.dcm")
        return summary

    monkeypatch.setattr(
        legacy_adapter,
        "_legacy_api",
        lambda: (
            FakeConfig,
            FakeSpec,
            run,
            "3.1.4",
            lambda _path: _model(),
            save_model,
        ),
    )
    events: list[tuple[int, int, str]] = []
    result = LegacyLesionInsertionAdapter().run(
        session,
        _volume(),
        _raw(raw),
        cancel_event=threading.Event(),
        progress=lambda completed, total, text: events.append((completed, total, text)),
        log=lambda _message: None,
    )

    assert events == [(2, 2, "projection.dcm")]
    assert (output / "dicom_ctpd_modified" / "projection.dcm").read_bytes() == b"modified"
    assert (raw / "input.dcm").read_bytes() == b"unchanged"
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["algorithm"]["version"] == "3.1.4"
    assert manifest["reconstruction"]["status"] == "not-configured"
    summary_text = Path(result.summary_path).read_text(encoding="utf-8")
    assert ".partial-" not in summary_text
    assert str(output) in summary_text


def test_adapter_honors_preexisting_cancellation(tmp_path: Path) -> None:
    event = threading.Event()
    event.set()
    with pytest.raises(InsertionCancelled, match="cancelled"):
        LegacyLesionInsertionAdapter().run(
            LesionSession(output_directory=str(tmp_path / "output")),
            _volume(),
            _raw(tmp_path),
            cancel_event=event,
            progress=lambda *_args: None,
            log=lambda _message: None,
        )


def test_session_rejects_duplicate_instance_ids(tmp_path: Path) -> None:
    model = tmp_path / "model.npz"
    model.write_bytes(b"x")
    lesion = LesionInstance.create(
        lesion_path=str(model),
        lesion_id="id",
        label="one",
        center_voxel_crs=(0.0, 0.0, 0.0),
        center_patient_lps_mm=(0.0, 0.0, 0.0),
        center_ctpd_mm=(0.0, 0.0, 0.0),
        background_hu=(0.0,),
    )
    session = LesionSession(lesions=(lesion, lesion))
    with pytest.raises(Exception, match="duplicate"):
        session.validate_basic()
