from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import (  # noqa: E402
    CTImageStorage,
    ExplicitVRLittleEndian,
    generate_uid,
)

from ct_pcd_gui.features.lesion_insertion._vendor.lesion_pipeline import (  # noqa: E402
    ctpd,
)
from ct_pcd_gui.features.lesion_insertion.domain.models import (  # noqa: E402
    LesionSession,
    ValidationError,
)
from ct_pcd_gui.features.lesion_insertion.infrastructure.dicom_service import (  # noqa: E402
    PydicomLesionDicomService,
)


def _write_slice(
    path: Path,
    *,
    uid: str | None,
    instance: int,
    z: float,
) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(
        str(path), {}, file_meta=meta, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = generate_uid()
    if uid:
        dataset.SeriesInstanceUID = uid
    dataset.StudyInstanceUID = "1.2.3"
    dataset.FrameOfReferenceUID = "1.2.4"
    dataset.SeriesNumber = 1
    dataset.AcquisitionNumber = 1
    dataset.SeriesDescription = "Synthetic CT"
    dataset.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    dataset.ConvolutionKernel = "STANDARD"
    dataset.Modality = "CT"
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.PixelSpacing = [2.0, 0.5]
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z]
    dataset.InstanceNumber = instance
    dataset.PatientPosition = "HFS"
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    dataset.RescaleSlope = 1
    dataset.RescaleIntercept = 0
    dataset.PixelData = np.full(
        (2, 3), instance, dtype=np.int16
    ).tobytes()
    dataset.save_as(path)


def test_missing_uid_series_uses_deterministic_surrogate_and_loads(
    tmp_path: Path,
) -> None:
    _write_slice(tmp_path / "z.dcm", uid=None, instance=1, z=0.0)
    _write_slice(tmp_path / "a.dcm", uid=None, instance=2, z=2.0)
    service = PydicomLesionDicomService()

    first_scan = service.discover_reconstruction_series(tmp_path)
    second_scan = service.discover_reconstruction_series(tmp_path)

    assert len(first_scan) == 1
    candidate = first_scan[0]
    assert candidate.series_instance_uid == ""
    assert candidate.uses_surrogate_key is True
    assert candidate.effective_selection_key.startswith(
        "missing-series-uid:"
    )
    assert (
        candidate.effective_selection_key
        == second_scan[0].effective_selection_key
    )
    assert candidate.missing_uid_instance_count == 2
    assert any(
        "SeriesInstanceUID" in warning
        for warning in candidate.warnings
    )
    assert {Path(path).name for path in candidate.source_paths} == {
        "z.dcm",
        "a.dcm",
    }

    volume = service.load_reconstruction(
        tmp_path, candidate.effective_selection_key
    )
    assert [Path(path).name for path in volume.source_paths] == [
        "z.dcm",
        "a.dcm",
    ]
    assert volume.hu[:, 0, 0].tolist() == [1.0, 2.0]
    assert volume.geometry.series_instance_uid == ""
    assert (
        volume.geometry.effective_series_key
        == candidate.effective_selection_key
    )
    assert (
        volume.geometry.series_grouping_method
        == "metadata_fingerprint_v1"
    )
    assert volume.geometry.series_missing_uid_instance_count == 2
    assert volume.geometry.source_was_nonconformant is True


def test_missing_uid_groups_are_separated_by_relative_parent(
    tmp_path: Path,
) -> None:
    for directory_name in ("first", "second"):
        directory = tmp_path / directory_name
        directory.mkdir()
        _write_slice(
            directory / "one.dcm", uid=None, instance=1, z=0.0
        )
        _write_slice(
            directory / "two.dcm", uid=None, instance=2, z=2.0
        )

    candidates = (
        PydicomLesionDicomService()
        .discover_reconstruction_series(tmp_path)
    )

    assert len(candidates) == 2
    assert all(candidate.uses_surrogate_key for candidate in candidates)
    assert len(
        {
            candidate.effective_selection_key
            for candidate in candidates
        }
    ) == 2


def test_missing_instances_attach_only_to_one_compatible_uid_series(
    tmp_path: Path,
) -> None:
    uid = generate_uid()
    _write_slice(tmp_path / "one.dcm", uid=uid, instance=1, z=0.0)
    _write_slice(tmp_path / "two.dcm", uid=None, instance=2, z=2.0)
    _write_slice(tmp_path / "three.dcm", uid=uid, instance=3, z=4.0)
    service = PydicomLesionDicomService()

    candidates = service.discover_reconstruction_series(tmp_path)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.series_instance_uid == uid
    assert candidate.effective_selection_key == uid
    assert candidate.uses_surrogate_key is False
    assert candidate.missing_uid_instance_count == 1
    volume = service.load_reconstruction(tmp_path, uid)
    assert volume.hu[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert volume.geometry.series_instance_uid == uid
    assert volume.geometry.series_missing_uid_instance_count == 1
    assert volume.geometry.source_was_nonconformant is True


def test_ambiguous_missing_instances_remain_a_separate_candidate(
    tmp_path: Path,
) -> None:
    _write_slice(
        tmp_path / "first.dcm",
        uid=generate_uid(),
        instance=1,
        z=0.0,
    )
    _write_slice(
        tmp_path / "second.dcm",
        uid=generate_uid(),
        instance=2,
        z=2.0,
    )
    _write_slice(
        tmp_path / "missing.dcm", uid=None, instance=3, z=4.0
    )

    candidates = (
        PydicomLesionDicomService()
        .discover_reconstruction_series(tmp_path)
    )

    assert len(candidates) == 3
    surrogate = [
        candidate
        for candidate in candidates
        if candidate.uses_surrogate_key
    ]
    assert len(surrogate) == 1
    assert surrogate[0].missing_uid_instance_count == 1


def test_session_migrates_uid_field_to_selection_key() -> None:
    old_session = LesionSession.from_dict(
        {"reconstruction_series_uid": "1.2.3.4"}
    )
    assert old_session.reconstruction_series_key == "1.2.3.4"

    new_session = LesionSession(
        reconstruction_series_uid="",
        reconstruction_series_selection_key=(
            "missing-series-uid:abc"
        ),
    )
    restored = LesionSession.from_dict(new_session.to_dict())
    assert restored.reconstruction_series_uid == ""
    assert (
        restored.reconstruction_series_key
        == "missing-series-uid:abc"
    )


def test_raw_ctpd_without_series_uid_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(
        ctpd,
        "discover_projection_records",
        lambda _source: [SimpleNamespace(series_instance_uid="")],
    )

    with pytest.raises(
        ValidationError, match="repair the source metadata"
    ):
        PydicomLesionDicomService().inspect_raw("unused")
