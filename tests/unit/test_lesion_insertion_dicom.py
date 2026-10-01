from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import FileDataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

from ct_pcd_gui.features.lesion_insertion.infrastructure.dicom_service import (  # noqa: E402
    PydicomLesionDicomService,
)


def _write_slice(path: Path, *, uid: str, instance: int, z: float) -> None:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = generate_uid()
    dataset.SOPInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = uid
    dataset.StudyInstanceUID = "1.2.3"
    dataset.FrameOfReferenceUID = "1.2.4"
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
    dataset.PixelData = np.full((2, 3), instance, dtype=np.int16).tobytes()
    dataset.save_as(path)


def test_series_discovery_and_physical_order_ignore_filenames(tmp_path: Path) -> None:
    uid = generate_uid()
    _write_slice(tmp_path / "z.dcm", uid=uid, instance=1, z=0.0)
    _write_slice(tmp_path / "a.dcm", uid=uid, instance=2, z=2.0)
    service = PydicomLesionDicomService()
    candidates = service.discover_reconstruction_series(tmp_path)
    assert len(candidates) == 1
    volume = service.load_reconstruction(tmp_path, uid)
    assert [Path(path).name for path in volume.source_paths] == ["z.dcm", "a.dcm"]
    assert volume.hu[:, 0, 0].tolist() == [1.0, 2.0]
