"""Synthetic CTPD staging tests; require the application's pydicom dependency."""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.sequence import Sequence  # noqa: E402
from pydicom.uid import (  # noqa: E402
    CTImageStorage, ExplicitVRLittleEndian, ImplicitVRLittleEndian, generate_uid,
)

from ct_pcd_gui.features.freect_reconstruction.dicom import (  # noqa: E402
    ANGLE, INSTANCE, NUMBER_OF_SPECTRA, SPECTRUM, Z_POSITION,
    index_frames, stage_projections,
)
from ct_pcd_gui.features.freect_reconstruction.models import (  # noqa: E402
    JobContext, ReconstructionCancelled,
)


@pytest.fixture
def context():
    return JobContext(threading.Event(), lambda event: None)


@pytest.fixture
def identity():
    return generate_uid(), generate_uid()


def dataset(identity, frames=1, order=1):
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPClassUID = ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.StudyInstanceUID, ds.SeriesInstanceUID = identity
    ds.InstanceNumber = order
    ds.Rows, ds.Columns = 16, 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope, ds.RescaleIntercept = 2.0, -1.0
    ds.add_new((0x7029, 0x0010), "LO", "DetectorSystemArrangementModule")
    ds.add_new((0x7031, 0x0010), "LO", "DetectorSystemGeometryModule")
    ds.add_new((0x7033, 0x0010), "LO", "SourceGeometryModule")
    ds.add_new((0x7029, 0x1010), "US", ds.Rows)
    ds.add_new((0x7029, 0x1011), "US", ds.Columns)
    ds.add_new(ANGLE, "FL", order * 0.1)
    ds.add_new(Z_POSITION, "FL", order * 0.01)
    ds.add_new(SPECTRUM, "US", 1)
    if frames > 1:
        ds.NumberOfFrames = frames
        ds.PerFrameFunctionalGroupsSequence = Sequence()
        for index in range(frames):
            item = Dataset()
            item.add_new(ANGLE, "FL", index * 0.1)
            item.add_new(Z_POSITION, "FL", index * 0.01)
            item.InstanceNumber = order + index
            ds.PerFrameFunctionalGroupsSequence.append(item)
    pixels = np.arange(frames * ds.Rows * ds.Columns, dtype="<u2")
    ds.PixelData = pixels.tobytes()
    return ds


def save(ds, path: Path, *, implicit=False):
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian if implicit else ExplicitVRLittleEndian
    ds.is_little_endian, ds.is_implicit_VR = True, implicit
    pydicom.dcmwrite(str(path), ds, write_like_original=False)
    return path


@pytest.mark.parametrize("implicit", [False, True])
def test_multiframe_preserves_pixels_and_per_frame_metadata(tmp_path, context, identity, implicit):
    ds = dataset(identity, frames=12)
    for i, item in enumerate(ds.PerFrameFunctionalGroupsSequence):
        # Nested standard functional groups must override root rescale values.
        transform = Dataset()
        transform.RescaleSlope, transform.RescaleIntercept = i + 1.0, -i
        item.PixelValueTransformationSequence = Sequence([transform])
    source = save(ds, tmp_path / "input.dcm", implicit=implicit)
    original_hash = hashlib.sha256(source.read_bytes()).digest()
    original_pixels = pydicom.dcmread(source).pixel_array.copy()
    out = tmp_path / "projections"
    assert stage_projections(source, out, context) == 12
    staged = sorted(out.glob("*.dcm"))
    assert len(staged) == 12
    uids = set()
    for i, path in enumerate(staged):
        frame = pydicom.dcmread(path)
        np.testing.assert_array_equal(frame.pixel_array, original_pixels[i])
        assert float(frame.RescaleSlope) == i + 1
        assert float(frame.RescaleIntercept) == -i
        assert frame.InstanceNumber == i + 1
        assert "NumberOfFrames" not in frame
        assert "PerFrameFunctionalGroupsSequence" not in frame
        assert frame.file_meta.TransferSyntaxUID == ImplicitVRLittleEndian
        uids.add(str(frame.SOPInstanceUID))
    assert len(uids) == 12
    assert hashlib.sha256(source.read_bytes()).digest() == original_hash
    assert {p.suffix for p in out.iterdir()} == {".dcm"}
    assert (tmp_path / "projection_manifest.json").exists()
    manifest = json.loads((tmp_path / "projection_manifest.json").read_text())
    assert len(manifest) == 12


def test_single_frames_sorted_by_instance_not_filename(tmp_path, context, identity):
    source = tmp_path / "input"
    source.mkdir()
    for order in range(1, 13):
        save(dataset(identity, order=order), source / f"reversed_{13-order:03d}.dcm")
    records = index_frames(source, context)
    assert [r.order for r in records] == list(range(1, 13))
    assert records[0].path.name == "reversed_012.dcm"


def test_missing_legacy_pixel_attributes_supported(tmp_path, context, identity):
    ds = dataset(identity, frames=12)
    del ds.PixelRepresentation
    del ds.PhotometricInterpretation
    del ds.SamplesPerPixel
    source = save(ds, tmp_path / "input.dcm", implicit=True)
    assert stage_projections(source, tmp_path / "projections", context) == 12
    staged = pydicom.dcmread(tmp_path / "projections" / "projection_000000000.dcm")
    assert staged.PixelRepresentation == 0
    np.testing.assert_array_equal(staged.pixel_array, np.arange(512).reshape(16, 32))


def test_multi_spectrum_requires_explicit_choice(tmp_path, context, identity):
    ds = dataset(identity, frames=20)
    ds.add_new(NUMBER_OF_SPECTRA, "US", 2)
    for index, frame in enumerate(ds.PerFrameFunctionalGroupsSequence):
        frame.add_new(SPECTRUM, "US", 1 + index % 2)
    source = save(ds, tmp_path / "input.dcm")
    with pytest.raises(ValueError, match="mixed acquisition"):
        index_frames(source, context)
    chosen = index_frames(source, context, spectrum=2)
    assert [r.order for r in chosen] == list(range(2, 21, 2))
    assert len(chosen) == 10


def test_declared_multi_spectrum_without_indices_rejected(tmp_path, context, identity):
    ds = dataset(identity, frames=12)
    ds.add_new(NUMBER_OF_SPECTRA, "US", 2)
    del ds[SPECTRUM]
    source = save(ds, tmp_path / "input.dcm")
    with pytest.raises(ValueError, match="Missing spectrum/source index"):
        index_frames(source, context)


def test_single_multiframe_can_use_stored_order(tmp_path, context, identity):
    ds = dataset(identity, frames=12)
    for frame in ds.PerFrameFunctionalGroupsSequence:
        del frame[INSTANCE]
    source = save(ds, tmp_path / "input.dcm")
    assert [r.order for r in index_frames(source, context)] == list(range(1, 13))


def test_chunked_multiframe_requires_global_order(tmp_path, context, identity):
    for index in range(2):
        ds = dataset(identity, frames=12)
        for frame in ds.PerFrameFunctionalGroupsSequence:
            del frame[INSTANCE]
        save(ds, tmp_path / f"chunk{index}.dcm")
    with pytest.raises(ValueError, match="global per-frame InstanceNumber"):
        index_frames(tmp_path, context)


@pytest.mark.parametrize("kind,match", [
    ("duplicate", "Duplicate"), ("gap", "irregular gaps"),
    ("angle", "Missing per-projection"), ("count", "one functional-group"),
])
def test_reject_ambiguous_frames(tmp_path, context, identity, kind, match):
    ds = dataset(identity, frames=12)
    frames = ds.PerFrameFunctionalGroupsSequence
    if kind == "duplicate":
        frames[1].InstanceNumber = 1
    elif kind == "gap":
        frames[-1].InstanceNumber = 99
    elif kind == "angle":
        del frames[0][ANGLE]
    elif kind == "count":
        frames.pop()
    source = save(ds, tmp_path / "input.dcm")
    with pytest.raises(ValueError, match=match):
        index_frames(source, context)


def test_mixed_series_rejected(tmp_path, context, identity):
    save(dataset(identity, frames=12), tmp_path / "one.dcm")
    save(dataset((identity[0], generate_uid()), frames=12, order=13), tmp_path / "two.dcm")
    with pytest.raises(ValueError, match="mixed acquisition"):
        index_frames(tmp_path, context)


def test_cancel_before_staging_does_not_write(tmp_path, context, identity):
    source = save(dataset(identity, frames=12), tmp_path / "input.dcm")
    context.cancel_event.set()
    out = tmp_path / "projections"
    with pytest.raises(ReconstructionCancelled):
        stage_projections(source, out, context)
    assert not out.exists()
