from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.tag import Tag
from pydicom.uid import (
    CTImageStorage,
    ExplicitVRLittleEndian,
    ImplicitVRLittleEndian,
    generate_uid,
)


def make_projection_file(
    path: Path,
    *,
    rows: int = 3,
    columns: int = 5,
    implicit_vr: bool = False,
    spectrum_index: int = 1,
    stored_transposed: bool = False,
    frame_count: int = 1,
    geometry_storage: str = "top_level",
) -> Path:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if geometry_storage not in {
        "top_level",
        "functional_groups",
        "functional_groups_missing_first",
        "alternate_sequence",
    }:
        raise ValueError("Unsupported geometry_storage")
    transfer_syntax = ImplicitVRLittleEndian if implicit_vr else ExplicitVRLittleEndian
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = transfer_syntax
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientName = "SYNTHETIC^TEST"
    ds.PatientID = "SYNTHETIC"
    ds.SeriesNumber = spectrum_index
    ds.InstanceNumber = 1
    ds.KVP = "120"
    # The private tags below always describe the physical detector. Some
    # DICOM-CT-PD generators store that array transposed in standard PixelData.
    ds.Rows = columns if stored_transposed else rows
    ds.Columns = rows if stored_transposed else columns
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleSlope = "0.001"
    ds.RescaleIntercept = "0"
    if frame_count > 1:
        ds.NumberOfFrames = str(frame_count)
    ds.add_new(Tag(0x7029, 0x0010), "LO", "DetectorSystemArrangementModule")
    ds.add_new(Tag(0x7029, 0x1010), "US", rows)
    ds.add_new(Tag(0x7029, 0x1011), "US", columns)
    ds.add_new(Tag(0x7029, 0x1002), "FL", 20.0)
    ds.add_new(Tag(0x7029, 0x1006), "FL", 20.0)
    ds.add_new(Tag(0x7029, 0x100B), "CS", "CYLINDRICAL")
    ds.add_new(Tag(0x7031, 0x0010), "LO", "DetectorDynamicsModule")
    ds.add_new(Tag(0x7031, 0x1031), "FL", 1000.0)
    ds.add_new(Tag(0x7031, 0x1033), "FL", [(columns + 1) / 2, (rows + 1) / 2])
    ds.add_new(Tag(0x7033, 0x0010), "LO", "SourceDynamicsModule")
    ds.add_new(Tag(0x7033, 0x100E), "CS", "FFSNONE")
    ds.add_new(Tag(0x7033, 0x105B), "US", 1)
    ds.add_new(Tag(0x7033, 0x1061), "US", 1)
    ds.add_new(Tag(0x7037, 0x1009), "CS", "HELICAL")
    ds.add_new(Tag(0x7037, 0x100A), "CS", "FANBEAM")
    ds.add_new(Tag(0x7041, 0x0010), "LO", "WaterAttenuationModule")

    angles = [0.1 * frame for frame in range(frame_count)]
    dynamic_values = [
        (Tag(0x7031, 0x1001), "FL", angles),
        (Tag(0x7031, 0x1002), "FL", [0.0] * frame_count),
        (Tag(0x7031, 0x1003), "FL", [600.0] * frame_count),
        (Tag(0x7033, 0x100B), "FL", [0.0] * frame_count),
        (Tag(0x7033, 0x100C), "FL", [0.0] * frame_count),
        (Tag(0x7033, 0x100D), "FL", [0.0] * frame_count),
        (Tag(0x7033, 0x105D), "US", [1] * frame_count),
        (Tag(0x7033, 0x1063), "US", [spectrum_index] * frame_count),
        (Tag(0x7041, 0x1001), "DS", ["0.02"] * frame_count),
    ]
    if frame_count == 1 or geometry_storage == "top_level":
        for tag, vr, values in dynamic_values:
            ds.add_new(tag, vr, values[0] if frame_count == 1 else values)
        if frame_count > 1:
            ds.add_new(Tag(0x0028, 0x0009), "AT", [Tag(0x7031, 0x1001)])
    else:
        frame_items = []
        for frame_index in range(frame_count):
            item = Dataset()
            if not (
                geometry_storage == "functional_groups_missing_first"
                and frame_index == 0
            ):
                for tag, vr, values in dynamic_values:
                    item.add_new(tag, vr, values[frame_index])
            frame_items.append(item)
        if geometry_storage in {
            "functional_groups",
            "functional_groups_missing_first",
        }:
            ds.PerFrameFunctionalGroupsSequence = Sequence(frame_items)
            increment_tag = Tag(0x5200, 0x9230)
        else:
            # Compatibility test: frame-aligned acquisition information in a
            # different sequence rather than fixed Per-frame Functional Groups.
            increment_tag = Tag(0x0040, 0x0555)  # Acquisition Context Sequence
            ds.add_new(increment_tag, "SQ", Sequence(frame_items))
        ds.add_new(Tag(0x0028, 0x0009), "AT", [increment_tag])

    pixel_shape = (frame_count, ds.Rows, ds.Columns)
    ds.PixelData = np.full(pixel_shape, 1000, dtype="<u2").tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def make_recon_series(directory: Path, base_hu: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    for slice_index in range(3):
        path = directory / f"slice_{slice_index + 1:03d}.dcm"
        file_meta = FileMetaDataset()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = CTImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.Modality = "CT"
        ds.PatientID = "SYNTHETIC"
        ds.PatientPosition = "HFS"
        ds.SeriesDescription = "T1" if base_hu >= 100 else "T2"
        ds.InstanceNumber = slice_index + 1
        ds.Rows = 4
        ds.Columns = 5
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [2.0, 3.0]
        ds.SliceThickness = 4.0
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, 4.0 * slice_index]
        ds.SliceLocation = 4.0 * slice_index
        ds.ReconstructionDiameter = 15.0
        ds.KVP = "120"
        ds.RescaleSlope = "1"
        ds.RescaleIntercept = "-1024"
        hu = np.full((4, 5), base_hu + 10 * slice_index, dtype=np.int32)
        hu += np.arange(4, dtype=np.int32)[:, None]
        hu += np.arange(5, dtype=np.int32)[None, :]
        ds.PixelData = (hu + 1024).astype("<u2").tobytes()
        ds.save_as(path, enforce_file_format=True)
    return directory
