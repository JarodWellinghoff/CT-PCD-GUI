"""Stage one ordered CTPD spectrum/source as FreeCT's single-frame Battelle input.

No source file is edited. Private bytes remain in their original little-endian
encoding; big-endian objects are rejected rather than incorrectly relabelled.
"""

from __future__ import annotations

import copy
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.tag import Tag
from pydicom.uid import ImplicitVRLittleEndian, generate_uid

from .models import JobContext

ANGLE = Tag(0x7031, 0x1001)
Z_POSITION = Tag(0x7031, 0x1002)
SPECTRUM = Tag(0x7033, 0x1063)
SOURCE = Tag(0x7033, 0x105D)
NUMBER_OF_SPECTRA = Tag(0x7033, 0x1061)
NUMBER_OF_SOURCES = Tag(0x7033, 0x105B)
INSTANCE = Tag(0x0020, 0x0013)
PIXELS = Tag(0x7FE0, 0x0010)
SHARED = Tag(0x5200, 0x9229)
PER_FRAME = Tag(0x5200, 0x9230)
# Only flatten CTPD modules and the public values relevant to this projection.
PUBLIC_FRAME_TAGS = {
    INSTANCE,
    Tag(0x0018, 0x1151),
    Tag(0x0018, 0x9311),
    Tag(0x0028, 0x1052),
    Tag(0x0028, 0x1053),
    Tag(0x0028, 0x0106),
    Tag(0x0028, 0x0107),
}
CTPD_GROUPS = {0x7029, 0x7031, 0x7033, 0x7037, 0x7041}
SKIP = {
    PIXELS,
    SHARED,
    PER_FRAME,
    Tag(0x0028, 0x0008),
    Tag(0x0028, 0x0009),
    Tag(0x0018, 0x1063),
    Tag(0x7FE0, 0x0001),
    Tag(0x7FE0, 0x0002),
    Tag(0x0020, 0x9221),
    Tag(0x0020, 0x9222),
    Tag(0x0020, 0x9161),
    Tag(0x0020, 0x9162),
    Tag(0x0020, 0x9163),
    Tag(0x0020, 0x9228),
}


@dataclass(frozen=True, slots=True)
class Frame:
    path: Path
    index: int
    order: int
    group: tuple[str, str, int, int]
    angle: float
    z: float


def _find(dataset: Dataset, tag: Tag):
    if tag in dataset:
        return dataset[tag]
    matches = [element for element in dataset.iterall() if element.tag == tag]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous repeated per-frame tag {tag}.")
    return matches[0] if matches else None


def _number(element, vr: str, default=None):
    if element is None:
        return default
    value = element.value
    if isinstance(value, bytes):
        if vr in {"IS", "DS"}:
            value = value.rstrip(b"\0 ").decode("ascii")
        else:
            fmt = {"US": "<H", "FL": "<f"}[vr]
            if len(value) != struct.calcsize(fmt):
                raise ValueError(f"Invalid private {vr} value in {element.tag}.")
            value = struct.unpack(fmt, value)[0]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite value in {element.tag}.")
    if vr in {"IS", "US"}:
        if value != int(value):
            raise ValueError(f"Non-integer value in {element.tag}.")
        return int(value)
    return value


def _groups(dataset: Dataset, count: int) -> tuple[Dataset | None, list]:
    shared = dataset.get("SharedFunctionalGroupsSequence", [])
    per_frame = dataset.get("PerFrameFunctionalGroupsSequence", [])
    if shared and len(shared) != 1:
        raise ValueError("Expected exactly one shared functional-group item.")
    if count > 1 and len(per_frame) != count:
        raise ValueError(
            "Multi-frame CTPD requires one functional-group item per frame."
        )
    if per_frame and len(per_frame) != count:
        raise ValueError("Functional-group count does not match NumberOfFrames.")
    return shared[0] if shared else None, per_frame


def _element(dataset: Dataset, shared, frame, tag: Tag):
    for scope in (frame, shared):
        if scope is not None:
            found = _find(scope, tag)
            if found is not None:
                return found
    return dataset.get(tag)


def _check_encoding(dataset: Dataset) -> None:
    syntax = getattr(getattr(dataset, "file_meta", None), "TransferSyntaxUID", None)
    if syntax is None:
        raise ValueError(
            "Missing TransferSyntaxUID; cannot safely decode this CTPD object."
        )
    if syntax.is_little_endian is False:
        raise ValueError(
            "Big-endian CTPD is unsupported (private byte encoding is ambiguous)."
        )
    if int(dataset.get("BitsAllocated", 0)) != 16:
        raise ValueError("This adapter requires 16-bit integer CTPD PixelData.")
    if int(dataset.get("SamplesPerPixel", 1)) != 1:
        raise ValueError("CTPD projections must be single-component grayscale data.")


def index_frames(
    source: Path,
    context: JobContext,
    spectrum: int = -1,
    source_index: int = -1,
) -> list[Frame]:
    candidates = (
        [source]
        if source.is_file()
        else sorted(
            p
            for p in source.rglob("*")
            if p.is_file()
            and not p.name.startswith(".")
            and p.suffix.lower() in {"", ".dcm", ".dicom", ".ima"}
        )
    )
    if not candidates:
        raise ValueError("No DICOM projection files were found.")
    records: list[Frame] = []
    for file_index, path in enumerate(candidates):
        context.report(f"Inspecting CTPD file {file_index + 1}/{len(candidates)}", 0)
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        _check_encoding(ds)
        count = int(ds.get("NumberOfFrames", 1) or 1)
        if count < 1:
            raise ValueError(f"Invalid NumberOfFrames in {path.name}.")
        shared, frames = _groups(ds, count)
        # study, series = str(ds.get("StudyInstanceUID", "")), str(
        #     ds.get("SeriesInstanceUID", "")
        # )
        study, series = str(ds.get("StudyInstanceUID", "")), "1"
        if not study or not series:
            raise ValueError(f"Missing Study/SeriesInstanceUID in {path.name}.")
        for index in range(count):
            context.check()
            frame = frames[index] if frames else None
            indices = []
            for index_tag, count_tag in (
                (SPECTRUM, NUMBER_OF_SPECTRA),
                (SOURCE, NUMBER_OF_SOURCES),
            ):
                element = _element(ds, shared, frame, index_tag)
                declared_count = _number(
                    _element(ds, shared, frame, count_tag), "US", 1
                )
                if declared_count < 1 or (declared_count > 1 and element is None):
                    raise ValueError(
                        f"Missing spectrum/source index {index_tag} or invalid count in {path.name}."
                    )
                indices.append(_number(element, "US", 1))
            channel, tube = indices
            if (spectrum >= 0 and channel != spectrum) or (
                source_index >= 0 and tube != source_index
            ):
                continue
            angle_element = _find(frame, ANGLE) if frame is not None else ds.get(ANGLE)
            if angle_element is None:
                raise ValueError(
                    f"Missing per-projection tube angle in {path.name}, frame {index+1}."
                )
            angle = _number(angle_element, "FL")
            z = _number(_element(ds, shared, frame, Z_POSITION), "FL")
            if z is None:
                raise ValueError(
                    f"Missing axial position in {path.name}, frame {index+1}."
                )
            # A batch's root InstanceNumber is NOT a frame's global projection number.
            order_element = (
                _find(frame, INSTANCE) if frame is not None else ds.get(INSTANCE)
            )
            order = _number(order_element, "IS")
            if order is None and len(candidates) == 1 and count > 1:
                order = index + 1
            if order is None:
                raise ValueError(
                    "Missing global per-frame InstanceNumber. Only a single multi-frame "
                    "object can fall back to its stored frame order."
                )
            records.append(
                Frame(path, index, order, (study, series, channel, tube), angle, z)
            )
    groups = {record.group for record in records}
    if not records:
        raise ValueError("No projections match the selected spectrum/source indices.")
    if len(groups) != 1:
        choices = sorted({(g[2], g[3]) for g in groups})
        raise ValueError(
            "Select one study/series and one spectrum/source, not a mixed acquisition. "
            f"Detected spectrum/source pairs: {choices}. Use a single-series folder."
        )
    records.sort(key=lambda item: item.order)
    orders = [record.order for record in records]
    if len(orders) != len(set(orders)):
        raise ValueError(
            "Duplicate projection InstanceNumbers; ambiguous ordering or mixed data."
        )
    steps = {b - a for a, b in zip(orders, orders[1:])}
    if len(steps) > 1:
        raise ValueError(
            "Projection InstanceNumbers have irregular gaps; check for missing frames."
        )
    if len(records) < 10:
        raise ValueError("At least 10 projections are required by this FreeCT build.")
    if len({round(record.angle, 7) for record in records}) < 3:
        raise ValueError("Projection angles do not vary; check per-frame geometry.")
    return records


def stage_projections(
    source: Path,
    destination: Path,
    context: JobContext,
    spectrum: int = -1,
    source_index: int = -1,
) -> int:
    records = index_frames(source, context, spectrum, source_index)
    destination.mkdir()
    by_file: dict[Path, list[tuple[int, Frame]]] = {}
    for output_index, record in enumerate(records):
        by_file.setdefault(record.path, []).append((output_index, record))
    completed = 0
    manifest = []
    geometry_shape = None
    for path, selected in by_file.items():
        context.check()
        ds = pydicom.dcmread(str(path), force=True)
        _check_encoding(ds)
        count = int(ds.get("NumberOfFrames", 1) or 1)
        shared, frames = _groups(ds, count)
        # Legacy CTPD commonly omits these conventional image attributes.
        ds.PixelRepresentation = int(ds.get("PixelRepresentation", 0))
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = str(
            ds.get("PhotometricInterpretation", "MONOCHROME2")
        )
        ds.HighBit = int(ds.get("HighBit", int(ds.BitsStored) - 1))
        try:
            pixels = ds.pixel_array
        except Exception as exc:
            raise ValueError(
                f"Could not decode {path.name}; a compression codec may be needed."
            ) from exc
        if pixels.ndim == 2 and count == 1:
            pixels = pixels[np.newaxis, ...]
        if pixels.shape != (count, int(ds.Rows), int(ds.Columns)):
            raise ValueError(f"Unexpected CTPD pixel shape: {pixels.shape}.")
        template = Dataset()
        for element in ds:
            if element.tag not in SKIP and element.tag.element != 0:
                template.add(copy.deepcopy(element))
        for output_index, record in selected:
            context.check()
            single = copy.deepcopy(template)
            frame = frames[record.index] if frames else None
            for scope in (shared, frame):
                if scope is not None:
                    for element in scope.iterall():
                        if element.VR != "SQ" and (
                            element.tag.group in CTPD_GROUPS
                            or element.tag in PUBLIC_FRAME_TAGS
                        ):
                            single.add(copy.deepcopy(element))
            for tag in (Tag(0x0028, 0x1052), Tag(0x0028, 0x1053)):
                if tag not in single or not math.isfinite(float(single[tag].value)):
                    raise ValueError(
                        "Each projection requires finite RescaleIntercept/RescaleSlope."
                    )
            if float(single.RescaleSlope) == 0:
                raise ValueError("Projection RescaleSlope cannot be zero.")
            detector_rows = _number(single.get(Tag(0x7029, 0x1010)), "US")
            detector_cols = _number(single.get(Tag(0x7029, 0x1011)), "US")
            shape = (detector_rows, detector_cols)
            if shape not in {
                (int(ds.Rows), int(ds.Columns)),
                (int(ds.Columns), int(ds.Rows)),
            }:
                raise ValueError(
                    "CTPD detector dimensions do not match stored pixel dimensions."
                )
            if geometry_shape is not None and shape != geometry_shape:
                raise ValueError(
                    "Detector dimensions change within the selected acquisition."
                )
            geometry_shape = shape
            single.InstanceNumber = record.order
            single.SOPInstanceUID = generate_uid()
            single.file_meta = FileMetaDataset()
            single.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            single.file_meta.MediaStorageSOPClassUID = single.SOPClassUID
            single.file_meta.MediaStorageSOPInstanceUID = single.SOPInstanceUID
            single.is_little_endian = True
            single.is_implicit_VR = True
            dtype = "<i2" if single.PixelRepresentation else "<u2"
            single.PixelData = np.ascontiguousarray(
                pixels[record.index], dtype=dtype
            ).tobytes()
            single[PIXELS].is_undefined_length = False
            single[PIXELS].VR = "OW"
            target = destination / f"projection_{output_index:09d}.dcm"
            # write_like_original=False supports pydicom 2.4 as well as 3.x.
            pydicom.dcmwrite(str(target), single, write_like_original=False)
            manifest.append(
                {
                    "file": target.name,
                    "source": str(path),
                    "frame_index": record.index,
                    "instance_number": record.order,
                    "tube_angle_rad": record.angle,
                    "table_position_mm": record.z,
                }
            )
            completed += 1
            if completed % 100 == 0 or completed == len(records):
                context.report(
                    f"Staged {completed}/{len(records)} projections",
                    100 * completed // len(records),
                )
        del pixels, ds
    manifest.sort(key=lambda item: item["file"])
    # Keep non-DICOM files OUT of the directory that FreeCT enumerates.
    (destination.parent / "projection_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return len(records)
