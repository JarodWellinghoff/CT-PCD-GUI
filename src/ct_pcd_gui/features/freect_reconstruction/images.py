"""Read the narrow float32 NIfTI format emitted by the supplied FreeCT v2 build.

Keep the engine's compressed file untouched. Write a separately validated .nii
with corrected NIfTI-1 magic/voxel spacing and no invented patient transform.
"""
from __future__ import annotations

import gzip
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import JobContext


@dataclass(frozen=True, slots=True)
class VolumeHeader:
    shape: tuple[int, int, int]
    offset: int
    slope: float
    intercept: float
    endian: str


def read_header(data: bytes | bytearray, *, allow_freect_magic: bool = False) -> VolumeHeader:
    if len(data) < 352:
        raise ValueError("Truncated NIfTI header.")
    endian = "<" if struct.unpack_from("<i", data)[0] == 348 else ">"
    if struct.unpack_from(endian + "i", data)[0] != 348:
        raise ValueError("Expected a 348-byte NIfTI-1 header from FreeCT.")
    allowed = {b"n+1\0", b"n+2\0"} if allow_freect_magic else {b"n+1\0"}
    if bytes(data[344:348]) not in allowed:
        raise ValueError("Invalid single-file NIfTI magic.")
    dims = struct.unpack_from(endian + "8h", data, 40)
    if dims[0] != 3 or min(dims[1:4]) < 1:
        raise ValueError("FreeCT output must be a nonempty three-dimensional volume.")
    if struct.unpack_from(endian + "2h", data, 70) != (16, 32):
        raise ValueError("Expected FreeCT float32 NIfTI output (datatype 16, bitpix 32).")
    offset, slope, intercept = struct.unpack_from(endian + "3f", data, 108)
    if offset != 352:
        raise ValueError("Unexpected voxel offset in FreeCT NIfTI output.")
    if slope == 0:
        slope, intercept = 1.0, 0.0
    if not math.isfinite(slope) or not math.isfinite(intercept):
        raise ValueError("Invalid NIfTI intensity scaling.")
    return VolumeHeader((dims[3], dims[2], dims[1]), 352, slope, intercept, endian)


def slice_positions(volume: dict) -> list[float]:
    start, end, pitch = (float(volume[key]) for key in ("StartPosition", "EndPosition", "SlicePitch"))
    if not all(math.isfinite(v) for v in (start, end, pitch)) or pitch <= 0:
        raise ValueError("Invalid slice range/pitch.")
    if start == end:
        return [start]
    step = pitch if end > start else -pitch
    positions = []
    current = start
    # Match FreeCT's iterative floating-point addition, including descending scans.
    while min(start, end) <= current <= max(start, end):
        positions.append(current)
        if len(positions) > 32767 or current + step == current:
            raise ValueError("Slice range is too large or pitch is below floating-point resolution.")
        current += step
    return positions


def normalize_volume(source: Path, target: Path, volume: dict, context: JobContext) -> VolumeHeader:
    context.check()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("FreeCT did not create a NIfTI volume; inspect its process logs.")
    with source.open("rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    temporary = target.with_suffix(".nii.part")
    try:
        with opener(source, "rb") as src, temporary.open("wb") as dst:
            raw = bytearray(src.read(352))
            header = read_header(raw, allow_freect_magic=True)
            expected = (len(slice_positions(volume)), int(volume["Ny"]), int(volume["Nx"]))
            if header.shape != expected:
                raise ValueError(f"FreeCT output shape {header.shape} does not match {expected}.")
            if raw[344:348] == b"n+2\0":
                context.report("Correcting the supplied build's NIfTI-1 magic (n+2 -> n+1).")
            raw[344:348] = b"n+1\0"
            # The supplied writer uses SliceThickness for pixdim[3], even when
            # SlicePitch differs, and marks a zero-origin identity as patient space.
            struct.pack_into(header.endian + "4f", raw, 76, 1.0,
                             float(volume["FieldOfView"]) / volume["Nx"],
                             float(volume["FieldOfView"]) / volume["Ny"],
                             float(volume["SlicePitch"]))
            struct.pack_into(header.endian + "2h", raw, 252, 0, 0)
            raw[256:328] = bytes(72)  # no invented quaternion, origin or sform
            raw[348:352] = bytes(4)  # no NIfTI extensions
            raw[123] = 2  # spatial units: millimeters; time is not represented
            description = b"FreeCT scanner grid; patient orientation unknown; see result.json"
            raw[148:228] = description.ljust(80, b"\0")
            dst.write(raw)
            slice_bytes = header.shape[1] * header.shape[2] * 4
            for index in range(header.shape[0]):
                context.check()
                data = src.read(slice_bytes)
                if len(data) != slice_bytes:
                    raise ValueError("FreeCT NIfTI voxel data is truncated.")
                if not np.isfinite(np.frombuffer(data, dtype=header.endian + "f4")).all():
                    raise ValueError(f"Non-finite reconstructed values in slice {index + 1}.")
                dst.write(data)
                if index % 50 == 0 or index + 1 == header.shape[0]:
                    context.report(f"Validating output slice {index+1}/{header.shape[0]}",
                                   100*(index+1)//header.shape[0])
            # Reading to EOF verifies a gzip trailer/CRC, not just pixel length.
            if src.read(1):
                raise ValueError("Unexpected trailing data in FreeCT NIfTI output.")
        context.check()
        os.replace(temporary, target)
        return header
    finally:
        temporary.unlink(missing_ok=True)


def read_slice(path: Path, index: int) -> np.ndarray:
    """Read only the requested slice; never load a full image volume into the UI."""
    with path.open("rb") as stream:
        header = read_header(stream.read(352))
        if not 0 <= index < header.shape[0]:
            raise IndexError("Slice index is out of range.")
        count = header.shape[1] * header.shape[2]
        stream.seek(header.offset + index * count * 4)
        data = stream.read(count * 4)
    if len(data) != count * 4:
        raise ValueError("Truncated NIfTI slice.")
    pixels = np.frombuffer(data, dtype=header.endian + "f4").reshape(header.shape[1:])
    return (pixels.astype(np.float32) * header.slope + header.intercept)
