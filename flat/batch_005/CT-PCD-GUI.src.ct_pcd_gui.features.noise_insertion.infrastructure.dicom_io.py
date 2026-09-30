from __future__ import annotations

import inspect
import os
from pathlib import Path
import numpy as np
from pydicom import dcmwrite
from pydicom.dataset import FileMetaDataset, Dataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

TAG_PHOTON_STATS = Tag(0x7033, 0x1065)
TAG_SMALLEST = Tag(0x0028, 0x0106)
TAG_LARGEST = Tag(0x0028, 0x0107)
TAG_TUBE_CURRENT = Tag(0x0018, 0x1151)
TAG_PIXEL_REPRESENTATION = Tag(0x0028, 0x0103)
TAG_PIXEL_DATA = Tag(0x7FE0, 0x0010)
TAG_EXTENDED_OFFSET_TABLE = Tag(0x7FE0, 0x0001)
TAG_EXTENDED_OFFSET_TABLE_LENGTHS = Tag(0x7FE0, 0x0002)
_DCMWRITE_SUPPORTS_ENFORCE = (
    "enforce_file_format" in inspect.signature(dcmwrite).parameters
)


def read_photon_statistics(dataset: Dataset) -> np.ndarray:
    if TAG_PHOTON_STATS not in dataset:
        raise KeyError("PhotonStatistics (7033,1065) was not found.")
    value = dataset[TAG_PHOTON_STATS].value
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=np.float32).copy()
    return np.asarray(value, dtype=np.float32).ravel()


def rescale_values(dataset: Dataset) -> tuple[float, float]:
    if not (hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept")):
        raise KeyError(
            "RescaleSlope / RescaleIntercept were not found in the input CTPD."
        )
    slope = float(dataset.RescaleSlope)
    intercept = float(dataset.RescaleIntercept)
    if slope == 0.0:
        raise ValueError("RescaleSlope is zero; pixel data cannot be encoded.")
    return intercept, slope


def encode_with_fixed_rescale(
    noisy_attenuation: np.ndarray,
    intercept: float,
    slope: float,
) -> tuple[np.ndarray, int]:
    raw = (noisy_attenuation - intercept) / slope
    clipped = int(np.count_nonzero((raw < 0.0) | (raw > 65535.0)))
    encoded = np.clip(np.rint(raw), 0.0, 65535.0).astype(np.uint16)
    return encoded, clipped


def force_minmax_us(dataset: Dataset) -> None:
    for element in dataset.iterall():
        if element.tag in (TAG_SMALLEST, TAG_LARGEST):
            element.VR = "US"


def set_uncompressed_output_transfer_syntax(dataset: Dataset) -> None:
    if not hasattr(dataset, "file_meta") or dataset.file_meta is None:
        dataset.file_meta = FileMetaDataset()

    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = dataset.get(
        "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2"
    )
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False


def write_dataset_atomic(dataset: Dataset, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".part")
    try:
        if _DCMWRITE_SUPPORTS_ENFORCE:
            dcmwrite(str(temporary_path), dataset, enforce_file_format=True)
        else:
            dcmwrite(str(temporary_path), dataset, write_like_original=False)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def set_uncompressed_pixel_data(dataset: Dataset, encoded: np.ndarray) -> None:
    for stale_tag in (TAG_EXTENDED_OFFSET_TABLE, TAG_EXTENDED_OFFSET_TABLE_LENGTHS):
        if stale_tag in dataset:
            del dataset[stale_tag]

    dataset.PixelData = np.ascontiguousarray(encoded).tobytes()
    element = dataset[TAG_PIXEL_DATA]
    element.is_undefined_length = False
    bits_allocated = int(getattr(dataset, "BitsAllocated", 16))
    element.VR = "OB" if bits_allocated <= 8 else "OW"


def update_tube_current(dataset: Dataset, mas_factor: float) -> None:
    if TAG_TUBE_CURRENT in dataset:
        try:
            original = float(dataset[TAG_TUBE_CURRENT].value)
            dataset[TAG_TUBE_CURRENT].value = int(round(original * mas_factor))
        except (TypeError, ValueError):
            pass


def new_output_identity(dataset: Dataset) -> None:
    dataset.SOPInstanceUID = generate_uid()
    set_uncompressed_output_transfer_syntax(dataset)


def prepare_single_frame_output(
    dataset: Dataset,
    encoded: np.ndarray,
    intercept: float,
    slope: float,
    mas_factor: float,
) -> Dataset:
    set_uncompressed_pixel_data(dataset, encoded)
    dataset.Rows, dataset.Columns = encoded.shape
    dataset.RescaleIntercept = intercept
    dataset.RescaleSlope = slope
    dataset.PixelRepresentation = 0

    if TAG_SMALLEST in dataset:
        del dataset[TAG_SMALLEST]
    if TAG_LARGEST in dataset:
        del dataset[TAG_LARGEST]
    dataset.add_new(TAG_SMALLEST, "US", int(encoded.min()))
    dataset.add_new(TAG_LARGEST, "US", int(encoded.max()))

    update_tube_current(dataset, mas_factor)
    new_output_identity(dataset)
    force_minmax_us(dataset)
    return dataset


def reshape_multiframe_pixel_array(raw: np.ndarray, frame_count: int) -> np.ndarray:
    if raw.ndim == 2 and frame_count == 1:
        return raw[np.newaxis, ...]
    if raw.ndim != 3:
        raise ValueError(
            f"Multi-frame input produced pixel array shape {raw.shape}; expected 3-D."
        )
    if raw.shape[0] == frame_count:
        return raw
    if raw.shape[-1] == frame_count:
        return np.moveaxis(raw, -1, 0)
    raise ValueError(
        f"Pixel array shape {raw.shape} does not contain {frame_count} frames."
    )


def prepare_multiframe_output(
    dataset: Dataset,
    encoded: np.ndarray,
    intercept: float,
    slope: float,
    mas_factor: float,
    had_root_smallest: bool,
    had_root_largest: bool,
) -> Dataset:
    frame_count, rows, columns = encoded.shape
    dataset.NumberOfFrames = frame_count
    dataset.Rows = rows
    dataset.Columns = columns
    set_uncompressed_pixel_data(dataset, encoded)
    dataset.RescaleIntercept = intercept
    dataset.RescaleSlope = slope

    if TAG_SMALLEST in dataset:
        del dataset[TAG_SMALLEST]
    if TAG_LARGEST in dataset:
        del dataset[TAG_LARGEST]
    if had_root_smallest:
        dataset.add_new(TAG_SMALLEST, "US", int(encoded.min()))
    if had_root_largest:
        dataset.add_new(TAG_LARGEST, "US", int(encoded.max()))

    update_tube_current(dataset, mas_factor)
    new_output_identity(dataset)
    force_minmax_us(dataset)
    return dataset
