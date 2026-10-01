from __future__ import annotations

import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.tag import Tag

from .config import InsertionConfig
from .coordinates import pixel_center_to_ctpd
from .ctpd import (
    ProjectionRecord,
    decode_projection_frame,
    discover_projection_records,
    encode_projection_frame,
    projection_frame_count,
    projection_pixel_dtype,
    read_geometry,
    source_and_detector_positions,
    stored_projection_shape,
)
from .dicom_series import load_recon_series
from .lesion_models import LesionModel, load_lesion_model
from .projector import VoxelVolume, prepare_difference_volume, project_volume


ProgressCallback = Callable[[int, int, str], None]
PIXEL_DATA_TAG = Tag(0x7FE0, 0x0010)


@dataclass(frozen=True)
class PreparedInsertion:
    model: LesionModel
    center_ctpd_mm: np.ndarray
    background_hu: tuple[float, ...]
    patient_position: str
    label: str


@dataclass(frozen=True)
class FileResult:
    relative_path: str
    spectrum_index: int
    spectrum_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    instance_number: int
    frame_count: int
    changed_frames: int
    changed_pixels: int
    clipped_pixels: int
    maximum_absolute_added_line_integral: float


_WORKER_INSERTIONS: list[PreparedInsertion] = []
_WORKER_SPECTRUM_MAP: dict[int, int] = {}
_WORKER_OUTPUT_ROOT: Path | None = None
_WORKER_OVERWRITE = False
_WORKER_VOLUME_CACHE: dict[tuple, VoxelVolume] = {}


def _metadata_signature(ds: Dataset) -> tuple[dict, dict]:
    header = Dataset()
    # Iterate by tag so a deferred multi-frame PixelData value is never loaded.
    for tag in ds.keys():
        if tag != PIXEL_DATA_TAG:
            header.add(ds[tag])
    file_meta = ds.file_meta.to_json_dict() if hasattr(ds, "file_meta") else {}
    return header.to_json_dict(), file_meta


def _initialize_worker(
    insertions: list[PreparedInsertion],
    spectrum_map: dict[int, int],
    output_root: str,
    overwrite: bool,
) -> None:
    global _WORKER_INSERTIONS, _WORKER_SPECTRUM_MAP, _WORKER_OUTPUT_ROOT
    global _WORKER_OVERWRITE, _WORKER_VOLUME_CACHE
    _WORKER_INSERTIONS = insertions
    _WORKER_SPECTRUM_MAP = spectrum_map
    _WORKER_OUTPUT_ROOT = Path(output_root)
    _WORKER_OVERWRITE = overwrite
    _WORKER_VOLUME_CACHE = {}


def _volume_for(
    insertion: PreparedInsertion,
    channel: int,
) -> VoxelVolume:
    if channel >= len(insertion.model.old_background_hu):
        raise IndexError(
            f"Insertion {insertion.label!r} model has "
            f"{len(insertion.model.old_background_hu)} old-background values but "
            f"spectrum mapping requests channel {channel + 1}."
        )
    key = (
        insertion.model.source_path,
        insertion.model.lesion_number,
        channel,
        float(insertion.model.old_background_hu[channel]),
        tuple(float(value) for value in insertion.center_ctpd_mm),
        insertion.patient_position,
    )
    volume = _WORKER_VOLUME_CACHE.get(key)
    if volume is None:
        volume = prepare_difference_volume(
            insertion.model,
            channel,
            insertion.center_ctpd_mm,
            insertion.patient_position,
        )
        _WORKER_VOLUME_CACHE[key] = volume
    return volume


def _pixel_data_span(ds: Dataset) -> tuple[int, int]:
    # Access the raw dictionary entry directly. Dataset.get_item() resolves a
    # deferred value and would load a large multi-frame PixelData block.
    raw_element = ds._dict.get(PIXEL_DATA_TAG)
    offset = getattr(raw_element, "value_tell", None)
    length = getattr(raw_element, "length", None)
    if offset is None or length is None:
        raise ValueError(
            "Could not determine the native Pixel Data byte range in the source DICOM."
        )
    return int(offset), int(length)


def _process_record(record: ProjectionRecord) -> FileResult:
    if _WORKER_OUTPUT_ROOT is None:
        raise RuntimeError("Projection worker was not initialized.")
    output_path = _WORKER_OUTPUT_ROOT / record.relative_path
    if output_path.exists() and not _WORKER_OVERWRITE:
        raise FileExistsError(f"Output already exists: {output_path}")

    # Keep potentially very large multi-frame Pixel Data deferred. The output
    # copy is memory-mapped and updated one frame at a time.
    ds = pydicom.dcmread(record.path, defer_size=1024)
    pixel_offset, pixel_length = _pixel_data_span(ds)
    original_signature = _metadata_signature(ds)
    first_geometry = read_geometry(ds, 0)
    frame_count = projection_frame_count(ds)
    storage_shape = stored_projection_shape(ds, first_geometry)
    pixel_dtype = projection_pixel_dtype(ds)
    expected_length = int(np.prod(storage_shape)) * pixel_dtype.itemsize
    if pixel_length != expected_length:
        raise ValueError(
            f"Pixel Data has {pixel_length} bytes; NumberOfFrames and private "
            f"detector dimensions require {expected_length} bytes."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    shutil.copy2(record.path, temporary)
    stored_frames: np.memmap | None = None
    changed = 0
    changed_frames = 0
    clipped = 0
    maximum_added = 0.0
    try:
        stored_frames = np.memmap(
            temporary,
            dtype=pixel_dtype,
            mode="r+",
            offset=pixel_offset,
            shape=storage_shape,
            order="C",
        )
        for frame_index in range(frame_count):
            geometry = read_geometry(ds, frame_index)
            if (
                geometry.rows != first_geometry.rows
                or geometry.columns != first_geometry.columns
                or geometry.stored_rows != first_geometry.stored_rows
                or geometry.stored_columns != first_geometry.stored_columns
                or geometry.pixel_data_transposed
                != first_geometry.pixel_data_transposed
            ):
                raise ValueError(
                    "Detector dimensions or Pixel Data orientation change between "
                    f"frames in {record.relative_path}."
                )
            if geometry.spectrum_index not in _WORKER_SPECTRUM_MAP:
                raise KeyError(
                    f"No lesion channel mapping for spectrum index "
                    f"{geometry.spectrum_index} in frame {frame_index + 1}."
                )
            channel = _WORKER_SPECTRUM_MAP[geometry.spectrum_index]
            source, detector = source_and_detector_positions(geometry)
            added = np.zeros((geometry.rows, geometry.columns), dtype=np.float64)
            for insertion in _WORKER_INSERTIONS:
                volume = _volume_for(
                    insertion,
                    channel,
                )
                added += project_volume(source, detector, volume)

            original = decode_projection_frame(
                ds, stored_frames, frame_index, geometry
            )
            encoded_frame, frame_clipped = encode_projection_frame(
                ds, original + added, frame_index, geometry
            )
            frame_changed = int(
                np.count_nonzero(encoded_frame != stored_frames[frame_index])
            )
            if frame_changed:
                stored_frames[frame_index] = encoded_frame
                changed_frames += 1
            changed += frame_changed
            clipped += frame_clipped
            maximum_added = max(
                maximum_added, float(np.max(np.abs(added), initial=0.0))
            )

        stored_frames.flush()
        del stored_frames
        stored_frames = None
        shutil.copystat(record.path, temporary)
        saved = pydicom.dcmread(temporary, stop_before_pixels=True)
        if _metadata_signature(saved) != original_signature:
            raise RuntimeError(
                f"Header-preservation check failed for {record.relative_path}."
            )
        temporary.replace(output_path)
    except Exception:
        if stored_frames is not None:
            del stored_frames
        temporary.unlink(missing_ok=True)
        raise

    return FileResult(
        relative_path=str(record.relative_path),
        spectrum_index=record.spectrum_index,
        spectrum_indices=record.spectrum_indices,
        source_indices=record.source_indices,
        instance_number=record.instance_number,
        frame_count=frame_count,
        changed_frames=changed_frames,
        changed_pixels=changed,
        clipped_pixels=clipped,
        maximum_absolute_added_line_integral=maximum_added,
    )


def _prepare_insertions(config: InsertionConfig) -> list[PreparedInsertion]:
    recon = load_recon_series(config.t1_series_dir) if config.t1_series_dir else None
    patient_position = (
        str(getattr(recon.first_header, "PatientPosition", config.patient_position)).upper()
        if recon is not None
        else config.patient_position.upper()
    )
    prepared: list[PreparedInsertion] = []
    for index, insertion in enumerate(config.insertions, start=1):
        model = load_lesion_model(insertion.lesion_model)
        if insertion.center_ctpd_mm is not None:
            center = np.asarray(insertion.center_ctpd_mm, dtype=float)
        else:
            if recon is None or insertion.center_pixel is None:
                raise ValueError("A reconstructed T1 series is required for pixel centers.")
            center = pixel_center_to_ctpd(insertion.center_pixel, recon)
        prepared.append(
            PreparedInsertion(
                model=model,
                center_ctpd_mm=center,
                background_hu=tuple(float(value) for value in insertion.background_hu),
                patient_position=patient_position,
                label=insertion.label or f"Lesion {index}",
            )
        )
    return prepared


def _resolve_spectrum_map(
    config: InsertionConfig,
    records: list[ProjectionRecord],
    insertions: list[PreparedInsertion],
) -> dict[int, int]:
    spectra = sorted(
        {spectrum for record in records for spectrum in record.spectrum_indices}
    )
    mapping = dict(config.spectrum_channel_map)
    if not mapping:
        mapping = {spectrum: channel for channel, spectrum in enumerate(spectra)}
    missing = [spectrum for spectrum in spectra if spectrum not in mapping]
    if missing:
        raise ValueError(f"Missing spectrum_channel_map entries for spectra {missing}.")
    maximum_channel = max(mapping[spectrum] for spectrum in spectra)
    for insertion in insertions:
        if maximum_channel >= insertion.model.channel_count:
            raise ValueError(
                f"{insertion.label} model has {insertion.model.channel_count} channels, "
                f"but mapping requests channel {maximum_channel + 1}."
            )
        if maximum_channel >= len(insertion.background_hu):
            raise ValueError(
                f"{insertion.label} has {len(insertion.background_hu)} background HU values, "
                f"but mapping requests channel {maximum_channel + 1}."
            )
    return mapping


def run_insertion(
    config: InsertionConfig,
    progress: ProgressCallback | None = None,
) -> dict:
    """Run the complete projection insertion and return the summary dictionary."""

    config.validate()
    start = time.perf_counter()
    records = discover_projection_records(config.ctpd_input)
    prepared = _prepare_insertions(config)
    spectrum_map = _resolve_spectrum_map(config, records, prepared)
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    if not config.overwrite:
        conflicts = [output_root / record.relative_path for record in records]
        conflicts = [path for path in conflicts if path.exists()]
        if conflicts:
            raise FileExistsError(
                f"{len(conflicts)} output projection file(s) already exist; "
                "choose an empty folder or enable overwrite."
            )

    results: list[FileResult] = []
    total_files = len(records)
    total_frames = sum(record.frame_count for record in records)
    _initialize_worker(prepared, spectrum_map, str(output_root), config.overwrite)
    if config.workers == 1:
        completed_frames = 0
        for record in records:
            result = _process_record(record)
            results.append(result)
            completed_frames += result.frame_count
            if progress:
                progress(completed_frames, total_frames, result.relative_path)
    else:
        with ProcessPoolExecutor(
            max_workers=config.workers,
            initializer=_initialize_worker,
            initargs=(prepared, spectrum_map, str(output_root), config.overwrite),
        ) as executor:
            futures = {executor.submit(_process_record, record): record for record in records}
            completed_frames = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed_frames += result.frame_count
                if progress:
                    progress(completed_frames, total_frames, result.relative_path)

    results.sort(key=lambda item: (item.spectrum_index, item.instance_number, item.relative_path))
    summary = {
        "pipeline_version": "3.1.4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(Path(config.ctpd_input).resolve()),
        "output": str(output_root.resolve()),
        "projection_file_count": total_files,
        "projection_frame_count": total_frames,
        "spectrum_channel_map": {str(key): value for key, value in spectrum_map.items()},
        "lesions": [
            {
                "label": item.label,
                "lesion_model": item.model.source_path,
                "center_ctpd_mm": item.center_ctpd_mm.tolist(),
                "background_hu": list(item.background_hu),
                "new_background_hu": list(item.background_hu),
                "old_background_hu": item.model.old_background_hu.tolist(),
                "old_background_method": item.model.old_background_method,
                "old_background_voxel_count": (
                    item.model.old_background_voxel_count
                ),
                "lesion_mean_hu_by_channel": (
                    item.model.lesion_mean_hu_by_channel.tolist()
                ),
                "original_contrast_hu_by_channel": (
                    item.model.lesion_mean_hu_by_channel
                    - item.model.old_background_hu
                ).tolist(),
                "expected_inserted_mean_hu_by_channel": (
                    [
                        float(new_background + lesion_mean - old_background)
                        for new_background, lesion_mean, old_background in zip(
                            item.background_hu,
                            item.model.lesion_mean_hu_by_channel,
                            item.model.old_background_hu,
                            strict=False,
                        )
                    ]
                ),
            }
            for item in prepared
        ],
        "changed_projection_files": sum(item.changed_pixels > 0 for item in results),
        "changed_projection_frames": sum(item.changed_frames for item in results),
        "changed_pixels": sum(item.changed_pixels for item in results),
        "clipped_pixels": sum(item.clipped_pixels for item in results),
        "elapsed_seconds": time.perf_counter() - start,
        "files": [item.__dict__ for item in results],
    }
    summary_path = output_root / "lesion_insertion_summary.json"
    with summary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary
