"""Backend for projection-domain noise insertion in DICOM-CT-PD data.

This module supports both single-frame folders and multi-frame DICOM files.  It
is GUI-agnostic and exposes progress, logging, preview, and cancellation hooks
so a Qt front end can remain responsive while processing large studies.
"""

from __future__ import annotations

import inspect
import math
import multiprocessing
import os
import threading
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np
from pydicom import dcmread, dcmwrite
from pydicom.dataset import FileMetaDataset, Dataset
from pydicom.tag import Tag
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from src.ct_pcd_gui.features.noise_insertion.application.models import (
    JobSummary,
    NoiseJobRequest as NoiseJobConfig,
    PreviewPayload,
)
from src.ct_pcd_gui.features.noise_insertion.domain.noise_model import (
    add_poisson_noise_log_domain,
)

TAG_NUMBER_OF_FRAMES = Tag(0x0028, 0x0008)
TAG_PHOTON_STATS = Tag(0x7033, 0x1065)
TAG_SMALLEST = Tag(0x0028, 0x0106)
TAG_LARGEST = Tag(0x0028, 0x0107)
TAG_TUBE_CURRENT = Tag(0x0018, 0x1151)
TAG_PIXEL_REPRESENTATION = Tag(0x0028, 0x0103)
TAG_PIXEL_DATA = Tag(0x7FE0, 0x0010)
TAG_EXTENDED_OFFSET_TABLE = Tag(0x7FE0, 0x0001)
TAG_EXTENDED_OFFSET_TABLE_LENGTHS = Tag(0x7FE0, 0x0002)

SUPPORTED_EXTENSIONS = {".dcm", ".ima"}
_DCMWRITE_SUPPORTS_ENFORCE = (
    "enforce_file_format" in inspect.signature(dcmwrite).parameters
)


class NoiseInsertionCancelled(RuntimeError):
    """Raised when a running noise insertion job is cancelled."""


@dataclass
class DicomWorkItem:
    input_path: Path
    output_path: Path
    frame_count: int
    is_multiframe: bool

    @property
    def work_units(self) -> int:
        # One unit per frame plus one final unit for DICOM serialization.
        return self.frame_count + 1


@dataclass(frozen=True)
class _ItemPlan:
    item_index: int
    item: DicomWorkItem
    global_frame_start: int


@dataclass(frozen=True)
class _SingleFrameTask:
    plan: _ItemPlan
    config: NoiseJobConfig
    frame_seed: int
    total_iterations: int


@dataclass
class _ItemResult:
    clipped_pixels: int
    previews: list[PreviewPayload]


@dataclass
class _FrameResult:
    frame_index: int
    encoded: np.ndarray
    clipped_pixels: int
    preview: Optional[PreviewPayload]


class _CancellationFlag(Protocol):
    def is_set(self) -> bool: ...


_PROCESS_CANCEL_EVENT: Optional[_CancellationFlag] = None


def _initialize_process_worker(cancel_event: _CancellationFlag) -> None:
    """Install a shared cancellation flag inside a spawned worker process."""
    global _PROCESS_CANCEL_EVENT
    _PROCESS_CANCEL_EVENT = cancel_event


LogCallback = Callable[[str], None]
StartedCallback = Callable[[int, int, int], None]
ProgressCallback = Callable[[int, int, str], None]
PreviewCallback = Callable[[PreviewPayload], None]


def _noop(*_args, **_kwargs) -> None:
    return None


def validate_config(config: NoiseJobConfig) -> None:
    input_path = Path(config.input_path).expanduser()
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    if not str(config.output_dir).strip():
        raise ValueError("An output directory is required.")

    if config.input_mode not in {"auto", "single", "multi"}:
        raise ValueError(f"Unknown input mode: {config.input_mode}")

    if not (0.0 < float(config.mas_factor) <= 1.0):
        raise ValueError("mAs factor must be greater than 0 and no greater than 1.")

    if float(config.electronic_noise) < 0.0:
        raise ValueError("Electronic noise (Ne) cannot be negative.")

    if int(config.preview_interval) < 0:
        raise ValueError("Preview interval cannot be negative.")

    if config.parallel_mode not in {"auto", "processes", "threads", "sequential"}:
        raise ValueError(f"Unknown parallel mode: {config.parallel_mode}")

    if int(config.max_workers) < 0:
        raise ValueError("Parallel worker count cannot be negative.")

    if any(sep in config.file_suffix for sep in (os.sep, os.altsep) if sep):
        raise ValueError("Output suffix cannot contain path separators.")


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _candidate_files(input_path: Path, output_dir: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    # Exclude generated output only when it is a strict child directory of the
    # input tree.  An output directory equal to, or above, the input directory
    # must not hide all source files.
    exclude_output_tree = output_dir != input_path and _path_is_within(
        output_dir, input_path
    )

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not (exclude_output_tree and _path_is_within(path, output_dir))
    ]
    return sorted(files, key=lambda p: str(p).lower())


def _frame_count_from_header(path: Path) -> int:
    ds = dcmread(
        str(path),
        force=True,
        stop_before_pixels=True,
        specific_tags=[TAG_NUMBER_OF_FRAMES],
    )
    value = ds.get(TAG_NUMBER_OF_FRAMES, 1)
    try:
        count = int(value.value if hasattr(value, "value") else value)
    except (TypeError, ValueError):
        count = 1
    return max(1, count)


def _output_path_for(
    source: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    if input_root.is_dir():
        relative = source.relative_to(input_root)
    else:
        relative = Path(source.name)

    extension = source.suffix or ".dcm"
    output_name = f"{source.stem}{suffix}{extension}"
    return output_root / relative.parent / output_name


def discover_work_items(
    config: NoiseJobConfig,
    log_callback: LogCallback = _noop,
    cancel_event: Optional[threading.Event] = None,
) -> list[DicomWorkItem]:
    """Inspect input headers and build a deterministic processing plan."""
    validate_config(config)

    input_path = Path(config.input_path).expanduser().resolve()
    output_root = Path(config.output_dir).expanduser().resolve()
    candidates = _candidate_files(input_path, output_root, config.recursive)
    if not candidates:
        raise FileNotFoundError(f"No .dcm or .ima files were found at: {input_path}")

    log_callback(f"Inspecting {len(candidates)} DICOM file(s)...")
    items: list[DicomWorkItem] = []

    for index, path in enumerate(candidates, start=1):
        if cancel_event is not None and cancel_event.is_set():
            raise NoiseInsertionCancelled("Cancelled while inspecting input files.")

        try:
            # Forced single-frame mode can avoid a second header read for every
            # projection file.  The pixel shape is still validated while processing.
            frame_count = (
                1 if config.input_mode == "single" else _frame_count_from_header(path)
            )
        except Exception as exc:
            message = f"Unable to read DICOM header for {path.name}: {exc}"
            if config.continue_on_error:
                log_callback(f"WARNING: {message}")
                continue
            raise RuntimeError(message) from exc

        detected_multiframe = frame_count > 1
        if config.input_mode == "single" and detected_multiframe:
            message = (
                f"{path.name} contains {frame_count} frames but the selected mode "
                "is Single-frame."
            )
            if config.continue_on_error:
                log_callback(f"WARNING: {message} File skipped.")
                continue
            raise ValueError(message)

        if config.input_mode == "multi" and not detected_multiframe:
            message = (
                f"{path.name} is single-frame but the selected mode is Multi-frame."
            )
            if config.continue_on_error:
                log_callback(f"WARNING: {message} File skipped.")
                continue
            raise ValueError(message)

        is_multiframe = (
            detected_multiframe
            if config.input_mode == "auto"
            else config.input_mode == "multi"
        )
        output_path = _output_path_for(
            path,
            input_path,
            output_root,
            config.file_suffix,
        )
        items.append(
            DicomWorkItem(
                input_path=path,
                output_path=output_path,
                frame_count=frame_count if is_multiframe else 1,
                is_multiframe=is_multiframe,
            )
        )

        if index % 100 == 0:
            log_callback(f"  Inspected {index}/{len(candidates)} files")

    if not items:
        raise RuntimeError("No compatible DICOM files remain after input inspection.")

    return items


def _read_photon_statistics(dataset: Dataset) -> np.ndarray:
    if TAG_PHOTON_STATS not in dataset:
        raise KeyError("PhotonStatistics (7033,1065) was not found.")
    value = dataset[TAG_PHOTON_STATS].value
    if isinstance(value, (bytes, bytearray)):
        return np.frombuffer(value, dtype=np.float32).copy()
    return np.asarray(value, dtype=np.float32).ravel()


def _rescale_values(dataset: Dataset) -> tuple[float, float]:
    if not (hasattr(dataset, "RescaleSlope") and hasattr(dataset, "RescaleIntercept")):
        raise KeyError(
            "RescaleSlope / RescaleIntercept were not found in the input CTPD."
        )
    slope = float(dataset.RescaleSlope)
    intercept = float(dataset.RescaleIntercept)
    if slope == 0.0:
        raise ValueError("RescaleSlope is zero; pixel data cannot be encoded.")
    return intercept, slope


def _encode_with_fixed_rescale(
    noisy_attenuation: np.ndarray,
    intercept: float,
    slope: float,
) -> tuple[np.ndarray, int]:
    raw = (noisy_attenuation - intercept) / slope
    clipped = int(np.count_nonzero((raw < 0.0) | (raw > 65535.0)))
    encoded = np.clip(np.rint(raw), 0.0, 65535.0).astype(np.uint16)
    return encoded, clipped


def _force_minmax_us(dataset: Dataset) -> None:
    for element in dataset.iterall():
        if element.tag in (TAG_SMALLEST, TAG_LARGEST):
            element.VR = "US"


def _set_uncompressed_output_transfer_syntax(dataset: Dataset) -> None:
    if not hasattr(dataset, "file_meta") or dataset.file_meta is None:
        dataset.file_meta = FileMetaDataset()

    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset.file_meta.MediaStorageSOPClassUID = dataset.get(
        "SOPClassUID", "1.2.840.10008.5.1.4.1.1.2"
    )
    dataset.file_meta.MediaStorageSOPInstanceUID = dataset.SOPInstanceUID
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False


def _write_dataset_atomic(dataset: Dataset, output_path: Path) -> None:
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


def _set_uncompressed_pixel_data(dataset: Dataset, encoded: np.ndarray) -> None:
    for stale_tag in (TAG_EXTENDED_OFFSET_TABLE, TAG_EXTENDED_OFFSET_TABLE_LENGTHS):
        if stale_tag in dataset:
            del dataset[stale_tag]

    dataset.PixelData = np.ascontiguousarray(encoded).tobytes()
    element = dataset[TAG_PIXEL_DATA]
    element.is_undefined_length = False
    bits_allocated = int(getattr(dataset, "BitsAllocated", 16))
    element.VR = "OB" if bits_allocated <= 8 else "OW"


def _update_tube_current(dataset: Dataset, mas_factor: float) -> None:
    if TAG_TUBE_CURRENT in dataset:
        try:
            original = float(dataset[TAG_TUBE_CURRENT].value)
            dataset[TAG_TUBE_CURRENT].value = int(round(original * mas_factor))
        except (TypeError, ValueError):
            pass


def _normalize_preview_pair(
    before: np.ndarray,
    after: np.ndarray,
    max_dimension: int = 640,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    before_f = np.asarray(before, dtype=np.float64)
    after_f = np.asarray(after, dtype=np.float64)
    finite = before_f[np.isfinite(before_f)]

    if finite.size:
        low, high = np.percentile(finite, [0.5, 99.5])
    else:
        low, high = 0.0, 1.0
    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.nanmin(finite)) if finite.size else 0.0
        high = float(np.nanmax(finite)) if finite.size else 1.0
    if high <= low:
        high = low + 1.0

    def convert(image: np.ndarray) -> np.ndarray:
        image = np.nan_to_num(image, nan=low, posinf=high, neginf=low)
        normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
        height, width = normalized.shape
        step = max(1, int(math.ceil(max(height, width) / max_dimension)))
        downsampled = normalized[::step, ::step]
        return np.ascontiguousarray(np.rint(downsampled * 255.0).astype(np.uint8))

    return convert(before_f), convert(after_f), low, high


def make_preview_payload(
    title: str,
    iteration: int,
    total_iterations: int,
    before: np.ndarray,
    after: np.ndarray,
) -> PreviewPayload:
    before_u8, after_u8, low, high = _normalize_preview_pair(before, after)
    return PreviewPayload(
        title=title,
        iteration=int(iteration),
        total_iterations=int(total_iterations),
        before_u8=before_u8,
        after_u8=after_u8,
        window_low=low,
        window_high=high,
        before_mean=float(np.nanmean(before)),
        before_std=float(np.nanstd(before)),
        after_mean=float(np.nanmean(after)),
        after_std=float(np.nanstd(after)),
    )


def _new_output_identity(dataset: Dataset) -> None:
    dataset.SOPInstanceUID = generate_uid()
    _set_uncompressed_output_transfer_syntax(dataset)


def _prepare_single_frame_output(
    dataset: Dataset,
    encoded: np.ndarray,
    intercept: float,
    slope: float,
    mas_factor: float,
    update_tube_current: bool,
) -> Dataset:
    _set_uncompressed_pixel_data(dataset, encoded)
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

    if update_tube_current:
        _update_tube_current(dataset, mas_factor)

    _new_output_identity(dataset)
    _force_minmax_us(dataset)
    return dataset


def _process_single_frame_item(
    item: DicomWorkItem,
    config: NoiseJobConfig,
    rng: np.random.Generator,
    preview_iteration: Optional[int],
    total_iterations: int,
    local_progress: Callable[[int, str], None],
    cancel_event: _CancellationFlag,
) -> tuple[int, list[PreviewPayload]]:
    if cancel_event.is_set():
        raise NoiseInsertionCancelled("Noise insertion was cancelled.")

    dataset = dcmread(str(item.input_path), force=True)
    raw = np.asarray(dataset.pixel_array)
    raw = np.squeeze(raw)
    if raw.ndim != 2:
        raise ValueError(
            f"Single-frame input produced pixel array shape {raw.shape}; expected 2-D."
        )

    intercept, slope = _rescale_values(dataset)
    before = raw.astype(np.float64) * slope + intercept
    photon_statistics = _read_photon_statistics(dataset)
    after = add_poisson_noise_log_domain(
        before,
        photon_statistics,
        config.mas_factor,
        electronic_noise=config.electronic_noise,
        rng=rng,
    )
    encoded, clipped = _encode_with_fixed_rescale(after, intercept, slope)
    local_progress(1, f"Noise generated: {item.input_path.name}")

    if cancel_event.is_set():
        raise NoiseInsertionCancelled("Noise insertion was cancelled.")

    output = _prepare_single_frame_output(
        dataset,
        encoded,
        intercept,
        slope,
        config.mas_factor,
        config.update_tube_current,
    )
    _write_dataset_atomic(output, item.output_path)
    local_progress(2, f"Wrote {item.output_path.name}")

    previews: list[PreviewPayload] = []
    if preview_iteration is not None:
        title = (
            f"Iteration {preview_iteration}/{total_iterations} - "
            f"{item.input_path.name}"
        )
        previews.append(
            make_preview_payload(
                title,
                preview_iteration,
                total_iterations,
                before,
                after,
            )
        )
    return clipped, previews


def _execute_single_frame_task(
    task: _SingleFrameTask,
    cancel_event: Optional[_CancellationFlag] = None,
) -> _ItemResult:
    """Process one single-frame file in a thread or spawned child process."""
    active_cancel_event = cancel_event or _PROCESS_CANCEL_EVENT
    if active_cancel_event is None:
        active_cancel_event = threading.Event()

    rng = np.random.default_rng(task.frame_seed)
    iteration = task.plan.global_frame_start + 1
    clipped, previews = _process_single_frame_item(
        task.plan.item,
        task.config,
        rng,
        (
            iteration
            if _is_preview_iteration(iteration, task.config.preview_interval)
            else None
        ),
        task.total_iterations,
        _noop,
        active_cancel_event,
    )
    return _ItemResult(clipped_pixels=clipped, previews=previews)


def _reshape_multiframe_pixel_array(raw: np.ndarray, frame_count: int) -> np.ndarray:
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


def _process_multiframe_frame(
    frame_index: int,
    raw_frame: np.ndarray,
    photon_statistics: np.ndarray,
    intercept: float,
    slope: float,
    mas_factor: float,
    electronic_noise: float,
    frame_seed: int,
    preview_title: Optional[str],
    preview_iteration: Optional[int],
    total_iterations: int,
    cancel_event: _CancellationFlag,
) -> _FrameResult:
    """Generate and encode one frame without mutating the pydicom dataset."""
    if cancel_event.is_set():
        raise NoiseInsertionCancelled("Noise insertion was cancelled.")

    before = np.asarray(raw_frame, dtype=np.float64) * slope + intercept
    after = add_poisson_noise_log_domain(
        before,
        photon_statistics,
        mas_factor,
        electronic_noise=electronic_noise,
        rng=np.random.default_rng(frame_seed),
    )
    encoded, clipped = _encode_with_fixed_rescale(after, intercept, slope)
    preview = (
        make_preview_payload(
            preview_title,
            preview_iteration,
            total_iterations,
            before,
            after,
        )
        if preview_title is not None and preview_iteration is not None
        else None
    )

    if cancel_event.is_set():
        raise NoiseInsertionCancelled("Noise insertion was cancelled.")

    return _FrameResult(
        frame_index=frame_index,
        encoded=encoded,
        clipped_pixels=clipped,
        preview=preview,
    )


def _prepare_multiframe_output(
    dataset: Dataset,
    encoded: np.ndarray,
    intercept: float,
    slope: float,
    mas_factor: float,
    update_tube_current: bool,
    had_root_smallest: bool,
    had_root_largest: bool,
) -> Dataset:
    frame_count, rows, columns = encoded.shape
    dataset.NumberOfFrames = frame_count
    dataset.Rows = rows
    dataset.Columns = columns
    _set_uncompressed_pixel_data(dataset, encoded)
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

    if update_tube_current:
        _update_tube_current(dataset, mas_factor)

    _new_output_identity(dataset)
    _force_minmax_us(dataset)
    return dataset


def _process_multiframe_item(
    item: DicomWorkItem,
    config: NoiseJobConfig,
    job_seed: int,
    global_frame_start: int,
    total_iterations: int,
    frame_workers: int,
    local_progress: Callable[[int, str], None],
    preview_callback: PreviewCallback,
    cancel_event: _CancellationFlag,
) -> int:
    dataset = dcmread(str(item.input_path), force=True)
    frame_count = int(dataset.NumberOfFrames)
    if frame_count != item.frame_count:
        raise ValueError(
            f"Frame count changed between inspection and processing for {item.input_path.name}: "
            f"{item.frame_count} -> {frame_count}."
        )

    if not (
        hasattr(dataset, "PerFrameFunctionalGroupsSequence")
        and dataset.PerFrameFunctionalGroupsSequence
    ):
        raise KeyError("PerFrameFunctionalGroupsSequence was not found.")
    if len(dataset.PerFrameFunctionalGroupsSequence) < frame_count:
        raise ValueError(
            "PerFrameFunctionalGroupsSequence has fewer items than NumberOfFrames."
        )

    raw = _reshape_multiframe_pixel_array(np.asarray(dataset.pixel_array), frame_count)
    intercept, slope = _rescale_values(dataset)
    had_root_smallest = TAG_SMALLEST in dataset
    had_root_largest = TAG_LARGEST in dataset

    encoded = np.empty(raw.shape, dtype=np.uint16)
    ordered_preview_indices = list(
        _local_preview_indices(
            global_frame_start,
            frame_count,
            config.preview_interval,
        )
    )
    preview_local_indices = set(ordered_preview_indices)
    preview_buffer: dict[int, PreviewPayload] = {}
    next_preview_position = 0
    total_clipped = 0
    completed_frames = 0

    def apply_result(result: _FrameResult) -> None:
        nonlocal total_clipped, completed_frames, next_preview_position
        frame_index = result.frame_index
        functional_group = dataset.PerFrameFunctionalGroupsSequence[frame_index]
        encoded[frame_index] = result.encoded
        total_clipped += result.clipped_pixels

        if TAG_SMALLEST in functional_group:
            del functional_group[TAG_SMALLEST]
        if TAG_LARGEST in functional_group:
            del functional_group[TAG_LARGEST]
        functional_group.add_new(TAG_SMALLEST, "US", int(result.encoded.min()))
        functional_group.add_new(TAG_LARGEST, "US", int(result.encoded.max()))
        if config.update_tube_current:
            _update_tube_current(functional_group, config.mas_factor)

        if result.preview is not None:
            preview_buffer[frame_index] = result.preview

        # Threaded frames can finish out of order. Release scheduled previews
        # only when all earlier preview targets have also completed so the UI
        # advances monotonically through the source projection order.
        while next_preview_position < len(ordered_preview_indices):
            expected_index = ordered_preview_indices[next_preview_position]
            payload = preview_buffer.get(expected_index)
            if payload is None:
                break
            preview_callback(preview_buffer.pop(expected_index))
            next_preview_position += 1

        completed_frames += 1
        local_progress(
            completed_frames,
            f"{item.input_path.name}: completed frame "
            f"{completed_frames}/{frame_count} (source frame {frame_index + 1})",
        )

    def make_frame_call(frame_index: int) -> tuple:
        functional_group = dataset.PerFrameFunctionalGroupsSequence[frame_index]
        photon_statistics = _read_photon_statistics(functional_group)
        preview_title = None
        preview_iteration = None
        if frame_index in preview_local_indices:
            preview_iteration = global_frame_start + frame_index + 1
            preview_title = (
                f"Iteration {preview_iteration}/{total_iterations} - "
                f"{item.input_path.name}, frame {frame_index + 1}/{frame_count}"
            )
        return (
            frame_index,
            raw[frame_index],
            photon_statistics,
            intercept,
            slope,
            config.mas_factor,
            config.electronic_noise,
            job_seed + global_frame_start + frame_index,
            preview_title,
            preview_iteration,
            total_iterations,
            cancel_event,
        )

    frame_workers = max(1, min(int(frame_workers), frame_count))
    if frame_workers == 1:
        for frame_index in range(frame_count):
            if cancel_event.is_set():
                raise NoiseInsertionCancelled("Noise insertion was cancelled.")
            apply_result(_process_multiframe_frame(*make_frame_call(frame_index)))
    else:
        executor = ThreadPoolExecutor(
            max_workers=frame_workers,
            thread_name_prefix="ctpd-noise-frame",
        )
        in_flight: dict[Future, int] = {}
        next_frame = 0
        completed_normally = False
        queue_limit = max(frame_workers, frame_workers * 2)

        def submit_available() -> None:
            nonlocal next_frame
            while (
                next_frame < frame_count
                and len(in_flight) < queue_limit
                and not cancel_event.is_set()
            ):
                frame_index = next_frame
                next_frame += 1
                future = executor.submit(
                    _process_multiframe_frame,
                    *make_frame_call(frame_index),
                )
                in_flight[future] = frame_index

        try:
            submit_available()
            while in_flight:
                if cancel_event.is_set():
                    raise NoiseInsertionCancelled("Noise insertion was cancelled.")

                completed, _ = wait(
                    in_flight,
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not completed:
                    continue

                for future in completed:
                    in_flight.pop(future, None)
                    apply_result(future.result())
                submit_available()
            completed_normally = True
        finally:
            if not completed_normally:
                for future in in_flight:
                    future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

    if cancel_event.is_set():
        raise NoiseInsertionCancelled("Noise insertion was cancelled.")

    output = _prepare_multiframe_output(
        dataset,
        encoded,
        intercept,
        slope,
        config.mas_factor,
        config.update_tube_current,
        had_root_smallest,
        had_root_largest,
    )
    _write_dataset_atomic(output, item.output_path)
    local_progress(frame_count + 1, f"Wrote {item.output_path.name}")
    return total_clipped


def _is_preview_iteration(iteration: int, preview_interval: int) -> bool:
    """Return whether a one-based ordered projection iteration is scheduled."""
    interval = int(preview_interval)
    return interval > 0 and int(iteration) > 0 and int(iteration) % interval == 0


def _local_preview_indices(
    global_frame_start: int,
    frame_count: int,
    preview_interval: int,
) -> range:
    """Return local zero-based frame indices matching the global interval."""
    interval = int(preview_interval)
    if interval <= 0 or frame_count <= 0:
        return range(0)

    # global_frame_start is the number of projections before this item. Preview
    # iterations are one-based multiples of the selected interval.
    first_iteration = ((global_frame_start // interval) + 1) * interval
    item_last_iteration = global_frame_start + frame_count
    if first_iteration > item_last_iteration:
        return range(0)

    first_local_index = first_iteration - global_frame_start - 1
    return range(first_local_index, frame_count, interval)


def _effective_worker_count(
    config: NoiseJobConfig,
    task_count: int,
    workload: str,
) -> int:
    """Resolve the user setting to a bounded worker count."""
    task_count = max(1, int(task_count))
    if config.parallel_mode == "sequential" or task_count == 1:
        return 1

    requested = int(config.max_workers)
    if requested > 0:
        return max(1, min(requested, task_count))

    cpu_count = os.cpu_count() or 1
    usable_cpus = cpu_count - 1 if cpu_count > 2 else cpu_count
    # Frame tasks hold several float64 arrays.  A lower automatic frame cap
    # prevents memory pressure while still using multiple cores.
    automatic_cap = 32
    return max(1, min(task_count, usable_cpus, automatic_cap))


def _process_one_plan(
    plan: _ItemPlan,
    config: NoiseJobConfig,
    job_seed: int,
    cancel_event: _CancellationFlag,
    summary: JobSummary,
    completed_units: int,
    total_units: int,
    log_callback: LogCallback,
    progress_callback: ProgressCallback,
    preview_callback: PreviewCallback,
) -> int:
    """Process one item, using frame threads only when it is multi-frame."""
    item = plan.item
    item_unit_start = completed_units
    last_local_unit = 0

    def local_progress(local_unit: int, status: str) -> None:
        nonlocal last_local_unit
        last_local_unit = max(last_local_unit, local_unit)
        progress_callback(item_unit_start + local_unit, total_units, status)

    frame_workers = 1
    if item.is_multiframe:
        frame_workers = _effective_worker_count(config, item.frame_count, "frames")
        summary.max_workers = max(summary.max_workers, frame_workers)

    log_callback(
        f"[{plan.item_index + 1}/{summary.total_files}] Processing "
        f"{item.input_path.name} "
        f"({'multi-frame' if item.is_multiframe else 'single-frame'}, "
        f"{item.frame_count} frame(s))"
    )
    if item.is_multiframe and frame_workers > 1:
        if config.parallel_mode == "processes":
            log_callback(
                f"  Using {frame_workers} frame threads. Multi-frame pixel data stays "
                "in one process to avoid copying the full array to child processes."
            )
        else:
            log_callback(f"  Using {frame_workers} parallel frame threads.")

    cancelled_during_item = False
    try:
        if item.is_multiframe:
            clipped = _process_multiframe_item(
                item,
                config,
                job_seed,
                plan.global_frame_start,
                summary.total_frames,
                frame_workers,
                local_progress,
                preview_callback,
                cancel_event,
            )
            previews: list[PreviewPayload] = []
        else:
            rng = np.random.default_rng(job_seed + plan.global_frame_start)
            iteration = plan.global_frame_start + 1
            clipped, previews = _process_single_frame_item(
                item,
                config,
                rng,
                (
                    iteration
                    if _is_preview_iteration(iteration, config.preview_interval)
                    else None
                ),
                summary.total_frames,
                local_progress,
                cancel_event,
            )

        summary.written_files += 1
        summary.clipped_pixels += clipped
        summary.output_paths.append(str(item.output_path))
        log_callback(
            f"OK: {item.output_path.name}"
            + (f" ({clipped:,} clipped pixel(s))" if clipped else "")
        )
        for preview in previews:
            preview_callback(preview)

    except NoiseInsertionCancelled:
        cancelled_during_item = True
        raise
    except Exception as exc:
        summary.failed_files += 1
        failure = f"{item.input_path.name}: {exc}"
        summary.failures.append(failure)
        log_callback(f"FAILED: {failure}")
        log_callback(traceback.format_exc().rstrip())
        if not config.continue_on_error:
            raise
    finally:
        if not cancelled_during_item:
            completed_units = item_unit_start + item.work_units
            if last_local_unit < item.work_units:
                progress_callback(
                    completed_units,
                    total_units,
                    f"Finished attempt: {item.input_path.name}",
                )

    return completed_units


def _run_single_frame_batch(
    plans: list[_ItemPlan],
    config: NoiseJobConfig,
    job_seed: int,
    cancel_event: _CancellationFlag,
    summary: JobSummary,
    completed_units: int,
    total_units: int,
    log_callback: LogCallback,
    progress_callback: ProgressCallback,
    preview_callback: PreviewCallback,
) -> int:
    """Process independent single-frame files concurrently."""
    worker_count = _effective_worker_count(config, len(plans), "files")
    if worker_count <= 1:
        for plan in plans:
            completed_units = _process_one_plan(
                plan,
                config,
                job_seed,
                cancel_event,
                summary,
                completed_units,
                total_units,
                log_callback,
                progress_callback,
                preview_callback,
            )
        return completed_units

    executor_kind = "threads" if config.parallel_mode == "threads" else "processes"
    summary.max_workers = max(summary.max_workers, worker_count)
    log_callback(
        f"Parallel single-frame batch: {len(plans)} file(s) using "
        f"{worker_count} {executor_kind}."
    )

    process_cancel_event: Optional[_CancellationFlag] = None
    executor: ProcessPoolExecutor | ThreadPoolExecutor
    if executor_kind == "processes":
        context = multiprocessing.get_context("spawn")
        process_cancel_event = context.Event()
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=_initialize_process_worker,
            initargs=(process_cancel_event,),
        )
    else:
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="ctpd-noise-file",
        )

    tasks = [
        _SingleFrameTask(
            plan=plan,
            config=config,
            frame_seed=job_seed + plan.global_frame_start,
            total_iterations=summary.total_frames,
        )
        for plan in plans
    ]
    in_flight: dict[Future, _SingleFrameTask] = {}
    next_task = 0
    queue_limit = max(worker_count, worker_count * 2)
    abort_exception: Optional[BaseException] = None
    cancellation_requested = False
    ordered_preview_iterations = sorted(
        plan.global_frame_start + 1
        for plan in plans
        if _is_preview_iteration(
            plan.global_frame_start + 1,
            config.preview_interval,
        )
    )
    resolved_preview_iterations: set[int] = set()
    preview_buffer: dict[int, PreviewPayload] = {}
    next_preview_position = 0

    def resolve_preview(
        plan: _ItemPlan,
        result: Optional[_ItemResult] = None,
    ) -> None:
        nonlocal next_preview_position
        if not _is_preview_iteration(
            plan.global_frame_start + 1,
            config.preview_interval,
        ):
            return

        iteration = plan.global_frame_start + 1
        resolved_preview_iterations.add(iteration)
        if result is not None and result.previews:
            preview_buffer[iteration] = result.previews[-1]

        # Parallel files can complete out of order. A failed preview target is
        # treated as resolved without an image so later scheduled previews are
        # not blocked indefinitely.
        while next_preview_position < len(ordered_preview_iterations):
            expected_iteration = ordered_preview_iterations[next_preview_position]
            if expected_iteration not in resolved_preview_iterations:
                break
            payload = preview_buffer.pop(expected_iteration, None)
            if payload is not None:
                preview_callback(payload)
            next_preview_position += 1

    def submit_available() -> None:
        nonlocal next_task
        while (
            next_task < len(tasks)
            and len(in_flight) < queue_limit
            and abort_exception is None
            and not cancel_event.is_set()
        ):
            task = tasks[next_task]
            next_task += 1
            plan = task.plan
            log_callback(
                f"[{plan.item_index + 1}/{summary.total_files}] Queued "
                f"{plan.item.input_path.name}"
            )
            if executor_kind == "processes":
                future = executor.submit(_execute_single_frame_task, task)
            else:
                future = executor.submit(
                    _execute_single_frame_task,
                    task,
                    cancel_event,
                )
            in_flight[future] = task

    try:
        submit_available()
        while in_flight:
            if cancel_event.is_set():
                cancellation_requested = True
                if process_cancel_event is not None:
                    process_cancel_event.set()
                for future in in_flight:
                    future.cancel()
                break

            completed, _ = wait(
                in_flight,
                timeout=0.1,
                return_when=FIRST_COMPLETED,
            )
            if not completed:
                continue

            for future in completed:
                task = in_flight.pop(future)
                plan = task.plan
                item = plan.item
                try:
                    result = future.result()
                except NoiseInsertionCancelled as exc:
                    resolve_preview(plan)
                    if cancel_event.is_set() or abort_exception is not None:
                        continue
                    failure = f"{item.input_path.name}: {exc}"
                    summary.failed_files += 1
                    summary.failures.append(failure)
                    log_callback(f"FAILED: {failure}")
                    if not config.continue_on_error and abort_exception is None:
                        abort_exception = exc
                except Exception as exc:
                    resolve_preview(plan)
                    summary.failed_files += 1
                    failure = f"{item.input_path.name}: {exc}"
                    summary.failures.append(failure)
                    log_callback(f"FAILED: {failure}")
                    log_callback(
                        "".join(
                            traceback.format_exception(
                                type(exc),
                                exc,
                                exc.__traceback__,
                            )
                        ).rstrip()
                    )
                    if not config.continue_on_error and abort_exception is None:
                        abort_exception = exc
                else:
                    resolve_preview(plan, result)
                    summary.written_files += 1
                    summary.clipped_pixels += result.clipped_pixels
                    summary.output_paths.append(str(item.output_path))
                    log_callback(
                        f"OK: {item.output_path.name}"
                        + (
                            f" ({result.clipped_pixels:,} clipped pixel(s))"
                            if result.clipped_pixels
                            else ""
                        )
                    )
                completed_units += item.work_units
                progress_callback(
                    completed_units,
                    total_units,
                    f"Finished attempt: {item.input_path.name}",
                )

            if abort_exception is not None:
                if process_cancel_event is not None:
                    process_cancel_event.set()
                for future in in_flight:
                    future.cancel()
                break

            submit_available()
    finally:
        if cancellation_requested or abort_exception is not None:
            if process_cancel_event is not None:
                process_cancel_event.set()
            for future in in_flight:
                future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    if cancellation_requested or cancel_event.is_set():
        raise NoiseInsertionCancelled(
            "Noise insertion was cancelled. Active parallel tasks were allowed to stop safely."
        )
    if abort_exception is not None:
        raise RuntimeError(
            "Parallel processing stopped after an individual file failed."
        ) from abort_exception

    return completed_units


def run_noise_job(
    config: NoiseJobConfig,
    *,
    cancel_event: Optional[threading.Event] = None,
    log_callback: LogCallback = _noop,
    started_callback: StartedCallback = _noop,
    progress_callback: ProgressCallback = _noop,
    preview_callback: PreviewCallback = _noop,
) -> JobSummary:
    """Run a complete batch noise insertion job.

    All callbacks are invoked synchronously in the calling thread.  A GUI should
    call this function from a worker thread and bridge callbacks to signals.
    """
    if cancel_event is None:
        cancel_event = threading.Event()

    items = discover_work_items(config, log_callback, cancel_event)
    total_frames = sum(item.frame_count for item in items)
    total_units = sum(item.work_units for item in items)
    scheduled_preview_updates = (
        total_frames // int(config.preview_interval)
        if int(config.preview_interval) > 0
        else 0
    )

    plans: list[_ItemPlan] = []
    global_frame_start = 0
    for item_index, item in enumerate(items):
        plans.append(
            _ItemPlan(
                item_index=item_index,
                item=item,
                global_frame_start=global_frame_start,
            )
        )
        global_frame_start += item.frame_count

    summary = JobSummary(
        total_files=len(items),
        total_frames=total_frames,
        parallel_mode=config.parallel_mode,
        max_workers=1,
    )
    started_callback(total_units, len(items), total_frames)

    input_kind_counts = {
        "single": sum(not item.is_multiframe for item in items),
        "multi": sum(item.is_multiframe for item in items),
    }
    log_callback(
        f"Plan: {len(items)} file(s), {total_frames} frame(s), "
        f"{input_kind_counts['single']} single-frame and "
        f"{input_kind_counts['multi']} multi-frame file(s)."
    )
    log_callback(
        f"Noise settings: mAs factor={config.mas_factor:.6g} "
        f"({config.mas_factor * 100.0:.3g}% dose), Ne={config.electronic_noise:.6g}, "
        f"seed={'random' if config.seed is None else config.seed}."
    )
    worker_setting = "automatic" if config.max_workers == 0 else str(config.max_workers)
    log_callback(
        f"Parallel settings: mode={config.parallel_mode}, workers={worker_setting}. "
        "Auto mode uses processes for independent single-frame files and threads "
        "for frames inside a multi-frame DICOM."
    )
    if config.preview_interval > 0:
        log_callback(
            f"Preview cadence: every {config.preview_interval:,} ordered projection "
            f"iteration(s), up to {scheduled_preview_updates:,} live update(s)."
        )
    else:
        log_callback("Preview cadence: disabled.")
    log_callback(f"Output directory: {Path(config.output_dir).expanduser().resolve()}")

    # A random job seed is generated once so every frame still receives an
    # independent deterministic sub-seed during this run.
    job_seed = (
        int(config.seed)
        if config.seed is not None
        else int.from_bytes(os.urandom(16), byteorder="little", signed=False)
    )

    completed_units = 0
    plan_index = 0

    def mark_skipped(plan: _ItemPlan) -> None:
        nonlocal completed_units
        item = plan.item
        summary.skipped_files += 1
        completed_units += item.work_units
        message = f"Skipped existing output: {item.output_path}"
        log_callback(f"SKIPPED: {message}")
        progress_callback(completed_units, total_units, message)

    while plan_index < len(plans):
        if cancel_event.is_set():
            raise NoiseInsertionCancelled("Noise insertion was cancelled.")

        plan = plans[plan_index]
        item = plan.item
        if item.output_path.exists() and not config.overwrite_existing:
            mark_skipped(plan)
            plan_index += 1
            continue

        # Consecutive single-frame files are independent and can be dispatched
        # as a bounded parallel batch. Multi-frame files are kept in one process
        # and their frames are parallelized with threads.
        if not item.is_multiframe and config.parallel_mode != "sequential":
            batch: list[_ItemPlan] = []
            batch_end = plan_index
            while batch_end < len(plans) and not plans[batch_end].item.is_multiframe:
                candidate = plans[batch_end]
                if (
                    candidate.item.output_path.exists()
                    and not config.overwrite_existing
                ):
                    mark_skipped(candidate)
                else:
                    batch.append(candidate)
                batch_end += 1

            if batch:
                completed_units = _run_single_frame_batch(
                    batch,
                    config,
                    job_seed,
                    cancel_event,
                    summary,
                    completed_units,
                    total_units,
                    log_callback,
                    progress_callback,
                    preview_callback,
                )
            plan_index = batch_end
            continue

        completed_units = _process_one_plan(
            plan,
            config,
            job_seed,
            cancel_event,
            summary,
            completed_units,
            total_units,
            log_callback,
            progress_callback,
            preview_callback,
        )
        plan_index += 1

    log_callback(
        f"Finished: {summary.written_files}/{summary.total_files} file(s) written, "
        f"{summary.failed_files} failed, {summary.skipped_files} skipped."
    )
    log_callback(
        f"Maximum parallel workers used: {summary.max_workers} "
        f"(requested mode: {summary.parallel_mode})."
    )
    if summary.clipped_pixels:
        log_callback(
            f"Total clipped pixels when re-encoding with the original rescale: "
            f"{summary.clipped_pixels:,}."
        )
    else:
        log_callback(
            "No pixels were clipped when re-encoding with the original rescale."
        )
    return summary
