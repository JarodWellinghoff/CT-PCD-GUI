from __future__ import annotations

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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from pydicom import dcmread
from ..application.models import (
    JobSummary,
    NoiseJobConfig,
    PreviewPayload,
    DicomWorkItem,
)
from ..domain.noise_model import add_poisson_noise_log_domain
from ..application.preview import (
    make_preview_payload,
    is_preview_iteration,
    local_preview_indices,
)
from ..infrastructure.dicom_io import (
    TAG_SMALLEST,
    TAG_LARGEST,
    read_photon_statistics,
    rescale_values,
    encode_with_fixed_rescale,
    write_dataset_atomic,
    update_tube_current,
    prepare_single_frame_output,
    reshape_multiframe_pixel_array,
    prepare_multiframe_output,
)
from ..infrastructure.discovery import discover_work_items, _noop
from ..application.errors import NoiseInsertionCancelled
from ..application.ports import (
    CancellationFlag,
    NoiseJobCallbacks,
    LogCallback,
    StartedCallback,
    ProgressCallback,
    PreviewCallback,
)


class DefaultNoiseJobExecutor:
    def run(
        self,
        config: NoiseJobConfig,
        *,
        cancel_event: CancellationFlag,
        callbacks: NoiseJobCallbacks,
    ) -> JobSummary:
        return run_noise_job(
            config,
            cancel_event=cancel_event,
            log_callback=callbacks.log,
            started_callback=callbacks.started,
            progress_callback=callbacks.progress,
            preview_callback=callbacks.preview,
        )


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


_PROCESS_CANCEL_EVENT: Optional[CancellationFlag] = None


def _initialize_process_worker(cancel_event: CancellationFlag) -> None:
    """Install a shared cancellation flag inside a spawned worker process."""
    global _PROCESS_CANCEL_EVENT
    _PROCESS_CANCEL_EVENT = cancel_event


def _process_single_frame_item(
    item: DicomWorkItem,
    config: NoiseJobConfig,
    rng: np.random.Generator,
    preview_iteration: Optional[int],
    total_iterations: int,
    local_progress: Callable[[int, str], None],
    cancel_event: CancellationFlag,
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

    intercept, slope = rescale_values(dataset)
    before = raw.astype(np.float64) * slope + intercept
    photon_statistics = read_photon_statistics(dataset)
    after = add_poisson_noise_log_domain(
        before,
        photon_statistics,
        config.mas_factor,
        electronic_noise=config.electronic_noise,
        rng=rng,
    )
    encoded, clipped = encode_with_fixed_rescale(after, intercept, slope)
    local_progress(1, f"Noise generated: {item.input_path.name}")

    if cancel_event.is_set():
        raise NoiseInsertionCancelled("Noise insertion was cancelled.")

    output = prepare_single_frame_output(
        dataset,
        encoded,
        intercept,
        slope,
        config.mas_factor,
    )
    write_dataset_atomic(output, item.output_path)
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
    cancel_event: Optional[CancellationFlag] = None,
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
            if is_preview_iteration(iteration, task.config.preview_interval)
            else None
        ),
        task.total_iterations,
        _noop,
        active_cancel_event,
    )
    return _ItemResult(clipped_pixels=clipped, previews=previews)


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
    cancel_event: CancellationFlag,
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
    encoded, clipped = encode_with_fixed_rescale(after, intercept, slope)
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


def _process_multiframe_item(
    item: DicomWorkItem,
    config: NoiseJobConfig,
    job_seed: int,
    global_frame_start: int,
    total_iterations: int,
    frame_workers: int,
    local_progress: Callable[[int, str], None],
    preview_callback: PreviewCallback,
    cancel_event: CancellationFlag,
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

    raw = reshape_multiframe_pixel_array(np.asarray(dataset.pixel_array), frame_count)
    intercept, slope = rescale_values(dataset)
    had_root_smallest = TAG_SMALLEST in dataset
    had_root_largest = TAG_LARGEST in dataset

    encoded = np.empty(raw.shape, dtype=np.uint16)
    ordered_preview_indices = list(
        local_preview_indices(
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
        update_tube_current(functional_group, config.mas_factor)

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
        photon_statistics = read_photon_statistics(functional_group)
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

    output = prepare_multiframe_output(
        dataset,
        encoded,
        intercept,
        slope,
        config.mas_factor,
        had_root_smallest,
        had_root_largest,
    )
    write_dataset_atomic(output, item.output_path)
    local_progress(frame_count + 1, f"Wrote {item.output_path.name}")
    return total_clipped


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
    cancel_event: CancellationFlag,
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
                    if is_preview_iteration(iteration, config.preview_interval)
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
    cancel_event: CancellationFlag,
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

    process_cancel_event: Optional[CancellationFlag] = None
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
        if is_preview_iteration(
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
        if not is_preview_iteration(
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
    cancel_event: CancellationFlag,
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

    items = discover_work_items(config, CancellationFlag, log_callback)
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
