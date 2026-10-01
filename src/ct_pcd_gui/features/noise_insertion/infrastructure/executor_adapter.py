from __future__ import annotations

import threading
from dataclasses import replace

from pydicom import dcmread

from ..application.models import JobSummary, MasFactor, NoiseJobConfig
from ..application.ports import CancellationFlag, NoiseJobCallbacks
from . import executor
from .dicom_io import simulated_series_description


_RUN_LOCK = threading.Lock()


class DefaultNoiseJobExecutor:
    """Run the established backend with v2-compatible job metadata.

    Discovery is performed once with the live cancellation event. The resulting
    work list is then reused by the established backend, and the first input
    header supplies one shared v2 SeriesDescription for every output file.
    """

    def run(
        self,
        config: NoiseJobConfig,
        *,
        cancel_event: CancellationFlag,
        callbacks: NoiseJobCallbacks,
    ) -> JobSummary:
        with _RUN_LOCK:
            original_discover = executor.discover_work_items
            items = original_discover(config, cancel_event, callbacks.log)

            if items:
                first_dataset = dcmread(
                    str(items[0].input_path),
                    force=True,
                    stop_before_pixels=True,
                )
                series_description = simulated_series_description(
                    first_dataset,
                    config.mas_factor,
                )
                config = replace(
                    config,
                    mas_factor=MasFactor(
                        float(config.mas_factor),
                        config.fine_tune_factor,
                        series_description,
                    ),
                )
                callbacks.log(
                    "v2 calibration: "
                    f"fine-tune factor={config.fine_tune_factor:.6g}."
                )
                callbacks.log(f"SeriesDescription: {series_description}")

            def reuse_discovered_items(
                _config: NoiseJobConfig,
                _incorrect_flag: object,
                _log_callback,
            ):
                return items

            executor.discover_work_items = reuse_discovered_items
            try:
                return executor.run_noise_job(
                    config,
                    cancel_event=cancel_event,
                    log_callback=callbacks.log,
                    started_callback=callbacks.started,
                    progress_callback=callbacks.progress,
                    preview_callback=callbacks.preview,
                )
            finally:
                executor.discover_work_items = original_discover
