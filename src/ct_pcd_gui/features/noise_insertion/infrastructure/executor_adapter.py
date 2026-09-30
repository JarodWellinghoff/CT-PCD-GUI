from __future__ import annotations

import os
import threading

from ..application.models import JobSummary, NoiseJobConfig
from ..application.ports import CancellationFlag, NoiseJobCallbacks
from . import executor
from .dicom_io import TUBE_CURRENT_UPDATE_ENV


_RUN_LOCK = threading.Lock()


class DefaultNoiseJobExecutor:
    """Run the established backend with the job's real cancellation context.

    The existing backend accidentally passes the ``CancellationFlag`` protocol
    object to discovery instead of the event supplied by the task runner. This
    adapter keeps the processing implementation intact while supplying the live
    event and propagating the tube-current option to spawned worker processes.
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
            previous_tube_current_setting = os.environ.get(TUBE_CURRENT_UPDATE_ENV)

            def discover_with_job_event(
                discovery_config: NoiseJobConfig,
                _incorrect_flag: object,
                log_callback,
            ):
                return original_discover(
                    discovery_config,
                    cancel_event,
                    log_callback,
                )

            executor.discover_work_items = discover_with_job_event
            os.environ[TUBE_CURRENT_UPDATE_ENV] = (
                "1" if config.update_tube_current else "0"
            )
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
                if previous_tube_current_setting is None:
                    os.environ.pop(TUBE_CURRENT_UPDATE_ENV, None)
                else:
                    os.environ[TUBE_CURRENT_UPDATE_ENV] = (
                        previous_tube_current_setting
                    )
