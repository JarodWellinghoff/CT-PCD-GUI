from __future__ import annotations

import threading

from .models import JobSummary, NoiseJobConfig
from .ports import NoiseJobCallbacks, NoiseJobExecutor
from .validation import validate_config


class RunNoiseInsertion:
    def __init__(self, executor: NoiseJobExecutor) -> None:
        self._executor = executor

    def execute(
        self,
        config: NoiseJobConfig,
        *,
        cancel_event: threading.Event,
        callbacks: NoiseJobCallbacks,
    ) -> JobSummary:
        validate_config(config)
        return self._executor.run(
            config,
            cancel_event=cancel_event,
            callbacks=callbacks,
        )
