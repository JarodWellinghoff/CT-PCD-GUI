from dataclasses import dataclass
from enum import Enum, auto


class NoiseJobPhase(Enum):
    IDLE = auto()
    VALIDATING = auto()
    INSPECTING = auto()
    RUNNING = auto()
    CANCELLING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass(frozen=True, slots=True)
class NoiseJobDraft:
    input_path: str
    output_dir: str
    input_mode: str
    mas_factor: float
    electronic_noise: float
    seed: int | None
    file_suffix: str
    overwrite_existing: bool
    recursive: bool
    continue_on_error: bool
    preview_interval: int
    parallel_mode: str
    max_workers: int


@dataclass(frozen=True, slots=True)
class NoiseViewState:
    phase: NoiseJobPhase = NoiseJobPhase.IDLE
    status: str = "Ready"
    completed_units: int = 0
    total_units: int = 0
    inputs_enabled: bool = True
    start_enabled: bool = True
    cancel_enabled: bool = False
