from __future__ import annotations

import pickle
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from ct_pcd_gui.features.noise_insertion.application.models import (
    DicomWorkItem,
    JobSummary,
    MasFactor,
    NoiseJobConfig,
)
from ct_pcd_gui.features.noise_insertion.application.ports import NoiseJobCallbacks
from ct_pcd_gui.features.noise_insertion.application.validation import validate_config
from ct_pcd_gui.features.noise_insertion.domain.noise_model import (
    add_poisson_noise_log_domain,
)
from ct_pcd_gui.features.noise_insertion.infrastructure import executor_adapter
from ct_pcd_gui.features.noise_insertion.infrastructure.dicom_io import (
    TAG_LARGEST,
    TAG_PIXEL_REPRESENTATION,
    TAG_SMALLEST,
    TAG_TUBE_CURRENT,
    prepare_multiframe_output,
    prepare_single_frame_output,
    simulated_series_description,
)


class _OnesRng:
    def standard_normal(self, size: tuple[int, ...]) -> np.ndarray:
        return np.ones(size, dtype=np.float64)


def _base_dataset() -> Dataset:
    dataset = Dataset()
    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    dataset.SOPInstanceUID = "1.2.3.4.5"
    dataset.BitsAllocated = 16
    dataset.RescaleIntercept = -1.0
    dataset.RescaleSlope = 0.001
    return dataset


def test_v2_noise_formula_uses_fine_tune_factor_and_returns_float32() -> None:
    attenuation = np.zeros((2, 3), dtype=np.float32)
    incident = np.array([100.0, 100.0], dtype=np.float32)
    mas_factor = 0.25
    fine_tune_factor = 2.0
    electronic_noise = 3.0

    actual = add_poisson_noise_log_domain(
        attenuation,
        incident,
        mas_factor,
        electronic_noise=electronic_noise,
        rng=_OnesRng(),
        fine_tune_factor=fine_tune_factor,
    )

    n1_detected = 100.0
    n2_detected = n1_detected * mas_factor * fine_tune_factor
    variance = (1.0 / n2_detected - 1.0 / n1_detected) * (
        1.0
        + electronic_noise / n2_detected
        + electronic_noise / n1_detected
    )
    expected = np.full(attenuation.shape, np.sqrt(variance), dtype=np.float32)

    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_config_carries_v2_calibration_through_process_pickling() -> None:
    config = NoiseJobConfig(
        input_path="input",
        output_dir="output",
        mas_factor=0.25,
        fine_tune_factor=1.7,
    )
    restored = pickle.loads(pickle.dumps(config))

    assert isinstance(restored.mas_factor, MasFactor)
    assert float(restored.mas_factor) == 0.25
    assert restored.mas_factor.fine_tune_factor == 1.7

    actual = add_poisson_noise_log_domain(
        np.zeros((1, 1), dtype=np.float32),
        np.array([100.0], dtype=np.float32),
        restored.mas_factor,
        rng=_OnesRng(),
    )
    expected_variance = 1.0 / (100.0 * 0.25 * 1.7) - 1.0 / 100.0
    np.testing.assert_allclose(actual, np.sqrt(expected_variance), rtol=1e-6)


def test_validation_rejects_non_positive_fine_tune_factor(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="Fine-tune factor"):
        validate_config(
            NoiseJobConfig(
                input_path=str(input_dir),
                output_dir=str(tmp_path / "output"),
                fine_tune_factor=0.0,
            )
        )


def test_multiframe_output_matches_v2_header_rules() -> None:
    dataset = _base_dataset()
    dataset.SeriesDescription = "PCD FD"
    dataset.PixelRepresentation = 1
    dataset.add_new(TAG_SMALLEST, "SS", -5)
    dataset.add_new(TAG_LARGEST, "SS", 5)
    dataset.add_new(TAG_TUBE_CURRENT, "IS", 200)

    first_group = Dataset()
    first_group.InstanceNumber = 101
    second_group = Dataset()
    second_group.InstanceNumber = 102
    dataset.PerFrameFunctionalGroupsSequence = Sequence(
        [first_group, second_group]
    )

    encoded = np.array(
        [
            [[2, 3], [4, 5]],
            [[6, 7], [8, 9]],
        ],
        dtype=np.uint16,
    )
    original_uid = str(dataset.SOPInstanceUID)

    output = prepare_multiframe_output(
        dataset,
        encoded,
        intercept=-1.0,
        slope=0.001,
        mas_factor=0.25,
        had_root_smallest=True,
        had_root_largest=True,
    )

    assert output.SeriesDescription == "PCD SIM 25pct"
    assert int(output.PixelRepresentation) == 1
    assert int(output[TAG_SMALLEST].value) == 2
    assert int(output[TAG_LARGEST].value) == 9
    assert output[TAG_SMALLEST].VR == "US"
    assert output[TAG_LARGEST].VR == "US"
    assert int(output[TAG_TUBE_CURRENT].value) == 50
    assert int(output.PerFrameFunctionalGroupsSequence[0].InstanceNumber) == 101
    assert int(output.PerFrameFunctionalGroupsSequence[1].InstanceNumber) == 102
    assert str(output.SOPInstanceUID) != original_uid
    assert float(output.RescaleIntercept) == -1.0
    assert float(output.RescaleSlope) == 0.001


def test_output_does_not_add_root_layout_tags_missing_from_input() -> None:
    dataset = _base_dataset()
    dataset.SeriesDescription = "PCD FD"
    encoded = np.array([[1, 2], [3, 4]], dtype=np.uint16)

    output = prepare_single_frame_output(
        dataset,
        encoded,
        intercept=-1.0,
        slope=0.001,
        mas_factor=0.25,
    )

    assert TAG_PIXEL_REPRESENTATION not in output
    assert TAG_SMALLEST not in output
    assert TAG_LARGEST not in output
    assert output.SeriesDescription == "PCD SIM 25pct"


def test_shared_series_description_is_carried_by_mas_factor() -> None:
    factor = MasFactor(0.25, 1.0, "First SIM 25pct")
    dataset = _base_dataset()
    dataset.SeriesDescription = "Different FD"

    output = prepare_single_frame_output(
        dataset,
        np.array([[1, 2], [3, 4]], dtype=np.uint16),
        intercept=-1.0,
        slope=0.001,
        mas_factor=factor,
    )

    assert output.SeriesDescription == "First SIM 25pct"


def test_executor_adapter_uses_first_input_description_and_live_event(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cancel_event = threading.Event()
    item = DicomWorkItem(
        input_path=tmp_path / "first.dcm",
        output_path=tmp_path / "output.dcm",
        frame_count=1,
        is_multiframe=False,
    )
    observed: dict[str, object] = {}

    def discover(config, received_cancel_event, log_callback):
        observed["discovery_cancel_event"] = received_cancel_event
        return [item]

    def run_noise_job(config, **kwargs):
        observed["run_config"] = config
        observed["run_cancel_event"] = kwargs["cancel_event"]
        assert executor_adapter.executor.discover_work_items(
            config,
            object(),
            kwargs["log_callback"],
        ) == [item]
        return JobSummary()

    monkeypatch.setattr(executor_adapter.executor, "discover_work_items", discover)
    monkeypatch.setattr(executor_adapter.executor, "run_noise_job", run_noise_job)
    monkeypatch.setattr(
        executor_adapter,
        "dcmread",
        lambda *_args, **_kwargs: SimpleNamespace(SeriesDescription="Study FD"),
    )

    messages: list[str] = []
    callbacks = NoiseJobCallbacks(
        log=messages.append,
        started=lambda _total, _files, _frames: None,
        progress=lambda _completed, _total, _status: None,
        preview=lambda _payload: None,
    )
    config = NoiseJobConfig(
        input_path=str(tmp_path),
        output_dir=str(tmp_path / "out"),
        fine_tune_factor=1.2,
    )

    summary = executor_adapter.DefaultNoiseJobExecutor().run(
        config,
        cancel_event=cancel_event,
        callbacks=callbacks,
    )

    run_config = observed["run_config"]
    assert isinstance(run_config, NoiseJobConfig)
    assert run_config.mas_factor.series_description == "Study SIM 25pct"
    assert run_config.mas_factor.fine_tune_factor == 1.2
    assert observed["discovery_cancel_event"] is cancel_event
    assert observed["run_cancel_event"] is cancel_event
    assert summary == JobSummary()
    assert any("fine-tune factor=1.2" in message for message in messages)


def test_existing_simulated_series_description_is_preserved() -> None:
    dataset = Dataset()
    dataset.SeriesDescription = "PCD SIM 50pct"

    assert simulated_series_description(dataset, 0.25) == "PCD SIM 50pct"
