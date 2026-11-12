"""Contract tests for the adapter; these do not validate FreeCT reconstruction physics."""
from __future__ import annotations

import copy
import gzip
import struct
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from ct_pcd_gui.features.freect_reconstruction.config import (
    bind_paths, dump_config, load_config, validate_config,
)
from ct_pcd_gui.features.freect_reconstruction.images import (
    normalize_volume, read_header, read_slice, slice_positions,
)
from ct_pcd_gui.features.freect_reconstruction.models import (
    JobContext, PreparedJob, ReconstructionCancelled,
)
from ct_pcd_gui.features.freect_reconstruction.process import resolve_executable, run_process
from ct_pcd_gui.features.freect_reconstruction import service


@pytest.fixture
def context():
    return JobContext(threading.Event(), lambda event: None)


@pytest.fixture
def config():
    return {
        "RawInput": {"Battelle": "/old/input", "Binary": "", "Ptr": "", "Ctd": ""},
        "ScanGeometry": {
            "Manufacturer": "test", "DetectorRows": 16, "DetectorChannels": 32,
            "DetectorTransverseSpacing": 1.0, "DetectorAxialSpacing": 1.0,
            "DetectorShape": "CYLINDRICAL", "DistSourceToDetector": 1000.0,
            "DistSourceToIsocenter": 500.0, "DetectorCentralRow": 8.5,
            "DetectorCentralChannel": 16.5, "ProjectionGeometry": "FANBEAM",
            "ScanType": "HELICAL", "ProjectionsPerRotation": 40,
            "FlyingFocalSpotMode": "FFSNONE", "StartTablePosition": -10.0,
            "EndTablePosition": 10.0, "TableFeedPerRotation": 10.0,
            "AcquisitionFieldOfView": 300.0, "CollimatedSliceWidth": 0.5,
            "HUCalibrationFactor": 0.02, "NumProjectionsTotal": 80,
        },
        "Preprocessing": {
            "CorrectMATLABIndexing": True, "FlipDetectorChannels": True,
            "FlipDetectorRows": True, "NegateTubeAngle": False, "AdapativeFiltration": 0.0,
        },
        "ReconVolume": {
            "StartPosition": 0.0, "EndPosition": 1.0, "SlicePitch": 0.5,
            "SliceThickness": 2.0, "FieldOfView": 256.0, "Nx": 16, "Ny": 16,
            "XOrigin": 0.0, "YOrigin": 0.0,
        },
        "ReconSpecific": {"KernelA": 0.5, "KernelC": 0.5},
        "ImageOutput": {"RescaleToHU": True, "AngleOffset": 0.0, "DICOM": "",
                        "NIfTI1": "", "Binary": "", "Png": ""},
    }


def write_native(path: Path, shape=(3, 16, 16), *, invalid=None, truncated=False):
    header = bytearray(352)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, 3, shape[2], shape[1], shape[0], 0, 0, 0, 0)
    struct.pack_into("<2h", header, 70, 16, 32)
    struct.pack_into("<4f", header, 76, 0, 16, 16, 2.0)
    struct.pack_into("<3f", header, 108, 352.0, 1.0, 0.0)
    struct.pack_into("<2h", header, 252, 1, 0)
    header[344:348] = b"n+2\0"  # Exact defect in the supplied source.
    values = np.arange(np.prod(shape), dtype="<f4").reshape(shape)
    if invalid is not None:
        values[0, 0, 0] = invalid
    with gzip.open(path, "wb") as stream:
        stream.write(header)
        data = values.tobytes()
        stream.write(data[:-8] if truncated else data)
    return values


def test_config_round_trip(config):
    assert load_config(dump_config(config)) == config
    validate_config(config, 80)


def test_duplicate_yaml_key_rejected(config):
    text = dump_config(config).replace("  KernelA: 0.5", "  KernelA: 0.5\n  KernelA: 1.0")
    with pytest.raises(ValueError, match="duplicate"):
        load_config(text)


def test_unsafe_yaml_rejected():
    with pytest.raises(ValueError):
        load_config("!!python/object/apply:os.system ['echo unsafe']")


def test_unknown_field_rejected(config):
    config["ReconVolume"]["SliceSpacing"] = 1
    with pytest.raises(ValueError, match="Unknown ReconVolume"):
        load_config(dump_config(config))


@pytest.mark.parametrize("section,key,value", [
    ("ReconVolume", "Nx", 16.0), ("ReconVolume", "Nx", True),
    ("ReconVolume", "SlicePitch", 0), ("ReconVolume", "SliceThickness", -1),
    ("ReconVolume", "StartPosition", float("nan")),
    ("ReconSpecific", "KernelC", 0), ("ReconSpecific", "KernelA", 0.8),
    ("ScanGeometry", "HUCalibrationFactor", 0),
    ("ScanGeometry", "FlyingFocalSpotMode", "FFSZ"),
    ("ScanGeometry", "NumProjectionsTotal", 79),
    ("ScanGeometry", "DetectorRows", 8),
    ("ScanGeometry", "DistSourceToDetector", 400),
    ("ScanGeometry", "ProjectionsPerRotation", 41),
    ("Preprocessing", "FlipDetectorRows", "false"),
    ("Preprocessing", "AdapativeFiltration", 1.0),
    ("ImageOutput", "AngleOffset", float("inf")),
    ("ImageOutput", "RescaleToHU", "true"),
])
def test_invalid_config_values(config, section, key, value):
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_config(config, 80)


def test_integer_kernels_are_emitted_as_go_float64(config):
    config["ReconSpecific"] = {"KernelA": 1, "KernelC": 1}
    validate_config(config, 80)
    assert "KernelA: 1.0" in dump_config(config)
    assert "KernelC: 1.0" in dump_config(config)


def test_attenuation_does_not_require_hu_calibration(config):
    config["ImageOutput"]["RescaleToHU"] = False
    config["ScanGeometry"]["HUCalibrationFactor"] = 0.0
    validate_config(config, 80)
    config["ImageOutput"]["DICOM"] = "somewhere"
    with pytest.raises(ValueError, match="requires HU"):
        validate_config(config, 80)


def test_managed_paths_are_rebound_without_mutating_input(config, tmp_path):
    original = copy.deepcopy(config)
    config["RawInput"]["Binary"] = "/untrusted"
    config["ImageOutput"]["Binary"] = "/untrusted/output"
    bound = bind_paths(config, tmp_path / "projections", tmp_path, True)
    assert bound["RawInput"]["Binary"] == ""
    assert bound["RawInput"]["Battelle"] == str(tmp_path / "projections")
    assert bound["ImageOutput"]["Binary"] == ""
    assert bound["ImageOutput"]["NIfTI1"] == str(tmp_path / "freect_native.nii.gz")
    assert config["ImageOutput"]["Binary"] == "/untrusted/output"
    assert bound["ScanGeometry"] == original["ScanGeometry"]


def test_descending_and_single_slice_ranges(config):
    volume = config["ReconVolume"]
    volume.update(StartPosition=1, EndPosition=0, SlicePitch=0.5)
    assert slice_positions(volume) == [1, 0.5, 0]
    volume.update(EndPosition=1)
    assert slice_positions(volume) == [1]


def test_normalizes_exact_uploaded_nifti_defects(config, context, tmp_path):
    native = tmp_path / "native.nii.gz"
    values = write_native(native)
    original = native.read_bytes()
    target = tmp_path / "reconstruction.nii"
    header = normalize_volume(native, target, config["ReconVolume"], context)
    assert header.shape == (3, 16, 16)
    assert native.read_bytes() == original
    raw = target.read_bytes()
    assert raw[344:348] == b"n+1\0"
    assert struct.unpack_from("<2h", raw, 252) == (0, 0)
    assert struct.unpack_from("<f", raw, 88)[0] == 0.5  # spacing, NOT thickness 2.0
    np.testing.assert_array_equal(read_slice(target, 1), values[1])
    with pytest.raises(IndexError):
        read_slice(target, 3)


@pytest.mark.parametrize("invalid,truncated", [(float("nan"), False), (float("inf"), False), (None, True)])
def test_bad_voxels_or_truncation_never_publish_output(config, context, tmp_path, invalid, truncated):
    native, target = tmp_path / "native.nii.gz", tmp_path / "reconstruction.nii"
    write_native(native, invalid=invalid, truncated=truncated)
    with pytest.raises(ValueError):
        normalize_volume(native, target, config["ReconVolume"], context)
    assert not target.exists()
    assert not target.with_suffix(".nii.part").exists()


def test_wrong_output_dimensions_fail(config, context, tmp_path):
    native = tmp_path / "native.nii.gz"
    write_native(native, shape=(2, 16, 16))
    with pytest.raises(ValueError, match="shape"):
        normalize_volume(native, tmp_path / "output.nii", config["ReconVolume"], context)


def test_true_nifti2_is_not_mistaken_for_known_magic_bug():
    raw = bytearray(352)
    struct.pack_into("<i", raw, 0, 540)
    raw[344:348] = b"n+2\0"
    with pytest.raises(ValueError, match="348-byte"):
        read_header(raw, allow_freect_magic=True)


def test_process_preserves_argv_and_separates_stdout(context, tmp_path):
    argument = "path with spaces; not a shell command & test"
    path = run_process([sys.executable, "-c", "import sys; print(sys.argv[1]); print('diagnostic',file=sys.stderr)",
                        argument], tmp_path, "test", context, log_stdout=False)
    assert path.read_text().strip() == argument
    assert "diagnostic" in (tmp_path / "test.stderr.log").read_text()


def test_nonzero_process_exit_is_a_failure(context, tmp_path):
    with pytest.raises(RuntimeError, match="code 7"):
        run_process([sys.executable, "-c", "raise SystemExit(7)"], tmp_path, "bad", context)


def test_silent_process_can_be_cancelled(context, tmp_path):
    timer = threading.Timer(0.2, context.cancel_event.set)
    timer.start()
    try:
        with pytest.raises(ReconstructionCancelled):
            run_process([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, "silent", context)
    finally:
        timer.cancel()


def test_process_timeout_terminates_child(context, tmp_path):
    with pytest.raises(TimeoutError):
        run_process([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, "timeout", context,
                    timeout=0.2)


def test_pre_cancel_does_not_launch_or_create_logs(context, tmp_path):
    context.cancel_event.set()
    with pytest.raises(ReconstructionCancelled):
        run_process([sys.executable, "-c", "raise Exception('must not run')"], tmp_path, "cancelled", context)
    assert not list(tmp_path.iterdir())


def test_executable_validation(tmp_path):
    with pytest.raises(ValueError):
        resolve_executable("")
    with pytest.raises(ValueError):
        resolve_executable(str(tmp_path / "no-such-freect"))
    assert resolve_executable(sys.executable).is_file()


@pytest.fixture
def prepared(config, tmp_path):
    directory = tmp_path / "job with spaces"
    stage = directory / "projections"
    stage.mkdir(parents=True)
    for index in range(80):
        (stage / f"projection_{index:09d}.dcm").write_bytes(b"fixture")
    return PreparedJob(directory, Path(sys.executable), 80, dump_config(config))


def fake_reconstruction(command, directory, stem, context, **kwargs):
    assert command[1] == "--no-telemetry"
    assert command[2] == "recon"
    config = load_config(Path(command[3]).read_text())
    write_native(Path(config["ImageOutput"]["NIfTI1"]))
    (directory / f"{stem}.stdout.log").write_text("reconstruction complete")
    (directory / f"{stem}.stderr.log").write_text("")
    return directory / f"{stem}.stdout.log"


def test_service_creates_fresh_attempts_and_checks_output(prepared, context, monkeypatch):
    monkeypatch.setattr(service, "run_process", fake_reconstruction)
    first = service.reconstruct(prepared, prepared.config_text, False, context)
    second = service.reconstruct(prepared, prepared.config_text, False, context)
    assert first.directory != second.directory
    assert first.image_path.exists() and second.image_path.exists()
    assert first.shape == (3, 16, 16)
    assert first.rescaled_to_hu
    import json
    status = json.loads((first.directory / "status.json").read_text())
    assert status["state"] == "completed"
    assert status["patient_orientation"] == "unknown"
    config = load_config((first.directory / "reconstruction.yaml").read_text())
    assert config["RawInput"]["Battelle"] == str(prepared.directory / "projections")
    assert config["ImageOutput"]["DICOM"] == ""


def test_zero_exit_with_engine_error_is_not_success(prepared, context, monkeypatch):
    def error_writer(*args, **kwargs):
        path = fake_reconstruction(*args, **kwargs)
        path.write_text('level=error msg="Could not rescale to HU"')
        return path
    monkeypatch.setattr(service, "run_process", error_writer)
    with pytest.raises(RuntimeError, match="despite exiting zero"):
        service.reconstruct(prepared, prepared.config_text, False, context)
    assert not list(prepared.directory.rglob("reconstruction.nii"))


def test_zero_exit_without_output_is_not_success(prepared, context, monkeypatch):
    def no_writer(command, directory, stem, context):
        for suffix in ("stdout", "stderr"):
            (directory / f"{stem}.{suffix}.log").write_text("")
    monkeypatch.setattr(service, "run_process", no_writer)
    with pytest.raises(ValueError, match="did not create"):
        service.reconstruct(prepared, prepared.config_text, False, context)


def test_native_dicom_missing_files_is_not_success(prepared, context, monkeypatch):
    monkeypatch.setattr(service, "run_process", fake_reconstruction)
    with pytest.raises(ValueError, match="native DICOM"):
        service.reconstruct(prepared, prepared.config_text, True, context)


def test_changed_staged_acquisition_is_rejected(prepared, context):
    (prepared.directory / "projections" / "not-a-projection.txt").write_text("extra")
    with pytest.raises(ValueError, match="Staged projection directory has changed"):
        service.reconstruct(prepared, prepared.config_text, False, context)


def test_advanced_yaml_cannot_override_detector_dimensions(prepared, context):
    config = load_config(prepared.config_text)
    config["ScanGeometry"]["DetectorRows"] = 32
    with pytest.raises(ValueError, match="DetectorRows cannot differ"):
        service.reconstruct(prepared, dump_config(config), False, context)


def test_unsupported_projection_geometry_rejected(config):
    config["ScanGeometry"]["ProjectionGeometry"] = "PARALLEL"
    with pytest.raises(ValueError, match="FANBEAM"):
        validate_config(config, 80)


@pytest.mark.parametrize("axial", [False, True])
def test_prepare_generated_config_contract(tmp_path, context, config, monkeypatch, axial):
    import types

    source = tmp_path / "input"
    source.mkdir()
    output = tmp_path / "output"
    generated = copy.deepcopy(config)
    generated["ReconSpecific"] = {"KernelA": 0.0, "KernelC": 0.0}
    generated["ScanGeometry"]["HUCalibrationFactor"] = 0.0
    if axial:
        generated["ScanGeometry"].update(
            ScanType="AXIAL", StartTablePosition=4.0, EndTablePosition=4.0,
            TableFeedPerRotation=0.0,
        )
        generated["ReconVolume"].update(StartPosition=float("nan"), EndPosition=float("nan"))

    def stage(source, destination, context, spectrum, source_index):
        destination.mkdir()
        for i in range(80):
            (destination / f"{i:03d}.dcm").write_bytes(b"synthetic placeholder")
        return 80

    def generate(argv, directory, stem, context, **kwargs):
        assert argv[1:3] == ["--no-telemetry", "generate-base-config"]
        assert argv[3] == f"--input-battelle={directory / 'projections'}"
        assert kwargs["log_stdout"] is False
        path = directory / f"{stem}.stdout.log"
        path.write_text(dump_config(generated))
        return path

    module = types.ModuleType("ct_pcd_gui.features.freect_reconstruction.dicom")
    module.stage_projections = stage
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(service, "run_process", generate)
    prepared = service.prepare(sys.executable, source, output, context)
    result = load_config(prepared.config_text)
    assert result["ReconSpecific"] == {"KernelA": 0.5, "KernelC": 0.5}
    assert result["ImageOutput"]["RescaleToHU"] is False
    if axial:
        assert result["ReconVolume"]["StartPosition"] == 4.0
        assert result["ReconVolume"]["EndPosition"] == 4.0
    validate_config(result, 80)
    assert prepared.projections == 80
    assert (prepared.directory / "generated-config.yaml").is_file()


def test_gzip_crc_error_does_not_publish_volume(tmp_path, config, context):
    source = tmp_path / "native.nii.gz"
    write_native(source)
    damaged = bytearray(source.read_bytes())
    damaged[-8] ^= 0xFF
    source.write_bytes(damaged)
    target = tmp_path / "normalized.nii"
    with pytest.raises((gzip.BadGzipFile, EOFError, OSError)):
        normalize_volume(source, target, config["ReconVolume"], context)
    assert not target.exists()
    assert not target.with_suffix(".nii.part").exists()
