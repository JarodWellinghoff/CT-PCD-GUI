"""Qt smoke tests; no FreeCT executable or patient data is required."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from ct_pcd_gui.features.freect_reconstruction.models import ReconstructionResult  # noqa: E402
from ct_pcd_gui.features.freect_reconstruction.presentation import presenter as presenters  # noqa: E402
from ct_pcd_gui.features.freect_reconstruction.presentation.module import (  # noqa: E402
    build_freect_reconstruction_module,
)
from ct_pcd_gui.features.freect_reconstruction.presentation.view import (  # noqa: E402
    ReconstructionImageView, ReconstructionPanel,
)
from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner  # noqa: E402


@pytest.fixture
def module(qtbot):
    runner = QtTaskRunner()
    descriptor = build_freect_reconstruction_module(runner)
    qtbot.addWidget(descriptor.panel)
    qtbot.addWidget(descriptor.workspace)
    yield descriptor
    descriptor.request_cancel()
    qtbot.waitUntil(lambda: not descriptor.is_busy(), timeout=6000)


def test_module_initial_state_and_lifecycle(module):
    assert module.module_id == "freect-reconstruction"
    assert module.display_name == "FreeCT Reconstruction"
    assert not module.is_busy()
    assert module.panel.prepare_button.isEnabled()
    assert not module.panel.run_button.isEnabled()
    assert not module.panel.cancel_button.isEnabled()
    module.cleanup()


def test_fast_worker_signals_are_not_lost(module, qtbot):
    def operation(context):
        context.report("fast task finished", 100)
        return None

    module.owner._start(operation)
    qtbot.waitUntil(lambda: not module.is_busy(), timeout=6000)
    assert "fast task finished" in module.workspace.log.toPlainText()
    assert module.workspace.progress.value() == 100
    assert not module.workspace.config.isReadOnly()


def test_cancel_stays_busy_until_worker_finishes(module, qtbot):
    def operation(context):
        context.cancel_event.wait(3)
        context.check()

    module.owner._start(operation)
    assert module.is_busy()
    assert not module.panel.inputs.isEnabled()
    module.request_cancel()
    qtbot.waitUntil(lambda: not module.is_busy(), timeout=6000)
    assert module.panel.inputs.isEnabled()
    assert "cancel" in module.workspace.status.text().lower()
    assert not module.panel.run_button.isEnabled()


@pytest.mark.parametrize("hu", [False, True])
def test_output_view_uses_hu_presets_only_for_hu(module, monkeypatch, tmp_path, hu):
    monkeypatch.setattr(presenters, "read_slice", lambda path, index: np.ones((16, 32)))
    result = ReconstructionResult(tmp_path, tmp_path / "image.nii", (3, 16, 32), hu, 0.0, 0.5)
    module.owner.succeeded(result)
    viewer = module.workspace.viewer
    assert viewer.slice_count == 3
    assert viewer.preset_combo.isEnabled() is hu
    assert "patient orientation unknown" in viewer.info_label.text()
    assert module.workspace.image_view.pixel_aspect == 0.5


def test_canvas_preserves_non_square_voxel_aspect(qtbot):
    canvas = ReconstructionImageView()
    qtbot.addWidget(canvas)
    canvas.pixel_aspect = 0.5
    canvas.set_array(np.zeros((100, 200), dtype=np.uint8))
    assert canvas.sceneRect().width() == 100
    assert canvas.sceneRect().height() == 100


def test_panel_does_not_silently_clamp_generated_kernel(qtbot):
    panel = ReconstructionPanel()
    qtbot.addWidget(panel)
    config = {
        "ReconVolume": {key: widget.value() for key, widget in panel.volume_controls.items()},
        "ReconSpecific": {"KernelA": 0.5, "KernelC": float("nan")},
        "ImageOutput": {"RescaleToHU": False},
    }
    with pytest.raises(ValueError, match="KernelC"):
        panel.load_settings(config)
