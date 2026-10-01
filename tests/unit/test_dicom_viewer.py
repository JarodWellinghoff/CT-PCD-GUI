from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from ct_pcd_gui.features.lesion_extraction.presentation.image_view import (
    OverlayImageView,
)
from ct_pcd_gui.features.lesion_extraction.presentation.workspace import (
    LesionExtractionWorkspace,
)
from ct_pcd_gui.features.lesion_insertion.presentation.slice_view import SliceView
from ct_pcd_gui.features.lesion_insertion.presentation.workspace import (
    LesionInsertionWorkspace,
)
from ct_pcd_gui.shared.qt.dicom_viewer import (
    CUSTOM_WINDOW_PRESET,
    DicomImageView,
    DicomSliceViewer,
    window_hu_to_uint8,
)


def test_window_hu_to_uint8_uses_requested_window() -> None:
    values = np.asarray([[-200.0, 0.0, 200.0]], dtype=np.float32)

    result = window_hu_to_uint8(values, width=400.0, level=0.0)

    np.testing.assert_array_equal(result, np.asarray([[0, 128, 255]], dtype=np.uint8))
    assert result.flags.c_contiguous


def test_slice_controls_use_one_based_display_and_zero_based_signal(qtbot) -> None:
    viewer = DicomSliceViewer()
    qtbot.addWidget(viewer)
    changed: list[int] = []
    viewer.slice_changed.connect(changed.append)

    viewer.set_slice_count(5, initial_index=2, emit=False)

    assert viewer.current_slice == 2
    assert viewer.slice_spin.value() == 3
    assert viewer.slice_position_label.text() == "3 / 5"

    viewer.slice_spin.setValue(5)
    assert viewer.current_slice == 4
    assert changed[-1] == 4

    viewer.step_slice(-10)
    assert viewer.current_slice == 0
    assert changed[-1] == 0


def test_window_presets_and_reset_preserve_last_named_preset(qtbot) -> None:
    viewer = DicomSliceViewer()
    qtbot.addWidget(viewer)

    viewer.preset_combo.setCurrentText("Lung")
    assert viewer.window_width == pytest.approx(1500.0)
    assert viewer.window_level == pytest.approx(-600.0)

    viewer.set_window(333.0, 12.0)
    assert viewer.preset_combo.currentText() == CUSTOM_WINDOW_PRESET

    viewer.reset_view()
    assert viewer.window_width == pytest.approx(1500.0)
    assert viewer.window_level == pytest.approx(-600.0)
    assert viewer.preset_combo.currentText() == "Lung"


def test_feature_viewers_extend_the_shared_canvas(qtbot) -> None:
    insertion_view = SliceView()
    extraction_view = OverlayImageView()
    qtbot.addWidget(insertion_view)
    qtbot.addWidget(extraction_view)

    assert isinstance(insertion_view, DicomImageView)
    assert isinstance(extraction_view, DicomImageView)


def test_reconstructed_dicom_workspaces_share_the_standard_controls(qtbot) -> None:
    insertion = LesionInsertionWorkspace()
    extraction = LesionExtractionWorkspace()
    qtbot.addWidget(insertion)
    qtbot.addWidget(extraction)

    assert isinstance(insertion.viewer, DicomSliceViewer)
    assert isinstance(extraction.viewer, DicomSliceViewer)
    assert insertion.viewer.preset_combo.itemText(0) == "Soft tissue"
    assert extraction.viewer.preset_combo.itemText(0) == "Soft tissue"
    assert insertion.viewer.fit_action.text() == extraction.viewer.fit_action.text()
    assert insertion.viewer.reset_action.text() == extraction.viewer.reset_action.text()
