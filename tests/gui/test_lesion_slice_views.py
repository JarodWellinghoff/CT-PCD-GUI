"""Offscreen interaction tests. VTK is stubbed only in workspace-layout tests."""
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QBoxLayout, QWidget

from ct_pcd_gui.features.lesion_viewer.domain.inspection import SliceChannel
from ct_pcd_gui.features.lesion_viewer.domain.models import LesionFileData
from ct_pcd_gui.features.lesion_viewer.presentation.slice_view import LesionSliceViews


@pytest.fixture
def slices(qtbot):
    widget = LesionSliceViews()
    qtbot.addWidget(widget)
    widget.resize(800, 800)
    widget.show()
    values = np.arange(5 * 4 * 6, dtype=np.float64).reshape(5, 4, 6) - 1000.123456789
    mask = np.zeros_like(values, dtype=bool); mask[:, 1:3, 1:4] = True
    widget.set_channels([SliceChannel("Low", values, mask), SliceChannel("High", values + 100, mask)])
    return widget


def test_both_navigation_controls_and_wheel_signal_sync(slices):
    assert len(slices.viewers) == 2
    slices.viewers[0].slice_slider.setValue(3)
    assert [v.current_slice for v in slices.viewers] == [3, 3]
    slices.viewers[1].slice_spin.setValue(2)
    assert [v.current_slice for v in slices.viewers] == [1, 1]
    slices.viewers[1].image_view.slice_delta_requested.emit(1)
    assert [v.current_slice for v in slices.viewers] == [2, 2]
    slices.set_slice(-100)
    assert [v.current_slice for v in slices.viewers] == [0, 0]
    slices.set_slice(100)
    assert [v.current_slice for v in slices.viewers] == [4, 4]


def test_stacked_layout_and_single_channel_reload(slices):
    assert slices.views_layout.direction() == QBoxLayout.Direction.TopToBottom
    assert slices.views_layout.indexOf(slices._boxes[0]) == 0
    assert slices.views_layout.indexOf(slices._boxes[1]) == 1
    slices.set_channels([SliceChannel("Single", np.zeros((1, 4, 5)))])
    assert len(slices.viewers) == 1
    assert slices.viewers[0].slice_count == 1
    assert slices.viewers[0].current_slice == 0
    assert not slices.outline.isEnabled()


def test_raw_probe_survives_window_level_and_outline_changes(slices):
    before = [channel.values.copy() for channel in slices.channels]
    slices._show_probe(2, 1)
    text = [label.text() for label in slices._readouts]
    assert str(before[0][slices.current_slice, 1, 2]) in text[0]
    assert str(before[1][slices.current_slice, 1, 2]) in text[1]
    slices.viewers[0].set_window(1800, -600)
    assert [label.text() for label in slices._readouts] == text
    slices.outline.setChecked(False)
    assert all(v.image_view.overlay.pixmap().isNull() for v in slices.viewers)
    slices.outline.setChecked(True)
    assert all(not v.image_view.overlay.pixmap().isNull() for v in slices.viewers)
    for channel, original in zip(slices.channels, before):
        np.testing.assert_array_equal(channel.values, original)


def test_probe_follows_slice_and_axis_order(slices):
    slices._show_probe(2, 1)
    slices.set_slice(4)
    assert "Slice 5" in slices._readouts[0].text()
    slices.set_geometry("YXZ", (0.6, 0.8, 1.2))
    assert [v.slice_count for v in slices.viewers] == [6, 6]
    assert slices.viewers[0].image_view.pixmap_item.transform().m11() == 0.6
    assert slices.viewers[0].image_view.pixmap_item.transform().m22() == 0.8
    slices._show_probe(-1, -1)
    assert "Hover" in slices._readouts[0].text()


def test_clear_releases_images_and_controls(slices):
    slices.clear()
    assert not slices.channels and not slices.viewers and not slices._volumes
    assert not slices.outline.isEnabled()
    slices.set_slice(4)  # No stale volume access after switching files.


def test_mismatched_channel_shapes_are_rejected(slices):
    slices.set_channels([SliceChannel("a", np.zeros((2, 3, 4))),
                         SliceChannel("b", np.zeros((3, 3, 4)))])
    assert not slices.viewers
    assert "different shapes" in slices.notice.text()


@pytest.fixture
def workspace(qtbot, monkeypatch, tmp_path):
    from ct_pcd_gui.features.lesion_viewer.presentation import workspace as module

    class Scene(QWidget):
        surface = None
        def clear_surface(self): self.surface = None
        def set_surface(self, data, **kwargs): self.surface = data
        def apply_appearance(self, state): pass
        def cleanup(self): pass

    monkeypatch.setattr(module, "LesionVtkScene", Scene)
    # Avoid reading/writing the developer's persisted layout preference.
    monkeypatch.setattr(module, "QSettings", lambda: QSettings(str(tmp_path / "prefs.ini"), QSettings.Format.IniFormat))
    widget = module.LesionViewerWorkspace()
    qtbot.addWidget(widget)
    widget.resize(1200, 900)
    widget.show()
    return widget


@pytest.mark.parametrize("mode,show_2d,show_3d", [("both", True, True), ("2d", True, False), ("3d", False, True)])
def test_layout_modes_and_screenshot(workspace, tmp_path, mode, show_2d, show_3d):
    workspace.mode_combo.setCurrentIndex(workspace.mode_combo.findData(mode))
    assert workspace.slices.isVisible() == show_2d
    assert workspace.stack.isVisible() == show_3d
    assert workspace.show_3d == show_3d
    expected = (QBoxLayout.Direction.LeftToRight if mode == "2d"
                else QBoxLayout.Direction.TopToBottom)
    assert workspace.slices.views_layout.direction() == expected
    path = workspace.save_screenshot(tmp_path / f"{mode}.png")
    assert path.is_file() and path.stat().st_size > 0


def test_metadata_and_file_switch_clear_stale_state(workspace):
    values = np.zeros((3, 4, 5))
    data = LesionFileData(Path("synthetic.npz"), [], file_metadata={
        "slice_channels": [SliceChannel("Low", values)],
        "members": {"LesionMeanHU": -100.5, "DicomHeaderJSON": {"PixelSpacing": [0.7, 0.8]}},
    })
    workspace.set_file_data(data)
    assert "LesionMeanHU" in workspace.metadata.toPlainText()
    assert "PixelSpacing" in workspace.metadata.toPlainText()
    assert "slice_channels" not in workspace.metadata.toPlainText()
    workspace.set_surface(object(), reset_camera=True)
    workspace.clear_file()
    assert workspace.surface is None and not workspace.slices.viewers
    assert not workspace.metadata.toPlainText()


@pytest.mark.parametrize("side_by_side", [True, False])
def test_channel_layout_preserves_viewers_and_state(slices, side_by_side):
    viewers = list(slices.viewers)
    slices.set_slice(3)
    viewers[0].set_window(700, -150)
    viewers[1].set_window(1400, -600)
    slices.outline.setChecked(False)
    # Start in the opposite layout so each case exercises an actual reflow.
    slices.set_side_by_side(not side_by_side)
    slices.set_side_by_side(side_by_side)

    expected = (QBoxLayout.Direction.LeftToRight if side_by_side
                else QBoxLayout.Direction.TopToBottom)
    assert slices.views_layout.direction() == expected
    assert all(actual is original for actual, original in zip(slices.viewers, viewers))
    assert slices.current_slice == 3
    assert [v.current_slice for v in slices.viewers] == [3, 3]
    assert [(v.window_width, v.window_level) for v in viewers] == [(700, -150), (1400, -600)]
    assert not slices.outline.isChecked()
    viewers[1].slice_spin.setValue(2)
    assert [v.current_slice for v in viewers] == [1, 1]


def test_side_by_side_layout_survives_channel_reload(slices):
    slices.set_side_by_side(True)
    slices.set_channels([SliceChannel("Single", np.zeros((1, 4, 5)))])
    assert slices.views_layout.direction() == QBoxLayout.Direction.LeftToRight
    assert len(slices.viewers) == 1
    assert slices.views_layout.stretch(0) == 1
    slices.set_channels([SliceChannel("Low", np.zeros((3, 4, 5))),
                         SliceChannel("High", np.ones((3, 4, 5)))])
    assert slices.views_layout.direction() == QBoxLayout.Direction.LeftToRight
    assert len(slices.viewers) == 2
    assert [slices.views_layout.stretch(i) for i in range(2)] == [1, 1]


def test_workspace_mode_switch_reflows_channels_and_preserves_state(workspace, qtbot):
    values = np.zeros((5, 4, 6))
    workspace.slices.set_channels([SliceChannel("Low", values), SliceChannel("High", values)])
    viewers = list(workspace.slices.viewers)
    workspace.slices.set_slice(3)
    viewers[0].set_window(700, -150)
    viewers[1].set_window(1400, -600)
    workspace.slices.outline.setChecked(False)

    for mode in ("2d", "both", "3d", "2d"):
        workspace.mode_combo.setCurrentIndex(workspace.mode_combo.findData(mode))
        expected = (QBoxLayout.Direction.LeftToRight if mode == "2d"
                    else QBoxLayout.Direction.TopToBottom)
        assert workspace.slices.views_layout.direction() == expected
        assert all(actual is original for actual, original in zip(workspace.slices.viewers, viewers))
        assert workspace.slices.current_slice == 3
        assert [(v.window_width, v.window_level) for v in viewers] == [(700, -150), (1400, -600)]
        assert not workspace.slices.outline.isChecked()
        assert workspace.settings.value("lesion_viewer/view_mode") == mode
        if mode == "3d":
            continue
        first, second = workspace.slices._boxes
        if mode == "2d":
            qtbot.waitUntil(lambda: first.geometry().right() < second.geometry().left())
            assert first.y() == second.y()
        else:
            qtbot.waitUntil(lambda: first.geometry().bottom() < second.geometry().top())
            assert first.x() == second.x()

    viewers[1].image_view.slice_delta_requested.emit(1)
    assert [v.current_slice for v in viewers] == [4, 4]


def test_saved_2d_preference_applies_before_channels_are_loaded(workspace, qtbot):
    workspace.settings.setValue("lesion_viewer/view_mode", "2d")
    restored = type(workspace)()
    qtbot.addWidget(restored)
    assert restored.mode_combo.currentData() == "2d"
    assert restored.slices.views_layout.direction() == QBoxLayout.Direction.LeftToRight
    values = np.zeros((3, 4, 5))
    restored.slices.set_channels([SliceChannel("Low", values), SliceChannel("High", values)])
    assert restored.slices.views_layout.direction() == QBoxLayout.Direction.LeftToRight
    assert len(restored.slices.viewers) == 2
