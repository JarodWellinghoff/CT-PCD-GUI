"""Regression checks for compact canvas sizing and physical-coordinate fitting."""
# ruff: noqa: E402
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
from PySide6.QtCore import QPoint, Qt

from ct_pcd_gui.features.lesion_insertion.application.library_preview import prepare_library_preview
from ct_pcd_gui.features.lesion_insertion.domain.models import LesionLibraryItem
from ct_pcd_gui.features.lesion_insertion.presentation.library_preview import LibraryLesionPreview
from ct_pcd_gui.features.lesion_viewer.presentation.slice_view import HuImageView
from ct_pcd_gui.shared.qt.dicom_viewer import array_to_qimage


def data(channels=2):
    shape = (3, 4, 5, channels)
    model = SimpleNamespace(
        voi_hu=np.arange(np.prod(shape), dtype=float).reshape(shape) - 900,
        mask=np.ones(shape, dtype=bool),
        row_spacing_mm=0.8,
        column_spacing_mm=0.6,
        slice_spacing_mm=1.5,
    )
    item = LesionLibraryItem("a.npz", "L01", "a", "a.npz", channel_count=channels)
    return prepare_library_preview(model, item)


@pytest.fixture
def preview(qtbot):
    widget = LibraryLesionPreview()
    qtbot.addWidget(widget)
    widget.resize(520, 750)
    widget.show()
    widget.set_data(data())
    qtbot.waitUntil(lambda: not widget._layout_timer.isActive())
    yield widget
    widget.cleanup()


def test_image_height_control_shrinks_both_canvases_and_the_tab(preview, qtbot):
    assert [v.image_view.height() for v in preview.slices.viewers] == [140, 140]
    initial = preview.tabs.height()
    preview.image_height.setValue(80)
    qtbot.waitUntil(
        lambda: all(v.image_view.height() == 80 for v in preview.slices.viewers)
        and preview.tabs.height() < initial
        and not preview._layout_timer.isActive()
    )
    assert preview.slices.scroll.isHidden()
    assert preview.slices.scroll.widget() is None
    assert preview.slices.content.parentWidget() is preview.slices


def test_fitted_images_are_top_aligned_without_pixel_changes(preview, qtbot):
    originals = [channel.values.copy() for channel in preview.slices.channels]
    preview.image_height.setValue(96)
    qtbot.waitUntil(lambda: all(v.image_view.height() == 96 for v in preview.slices.viewers))
    for viewer in preview.slices.viewers:
        canvas = viewer.image_view
        assert canvas.alignment() == (
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
        )
        canvas.fit_image()
        top = canvas.mapFromScene(canvas.pixmap_item.sceneBoundingRect().topLeft()).y()
        assert 0 <= top <= 4  # Qt's small fitting margin is not a letterbox band.
    preview.slices.viewers[0].slice_spin.setValue(4)
    preview.slices.viewers[0].set_window(1800, -600)
    preview.slices.outline.setChecked(False)
    assert [v.current_slice for v in preview.slices.viewers] == [3, 3]
    for original, channel in zip(originals, preview.slices.channels, strict=True):
        np.testing.assert_array_equal(original, channel.values)


def test_more_panel_height_does_not_stretch_the_canvases_or_clip_controls(preview, qtbot):
    preview.resize(480, 1100)
    qtbot.waitUntil(lambda: not preview._layout_timer.isActive())
    for viewer, readout in zip(preview.slices.viewers, preview.slices._readouts, strict=True):
        assert viewer.image_view.height() == 140
        for control in (viewer.slice_spin, readout):
            bottom = control.mapTo(preview.slices, QPoint(0, control.height())).y()
            assert bottom <= preview.slices.height()


def test_3d_height_is_independent_of_2d_and_switching_restores_size(preview, qtbot):
    preview.image_height.setValue(80)
    qtbot.waitUntil(lambda: not preview._layout_timer.isActive())
    original_2d_height = preview.tabs.height()
    preview.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: preview.tabs.height() == 300)
    assert not preview.image_height.isEnabled()
    assert preview.scene is None  # No rendering context needed for this test.
    preview.tabs.setCurrentIndex(0)
    qtbot.waitUntil(lambda: preview.tabs.height() == original_2d_height)
    assert preview.image_height.isEnabled()


def test_reloading_keeps_height_and_cleanup_stops_deferred_layout(preview, qtbot):
    preview.image_height.setValue(64)
    preview.set_item(None)
    preview.set_data(data(channels=1))
    qtbot.waitUntil(lambda: not preview._layout_timer.isActive())
    assert len(preview.slices.viewers) == 1
    assert preview.slices.viewers[0].image_view.height() == 64
    preview.image_height.setValue(200)
    preview.cleanup()
    assert not preview._layout_timer.isActive()
    preview._resize_preview()  # A queued callback after cleanup is harmless.


@pytest.mark.parametrize("spacing", [(0.4, 0.8), (2.0, 3.0)])
def test_physical_scene_bounds_are_installed_before_first_fit(qtbot, spacing):
    fitted = []

    class InspectingView(HuImageView):
        def fit_image(self):
            if self.has_image:
                fitted.append((self.sceneRect(), self.pixmap_item.sceneBoundingRect()))
            super().fit_image()

    view = InspectingView()
    qtbot.addWidget(view)
    view.resize(220, 140)
    view.show()
    view.set_spacing(*spacing)
    image = array_to_qimage(np.arange(24, dtype=np.uint8).reshape(4, 6))
    view.set_qimage(image)
    assert fitted
    assert all(scene_rect == image_rect for scene_rect, image_rect in fitted)
    expected = view.pixmap_item.sceneBoundingRect()
    assert expected.width() == pytest.approx(6 * spacing[0])
    assert expected.height() == pytest.approx(4 * spacing[1])
    changed = []
    view.scene().sceneRectChanged.connect(changed.append)
    view.set_qimage(image)  # Slice/WL refresh must never install raw-pixel bounds.
    assert all(rect == expected for rect in changed)
    assert view.sceneRect() == expected
