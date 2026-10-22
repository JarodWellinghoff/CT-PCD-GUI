"""Offscreen preview interactions. Only the rendering scene is replaced in 3-D UI tests."""
# ruff: noqa: E402
import os
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6')
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QBoxLayout, QWidget

from ct_pcd_gui.features.lesion_insertion.application.library_preview import prepare_library_preview
from ct_pcd_gui.features.lesion_insertion.domain.models import LesionLibraryItem
from ct_pcd_gui.features.lesion_insertion.presentation.library_preview import LibraryLesionPreview
from ct_pcd_gui.features.lesion_insertion.presentation.library_preview_controller import (
    LibraryPreviewController,
)


def source(path='a.npz', channels=2):
    model = SimpleNamespace(
        voi_hu=np.arange(3 * 4 * 5 * channels, dtype=float).reshape(3, 4, 5, channels) - 900,
        mask=np.ones((3, 4, 5, channels), dtype=bool),
        row_spacing_mm=0.8, column_spacing_mm=0.6, slice_spacing_mm=1.5,
    )
    item = LesionLibraryItem(path, 'L01', path, path, channel_count=channels)
    return item, model


@pytest.fixture
def preview(qtbot):
    widget = LibraryLesionPreview()
    qtbot.addWidget(widget)
    widget.resize(520, 570)
    widget.show()
    yield widget
    widget.cleanup()


def test_compact_view_reuses_slices_and_preserves_hu(preview):
    item, model = source()
    preview.set_item(item)
    preview.set_data(prepare_library_preview(model, item))
    assert preview.scene is None  # No OpenGL context for 2-D-only inspection.
    assert preview.tabs.currentIndex() == 0
    assert preview.slices.views_layout.direction() == QBoxLayout.Direction.LeftToRight
    assert len(preview.slices.viewers) == 2
    preview.slices.viewers[0].slice_spin.setValue(4)
    assert [v.current_slice for v in preview.slices.viewers] == [3, 3]
    preview.slices._show_probe(2, 1)
    assert str(model.voi_hu[1, 2, 3, 0]) in preview.slices._readouts[0].text()
    old = model.voi_hu.copy()
    preview.slices.viewers[0].set_window(1800, -600)
    preview.slices.outline.setChecked(False)
    np.testing.assert_array_equal(model.voi_hu, old)
    assert 'Mask volume:' in preview.summary.toPlainText()
    assert 'spacing_rcs_mm' in preview.metadata.toPlainText()


def test_clear_and_single_threshold_reload(preview):
    item, model = source()
    preview.set_data(prepare_library_preview(model, item))
    preview.set_item(None)
    assert not preview.slices.viewers
    assert not preview.summary.toPlainText()
    assert not preview.metadata.toPlainText()
    assert not preview.tabs.isTabEnabled(1)
    item, model = source('single.npz', 1)
    preview.set_data(prepare_library_preview(model, item))
    assert len(preview.slices.viewers) == 1


def test_3d_requests_selected_source_only_and_cleans_scene(preview, monkeypatch):
    class Scene(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.surface = None
            self.cleaned = False
        def set_surface(self, polydata, **kwargs):
            self.surface = polydata
        def clear_surface(self):
            self.surface = None
        def reset_camera(self):
            pass
        def cleanup(self):
            self.cleaned = True
    module = ModuleType('ct_pcd_gui.features.lesion_viewer.presentation.vtk_scene')
    module.LesionVtkScene = Scene
    monkeypatch.setitem(sys.modules, module.__name__, module)
    item, model = source()
    preview.set_data(prepare_library_preview(model, item))
    requested = []
    preview.surface_requested.connect(requested.append)
    preview.tabs.setCurrentIndex(1)
    assert requested[-1] == 0
    mesh = object()
    preview.show_surface(0, mesh)
    assert preview.scene.surface is mesh
    preview.surface_channel.setCurrentIndex(1)
    assert requested[-1] == 1
    preview.show_surface_error('Unavailable')
    assert len(preview.slices.viewers) == 2
    preview.cleanup()
    assert preview.scene.cleaned


class Signals(QObject):
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    finished = Signal()


class Handle:
    def __init__(self, operation):
        self.operation = operation
        self.context = SimpleNamespace(signals=Signals())
        self.cancelled = False
    def cancel(self):
        self.cancelled = True


class DeferredRunner:
    def __init__(self):
        self.handles = []
    def start(self, operation):
        handle = Handle(operation)
        self.handles.append(handle)
        return handle


def test_late_success_and_failure_cannot_replace_new_selection(preview, qtbot):
    runner = DeferredRunner()
    controller = LibraryPreviewController(preview, SimpleNamespace(), runner)
    a, model_a = source('a.npz')
    b, model_b = source('b.npz')
    controller.select(a)
    old = runner.handles[-1]
    controller.select(b)
    current = runner.handles[-1]
    assert old.cancelled
    current.context.signals.succeeded.emit(prepare_library_preview(model_b, b))
    current.context.signals.finished.emit()
    qtbot.waitUntil(lambda: preview.data is not None)
    old.context.signals.succeeded.emit(prepare_library_preview(model_a, a))
    old.context.signals.failed.emit('late failure')
    old.context.signals.finished.emit()
    qtbot.waitUntil(lambda: not controller.is_running)
    assert preview.data.path == 'b.npz'
    assert 'late failure' not in preview.summary.toPlainText()
    controller.cleanup()


def test_deselection_invalidates_worker_results(preview, qtbot):
    runner = DeferredRunner()
    controller = LibraryPreviewController(preview, SimpleNamespace(), runner)
    item, model = source()
    controller.select(item)
    handle = runner.handles[-1]
    controller.select(None)
    handle.context.signals.succeeded.emit(prepare_library_preview(model, item))
    handle.context.signals.finished.emit()
    qtbot.waitUntil(lambda: not controller.is_running)
    assert preview.data is None and not preview.slices.viewers
    controller.cleanup()


def test_cleanup_ignores_pending_results(preview, qtbot):
    runner = DeferredRunner()
    controller = LibraryPreviewController(preview, SimpleNamespace(), runner)
    item, model = source()
    controller.select(item)
    handle = runner.handles[-1]
    controller.cleanup()
    assert handle.cancelled
    handle.context.signals.succeeded.emit(prepare_library_preview(model, item))
    handle.context.signals.finished.emit()
    qtbot.waitUntil(lambda: not controller.is_running)
    assert preview.data is None


def test_filtering_out_all_items_emits_none(qtbot):
    from ct_pcd_gui.features.lesion_insertion.presentation.library_preview_integration import (
        PreviewLesionInsertionPanel,
    )
    panel = PreviewLesionInsertionPanel()
    qtbot.addWidget(panel)
    selected = []
    panel.library_selected.connect(selected.append)
    item, _ = source()
    panel.set_library_items([item])
    panel.library_search.setText('no matching lesion')
    assert selected[-1] is None
    panel.library_preview.cleanup()


def test_stale_3d_result_is_ignored_after_lesion_change(preview, qtbot):
    runner = DeferredRunner()
    controller = LibraryPreviewController(preview, SimpleNamespace(), runner)
    item, model = source('a.npz')
    controller.select(item)
    load = runner.handles[-1]
    load.context.signals.succeeded.emit(prepare_library_preview(model, item))
    load.context.signals.finished.emit()
    qtbot.waitUntil(lambda: preview.data is not None)
    preview.tabs.setCurrentIndex(1)
    surface = runner.handles[-1]
    other, _ = source('b.npz')
    controller.select(other)
    assert surface.cancelled
    surface.context.signals.succeeded.emit(object())
    surface.context.signals.finished.emit()
    qtbot.waitUntil(lambda: surface not in controller._active)
    assert preview.scene is None and preview.data is None
    controller.cleanup()
    for handle in runner.handles:
        handle.context.signals.finished.emit()


def test_module_factory_uses_preview_extensions():
    from ct_pcd_gui.features.lesion_insertion.presentation import module
    from ct_pcd_gui.features.lesion_insertion.presentation.library_preview_integration import (
        PreviewLesionInsertionPanel,
        PreviewLesionInsertionPresenter,
    )
    assert module.LesionInsertionPanel is PreviewLesionInsertionPanel
    assert module.LesionInsertionPresenter is PreviewLesionInsertionPresenter
