"""Per-lesion threshold UI tests; run with the GUI dependencies installed."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from types import SimpleNamespace as NS
import pytest
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from ct_pcd_gui.features.lesion_insertion.presentation.threshold_selector import ThresholdSelector


@pytest.fixture
def selector():
    app = QApplication.instance() or QApplication([])
    widget = ThresholdSelector()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_widget_keeps_source_and_destination_separate(selector):
    selector.set_available_spectra((1, 2))
    selector.set_lesion(NS(background_hu=(0, 0), source_channel=1, target_spectrum_indices=(1, 2)))
    assert selector.source_combo.currentData() == 1
    assert tuple(selector.target_combo.currentData()) == (1, 2)
    assert "same lesion source threshold" in selector.summary.text()


def test_refresh_does_not_emit_an_edit(selector):
    events = []
    selector.selection_changed.connect(lambda *args: events.append(args))
    selector.set_available_spectra((1, 2))
    selector.set_lesion(NS(background_hu=(0,), source_channel=0, target_spectrum_indices=(2,)))
    assert not events


def test_unknown_saved_target_is_not_silently_replaced(selector):
    selector.set_available_spectra((1, 2))
    selector.set_lesion(NS(background_hu=(0, 0), source_channel=0, target_spectrum_indices=(7,)))
    assert tuple(selector.target_combo.currentData()) == (7,)
    assert "WARNING" in selector.summary.text()


def test_no_lesion_disables_controls(selector):
    selector.set_lesion(None)
    assert not selector.isEnabled()
