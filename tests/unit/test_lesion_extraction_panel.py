from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt

from ct_pcd_gui.features.lesion_extraction.models import (
    LesionCandidate,
    SegmentDefinition,
)
from ct_pcd_gui.features.lesion_extraction.presentation.panel import (
    LesionExtractionPanel,
)


def _candidate(candidate_id: str, name: str) -> LesionCandidate:
    part = int(candidate_id.rsplit(":", 1)[-1])
    return LesionCandidate(
        candidate_id=candidate_id,
        segment_key="Segment0",
        segment_name="Liver lesion",
        component_label=part,
        component_index=part,
        component_count=2,
        output_name=name,
        voxel_count=25,
        physical_size_mm3=12.5,
        centroid_xyz_mm=(10.0, 20.0, 30.0),
    )


def test_panel_defaults_to_labels_named_lesion(qtbot) -> None:
    panel = LesionExtractionPanel()
    qtbot.addWidget(panel)
    panel.set_segments(
        (
            SegmentDefinition("Segment0", "Liver", 1),
            SegmentDefinition("Segment1", "Liver lesion", 2),
            SegmentDefinition("Segment2", "Kidney Lesion", 3),
        )
    )
    assert panel.selected_segment_keys() == ("Segment1", "Segment2")


def test_panel_collects_omissions_and_edited_names(qtbot) -> None:
    panel = LesionExtractionPanel()
    qtbot.addWidget(panel)
    panel.set_candidates(
        (
            _candidate("Segment0:1", "Lesion-part-01"),
            _candidate("Segment0:2", "Lesion-part-02"),
        )
    )
    panel.candidate_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    panel.candidate_table.item(1, 1).setText("Renamed lesion")
    selections = panel.candidate_selections()
    assert selections[0].included is False
    assert selections[1].included is True
    assert selections[1].output_name == "Renamed lesion"
