from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPen
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QWidget,
)

from ct_pcd_gui.shared.qt.dicom_viewer import DicomImageView


@dataclass(frozen=True, slots=True)
class SliceMarker:
    instance_id: str
    column: float
    row: float
    selected: bool
    enabled: bool


class SliceView(DicomImageView):
    """Shared DICOM canvas extended with lesion placement annotations."""

    location_selected = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._overlay_items: list[QGraphicsItem] = []
        self.image_clicked.connect(self.location_selected.emit)

    def set_image(
        self,
        image: QImage,
        *,
        cursor_position: tuple[float, float] | None,
        markers: list[SliceMarker],
    ) -> None:
        """Compatibility wrapper for callers that already provide a QImage."""

        self.set_qimage(image)
        self.set_annotations(cursor_position=cursor_position, markers=markers)

    def set_annotations(
        self,
        *,
        cursor_position: tuple[float, float] | None,
        markers: list[SliceMarker],
    ) -> None:
        self._clear_overlays()
        if cursor_position is not None:
            self._add_crosshair(*cursor_position)
        for marker in markers:
            self._add_marker(marker)

    def clear_image(self) -> None:
        self._clear_overlays()
        super().clear_image()

    def _clear_overlays(self) -> None:
        scene = self.scene()
        if scene is not None:
            for item in self._overlay_items:
                scene.removeItem(item)
        self._overlay_items.clear()

    def _add_crosshair(self, column: float, row: float) -> None:
        pen = QPen(QColor(255, 230, 40), 1.2)
        pen.setCosmetic(True)
        horizontal = QGraphicsLineItem(column - 12, row, column + 12, row)
        vertical = QGraphicsLineItem(column, row - 12, column, row + 12)
        horizontal.setPen(pen)
        vertical.setPen(pen)
        horizontal.setZValue(20)
        vertical.setZValue(20)
        scene = self.scene()
        if scene is None:
            return
        scene.addItem(horizontal)
        scene.addItem(vertical)
        self._overlay_items.extend([horizontal, vertical])

    def _add_marker(self, marker: SliceMarker) -> None:
        color = QColor(0, 220, 255) if marker.selected else QColor(70, 255, 100)
        if not marker.enabled:
            color = QColor(160, 160, 160)
        pen = QPen(color, 2.2 if marker.selected else 1.2)
        pen.setCosmetic(True)
        radius = 8.0 if marker.selected else 6.0
        ellipse = QGraphicsEllipseItem(
            marker.column - radius,
            marker.row - radius,
            radius * 2,
            radius * 2,
        )
        ellipse.setPen(pen)
        ellipse.setBrush(Qt.BrushStyle.NoBrush)
        ellipse.setZValue(15)
        scene = self.scene()
        if scene is None:
            return
        scene.addItem(ellipse)
        self._overlay_items.append(ellipse)
