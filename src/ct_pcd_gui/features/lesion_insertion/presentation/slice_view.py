from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)


@dataclass(frozen=True, slots=True)
class SliceMarker:
    instance_id: str
    column: float
    row: float
    selected: bool
    enabled: bool


class SliceView(QGraphicsView):
    location_selected = Signal(float, float)
    slice_delta_requested = Signal(int)
    window_level_dragged = Signal(float, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._overlay_items: list[object] = []
        self._cursor_position: tuple[float, float] | None = None
        self._pan_origin: QPoint | None = None
        self._wl_origin: QPoint | None = None
        self.setBackgroundBrush(QBrush(QColor(10, 10, 10)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)

    def set_image(
        self,
        image: QImage,
        *,
        cursor_position: tuple[float, float] | None,
        markers: list[SliceMarker],
    ) -> None:
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._cursor_position = cursor_position
        self._clear_overlays()
        if cursor_position is not None:
            self._add_crosshair(*cursor_position)
        for marker in markers:
            self._add_marker(marker)

    def clear_image(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._clear_overlays()

    def fit_image(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(
                self._pixmap_item.boundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def reset_view(self) -> None:
        self.resetTransform()
        self.fit_image()

    def _clear_overlays(self) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)  # type: ignore[arg-type]
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
        self._scene.addItem(horizontal)
        self._scene.addItem(vertical)
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
        self._scene.addItem(ellipse)
        self._overlay_items.append(ellipse)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        steps = int(event.angleDelta().y() / 120)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15**steps
            self.scale(factor, factor)
        elif steps:
            self.slice_delta_requested.emit(steps)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._wl_origin = event.position().toPoint()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            if self._pixmap_item.boundingRect().contains(point):
                self.location_selected.emit(float(point.x()), float(point.y()))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        current = event.position().toPoint()
        if self._pan_origin is not None:
            delta = current - self._pan_origin
            self._pan_origin = current
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        if self._wl_origin is not None:
            delta = current - self._wl_origin
            self._wl_origin = current
            self.window_level_dragged.emit(float(delta.x()), float(-delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = None
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._wl_origin = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._pixmap_item.pixmap().isNull() and self.transform().m11() == 1.0:
            self.fit_image()
