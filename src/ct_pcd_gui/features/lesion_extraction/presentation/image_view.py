from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView


class OverlayImageView(QGraphicsView):
    slice_delta_requested = Signal(int)
    label_clicked = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._labels: np.ndarray | None = None
        self._fit_next = True
        self.setBackgroundBrush(QBrush(QColor(8, 8, 8)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)

    def set_rgb_array(self, pixels: np.ndarray, labels: np.ndarray) -> None:
        array = np.ascontiguousarray(pixels, dtype=np.uint8)
        height, width, _ = array.shape
        image = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self._labels = np.asarray(labels)
        if self._fit_next:
            self.fit_image()
            self._fit_next = False

    def clear_image(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._labels = None
        self._fit_next = True

    def fit_image(self) -> None:
        if not self._pixmap_item.pixmap().isNull():
            self.fitInView(
                self._pixmap_item.boundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def reset_view(self) -> None:
        self.resetTransform()
        self.fit_image()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        steps = int(event.angleDelta().y() / 120)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if steps:
                factor = 1.15**steps
                self.scale(factor, factor)
        elif steps:
            self.slice_delta_requested.emit(steps)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._labels is not None:
            point = self.mapToScene(event.position().toPoint())
            column = int(point.x())
            row = int(point.y())
            if 0 <= row < self._labels.shape[0] and 0 <= column < self._labels.shape[1]:
                label = int(self._labels[row, column])
                if label > 0:
                    self.label_clicked.emit(label)
                    event.accept()
                    return
        super().mousePressEvent(event)
