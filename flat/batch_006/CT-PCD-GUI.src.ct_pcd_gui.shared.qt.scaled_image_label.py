from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QWidget


class ScaledImageLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(190, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #090909; border: 1px solid #555;")
        self.setText("No image")

    def set_array(self, image: np.ndarray) -> None:
        array = np.ascontiguousarray(image, dtype=np.uint8)
        height, width = array.shape
        qimage = QImage(
            array.data,
            width,
            height,
            int(array.strides[0]),
            QImage.Format.Format_Grayscale8,
        ).copy()
        self._source_pixmap = QPixmap.fromImage(qimage)
        self.setText("")
        self._rescale()

    def clear_image(self, message: str = "No image") -> None:
        self._source_pixmap = None
        self.clear()
        self.setText(message)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap is None or self.width() < 2 or self.height() < 2:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
