from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import QWidget

from ct_pcd_gui.shared.qt.dicom_viewer import DicomImageView


class OverlayImageView(DicomImageView):
    """Shared DICOM canvas extended with segmentation-label hit testing."""

    label_clicked = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._labels: np.ndarray | None = None
        self.image_clicked.connect(self._label_at_position)

    def set_rgb_array(self, pixels: np.ndarray, labels: np.ndarray) -> None:
        label_array = np.asarray(labels)
        if label_array.ndim != 2:
            raise ValueError("Segmentation overlay labels must be two-dimensional.")
        if tuple(label_array.shape) != tuple(np.asarray(pixels).shape[:2]):
            raise ValueError("Overlay labels must match the displayed image dimensions.")
        self._labels = label_array
        self.set_array(pixels)

    def clear_image(self) -> None:
        self._labels = None
        super().clear_image()

    @Slot(float, float)
    def _label_at_position(self, column: float, row: float) -> None:
        if self._labels is None:
            return
        column_index = int(column)
        row_index = int(row)
        if (
            0 <= row_index < self._labels.shape[0]
            and 0 <= column_index < self._labels.shape[1]
        ):
            label = int(self._labels[row_index, column_index])
            if label > 0:
                self.label_clicked.emit(label)
