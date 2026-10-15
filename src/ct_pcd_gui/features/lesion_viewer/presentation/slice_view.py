"""Synchronized lesion slices built on the shared DICOM viewer controls."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QTransform
from PySide6.QtWidgets import (
    QBoxLayout, QCheckBox, QGraphicsPixmapItem, QGroupBox, QHBoxLayout, QLabel,
    QScrollArea, QVBoxLayout, QWidget,
)

from ct_pcd_gui.shared.qt.dicom_viewer import DicomImageView, DicomSliceViewer, array_to_qimage
from ..domain.inspection import SliceChannel, mask_outline, to_zyx


class HuImageView(DicomImageView):
    """Pixel probe and mask overlay; neither changes the underlying HU image."""

    pixel_hovered = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.overlay = QGraphicsPixmapItem(self.pixmap_item)
        self.overlay.setZValue(1)
        self.overlay.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self._spacing = (1.0, 1.0)

    def set_spacing(self, x: float, y: float) -> None:
        self._spacing = (x, y)
        self.pixmap_item.setTransform(QTransform.fromScale(x, y))
        if self.has_image:
            self.scene().setSceneRect(self.pixmap_item.sceneBoundingRect())
            self.fit_image()

    def set_qimage(self, image: QImage) -> None:
        super().set_qimage(image)
        self.scene().setSceneRect(self.pixmap_item.sceneBoundingRect())

    def fit_image(self) -> None:
        if self.has_image:
            self.fitInView(self.pixmap_item.sceneBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            self._auto_fit = True

    def set_mask(self, mask: np.ndarray | None) -> None:
        if mask is None:
            self.overlay.setPixmap(QPixmap())
            return
        boundary = mask_outline(mask)
        pixels = np.zeros((*boundary.shape, 4), dtype=np.uint8)
        pixels[boundary] = (255, 190, 55, 255)
        self.overlay.setPixmap(QPixmap.fromImage(array_to_qimage(pixels)))

    def clear_image(self) -> None:
        super().clear_image()
        self.overlay.setPixmap(QPixmap())

    def _pixel(self, event: QMouseEvent) -> tuple[int, int]:
        point = self.pixmap_item.mapFromScene(self.mapToScene(event.position().toPoint()))
        if self.has_image and self.pixmap_item.boundingRect().contains(point):
            return math.floor(point.x()), math.floor(point.y())
        return -1, -1

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.pixel_hovered.emit(*self._pixel(event))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = self._pixel(event)
            if x >= 0:
                self.pixel_hovered.emit(x, y)
                self.image_clicked.emit(float(x), float(y))
                event.accept()
                return
        super().mousePressEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self.pixel_hovered.emit(-1, -1)
        super().leaveEvent(event)


class LesionSliceViews(QWidget):
    """Arrange channels in a row or column with one shared slice position."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.channels: list[SliceChannel] = []
        self.viewers: list[DicomSliceViewer] = []
        self._volumes: list[np.ndarray] = []
        self._masks: list[np.ndarray | None] = []
        self._boxes: list[QGroupBox] = []
        self._readouts: list[QLabel] = []
        self._axis_order = "ZYX"
        self._spacing = (1.0, 1.0, 1.0)
        self._slice = 0
        self._probe: tuple[int, int] | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addWidget(QLabel("2-D stored voxel values"))
        row.addStretch(1)
        self.outline = QCheckBox("Lesion outline")
        self.outline.setToolTip("Show the stored LesionMask boundary; independent of 3-D surface processing.")
        self.outline.setChecked(True)
        self.outline.toggled.connect(self._render)
        row.addWidget(self.outline)
        layout.addLayout(row)
        self.notice = QLabel("Load a lesion file to inspect its slices.")
        self.notice.setWordWrap(True)
        layout.addWidget(self.notice)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content = QWidget()
        self.views_layout = QVBoxLayout(self.content)
        self.views_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)
        self.outline.setEnabled(False)

    @property
    def current_slice(self) -> int:
        return self._slice

    def set_side_by_side(self, enabled: bool) -> None:
        """Reflow existing viewers without resetting slices or window/level."""
        direction = (
            QBoxLayout.Direction.LeftToRight
            if enabled
            else QBoxLayout.Direction.TopToBottom
        )
        self.views_layout.setDirection(direction)

    def clear(self) -> None:
        for box in self._boxes:
            self.views_layout.removeWidget(box)
            box.hide()
            box.deleteLater()
        self.channels = []
        self.viewers = []
        self._volumes = []
        self._masks = []
        self._boxes = []
        self._readouts = []
        self._slice = 0
        self._probe = None
        self.notice.setText("Load a lesion file to inspect its slices.")
        self.outline.setEnabled(False)

    def set_channels(self, channels: list[SliceChannel]) -> None:
        self.clear()
        if not channels:
            self.notice.setText("Original HU unavailable: this file has no recognized VOI/HU array.")
            return
        shape = channels[0].values.shape
        if any(channel.values.shape != shape for channel in channels):
            self.notice.setText("Channels have different shapes; synchronized views are unavailable.")
            return
        self.channels = channels
        for channel in channels:
            box = QGroupBox(f"{channel.name} | {channel.units}")
            box.setMinimumHeight(230)
            box_layout = QVBoxLayout(box)
            canvas = HuImageView()
            viewer = DicomSliceViewer(image_view=canvas)
            viewer.slice_changed.connect(self.set_slice)
            canvas.pixel_hovered.connect(self._show_probe)
            viewer.set_info_text("Stored VOI" if channel.units == "HU" else "Stored array")
            if channel.units != "HU":
                step = max(1, channel.values.size // 100000)
                sample = channel.values.flat[::step][:100000]
                finite = sample[np.isfinite(sample)]
                if finite.size:
                    low, high = float(finite.min()), float(finite.max())
                    viewer.set_window(max(1.0, high - low), (low + high) / 2.0,
                                      emit=False, remember_for_reset=True)
            box_layout.addWidget(viewer, 1)
            readout = QLabel("Hover over the image to inspect a stored value.")
            readout.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box_layout.addWidget(readout)
            self.views_layout.addWidget(box, 1)
            self._boxes.append(box)
            self._readouts.append(readout)
            self.viewers.append(viewer)
        self.outline.setEnabled(any(channel.mask is not None for channel in channels))
        missing_mask = any(channel.mask is None for channel in channels)
        note = "Image numbers are synchronized across channels. " if len(channels) > 1 else ""
        note += "Displayed values are stored values; window/level only changes their appearance."
        if channels[0].units != "HU":
            note = "Original HU unavailable. Showing the selected stored array without assuming HU units."
        if missing_mask:
            note += " Outline unavailable for channels without a matching mask."
        self.notice.setText(note)
        self.set_geometry(self._axis_order, self._spacing, reset=True)

    def set_geometry(self, axis_order: str, spacing_xyz: tuple[float, float, float], *, reset: bool = False) -> None:
        changed = self._axis_order != axis_order
        spacing_changed = self._spacing != spacing_xyz
        self._axis_order, self._spacing = axis_order, spacing_xyz
        if not self.channels or not (reset or changed or spacing_changed):
            return
        self._volumes = [to_zyx(channel.values, axis_order) for channel in self.channels]
        self._masks = [to_zyx(channel.mask, axis_order) if channel.mask is not None else None
                       for channel in self.channels]
        count = self._volumes[0].shape[0]
        if reset or changed:
            self._slice = count // 2
            self._probe = None
        self._slice = max(0, min(self._slice, count - 1))
        for viewer in self.viewers:
            viewer.set_slice_count(count, initial_index=self._slice)
            viewer.image_view.set_spacing(spacing_xyz[0], spacing_xyz[1])
        self._render()
        for viewer in self.viewers:
            viewer.image_view.fit_image()

    def set_slice(self, index: int) -> None:
        if not self._volumes:
            return
        self._slice = max(0, min(int(index), self._volumes[0].shape[0] - 1))
        for viewer in self.viewers:
            viewer.set_slice(self._slice, emit=False)
        self._render()

    def _render(self, *_args) -> None:
        for viewer, volume, mask in zip(self.viewers, self._volumes, self._masks):
            viewer.set_hu_image(volume[self._slice])
            viewer.image_view.set_mask(mask[self._slice] if self.outline.isChecked() and mask is not None else None)
        self._show_probe(*(self._probe or (-1, -1)))

    def _show_probe(self, x: int, y: int) -> None:
        self._probe = (x, y) if x >= 0 and y >= 0 else None
        for label, channel, volume in zip(self._readouts, self.channels, self._volumes):
            if not (0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]):
                label.setText("Hover over the image to inspect a stored value.")
                continue
            value = volume[self._slice, y, x].item()
            unit = " HU" if channel.units == "HU" else " (stored value)"
            valid = "" if np.isfinite(value) else " [non-finite]"
            label.setText(f"Slice {self._slice + 1}, row {y + 1}, column {x + 1}: {value}{unit}{valid}")
