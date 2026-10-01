from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPoint, QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True, slots=True)
class WindowPreset:
    """Named DICOM window/level values shown in the shared viewer toolbar."""

    name: str
    width: float
    level: float


DEFAULT_WINDOW_PRESETS: tuple[WindowPreset, ...] = (
    WindowPreset("Soft tissue", 400.0, 40.0),
    WindowPreset("Liver", 150.0, 70.0),
    WindowPreset("Lung", 1500.0, -600.0),
    WindowPreset("Bone", 2000.0, 500.0),
)
CUSTOM_WINDOW_PRESET = "Custom"


def window_hu_to_uint8(values: np.ndarray, width: float, level: float) -> np.ndarray:
    """Window a two-dimensional HU image into a contiguous uint8 array."""

    source = np.asarray(values, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("DICOM viewer images must be two-dimensional.")
    raw_width = float(width)
    safe_level = float(level)
    if not math.isfinite(raw_width) or not math.isfinite(safe_level):
        raise ValueError("DICOM window width and level must be finite.")
    safe_width = max(1.0, raw_width)
    lower = safe_level - safe_width / 2.0
    finite = np.nan_to_num(
        source,
        nan=lower,
        posinf=lower + safe_width,
        neginf=lower,
    )
    scaled = np.clip((finite - lower) / safe_width, 0.0, 1.0)
    return np.ascontiguousarray(np.rint(scaled * 255.0).astype(np.uint8))


def array_to_qimage(pixels: np.ndarray) -> QImage:
    """Convert a grayscale, RGB, or RGBA uint8 array into an owned QImage."""

    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if array.ndim == 2:
        height, width = array.shape
        image_format = QImage.Format.Format_Grayscale8
    elif array.ndim == 3 and array.shape[2] == 3:
        height, width, _ = array.shape
        image_format = QImage.Format.Format_RGB888
    elif array.ndim == 3 and array.shape[2] == 4:
        height, width, _ = array.shape
        image_format = QImage.Format.Format_RGBA8888
    else:
        raise ValueError("Expected a 2-D grayscale, RGB, or RGBA uint8 image.")
    return QImage(
        array.data,
        width,
        height,
        int(array.strides[0]),
        image_format,
    ).copy()


class DicomImageView(QGraphicsView):
    """Reusable image canvas with the lesion-insertion viewer interactions."""

    image_clicked = Signal(float, float)
    slice_delta_requested = Signal(int)
    window_level_dragged = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._pan_origin: QPoint | None = None
        self._window_level_origin: QPoint | None = None
        self._fit_next_image = True
        self._auto_fit = True

        self.setBackgroundBrush(QBrush(QColor(10, 10, 10)))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setMouseTracking(True)
        self.setToolTip(
            "Wheel: change slice | Ctrl+wheel: zoom | Middle drag: pan | "
            "Right drag: window/level"
        )

    @property
    def pixmap_item(self) -> QGraphicsPixmapItem:
        """The base image item, exposed for feature-specific overlay subclasses."""

        return self._pixmap_item

    @property
    def has_image(self) -> bool:
        return not self._pixmap_item.pixmap().isNull()

    def set_qimage(self, image: QImage) -> None:
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if self._fit_next_image:
            self.fit_image()
            self._fit_next_image = False

    def set_array(self, pixels: np.ndarray) -> None:
        self.set_qimage(array_to_qimage(pixels))

    def clear_image(self) -> None:
        self._pixmap_item.setPixmap(QPixmap())
        self._scene.setSceneRect(0.0, 0.0, 0.0, 0.0)
        self.resetTransform()
        self._fit_next_image = True
        self._auto_fit = True

    @Slot()
    def fit_image(self) -> None:
        if not self.has_image:
            return
        self.fitInView(
            self._pixmap_item.boundingRect(),
            Qt.AspectRatioMode.KeepAspectRatio,
        )
        self._auto_fit = True

    @Slot()
    def reset_view(self) -> None:
        self.resetTransform()
        self.fit_image()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt API
        steps = int(event.angleDelta().y() / 120)
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if steps:
                factor = 1.15**steps
                current_scale = abs(float(self.transform().m11()))
                next_scale = current_scale * factor
                if 0.02 <= next_scale <= 100.0:
                    self.scale(factor, factor)
                    self._auto_fit = False
        elif steps:
            self.slice_delta_requested.emit(steps)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = event.position().toPoint()
            self._auto_fit = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._window_level_origin = event.position().toPoint()
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self.has_image:
            point = self.mapToScene(event.position().toPoint())
            if self._pixmap_item.boundingRect().contains(point):
                self.image_clicked.emit(float(point.x()), float(point.y()))
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
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
        if self._window_level_origin is not None:
            delta = current - self._window_level_origin
            self._window_level_origin = current
            self.window_level_dragged.emit(float(delta.x()), float(-delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = None
            self.unsetCursor()
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._window_level_origin = None
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self._auto_fit and self.has_image:
            self.fit_image()


class DicomSliceViewer(QWidget):
    """Standard DICOM slice controls wrapped around a customizable image view."""

    slice_changed = Signal(int)
    image_clicked = Signal(float, float)
    window_level_changed = Signal(float, float)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        image_view: DicomImageView | None = None,
        window_presets: Sequence[WindowPreset] = DEFAULT_WINDOW_PRESETS,
    ) -> None:
        super().__init__(parent)
        presets = tuple(window_presets)
        if not presets:
            raise ValueError("At least one DICOM window preset is required.")
        self._presets = presets
        self._preset_names = {preset.name for preset in presets}
        if len(self._preset_names) != len(presets) or CUSTOM_WINDOW_PRESET in self._preset_names:
            raise ValueError("DICOM window preset names must be unique and cannot be 'Custom'.")
        self._window_width = float(presets[0].width)
        self._window_level = float(presets[0].level)
        self._reset_window = (
            float(presets[0].width),
            float(presets[0].level),
            presets[0].name,
        )
        self._slice_count = 0
        self._source_hu: np.ndarray | None = None
        self.image_view = image_view or DicomImageView(self)
        if self.image_view.parent() is None:
            self.image_view.setParent(self)
        self._build_ui()
        self._connect()
        self.set_slice_count(0)
        self.set_window(
            presets[0].width,
            presets[0].level,
            preset_name=presets[0].name,
            emit=False,
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        self.toolbar = QToolBar("DICOM viewer", self)
        self.toolbar.setMovable(False)
        self.fit_action = self.toolbar.addAction("Fit")
        self.fit_action.setShortcut(QKeySequence("F"))
        self.fit_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.reset_action = self.toolbar.addAction("Reset view")
        self.reset_action.setShortcut(QKeySequence("R"))
        self.reset_action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(QLabel("Window preset:"))
        self.preset_combo = QComboBox(self.toolbar)
        for preset in self._presets:
            self.preset_combo.addItem(preset.name)
        self.preset_combo.addItem(CUSTOM_WINDOW_PRESET)
        self.toolbar.addWidget(self.preset_combo)
        self.window_label = QLabel(self.toolbar)
        self.window_label.setMinimumWidth(120)
        self.toolbar.addWidget(self.window_label)
        root.addWidget(self.toolbar)

        root.addWidget(self.image_view, 1)

        navigation = QHBoxLayout()
        navigation.setContentsMargins(0, 0, 0, 0)
        navigation.addWidget(QLabel("Slice"))
        self.slice_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.slice_slider.setRange(0, 0)
        navigation.addWidget(self.slice_slider, 1)
        self.slice_spin = QSpinBox(self)
        self.slice_spin.setRange(0, 0)
        self.slice_spin.setSpecialValueText("--")
        navigation.addWidget(self.slice_spin)
        self.slice_position_label = QLabel("0 / 0", self)
        self.slice_position_label.setMinimumWidth(70)
        navigation.addWidget(self.slice_position_label)
        self.info_label = QLabel("No volume loaded", self)
        self.info_label.setWordWrap(True)
        navigation.addWidget(self.info_label)
        root.addLayout(navigation)

    def _connect(self) -> None:
        self.fit_action.triggered.connect(self.image_view.fit_image)
        self.reset_action.triggered.connect(self.reset_view)
        self.preset_combo.currentTextChanged.connect(self._preset_selected)
        self.slice_slider.valueChanged.connect(self._slider_changed)
        self.slice_spin.valueChanged.connect(self._spin_changed)
        self.image_view.slice_delta_requested.connect(self.step_slice)
        self.image_view.window_level_dragged.connect(self._adjust_window_level)
        self.image_view.image_clicked.connect(self.image_clicked.emit)

    @property
    def current_slice(self) -> int:
        return self.slice_slider.value() if self._slice_count else 0

    @property
    def slice_count(self) -> int:
        return self._slice_count

    @property
    def window_width(self) -> float:
        return self._window_width

    @property
    def window_level(self) -> float:
        return self._window_level

    def add_toolbar_separator(self) -> None:
        self.toolbar.addSeparator()

    def add_toolbar_widget(self, widget: QWidget) -> None:
        self.toolbar.addWidget(widget)

    def set_info_text(self, text: str) -> None:
        self.info_label.setText(text)

    def set_slice_count(
        self,
        count: int,
        *,
        initial_index: int = 0,
        emit: bool = False,
    ) -> None:
        self._slice_count = max(0, int(count))
        enabled = self._slice_count > 0
        target = (
            max(0, min(int(initial_index), self._slice_count - 1))
            if enabled
            else 0
        )
        with QSignalBlocker(self.slice_slider):
            self.slice_slider.setRange(0, max(0, self._slice_count - 1))
            self.slice_slider.setValue(target)
            self.slice_slider.setEnabled(enabled)
        with QSignalBlocker(self.slice_spin):
            if enabled:
                self.slice_spin.setSpecialValueText("")
                self.slice_spin.setRange(1, self._slice_count)
                self.slice_spin.setValue(target + 1)
            else:
                self.slice_spin.setRange(0, 0)
                self.slice_spin.setSpecialValueText("--")
                self.slice_spin.setValue(0)
            self.slice_spin.setEnabled(enabled)
        self._update_slice_position(target)
        if emit and enabled:
            self.slice_changed.emit(target)

    def set_slice(self, index: int, *, emit: bool = True) -> None:
        if self._slice_count <= 0:
            return
        target = max(0, min(int(index), self._slice_count - 1))
        if emit:
            self.slice_slider.setValue(target)
            return
        with QSignalBlocker(self.slice_slider):
            self.slice_slider.setValue(target)
        with QSignalBlocker(self.slice_spin):
            self.slice_spin.setValue(target + 1)
        self._update_slice_position(target)

    @Slot(int)
    def step_slice(self, delta: int) -> None:
        self.set_slice(self.current_slice + int(delta))

    def set_window(
        self,
        width: float,
        level: float,
        *,
        preset_name: str | None = None,
        emit: bool = True,
        remember_for_reset: bool = False,
    ) -> None:
        raw_width = float(width)
        safe_level = float(level)
        if not math.isfinite(raw_width) or not math.isfinite(safe_level):
            raise ValueError("DICOM window width and level must be finite.")
        safe_width = max(1.0, raw_width)
        self._window_width = safe_width
        self._window_level = safe_level

        selected_name = preset_name or self._matching_preset_name(
            safe_width, safe_level
        )
        if selected_name not in self._preset_names | {CUSTOM_WINDOW_PRESET}:
            selected_name = CUSTOM_WINDOW_PRESET
        if selected_name != CUSTOM_WINDOW_PRESET or remember_for_reset:
            self._reset_window = (safe_width, safe_level, selected_name)
        index = self.preset_combo.findText(selected_name)
        if index >= 0:
            with QSignalBlocker(self.preset_combo):
                self.preset_combo.setCurrentIndex(index)
        self.window_label.setText(
            f"W/L: {self._format_number(safe_width)} / "
            f"{self._format_number(safe_level)}"
        )
        self._render_hu_source()
        if emit:
            self.window_level_changed.emit(safe_width, safe_level)

    def set_hu_image(self, values: np.ndarray) -> None:
        source = np.asarray(values)
        if source.ndim != 2:
            raise ValueError("DICOM viewer images must be two-dimensional.")
        self._source_hu = source
        self._render_hu_source()

    def set_rgb_image(self, pixels: np.ndarray) -> None:
        self._source_hu = None
        self.image_view.set_array(pixels)

    def clear_image(self) -> None:
        self._source_hu = None
        self.image_view.clear_image()

    @Slot()
    def reset_view(self) -> None:
        self.image_view.reset_view()
        width, level, preset_name = self._reset_window
        self.set_window(
            width,
            level,
            preset_name=preset_name,
        )

    @Slot(int)
    def _slider_changed(self, index: int) -> None:
        if self._slice_count <= 0:
            return
        with QSignalBlocker(self.slice_spin):
            self.slice_spin.setValue(index + 1)
        self._update_slice_position(index)
        self.slice_changed.emit(index)

    @Slot(int)
    def _spin_changed(self, display_index: int) -> None:
        if self._slice_count <= 0:
            return
        self.slice_slider.setValue(display_index - 1)

    @Slot(str)
    def _preset_selected(self, name: str) -> None:
        preset = self._preset_by_name(name)
        if preset is not None:
            self.set_window(
                preset.width,
                preset.level,
                preset_name=preset.name,
            )

    @Slot(float, float)
    def _adjust_window_level(self, width_delta: float, level_delta: float) -> None:
        self.set_window(
            self._window_width + width_delta * 2.0,
            self._window_level + level_delta * 2.0,
            preset_name=CUSTOM_WINDOW_PRESET,
        )

    def _render_hu_source(self) -> None:
        if self._source_hu is None:
            return
        self.image_view.set_array(
            window_hu_to_uint8(
                self._source_hu,
                self._window_width,
                self._window_level,
            )
        )

    def _update_slice_position(self, index: int) -> None:
        if self._slice_count <= 0:
            self.slice_position_label.setText("0 / 0")
        else:
            self.slice_position_label.setText(f"{index + 1} / {self._slice_count}")

    def _matching_preset_name(self, width: float, level: float) -> str:
        for preset in self._presets:
            if math.isclose(width, preset.width) and math.isclose(level, preset.level):
                return preset.name
        return CUSTOM_WINDOW_PRESET

    def _preset_by_name(self, name: str) -> WindowPreset | None:
        return next((preset for preset in self._presets if preset.name == name), None)

    @staticmethod
    def _format_number(value: float) -> str:
        rounded = round(value)
        return str(rounded) if math.isclose(value, rounded, abs_tol=1e-6) else f"{value:.1f}"
