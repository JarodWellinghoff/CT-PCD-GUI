"""Compact library inspection using the lesion viewer's existing 2-D/3-D views."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ct_pcd_gui.features.lesion_viewer.presentation.slice_view import LesionSliceViews

from ..application.library_preview import LibraryPreviewData
from ..domain.models import LesionLibraryItem

if TYPE_CHECKING:
    from ct_pcd_gui.features.lesion_viewer.presentation.vtk_scene import LesionVtkScene


class LibraryLesionPreview(QWidget):
    surface_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.data: LibraryPreviewData | None = None
        self.scene: LesionVtkScene | None = None  # No VTK/OpenGL initialization for 2-D.
        self._surface_channel: int | None = None
        self._closed = False
        self._layout_timer = QTimer(self)
        self._layout_timer.setSingleShot(True)
        self._layout_timer.timeout.connect(self._resize_preview)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        notice = QLabel(
            "Source lesion preview only. Choose the lesion's source threshold and "
            "target scan thresholds separately in 'Selected lesion'."
        )
        notice.setWordWrap(True)
        layout.addWidget(notice)
        height_row = QHBoxLayout()
        height_row.addWidget(QLabel("2-D image height:"))
        self.image_height = QSpinBox()
        self.image_height.setRange(64, 320)
        self.image_height.setSingleStep(16)
        self.image_height.setValue(140)
        self.image_height.setSuffix(" px")
        self.image_height.setToolTip(
            "Height of each 2-D image canvas. The full image is fitted without "
            "stretching or cropping. This does not change the 3-D view."
        )
        height_row.addWidget(self.image_height)
        height_row.addStretch(1)
        layout.addLayout(height_row)
        self.tabs = QTabWidget()
        self.slices = LesionSliceViews()
        self.slices.set_side_by_side(True)
        # The insertion panel already scrolls. A second scroll area here can
        # clip a tall inner viewer instead of making the viewer itself smaller.
        self.slices.scroll.takeWidget()
        self.slices.layout().replaceWidget(self.slices.scroll, self.slices.content)
        self.slices.scroll.hide()
        self.slices.views_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.tabs.addTab(self.slices, "2-D HU")
        surface_page = QWidget()
        surface_layout = QVBoxLayout(surface_page)
        surface_layout.setContentsMargins(4, 4, 4, 4)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("3-D source:"))
        self.surface_channel = QComboBox()
        self.surface_channel.setToolTip(
            "Inspect a source mask; this does not select a threshold for insertion."
        )
        controls.addWidget(self.surface_channel, 1)
        self.reset_camera = QPushButton("Reset view")
        self.reset_camera.clicked.connect(self._reset_camera)
        controls.addWidget(self.reset_camera)
        self.retry_surface = QPushButton("Refresh")
        self.retry_surface.clicked.connect(lambda: self._request_surface(force=True))
        controls.addWidget(self.retry_surface)
        surface_layout.addLayout(controls)
        self.surface_stack = QStackedWidget()
        self.surface_notice = QLabel("Select a lesion to inspect its stored segmentation in 3-D.")
        self.surface_notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.surface_notice.setWordWrap(True)
        self.surface_stack.addWidget(self.surface_notice)
        surface_layout.addWidget(self.surface_stack, 1)
        surface_help = QLabel("Stored mask surface, in mm. Drag to rotate; use the wheel to zoom.")
        surface_help.setWordWrap(True)
        surface_layout.addWidget(surface_help)
        self.tabs.addTab(surface_page, "3-D shape")
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.tabs)
        details_row = QHBoxLayout()
        self.show_information = QCheckBox("Show lesion information")
        self.show_information.setChecked(True)
        details_row.addWidget(self.show_information)
        details_row.addStretch(1)
        layout.addLayout(details_row)
        self.details = QTabWidget()
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.details.addTab(self.summary, "Summary")
        self.details.addTab(self.metadata, "Model metadata")
        self.details.setMinimumHeight(140)
        self.details.setMaximumHeight(200)
        layout.addWidget(self.details)
        self.show_information.toggled.connect(self.details.setVisible)
        self.tabs.currentChanged.connect(lambda _index: self._request_surface())
        self.tabs.currentChanged.connect(self._schedule_preview_resize)
        self.image_height.valueChanged.connect(self._schedule_preview_resize)
        self.tabs.installEventFilter(self)
        self.slices.installEventFilter(self)
        self.surface_channel.currentIndexChanged.connect(lambda _index: self._request_surface())
        self.set_item(None)

    def set_item(self, item: LesionLibraryItem | None) -> None:
        """Clear all old views immediately, including when a search has no matches."""
        self.data = None
        self._surface_channel = None
        self.slices.clear()
        self.summary.clear()
        self.metadata.clear()
        with QSignalBlocker(self.surface_channel):
            self.surface_channel.clear()
        self.tabs.setTabEnabled(1, False)
        self.reset_camera.setEnabled(False)
        self.retry_surface.setEnabled(False)
        if self.scene is not None:
            self.scene.clear_surface()
        self.surface_stack.setCurrentWidget(self.surface_notice)
        if item is None:
            message = "Select a lesion from the library."
        elif not item.compatible:
            message = f"Incompatible lesion model: {item.warning}"
            self.summary.setPlainText(f"File: {item.path}\n\n{message}")
        else:
            message = f"Loading source HU and mask for {item.display_name}..."
            self.summary.setPlainText(
                f"Lesion: {item.display_name} ({item.lesion_id})\nFile: {item.path}"
            )
        self.slices.notice.setText(message)
        self.surface_notice.setText(message)
        self._schedule_preview_resize()

    def set_data(self, data: LibraryPreviewData) -> None:
        if self._closed:
            return
        self.data = data
        # The insertion loader explicitly uses row/column/slice: Y/X/Z, not Z/Y/X.
        self.slices.set_geometry("YXZ", data.spacing_xyz)
        self.slices.set_channels(list(data.channels))
        self.slices.set_side_by_side(True)
        for viewer in self.slices.viewers:
            viewer.image_view.setMinimumWidth(96)
            viewer.image_view.setFixedHeight(self.image_height.value())
            viewer.image_view.setAlignment(
                Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter
            )
            viewer.info_label.hide()  # The source-HU notice already supplies this context.
            viewer.slice_position_label.setMinimumWidth(40)
            viewer.window_label.setMinimumWidth(0)
            viewer.reset_action.setText("Reset")
            for action in viewer.toolbar.actions():
                widget = viewer.toolbar.widgetForAction(action)
                if isinstance(widget, QLabel) and widget is not viewer.window_label:
                    action.setVisible(False)  # Hide the redundant "Window preset:" caption.
        for box in self.slices.content.findChildren(QGroupBox):
            box.setMinimumHeight(0)
            box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        for label in self.slices.content.findChildren(QLabel):
            label.setWordWrap(True)
        self.summary.setPlainText(data.summary)
        self.metadata.setPlainText(data.metadata)
        with QSignalBlocker(self.surface_channel):
            self.surface_channel.clear()
            for index, channel in enumerate(data.channels):
                self.surface_channel.addItem(channel.name, index)
        self.tabs.setTabEnabled(1, True)
        self.retry_surface.setEnabled(True)
        self.surface_notice.setText("Open this tab to build the stored-mask 3-D preview.")
        self._request_surface()
        self._schedule_preview_resize()

    def show_load_error(self, message: str) -> None:
        self.slices.notice.setText(f"Source preview unavailable: {message}")
        self.surface_notice.setText(f"Source preview unavailable: {message}")
        self.summary.appendPlainText(f"\nPreview error: {message}")

    def _schedule_preview_resize(self, *_args) -> None:
        if not self._closed and not self._layout_timer.isActive():
            # Let Qt settle widths/font metrics before sizing wrapped labels.
            self._layout_timer.start(0)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if watched is self.tabs and event.type() == QEvent.Type.Resize:
            if event.size().width() != event.oldSize().width():
                self._schedule_preview_resize()
        elif watched is self.slices and event.type() == QEvent.Type.LayoutRequest:
            self._schedule_preview_resize()
        return super().eventFilter(watched, event)

    def _resize_preview(self) -> None:
        if self._closed:
            return
        in_2d = self.tabs.currentIndex() == 0
        self.image_height.setEnabled(in_2d and bool(self.slices.viewers))
        for viewer in self.slices.viewers:
            canvas = viewer.image_view
            height = self.image_height.value()
            if canvas.minimumHeight() != height or canvas.maximumHeight() != height:
                canvas.setFixedHeight(height)
                # DicomImageView.resizeEvent refits only in auto-fit mode;
                # a user's deliberate zoom/pan is not reset on slice changes.

        if in_2d:
            frame = self.tabs.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)
            width = max(1, self.tabs.width() - 2 * frame)
            page_layout = self.slices.layout()
            page_layout.activate()
            body = (
                page_layout.totalHeightForWidth(width)
                if page_layout.hasHeightForWidth()
                else page_layout.totalSizeHint().height()
            )
            # Size the page around the actual canvases AND all wrapped controls;
            # do not impose the hidden 3-D page's preferred height on this tab.
            height = max(60, body) + self.tabs.tabBar().sizeHint().height() + 2 * frame
        else:
            height = 300
        if self.tabs.minimumHeight() != height or self.tabs.maximumHeight() != height:
            self.tabs.setFixedHeight(height)

    def _request_surface(self, *, force: bool = False) -> None:
        if self._closed or self.data is None or self.tabs.currentIndex() != 1:
            return
        index = self.surface_channel.currentIndex()
        if index < 0 or (not force and index == self._surface_channel):
            return
        self._surface_channel = None
        self.reset_camera.setEnabled(False)
        self.surface_stack.setCurrentWidget(self.surface_notice)
        self.surface_notice.setText("Building source mask surface...")
        self.surface_requested.emit(index)

    def show_surface(self, channel_index: int, polydata: object) -> None:
        if (
            self._closed or self.data is None
            or channel_index != self.surface_channel.currentIndex()
        ):
            return
        try:
            if self.scene is None:
                from ct_pcd_gui.features.lesion_viewer.presentation.vtk_scene import LesionVtkScene

                self.scene = LesionVtkScene(self.surface_stack)
                self.surface_stack.addWidget(self.scene)
            self.scene.set_surface(polydata, reset_camera=True)
        except Exception as exc:
            self.show_surface_error(f"3-D rendering unavailable: {exc}")
            return
        self._surface_channel = channel_index
        self.surface_stack.setCurrentWidget(self.scene)
        self.reset_camera.setEnabled(True)

    def show_surface_error(self, message: str) -> None:
        self._surface_channel = None
        self.reset_camera.setEnabled(False)
        self.surface_stack.setCurrentWidget(self.surface_notice)
        self.surface_notice.setText(f"{message}\n\nThe 2-D HU views remain available.")

    def _reset_camera(self) -> None:
        if self.scene is not None:
            self.scene.reset_camera()

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._layout_timer.stop()
        self.data = None
        self.slices.clear()
        if self.scene is not None:
            self.scene.cleanup()
