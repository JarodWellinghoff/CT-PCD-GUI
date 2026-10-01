from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)
from ct_pcd_gui.shared.qt.path_selector import PathSelector


class PanelUiMixin:
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(7, 7, 7, 7)
        root.setSpacing(6)
        title = QLabel("Lesion Viewer")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        root.addWidget(title)

        self.source_selector = PathSelector(
            "MAT/NPZ lesion folder or file",
            allow_file=True,
            file_caption="Select lesion file",
            folder_caption="Select lesion library",
            file_filter="Lesion files (*.mat *.npz)",
            allow_all_files=False,
        )
        root.addWidget(self.source_selector)
        row = QHBoxLayout()
        self.open_button = QPushButton("Open / Scan")
        self.refresh_button = QPushButton("Refresh")
        row.addWidget(self.open_button, 1)
        row.addWidget(self.refresh_button)
        root.addLayout(row)

        self.filter_field = QLineEdit()
        self.filter_field.setPlaceholderText("Filter library")
        self.filter_field.setClearButtonEnabled(True)
        root.addWidget(self.filter_field)
        self.file_table = QTableView()
        self.file_table.setModel(self.file_proxy)
        self.file_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.file_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.file_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.file_table.setSortingEnabled(True)
        self.file_table.verticalHeader().hide()
        self.file_table.setMinimumHeight(170)
        self.file_table.horizontalHeader().setSectionResizeMode(
            self.file_table.horizontalHeader().ResizeMode.Stretch
        )
        root.addWidget(self.file_table)
        row = QHBoxLayout()
        self.load_button = QPushButton("Load Selected")
        self.file_count_label = QLabel("0 files")
        self.file_count_label.setStyleSheet("color: #bdbdbd;")
        row.addWidget(self.load_button)
        row.addStretch(1)
        row.addWidget(self.file_count_label)
        root.addLayout(row)

        form = QFormLayout()
        self.candidate_combo = QComboBox()
        form.addRow("Array:", self.candidate_combo)
        self.axis_order_combo = QComboBox()
        for order in ("ZYX", "YXZ", "XYZ", "ZXY", "YZX", "XZY"):
            self.axis_order_combo.addItem(order, order)
        form.addRow("Axis order:", self.axis_order_combo)
        spacing = QWidget()
        spacing_row = QHBoxLayout(spacing)
        spacing_row.setContentsMargins(0, 0, 0, 0)
        self.spacing_x, self.spacing_y, self.spacing_z = (
            self._spacing_box(),
            self._spacing_box(),
            self._spacing_box(),
        )
        for box in (self.spacing_x, self.spacing_y, self.spacing_z):
            spacing_row.addWidget(box)
        form.addRow("Spacing X/Y/Z:", spacing)
        self.spacing_source_label = QLabel("Manual")
        form.addRow("", self.spacing_source_label)
        root.addLayout(form)

        form = QFormLayout()
        threshold = QWidget()
        threshold_row = QHBoxLayout(threshold)
        threshold_row.setContentsMargins(0, 0, 0, 0)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 1000)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setDecimals(5)
        self.threshold_spin.setKeyboardTracking(False)
        self.auto_threshold_button = QPushButton("Auto")
        threshold_row.addWidget(self.threshold_slider, 1)
        threshold_row.addWidget(self.threshold_spin)
        threshold_row.addWidget(self.auto_threshold_button)
        form.addRow("Threshold:", threshold)
        self.foreground_below_checkbox = QCheckBox("Lesion values are below threshold")
        form.addRow("", self.foreground_below_checkbox)
        self.downsample_combo = QComboBox()
        for text, value in (("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)):
            self.downsample_combo.addItem(text, value)
        form.addRow("Downsample:", self.downsample_combo)
        self.smoothing_spin = QSpinBox()
        self.smoothing_spin.setRange(0, 100)
        self.smoothing_spin.setValue(20)
        self.smoothing_spin.setSingleStep(5)
        form.addRow("Smoothing:", self.smoothing_spin)
        self.reduction_spin = QSpinBox()
        self.reduction_spin.setRange(0, 95)
        self.reduction_spin.setSuffix(" %")
        form.addRow("Mesh reduction:", self.reduction_spin)
        self.largest_component_checkbox = QCheckBox("Largest component only")
        self.largest_component_checkbox.setChecked(True)
        self.pad_border_checkbox = QCheckBox("Pad border for closed surface")
        self.pad_border_checkbox.setChecked(True)
        form.addRow("", self.largest_component_checkbox)
        form.addRow("", self.pad_border_checkbox)
        root.addLayout(form)

        form = QFormLayout()
        self.color_button = QPushButton("Choose…")
        form.addRow("Color:", self.color_button)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(100)
        form.addRow("Opacity:", self.opacity_slider)
        self.representation_combo = QComboBox()
        self.representation_combo.addItems(["Surface", "Wireframe", "Points"])
        form.addRow("Representation:", self.representation_combo)
        self.edge_checkbox = QCheckBox("Show edges")
        self.parallel_projection_checkbox = QCheckBox("Parallel projection")
        form.addRow("", self.edge_checkbox)
        form.addRow("", self.parallel_projection_checkbox)
        root.addLayout(form)

        row = QHBoxLayout()
        self.reset_camera_button = QPushButton("Reset")
        self.view_x_button, self.view_y_button, self.view_z_button = (
            QPushButton("+X"),
            QPushButton("+Y"),
            QPushButton("+Z"),
        )
        for button in (
            self.reset_camera_button,
            self.view_x_button,
            self.view_y_button,
            self.view_z_button,
        ):
            row.addWidget(button)
        root.addLayout(row)
        row = QHBoxLayout()
        self.screenshot_button = QPushButton("Screenshot…")
        self.export_button = QPushButton("Export Mesh…")
        row.addWidget(self.screenshot_button)
        row.addWidget(self.export_button)
        root.addLayout(row)
        root.addStretch(1)

    @staticmethod
    def _spacing_box() -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(0.0001, 1000.0)
        box.setDecimals(5)
        box.setValue(1.0)
        box.setSuffix(" mm")
        box.setKeyboardTracking(False)
        return box

    def _connect(self) -> None:
        self.open_button.clicked.connect(
            lambda: self.source_requested.emit(self.source_path())
        )
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.filter_field.textChanged.connect(self._filter_changed)
        self.load_button.clicked.connect(self._emit_selected)
        self.file_table.doubleClicked.connect(lambda _index: self._emit_selected())
        self.candidate_combo.currentIndexChanged.connect(self._candidate_selected)
        self.threshold_slider.valueChanged.connect(self._slider_changed)
        self.threshold_spin.valueChanged.connect(self._spin_changed)
        self.auto_threshold_button.clicked.connect(self._auto_threshold)
        for widget in (
            self.spacing_x,
            self.spacing_y,
            self.spacing_z,
            self.smoothing_spin,
            self.reduction_spin,
        ):
            widget.valueChanged.connect(self._surface_changed)
        for widget in (self.axis_order_combo, self.downsample_combo):
            widget.currentIndexChanged.connect(self._surface_changed)
        for widget in (
            self.foreground_below_checkbox,
            self.largest_component_checkbox,
            self.pad_border_checkbox,
        ):
            widget.toggled.connect(self._surface_changed)
        self.color_button.clicked.connect(self._choose_color)
        self.opacity_slider.valueChanged.connect(self._appearance_changed)
        self.representation_combo.currentIndexChanged.connect(self._appearance_changed)
        self.edge_checkbox.toggled.connect(self._appearance_changed)
        self.parallel_projection_checkbox.toggled.connect(self._appearance_changed)
        self.reset_camera_button.clicked.connect(self.reset_camera_requested.emit)
        self.view_x_button.clicked.connect(lambda: self.axis_view_requested.emit("X"))
        self.view_y_button.clicked.connect(lambda: self.axis_view_requested.emit("Y"))
        self.view_z_button.clicked.connect(lambda: self.axis_view_requested.emit("Z"))
        self.screenshot_button.clicked.connect(self._screenshot)
        self.export_button.clicked.connect(self._export)
