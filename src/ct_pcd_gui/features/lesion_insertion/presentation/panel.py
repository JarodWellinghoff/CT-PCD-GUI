from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..domain.models import (
    DicomSeriesCandidate,
    LesionInstance,
    LesionLibraryItem,
    LesionParameters,
)


class DropLineEdit(QLineEdit):
    path_dropped = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setClearButtonEnabled(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and len(event.mimeData().urls()) == 1:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            path = urls[0].toLocalFile()
            self.setText(path)
            self.path_dropped.emit(path)
            event.acceptProposedAction()
            return
        event.ignore()


class PathRow(QWidget):
    changed = Signal(str)
    action_requested = Signal(str)

    def __init__(
        self,
        *,
        placeholder: str,
        action_text: str,
        directory: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.directory = directory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit = DropLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        self.edit.path_dropped.connect(self.action_requested)
        layout.addWidget(self.edit, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(32)
        browse.setToolTip("Browse")
        browse.clicked.connect(self._browse)
        layout.addWidget(browse)
        self.action = QPushButton(action_text)
        self.action.clicked.connect(lambda: self.action_requested.emit(self.path()))
        layout.addWidget(self.action)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str) -> None:
        self.edit.setText(path)

    @Slot()
    def _browse(self) -> None:
        current = self.path() or str(Path.home())
        if self.directory:
            selected = QFileDialog.getExistingDirectory(
                self, "Select directory", current
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(self, "Select file", current)
        if selected:
            self.set_path(selected)
            self.action_requested.emit(selected)


class LesionInsertionPanel(QWidget):
    reconstruction_requested = Signal(str)
    series_selected = Signal(str)
    raw_requested = Signal(str)
    library_requested = Signal(str)
    library_selected = Signal(object)
    lesion_selected = Signal(str)
    add_requested = Signal()
    move_to_cursor_requested = Signal()
    duplicate_requested = Signal()
    remove_requested = Signal()
    move_up_requested = Signal()
    move_down_requested = Signal()
    show_all_requested = Signal(bool)
    lesion_flags_changed = Signal(str, bool, bool)
    parameters_changed = Signal(object)
    position_changed = Signal(object)
    background_changed = Signal(str)
    reset_parameters_requested = Signal()
    save_session_requested = Signal()
    load_session_requested = Signal()
    validate_requested = Signal()
    run_requested = Signal()
    cancel_requested = Signal()
    session_fields_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._library_items: list[LesionLibraryItem] = []
        self._lesions: list[LesionInstance] = []
        self._updating = False
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)

        workflow = QLabel(
            "1. Load reconstruction and raw CTPD  •  2. Choose a lesion  •  "
            "3. Click a 3-D location  •  4. Add/edit lesions  •  5. Validate and process"
        )
        workflow.setWordWrap(True)
        layout.addWidget(workflow)

        data_group = QGroupBox("1. Case data")
        data_layout = QFormLayout(data_group)
        self.reconstruction_row = PathRow(
            placeholder="Reconstructed DICOM file or directory",
            action_text="Discover",
        )
        self.reconstruction_row.action_requested.connect(self.reconstruction_requested)
        self.reconstruction_row.changed.connect(
            lambda _text: self.session_fields_changed.emit()
        )
        data_layout.addRow("Reconstruction", self.reconstruction_row)
        self.series_combo = QComboBox()
        self.series_combo.setEnabled(False)
        self.series_combo.currentIndexChanged.connect(self._series_changed)
        data_layout.addRow("DICOM series", self.series_combo)
        self.raw_row = PathRow(
            placeholder="DICOM-CT-PD file or directory",
            action_text="Inspect",
        )
        self.raw_row.action_requested.connect(self.raw_requested)
        self.raw_row.changed.connect(lambda _text: self.session_fields_changed.emit())
        data_layout.addRow("CTPD raw", self.raw_row)
        self.data_status = QLabel("No case loaded.")
        self.data_status.setWordWrap(True)
        data_layout.addRow(self.data_status)
        layout.addWidget(data_group)

        library_group = QGroupBox("2. Lesion library")
        library_layout = QVBoxLayout(library_group)
        self.library_row = PathRow(
            placeholder="MAT/NPZ lesion-model library",
            action_text="Scan",
        )
        self.library_row.action_requested.connect(self.library_requested)
        self.library_row.changed.connect(
            lambda _text: self.session_fields_changed.emit()
        )
        library_layout.addWidget(self.library_row)
        self.library_search = QLineEdit()
        self.library_search.setPlaceholderText("Search lesions")
        self.library_search.setClearButtonEnabled(True)
        self.library_search.textChanged.connect(self._filter_library)
        library_layout.addWidget(self.library_search)
        self.library_list = QListWidget()
        self.library_list.setMinimumHeight(130)
        self.library_list.currentRowChanged.connect(self._library_row_changed)
        library_layout.addWidget(self.library_list)
        self.library_thumbnail = QLabel("No thumbnail")
        self.library_thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.library_thumbnail.setMinimumHeight(96)
        library_layout.addWidget(self.library_thumbnail)
        self.library_metadata = QPlainTextEdit()
        self.library_metadata.setReadOnly(True)
        self.library_metadata.setMaximumHeight(130)
        library_layout.addWidget(self.library_metadata)
        layout.addWidget(library_group)

        lesions_group = QGroupBox("3. Lesions in this case")
        lesions_layout = QVBoxLayout(lesions_group)
        action_row = QHBoxLayout()
        for text, signal in (
            ("Add at crosshair", self.add_requested),
            ("Move selected", self.move_to_cursor_requested),
            ("Duplicate", self.duplicate_requested),
            ("Remove", self.remove_requested),
        ):
            button = QPushButton(text)
            button.clicked.connect(signal)
            action_row.addWidget(button)
        lesions_layout.addLayout(action_row)
        order_row = QHBoxLayout()
        up = QPushButton("Move up")
        up.clicked.connect(self.move_up_requested)
        down = QPushButton("Move down")
        down.clicked.connect(self.move_down_requested)
        show = QPushButton("Show all")
        show.clicked.connect(lambda: self.show_all_requested.emit(True))
        hide = QPushButton("Hide all")
        hide.clicked.connect(lambda: self.show_all_requested.emit(False))
        for button in (up, down, show, hide):
            order_row.addWidget(button)
        lesions_layout.addLayout(order_row)
        self.lesion_table = QTableWidget(0, 7)
        self.lesion_table.setHorizontalHeaderLabels(
            ["On", "Show", "Label", "Lesion", "Column", "Row", "Slice"]
        )
        self.lesion_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.lesion_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.lesion_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.lesion_table.itemSelectionChanged.connect(self._lesion_selection_changed)
        self.lesion_table.itemChanged.connect(self._lesion_item_changed)
        self.lesion_table.verticalHeader().setVisible(False)

        self.lesion_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.lesion_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        lesions_layout.addWidget(self.lesion_table)
        layout.addWidget(lesions_group)

        parameter_group = QGroupBox("4. Selected lesion")
        parameter_layout = QGridLayout(parameter_group)
        self.column_spin = self._position_spin(0.0, 99999.0, "voxel column")
        self.row_spin = self._position_spin(0.0, 99999.0, "voxel row")
        self.slice_spin = self._position_spin(0.0, 99999.0, "zero-based slice")
        parameter_layout.addWidget(QLabel("Column"), 0, 0)
        parameter_layout.addWidget(self.column_spin, 0, 1)
        parameter_layout.addWidget(QLabel("Row"), 0, 2)
        parameter_layout.addWidget(self.row_spin, 0, 3)
        parameter_layout.addWidget(QLabel("Slice"), 1, 0)
        parameter_layout.addWidget(self.slice_spin, 1, 1)
        self.contrast_spin = self._double_spin(0.0, 5.0, 0.05, "×")
        parameter_layout.addWidget(QLabel("Contrast"), 1, 2)
        parameter_layout.addWidget(self.contrast_spin, 1, 3)
        self.scale_spins = [self._double_spin(0.25, 4.0, 0.05, "×") for _ in range(3)]
        for index, name in enumerate("XYZ"):
            row = 2 + index // 2
            column = (index % 2) * 2
            parameter_layout.addWidget(QLabel(f"Scale {name}"), row, column)
            parameter_layout.addWidget(self.scale_spins[index], row, column + 1)
        self.rotation_spins = [
            self._double_spin(-180.0, 180.0, 1.0, "°") for _ in range(3)
        ]
        for index, name in enumerate("XYZ"):
            row = 4 + index // 2
            column = (index % 2) * 2
            parameter_layout.addWidget(QLabel(f"Rotation {name}"), row, column)
            parameter_layout.addWidget(self.rotation_spins[index], row, column + 1)
        self.opacity_spin = self._double_spin(0.0, 1.0, 0.05, "")
        parameter_layout.addWidget(QLabel("Preview opacity"), 6, 0)
        parameter_layout.addWidget(self.opacity_spin, 6, 1)
        reset = QPushButton("Reset parameters")
        reset.clicked.connect(self.reset_parameters_requested)
        parameter_layout.addWidget(reset, 6, 2, 1, 2)
        self.background_hu_edit = QLineEdit()
        self.background_hu_edit.setPlaceholderText("One HU value per channel")
        self.background_hu_edit.setToolTip(
            "New-location background HU supplied to pipeline 3.1.4. "
            "A single value is repeated for multi-channel lesions."
        )
        self.background_hu_edit.editingFinished.connect(self._emit_background)
        parameter_layout.addWidget(QLabel("Background HU"), 7, 0)
        parameter_layout.addWidget(self.background_hu_edit, 7, 1, 1, 3)
        for spin in (self.column_spin, self.row_spin, self.slice_spin):
            spin.valueChanged.connect(self._emit_position)
        for spin in (
            self.contrast_spin,
            *self.scale_spins,
            *self.rotation_spins,
            self.opacity_spin,
        ):
            spin.valueChanged.connect(self._emit_parameters)
        layout.addWidget(parameter_group)

        output_group = QGroupBox("5. Final processing")
        output_layout = QFormLayout(output_group)
        self.output_row = PathRow(
            placeholder="New or empty output directory",
            action_text="Select",
        )
        self.output_row.action.setVisible(False)
        self.output_row.changed.connect(
            lambda _text: self.session_fields_changed.emit()
        )
        output_layout.addRow("Output", self.output_row)
        self.spectrum_map = QLineEdit('{"1": 0, "2": 1}')
        self.spectrum_map.setToolTip(
            "JSON mapping from 1-based CTPD spectrum index to 0-based lesion-model channel."
        )
        self.spectrum_map.textChanged.connect(
            lambda _text: self.session_fields_changed.emit()
        )
        output_layout.addRow("Spectrum → channel", self.spectrum_map)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, 64)
        self.workers_spin.setValue(1)
        self.workers_spin.setToolTip(
            "Validate with one worker first. Workers parallelize DICOM files, not "
            "frames within one multi-frame file."
        )
        self.workers_spin.valueChanged.connect(
            lambda _value: self.session_fields_changed.emit()
        )
        output_layout.addRow("Projection workers", self.workers_spin)
        self.reconstruction_command = QLineEdit()
        self.reconstruction_command.setPlaceholderText(
            "Optional executable with {input}, {output}, {session} placeholders"
        )
        self.reconstruction_command.textChanged.connect(
            lambda _text: self.session_fields_changed.emit()
        )
        output_layout.addRow("Reconstruction command", self.reconstruction_command)
        self.reconstruction_output_row = PathRow(
            placeholder="Optional reconstructed-output directory",
            action_text="Select",
        )
        self.reconstruction_output_row.action.setVisible(False)
        self.reconstruction_output_row.changed.connect(
            lambda _text: self.session_fields_changed.emit()
        )
        output_layout.addRow("Reconstruction output", self.reconstruction_output_row)
        layout.addWidget(output_group)

        session_row = QHBoxLayout()
        save = QPushButton("Save session…")
        save.clicked.connect(self.save_session_requested)
        load = QPushButton("Load session…")
        load.clicked.connect(self.load_session_requested)
        validate = QPushButton("Validate")
        validate.clicked.connect(self.validate_requested)
        session_row.addWidget(save)
        session_row.addWidget(load)
        session_row.addWidget(validate)
        layout.addLayout(session_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run final insertion")
        self.run_button.clicked.connect(self.run_requested)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_requested)
        run_row.addWidget(self.run_button, 1)
        run_row.addWidget(self.cancel_button)
        layout.addLayout(run_row)
        layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll)

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, step: float, suffix: str
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(1.0 if minimum <= 1.0 <= maximum else minimum)
        if suffix:
            spin.setSuffix(f" {suffix}")
        return spin

    @staticmethod
    def _position_spin(minimum: float, maximum: float, tooltip: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(1.0)
        spin.setToolTip(tooltip)
        return spin

    def reconstruction_path(self) -> str:
        return self.reconstruction_row.path()

    def raw_path(self) -> str:
        return self.raw_row.path()

    def library_path(self) -> str:
        return self.library_row.path()

    def output_path(self) -> str:
        return self.output_row.path()

    def reconstruction_output_path(self) -> str:
        return self.reconstruction_output_row.path()

    def selected_series_uid(self) -> str:
        return str(self.series_combo.currentData() or "")

    def selected_lesion_id(self) -> str:
        selected = self.lesion_table.selectionModel().selectedRows()
        return str(selected[0].data(Qt.ItemDataRole.UserRole) or "") if selected else ""

    def selected_library_item(self) -> LesionLibraryItem | None:
        item = self.library_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def spectrum_mapping(self) -> dict[int, int]:
        try:
            value = json.loads(self.spectrum_map.text().strip() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Spectrum mapping is not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError("Spectrum mapping must be a JSON object.")
        return {int(key): int(channel) for key, channel in value.items()}

    def set_series(
        self, values: list[DicomSeriesCandidate], selected_uid: str = ""
    ) -> None:
        selected_index = 0
        with QSignalBlocker(self.series_combo):
            self.series_combo.clear()
            for index, candidate in enumerate(values):
                selection_key = candidate.effective_selection_key
                self.series_combo.addItem(candidate.display_name, selection_key)
                if selection_key == selected_uid:
                    selected_index = index
            self.series_combo.setEnabled(bool(values))
            if values:
                self.series_combo.setCurrentIndex(selected_index)
        if values:
            self.series_selected.emit(str(self.series_combo.currentData()))

    def set_position_ranges(self, columns: int, rows: int, slices: int) -> None:
        self.column_spin.setRange(0.0, max(0.0, float(columns - 1)))
        self.row_spin.setRange(0.0, max(0.0, float(rows - 1)))
        self.slice_spin.setRange(0.0, max(0.0, float(slices - 1)))

    def set_library_items(self, values: list[LesionLibraryItem]) -> None:
        self._library_items = values
        self._filter_library(self.library_search.text())

    @Slot(str)
    def _filter_library(self, query: str) -> None:
        normalized = query.strip().casefold()
        current = self.selected_library_item()
        current_path = current.path if current else ""
        with QSignalBlocker(self.library_list):
            self.library_list.clear()
            selected_row = -1
            for value in self._library_items:
                haystack = (
                    f"{value.display_name} {value.relative_path} {value.lesion_id}"
                ).casefold()
                if normalized and normalized not in haystack:
                    continue
                label = value.display_name
                if not value.compatible:
                    label += "  ⚠ incompatible"
                item = QListWidgetItem(label)
                item.setData(Qt.ItemDataRole.UserRole, value)
                item.setToolTip(value.warning or value.relative_path)
                self.library_list.addItem(item)
                if value.path == current_path:
                    selected_row = self.library_list.count() - 1
            if self.library_list.count():
                self.library_list.setCurrentRow(
                    selected_row if selected_row >= 0 else 0
                )
        self._library_row_changed(self.library_list.currentRow())

    @Slot(int)
    def _library_row_changed(self, row: int) -> None:
        item = self.library_list.item(row) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.library_thumbnail.setText("Loading thumbnail…")
        self.library_thumbnail.setPixmap(QPixmap())
        if not isinstance(value, LesionLibraryItem):
            self.library_metadata.clear()
            self.library_thumbnail.setText("No thumbnail")
            return
        if value.compatible:
            lines = [
                f"ID: {value.lesion_id}",
                f"Channels: {value.channel_count}",
                f"Shape (row, column, slice): {value.dimensions_rcs}",
                f"Spacing (mm): {value.spacing_rcs_mm}",
            ]
            for key in (
                "contrast_hu_by_channel",
                "old_background_method",
                "old_background_voxel_count",
                "kvp",
            ):
                if key in value.metadata:
                    lines.append(f"{key}: {value.metadata[key]}")
        else:
            lines = ["Incompatible lesion model", value.warning]
        self.library_metadata.setPlainText("\n".join(lines))
        self.library_selected.emit(value)

    def set_library_thumbnail(self, array: np.ndarray | None) -> None:
        if array is None or array.ndim != 2 or array.size == 0:
            self.library_thumbnail.setPixmap(QPixmap())
            self.library_thumbnail.setText("No thumbnail")
            return
        pixels = np.ascontiguousarray(array.astype(np.uint8, copy=False))
        image = QImage(
            pixels.data,
            pixels.shape[1],
            pixels.shape[0],
            pixels.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            128,
            128,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.library_thumbnail.setText("")
        self.library_thumbnail.setPixmap(pixmap)

    def set_lesions(self, values: tuple[LesionInstance, ...], selected_id: str) -> None:
        self._lesions = list(values)
        self._updating = True
        try:
            self.lesion_table.setRowCount(len(values))
            selected_row = -1
            for row, lesion in enumerate(values):
                enabled = QTableWidgetItem()
                enabled.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                enabled.setCheckState(
                    Qt.CheckState.Checked if lesion.enabled else Qt.CheckState.Unchecked
                )
                enabled.setData(Qt.ItemDataRole.UserRole, lesion.instance_id)
                visible = QTableWidgetItem()
                visible.setFlags(enabled.flags())
                visible.setCheckState(
                    Qt.CheckState.Checked if lesion.visible else Qt.CheckState.Unchecked
                )
                self.lesion_table.setItem(row, 0, enabled)
                self.lesion_table.setItem(row, 1, visible)
                for column, text in enumerate(
                    (
                        lesion.label,
                        lesion.lesion_id,
                        f"{lesion.center_voxel_crs[0]:.1f}",
                        f"{lesion.center_voxel_crs[1]:.1f}",
                        f"{lesion.center_voxel_crs[2]:.1f}",
                    ),
                    start=2,
                ):
                    cell = QTableWidgetItem(text)
                    cell.setData(Qt.ItemDataRole.UserRole, lesion.instance_id)
                    self.lesion_table.setItem(row, column, cell)
                if lesion.instance_id == selected_id:
                    selected_row = row
            self.lesion_table.resizeColumnsToContents()
            if selected_row >= 0:
                self.lesion_table.selectRow(selected_row)
        finally:
            self._updating = False

    def set_selected_lesion(self, lesion: LesionInstance | None) -> None:
        controls = [
            self.column_spin,
            self.row_spin,
            self.slice_spin,
            self.contrast_spin,
            *self.scale_spins,
            *self.rotation_spins,
            self.opacity_spin,
            self.background_hu_edit,
        ]
        self._updating = True
        try:
            for control in controls:
                control.setEnabled(lesion is not None)
            if lesion is None:
                self.background_hu_edit.clear()
                return
            self.column_spin.setValue(lesion.center_voxel_crs[0])
            self.row_spin.setValue(lesion.center_voxel_crs[1])
            self.slice_spin.setValue(lesion.center_voxel_crs[2])
            self.contrast_spin.setValue(lesion.parameters.contrast_scale)
            for spin, value in zip(
                self.scale_spins, lesion.parameters.scale_xyz, strict=True
            ):
                spin.setValue(value)
            for spin, value in zip(
                self.rotation_spins, lesion.parameters.rotation_deg_xyz, strict=True
            ):
                spin.setValue(value)
            self.opacity_spin.setValue(lesion.parameters.preview_opacity)
            self.background_hu_edit.setText(
                ", ".join(f"{value:g}" for value in lesion.background_hu)
            )
        finally:
            self._updating = False

    @Slot()
    def _lesion_selection_changed(self) -> None:
        if not self._updating:
            self.lesion_selected.emit(self.selected_lesion_id())

    @Slot(QTableWidgetItem)
    def _lesion_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() not in {0, 1}:
            return
        row = item.row()
        instance_id = str(
            self.lesion_table.item(row, 0).data(Qt.ItemDataRole.UserRole) or ""
        )
        enabled = self.lesion_table.item(row, 0).checkState() == Qt.CheckState.Checked
        visible = self.lesion_table.item(row, 1).checkState() == Qt.CheckState.Checked
        self.lesion_flags_changed.emit(instance_id, enabled, visible)

    @Slot(int)
    def _series_changed(self, _index: int) -> None:
        uid = self.selected_series_uid()
        if uid:
            self.series_selected.emit(uid)
            self.session_fields_changed.emit()

    @Slot()
    def _emit_position(self) -> None:
        if not self._updating:
            self.position_changed.emit(
                (
                    self.column_spin.value(),
                    self.row_spin.value(),
                    self.slice_spin.value(),
                )
            )

    @Slot()
    def _emit_background(self) -> None:
        if not self._updating:
            self.background_changed.emit(self.background_hu_edit.text().strip())

    @Slot()
    def _emit_parameters(self) -> None:
        if self._updating:
            return
        self.parameters_changed.emit(
            LesionParameters(
                contrast_scale=self.contrast_spin.value(),
                scale_xyz=(
                    self.scale_spins[0].value(),
                    self.scale_spins[1].value(),
                    self.scale_spins[2].value(),
                ),
                rotation_deg_xyz=(
                    self.rotation_spins[0].value(),
                    self.rotation_spins[1].value(),
                    self.rotation_spins[2].value(),
                ),
                preview_opacity=self.opacity_spin.value(),
            )
        )

    def set_running(self, running: bool) -> None:
        self.run_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def show_validation(self, title: str, text: str, *, error: bool = False) -> None:
        if error:
            QMessageBox.critical(self, title, text)
        else:
            QMessageBox.information(self, title, text)
