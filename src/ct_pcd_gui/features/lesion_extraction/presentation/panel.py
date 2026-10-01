from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..models import CandidateSelection, DicomSeriesSpec, LesionCandidate, SegmentDefinition


class LesionExtractionPanel(QWidget):
    add_dicom_requested = Signal()
    browse_segmentation_requested = Signal()
    browse_output_requested = Signal()
    load_requested = Signal()
    split_requested = Signal()
    export_requested = Signal()
    cancel_requested = Signal()
    inputs_changed = Signal()
    segment_selection_changed = Signal()
    candidate_state_changed = Signal()
    candidate_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._stage = "empty"
        self._busy = False
        self._build_ui()
        self._connect()
        self.set_stage("empty")

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll)
        content = QWidget(scroll)
        scroll.setWidget(content)
        root = QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        heading = QLabel("Lesion Extraction", content)
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        root.addWidget(heading)
        description = QLabel(
            "Load DICOM series and a NRRD segmentation, select lesion labels, "
            "review connected components, and export compatible NPZ models.",
            content,
        )
        description.setWordWrap(True)
        root.addWidget(description)

        self.input_group = QGroupBox("1. Inputs", content)
        input_layout = QVBoxLayout(self.input_group)
        input_layout.addWidget(QLabel("DICOM series folders:"))
        self.dicom_list = QListWidget(self.input_group)
        self.dicom_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.dicom_list.setMinimumHeight(90)
        input_layout.addWidget(self.dicom_list)
        buttons = QHBoxLayout()
        self.add_dicom_button = QPushButton("Add folder...", self.input_group)
        self.remove_dicom_button = QPushButton("Remove", self.input_group)
        buttons.addWidget(self.add_dicom_button)
        buttons.addWidget(self.remove_dicom_button)
        buttons.addStretch(1)
        input_layout.addLayout(buttons)
        input_layout.addWidget(QLabel("NRRD segmentation:"))
        row = QHBoxLayout()
        self.segmentation_path = QLineEdit(self.input_group)
        self.segmentation_path.setPlaceholderText("Select a .nrrd or .seg.nrrd file")
        self.segmentation_browse = QPushButton("Browse...", self.input_group)
        row.addWidget(self.segmentation_path, 1)
        row.addWidget(self.segmentation_browse)
        input_layout.addLayout(row)
        self.load_button = QPushButton("Load and preview", self.input_group)
        self.load_button.setDefault(True)
        input_layout.addWidget(self.load_button)
        self.series_summary = QLabel("No DICOM series loaded", self.input_group)
        self.series_summary.setWordWrap(True)
        input_layout.addWidget(self.series_summary)
        root.addWidget(self.input_group)

        self.segment_group = QGroupBox("2. Choose lesion labels", content)
        segment_layout = QVBoxLayout(self.segment_group)
        segment_layout.addWidget(
            QLabel("Checked labels are split into separate 3-D connected components.")
        )
        self.segment_tree = QTreeWidget(self.segment_group)
        self.segment_tree.setHeaderLabels(["Use", "Segment", "Value", "Layer"])
        self.segment_tree.setRootIsDecorated(False)
        self.segment_tree.setAlternatingRowColors(True)
        self.segment_tree.setMinimumHeight(150)
        segment_layout.addWidget(self.segment_tree)
        buttons = QHBoxLayout()
        self.select_all_segments = QPushButton("Select all", self.segment_group)
        self.select_no_segments = QPushButton("Select none", self.segment_group)
        buttons.addWidget(self.select_all_segments)
        buttons.addWidget(self.select_no_segments)
        buttons.addStretch(1)
        segment_layout.addLayout(buttons)
        form = QFormLayout()
        self.minimum_voxels = QSpinBox(self.segment_group)
        self.minimum_voxels.setRange(1, 1_000_000_000)
        self.minimum_voxels.setValue(1)
        form.addRow("Minimum component voxels:", self.minimum_voxels)
        segment_layout.addLayout(form)
        self.split_button = QPushButton("Preview lesion split", self.segment_group)
        segment_layout.addWidget(self.split_button)
        root.addWidget(self.segment_group)

        self.candidate_group = QGroupBox("3. Review lesions", content)
        candidate_layout = QVBoxLayout(self.candidate_group)
        candidate_layout.addWidget(
            QLabel("Uncheck lesions to omit them. Output names can be edited.")
        )
        self.candidate_table = QTableWidget(0, 6, self.candidate_group)
        self.candidate_table.setHorizontalHeaderLabels(
            ["Use", "Output name", "Segment", "Part", "Voxels", "Volume (mm3)"]
        )
        self.candidate_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.candidate_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setMinimumHeight(180)
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        candidate_layout.addWidget(self.candidate_table)
        buttons = QHBoxLayout()
        self.include_all_candidates = QPushButton("Include all", self.candidate_group)
        self.include_no_candidates = QPushButton("Omit all", self.candidate_group)
        buttons.addWidget(self.include_all_candidates)
        buttons.addWidget(self.include_no_candidates)
        buttons.addStretch(1)
        self.candidate_count = QLabel("0 lesions", self.candidate_group)
        buttons.addWidget(self.candidate_count)
        candidate_layout.addLayout(buttons)
        root.addWidget(self.candidate_group)

        self.output_group = QGroupBox("4. Export", content)
        output_layout = QVBoxLayout(self.output_group)
        output_layout.addWidget(QLabel("Output folder:"))
        row = QHBoxLayout()
        self.output_path = QLineEdit(self.output_group)
        self.output_path.setPlaceholderText("Choose an output folder")
        self.output_browse = QPushButton("Browse...", self.output_group)
        row.addWidget(self.output_path, 1)
        row.addWidget(self.output_browse)
        output_layout.addLayout(row)
        self.export_button = QPushButton("Export included lesions", self.output_group)
        output_layout.addWidget(self.export_button)
        root.addWidget(self.output_group)

        self.status_label = QLabel("Add inputs to begin.", content)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)
        self.progress = QProgressBar(content)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)
        self.cancel_button = QPushButton("Cancel", content)
        self.cancel_button.setEnabled(False)
        root.addWidget(self.cancel_button)
        self.log = QPlainTextEdit(content)
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        self.log.setMaximumHeight(110)
        root.addWidget(self.log)
        root.addStretch(1)

    def _connect(self) -> None:
        self.add_dicom_button.clicked.connect(self.add_dicom_requested)
        self.remove_dicom_button.clicked.connect(self._remove_selected_dicom)
        self.segmentation_browse.clicked.connect(self.browse_segmentation_requested)
        self.output_browse.clicked.connect(self.browse_output_requested)
        self.load_button.clicked.connect(self.load_requested)
        self.split_button.clicked.connect(self.split_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.cancel_button.clicked.connect(self.cancel_requested)
        self.segmentation_path.textChanged.connect(self.inputs_changed)
        self.dicom_list.model().rowsInserted.connect(lambda *_: self.inputs_changed.emit())
        self.dicom_list.model().rowsRemoved.connect(lambda *_: self.inputs_changed.emit())
        self.segment_tree.itemChanged.connect(
            lambda *_: self.segment_selection_changed.emit()
        )
        self.minimum_voxels.valueChanged.connect(
            lambda *_: self.segment_selection_changed.emit()
        )
        self.select_all_segments.clicked.connect(lambda: self._check_segments(True))
        self.select_no_segments.clicked.connect(lambda: self._check_segments(False))
        self.candidate_table.itemChanged.connect(lambda *_: self._candidate_changed())
        self.candidate_table.itemSelectionChanged.connect(self._candidate_selected)
        self.include_all_candidates.clicked.connect(lambda: self._check_candidates(True))
        self.include_no_candidates.clicked.connect(lambda: self._check_candidates(False))

    def add_dicom_folder(self, folder: str | Path) -> None:
        path = str(Path(folder).expanduser().resolve())
        for index in range(self.dicom_list.count()):
            if self.dicom_list.item(index).data(Qt.ItemDataRole.UserRole) == path:
                self.dicom_list.setCurrentRow(index)
                return
        item = QListWidgetItem(path)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.dicom_list.addItem(item)

    def _remove_selected_dicom(self) -> None:
        rows = sorted(
            {self.dicom_list.row(item) for item in self.dicom_list.selectedItems()},
            reverse=True,
        )
        for row in rows:
            self.dicom_list.takeItem(row)

    def dicom_folders(self) -> tuple[str, ...]:
        return tuple(
            str(self.dicom_list.item(index).data(Qt.ItemDataRole.UserRole))
            for index in range(self.dicom_list.count())
        )

    def set_discovered_series(self, series: Sequence[DicomSeriesSpec]) -> None:
        names = ", ".join(item.display_name for item in series[:3])
        if len(series) > 3:
            names += f", and {len(series) - 3} more"
        self.series_summary.setText(f"Discovered {len(series)} series: {names}")

    def set_segments(self, segments: Sequence[SegmentDefinition]) -> None:
        lesion_named = any("lesion" in segment.name.casefold() for segment in segments)
        with QSignalBlocker(self.segment_tree):
            self.segment_tree.clear()
            for segment in segments:
                item = QTreeWidgetItem(
                    ["", segment.name, str(segment.label_value), str(segment.layer)]
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = "lesion" in segment.name.casefold() if lesion_named else True
                item.setCheckState(
                    0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
                )
                item.setData(0, Qt.ItemDataRole.UserRole, segment.key)
                self.segment_tree.addTopLevelItem(item)
        for column in range(4):
            self.segment_tree.resizeColumnToContents(column)

    def selected_segment_keys(self) -> tuple[str, ...]:
        return tuple(
            str(item.data(0, Qt.ItemDataRole.UserRole))
            for index in range(self.segment_tree.topLevelItemCount())
            if (item := self.segment_tree.topLevelItem(index)).checkState(0)
            == Qt.CheckState.Checked
        )

    def _check_segments(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        with QSignalBlocker(self.segment_tree):
            for index in range(self.segment_tree.topLevelItemCount()):
                self.segment_tree.topLevelItem(index).setCheckState(0, state)
        self.segment_selection_changed.emit()

    def set_candidates(self, candidates: Sequence[LesionCandidate]) -> None:
        with QSignalBlocker(self.candidate_table):
            self.candidate_table.setRowCount(len(candidates))
            for row, candidate in enumerate(candidates):
                use = QTableWidgetItem()
                use.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                use.setCheckState(Qt.CheckState.Checked)
                use.setData(Qt.ItemDataRole.UserRole, candidate.candidate_id)
                self.candidate_table.setItem(row, 0, use)
                self.candidate_table.setItem(row, 1, QTableWidgetItem(candidate.output_name))
                self.candidate_table.setItem(row, 2, QTableWidgetItem(candidate.segment_name))
                self.candidate_table.setItem(
                    row,
                    3,
                    QTableWidgetItem(
                        f"{candidate.component_index}/{candidate.component_count}"
                    ),
                )
                self.candidate_table.setItem(
                    row, 4, QTableWidgetItem(f"{candidate.voxel_count:,}")
                )
                self.candidate_table.setItem(
                    row, 5, QTableWidgetItem(f"{candidate.physical_size_mm3:,.2f}")
                )
                for column in (2, 3, 4, 5):
                    item = self.candidate_table.item(row, column)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._candidate_changed()
        if candidates:
            self.candidate_table.selectRow(0)

    def clear_candidates(self) -> None:
        with QSignalBlocker(self.candidate_table):
            self.candidate_table.setRowCount(0)
        self.candidate_count.setText("0 lesions")

    def candidate_selections(self) -> tuple[CandidateSelection, ...]:
        result: list[CandidateSelection] = []
        for row in range(self.candidate_table.rowCount()):
            use = self.candidate_table.item(row, 0)
            name = self.candidate_table.item(row, 1)
            result.append(
                CandidateSelection(
                    candidate_id=str(use.data(Qt.ItemDataRole.UserRole)),
                    included=use.checkState() == Qt.CheckState.Checked,
                    output_name=name.text().strip(),
                )
            )
        return tuple(result)

    def select_candidate(self, candidate_id: str) -> None:
        with QSignalBlocker(self.candidate_table):
            for row in range(self.candidate_table.rowCount()):
                item = self.candidate_table.item(row, 0)
                if item.data(Qt.ItemDataRole.UserRole) == candidate_id:
                    self.candidate_table.selectRow(row)
                    self.candidate_table.scrollToItem(item)
                    break

    def _candidate_changed(self) -> None:
        included = sum(selection.included for selection in self.candidate_selections())
        self.candidate_count.setText(
            f"{self.candidate_table.rowCount()} lesion(s), {included} included"
        )
        self.candidate_state_changed.emit()

    def _candidate_selected(self) -> None:
        row = self.candidate_table.currentRow()
        if row >= 0:
            item = self.candidate_table.item(row, 0)
            self.candidate_selected.emit(str(item.data(Qt.ItemDataRole.UserRole)))

    def _check_candidates(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        with QSignalBlocker(self.candidate_table):
            for row in range(self.candidate_table.rowCount()):
                self.candidate_table.item(row, 0).setCheckState(state)
        self._candidate_changed()

    def set_stage(self, stage: str) -> None:
        self._stage = stage
        self._apply_enabled()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.cancel_button.setEnabled(busy)
        self._apply_enabled()

    def _apply_enabled(self) -> None:
        self.input_group.setEnabled(not self._busy)
        self.segment_group.setEnabled(
            not self._busy and self._stage in {"inspected", "prepared"}
        )
        self.candidate_group.setEnabled(not self._busy and self._stage == "prepared")
        self.output_group.setEnabled(not self._busy and self._stage == "prepared")

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled and not self._busy)

    def set_status(self, text: str, *, log: bool = False) -> None:
        self.status_label.setText(text)
        if log:
            self.append_log(text)

    def append_log(self, text: str) -> None:
        if text.strip():
            self.log.appendPlainText(text.strip())

    def set_progress(self, completed: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(max(0, min(completed, total)))

    def reset_progress(self) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
