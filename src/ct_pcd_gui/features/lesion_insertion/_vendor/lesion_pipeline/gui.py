from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWizard,
    QWizardPage,
    QWidget,
)
from PyQt6.QtCore import Qt

from .config import InsertionConfig, InsertionSpec
from .coordinates import pixel_center_to_ctpd
from .ctpd import discover_projection_records
from .lesion_models import generate_lesion_models, load_lesion_model
from .pipeline import run_insertion


class PathRow(QWidget):
    def __init__(
        self,
        *,
        directory: bool,
        save_directory: bool = False,
        file_or_directory: bool = False,
    ) -> None:
        super().__init__()
        self.directory = directory
        self.save_directory = save_directory
        self.file_or_directory = file_or_directory
        self.edit = QLineEdit()
        button = QPushButton("Browse folder" if file_or_directory else "Browse")
        button.clicked.connect(self.browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)
        if file_or_directory:
            file_button = QPushButton("Browse file")
            file_button.clicked.connect(self.browse_file)
            layout.addWidget(file_button)

    @property
    def value(self) -> str:
        return self.edit.text().strip().strip('"')

    def browse(self) -> None:
        if self.directory or self.save_directory:
            selected = QFileDialog.getExistingDirectory(self, "Select directory")
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Select NRRD segmentation",
                "",
                "NRRD files (*.nrrd);;All files (*)",
            )
        if selected:
            self.edit.setText(selected)

    def browse_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select DICOM-CT-PD projection",
            "",
            "DICOM files (*.dcm *.ima *.IMA);;All files (*)",
        )
        if selected:
            self.edit.setText(selected)


class InputsPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Inputs")
        self.setSubTitle(
            "Select the two reconstructed threshold series, lesion segmentation, "
            "DICOM-CT-PD projection series, and output directory."
        )
        self.t1 = PathRow(directory=True)
        self.t2 = PathRow(directory=True)
        self.nrrd = PathRow(directory=False)
        self.ctpd = PathRow(directory=True, file_or_directory=True)
        self.output = PathRow(directory=True, save_directory=True)
        self.alignment = QComboBox()
        self.alignment.addItem("Physical NRRD/DICOM coordinates (recommended)", "physical")
        self.alignment.addItem("Legacy flip/rotate alignment", "legacy")
        self.workers = QSpinBox()
        self.workers.setRange(1, 64)
        self.workers.setValue(1)
        form = QFormLayout(self)
        form.addRow("T1 reconstructed DICOM series:", self.t1)
        form.addRow("T2 reconstructed DICOM series:", self.t2)
        form.addRow("Lesion segmentation (.nrrd):", self.nrrd)
        form.addRow("DICOM-CT-PD input directory:", self.ctpd)
        form.addRow("Output directory:", self.output)
        form.addRow("Mask alignment:", self.alignment)
        form.addRow("Projection workers:", self.workers)

    def validatePage(self) -> bool:
        required = {
            "T1 series": self.t1.value,
            "T2 series": self.t2.value,
            "NRRD segmentation": self.nrrd.value,
            "DICOM-CT-PD input": self.ctpd.value,
            "Output directory": self.output.value,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            QMessageBox.warning(self, "Missing input", "Select: " + ", ".join(missing))
            return False
        for name in ("T1 series", "T2 series"):
            if not Path(required[name]).is_dir():
                QMessageBox.warning(self, "Invalid input", f"{name} is not a directory.")
                return False
        if not Path(required["DICOM-CT-PD input"]).exists():
            QMessageBox.warning(self, "Invalid input", "The DICOM-CT-PD input was not found.")
            return False
        if not Path(required["NRRD segmentation"]).is_file():
            QMessageBox.warning(self, "Invalid input", "The NRRD segmentation was not found.")
            return False
        Path(self.output.value).mkdir(parents=True, exist_ok=True)
        return True


class PlacementPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Select lesion locations")
        self.setSubTitle(
            "Draw a rectangle on T1, choose the lesion model, and add the placement. "
            "New-location T1/T2 background HU values are measured from the same ROI; "
            "the projected lesion contrast uses the model's original background."
        )
        self.t1_series = None
        self.t2_series = None
        self.model_paths: list[Path] = []
        self.insertions: list[InsertionSpec] = []
        self.current_roi: tuple[int, int, int, int] | None = None
        self.current_slice = 0
        self.roi_means: tuple[float, float] | None = None

        self.figure_t1 = Figure(figsize=(5, 5), tight_layout=True)
        self.figure_t2 = Figure(figsize=(5, 5), tight_layout=True)
        self.canvas_t1 = FigureCanvas(self.figure_t1)
        self.canvas_t2 = FigureCanvas(self.figure_t2)
        self.ax_t1 = self.figure_t1.add_subplot(111)
        self.ax_t2 = self.figure_t2.add_subplot(111)
        self.selector = RectangleSelector(
            self.ax_t1,
            self._roi_selected,
            useblit=True,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords="pixels",
            interactive=True,
        )
        image_layout = QHBoxLayout()
        image_layout.addWidget(self.canvas_t1)
        image_layout.addWidget(self.canvas_t2)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._slice_changed)
        self.slice_label = QLabel("Slice: -")
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slice_label)
        slider_layout.addWidget(self.slider, 1)

        self.model_combo = QComboBox()
        self.model_info = QLabel("Model: -")
        self.model_combo.currentIndexChanged.connect(self._update_model_info)
        self.roi_info = QLabel("Draw an ROI on T1.")
        add_button = QPushButton("Add this placement")
        add_button.clicked.connect(self._add_placement)
        remove_button = QPushButton("Remove selected placement")
        remove_button.clicked.connect(self._remove_placement)
        controls = QGridLayout()
        controls.addWidget(QLabel("Lesion model:"), 0, 0)
        controls.addWidget(self.model_combo, 0, 1)
        controls.addWidget(self.model_info, 0, 2)
        controls.addWidget(self.roi_info, 1, 0, 1, 3)
        controls.addWidget(add_button, 2, 1)
        controls.addWidget(remove_button, 2, 2)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            [
                "Label",
                "Model",
                "Column",
                "Row",
                "Slice",
                "New BG T1",
                "New BG T2",
            ]
        )
        layout = QVBoxLayout(self)
        layout.addLayout(image_layout, 1)
        layout.addLayout(slider_layout)
        layout.addLayout(controls)
        layout.addWidget(self.table)

    def initializePage(self) -> None:
        if self.t1_series is not None:
            return
        inputs: InputsPage = self.wizard().page(0)  # type: ignore[assignment]
        try:
            model_dir = Path(inputs.output.value) / "lesion_models"
            self.model_paths, self.t1_series, self.t2_series = generate_lesion_models(
                inputs.t1.value,
                inputs.t2.value,
                inputs.nrrd.value,
                model_dir,
                mask_alignment=str(inputs.alignment.currentData()),
                write_mat=True,
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "Data preparation failed",
                f"{error}\n\nCheck that T1, T2, and NRRD describe the same patient and grid.",
            )
            self.wizard().back()
            return
        for path in self.model_paths:
            model = load_lesion_model(path)
            self.model_combo.addItem(f"Lesion {model.lesion_number:02d}", str(path))
        self.slider.setRange(0, self.t1_series.shape[0] - 1)
        self._update_model_info()
        self._update_images()

    def isComplete(self) -> bool:
        return bool(self.insertions)

    def _slice_changed(self, value: int) -> None:
        self.current_slice = value
        self.current_roi = None
        self.roi_means = None
        self.roi_info.setText("Draw an ROI on T1.")
        self._update_images()

    def _update_images(self) -> None:
        if self.t1_series is None or self.t2_series is None:
            return
        t1_image = self.t1_series.hu[self.current_slice]
        t2_image = self.t2_series.hu[self.current_slice]
        self.ax_t1.clear()
        self.ax_t2.clear()
        self.ax_t1.imshow(t1_image, cmap="gray", vmin=-140, vmax=260)
        self.ax_t2.imshow(t2_image, cmap="gray", vmin=-140, vmax=260)
        self.ax_t1.set_title("T1")
        self.ax_t2.set_title("T2")
        self.ax_t1.set_axis_off()
        self.ax_t2.set_axis_off()
        self.selector = RectangleSelector(
            self.ax_t1,
            self._roi_selected,
            useblit=True,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords="pixels",
            interactive=True,
        )
        self.canvas_t1.draw_idle()
        self.canvas_t2.draw_idle()
        self.slice_label.setText(
            f"Slice: {self.current_slice + 1}/{self.t1_series.shape[0]}"
        )

    def _roi_selected(self, click, release) -> None:
        if self.t1_series is None or self.t2_series is None:
            return
        if None in (click.xdata, click.ydata, release.xdata, release.ydata):
            return
        columns = self.t1_series.shape[2]
        rows = self.t1_series.shape[1]
        x0, x1 = sorted((int(round(click.xdata)), int(round(release.xdata))))
        y0, y1 = sorted((int(round(click.ydata)), int(round(release.ydata))))
        x0, x1 = max(0, x0), min(columns, x1 + 1)
        y0, y1 = max(0, y0), min(rows, y1 + 1)
        if x1 <= x0 or y1 <= y0:
            return
        self.current_roi = (x0, y0, x1, y1)
        t1_roi = self.t1_series.hu[self.current_slice, y0:y1, x0:x1]
        t2_roi = self.t2_series.hu[self.current_slice, y0:y1, x0:x1]
        self.roi_means = (float(np.mean(t1_roi)), float(np.mean(t2_roi)))
        self.roi_info.setText(
            f"ROI: columns {x0}-{x1 - 1}, rows {y0}-{y1 - 1}; "
            f"mean T1={self.roi_means[0]:.2f} HU, T2={self.roi_means[1]:.2f} HU"
        )

    def _update_model_info(self) -> None:
        if self.model_combo.currentIndex() < 0:
            return
        model = load_lesion_model(self.model_combo.currentData())
        lesion_means = ", ".join(
            f"{value:.1f}" for value in model.lesion_mean_hu_by_channel
        )
        old_backgrounds = ", ".join(
            f"{value:.1f}" for value in model.old_background_hu
        )
        self.model_info.setText(
            f"{model.channel_count} channels; lesion means [{lesion_means}] HU; "
            f"old backgrounds [{old_backgrounds}] HU from "
            f"{model.old_background_voxel_count} inverse-mask VOI voxels"
        )

    def _add_placement(self) -> None:
        if self.current_roi is None or self.roi_means is None or self.t1_series is None:
            QMessageBox.warning(self, "No ROI", "Draw a rectangular ROI on the T1 image first.")
            return
        x0, y0, x1, y1 = self.current_roi
        center_pixel = [(x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0, self.current_slice]
        model_path = str(self.model_combo.currentData())
        label = f"Placement {len(self.insertions) + 1}"
        insertion = InsertionSpec(
            lesion_model=model_path,
            center_pixel=center_pixel,
            background_hu=[self.roi_means[0], self.roi_means[1]],
            label=label,
        )
        # Perform the coordinate conversion now so the user can catch obvious errors.
        center_ctpd = pixel_center_to_ctpd(center_pixel, self.t1_series)
        self.insertions.append(insertion)
        row = self.table.rowCount()
        self.table.insertRow(row)
        values = [
            label,
            Path(model_path).name,
            f"{center_pixel[0]:.1f}",
            f"{center_pixel[1]:.1f}",
            str(self.current_slice),
            f"{self.roi_means[0]:.2f}",
            f"{self.roi_means[1]:.2f}",
        ]
        for column, value in enumerate(values):
            self.table.setItem(row, column, QTableWidgetItem(value))
        self.roi_info.setText(
            f"Added {label}; CT-PD center = "
            f"({center_ctpd[0]:.2f}, {center_ctpd[1]:.2f}, {center_ctpd[2]:.2f}) mm"
        )
        self.completeChanged.emit()

    def _remove_placement(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        self.table.removeRow(row)
        self.insertions.pop(row)
        self.completeChanged.emit()


class PipelineWorker(QObject):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, config: InsertionConfig) -> None:
        super().__init__()
        self.config = config

    def run(self) -> None:
        try:
            summary = run_insertion(
                self.config,
                progress=lambda done, total, name: self.progress.emit(done, total, name),
            )
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RunPage(QWizardPage):
    def __init__(self) -> None:
        super().__init__()
        self.setTitle("Map spectra and create the output")
        self.setSubTitle(
            "Confirm which lesion-model channel corresponds to each DICOM-CT-PD spectrum, "
            "then run the insertion."
        )
        self.mapping_table = QTableWidget(0, 4)
        self.mapping_table.setHorizontalHeaderLabels(
            ["Spectrum", "Source", "Projection files", "Lesion-model channel"]
        )
        self.run_button = QPushButton("Start lesion insertion")
        self.run_button.clicked.connect(self._start)
        self.progress = QProgressBar()
        self.status = QLabel("Ready")
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.mapping_table)
        layout.addWidget(self.run_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        layout.addWidget(self.log)
        self._done = False
        self._thread: QThread | None = None
        self._worker: PipelineWorker | None = None

    def initializePage(self) -> None:
        self.mapping_table.setRowCount(0)
        inputs: InputsPage = self.wizard().page(0)  # type: ignore[assignment]
        placements: PlacementPage = self.wizard().page(1)  # type: ignore[assignment]
        try:
            records = discover_projection_records(inputs.ctpd.value)
        except Exception as error:
            QMessageBox.critical(self, "DICOM-CT-PD inspection failed", str(error))
            self.wizard().back()
            return
        groups: dict[tuple[int, int], int] = {}
        for record in records:
            for spectrum, source, frame_count in record.frame_group_counts:
                key = (spectrum, source)
                groups[key] = groups.get(key, 0) + frame_count
        channel_count = min(
            load_lesion_model(spec.lesion_model).channel_count
            for spec in placements.insertions
        )
        spectra = sorted({key[0] for key in groups})
        default_channel = {
            spectrum: min(index, channel_count - 1)
            for index, spectrum in enumerate(spectra)
        }
        for row, ((spectrum, source), count) in enumerate(sorted(groups.items())):
            self.mapping_table.insertRow(row)
            self.mapping_table.setItem(row, 0, QTableWidgetItem(str(spectrum)))
            self.mapping_table.setItem(row, 1, QTableWidgetItem(str(source)))
            self.mapping_table.setItem(row, 2, QTableWidgetItem(str(count)))
            combo = QComboBox()
            for channel in range(channel_count):
                combo.addItem(f"Channel {channel + 1}", channel)
            combo.setCurrentIndex(default_channel[spectrum])
            self.mapping_table.setCellWidget(row, 3, combo)
        total_frames = sum(record.frame_count for record in records)
        self.progress.setRange(0, total_frames)
        self.progress.setValue(0)
        self.status.setText(
            f"Ready: {total_frames} projection frames in {len(records)} DICOM file(s)"
        )

    def isComplete(self) -> bool:
        return self._done

    def _mapping(self) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for row in range(self.mapping_table.rowCount()):
            spectrum = int(self.mapping_table.item(row, 0).text())
            combo: QComboBox = self.mapping_table.cellWidget(row, 3)  # type: ignore[assignment]
            selected = int(combo.currentData())
            if spectrum in mapping and mapping[spectrum] != selected:
                raise ValueError(
                    f"Spectrum {spectrum} was mapped to different channels across sources."
                )
            mapping[spectrum] = selected
        return mapping

    def _start(self) -> None:
        inputs: InputsPage = self.wizard().page(0)  # type: ignore[assignment]
        placements: PlacementPage = self.wizard().page(1)  # type: ignore[assignment]
        try:
            mapping = self._mapping()
        except Exception as error:
            QMessageBox.warning(self, "Invalid mapping", str(error))
            return
        output = Path(inputs.output.value) / "dicom_ctpd_modified"
        config = InsertionConfig(
            ctpd_input=inputs.ctpd.value,
            output_dir=str(output),
            t1_series_dir=inputs.t1.value,
            insertions=list(placements.insertions),
            spectrum_channel_map=mapping,
            workers=inputs.workers.value(),
            overwrite=False,
        )
        config.save(Path(inputs.output.value) / "lesion_insertion_config.json")
        self.run_button.setEnabled(False)
        self.log.clear()
        self.status.setText("Running...")
        self._thread = QThread(self)
        self._worker = PipelineWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_progress(self, done: int, total: int, name: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"{done}/{total}: {name}")
        self.log.append(f"{done}/{total}  {name}")

    def _on_finished(self, summary: dict) -> None:
        self._done = True
        self.status.setText(
            f"Complete: {summary['changed_projection_frames']} projection frames changed; "
            f"{summary['clipped_pixels']} clipped pixels."
        )
        self.log.append(f"Output: {summary['output']}")
        self.completeChanged.emit()

    def _on_failed(self, details: str) -> None:
        self.run_button.setEnabled(True)
        self.status.setText("Failed")
        self.log.setPlainText(details)
        QMessageBox.critical(self, "Insertion failed", details.splitlines()[-1])


class LesionInsertionWizard(QWizard):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DICOM-CT-PD Lesion Insertion")
        self.resize(1300, 900)
        self.setPage(0, InputsPage())
        self.setPage(1, PlacementPage())
        self.setPage(2, RunPage())
        self.setStartId(0)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    wizard = LesionInsertionWizard()
    wizard.show()
    return app.exec()
