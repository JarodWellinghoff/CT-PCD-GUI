from pathlib import Path
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QScrollBar,
    QApplication,
)
from PyQt6.QtCore import Qt, pyqtSignal
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkIOImage import vtkDICOMImageReader
from vtkmodules.vtkInteractionImage import vtkImageViewer2
from vtkmodules.vtkRenderingCore import (
    vtkActor2D,
    vtkTextMapper,
    vtkTextProperty,
)
from vtkmodules.vtkImagingCore import vtkImageReslice
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from src.models import ROIManager
from src.models import DrawTool
from src.interaction.drawing_interactor_style import DrawingInteractorStyle
import vtkmodules.all as vtk
import pydicom


# ===========================================================================
# DICOM Viewer widget – multi-series
# ===========================================================================


class DicomViewerWidget(QWidget):
    drawing_cancelled = pyqtSignal()
    dicom_spacing_changed = pyqtSignal(list)  # [sx, sy, sz]

    # Emitted during series load: label, 0-100 int percent, then finished
    loading_started = pyqtSignal(str)  # message describing what is loading
    loading_progress = pyqtSignal(int)  # 0-100
    loading_finished = pyqtSignal()

    def __init__(self, dicom_folder=None, view_orientation="axial", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- Series selector bar  ----
        self._series_bar = QWidget()
        bar_lay = QHBoxLayout(self._series_bar)
        bar_lay.setContentsMargins(6, 3, 6, 3)
        bar_lay.addWidget(QLabel("Series:"))
        self._series_bar.setFixedHeight(32)
        self.series_combo = QComboBox()
        self.series_combo.setMinimumWidth(260)
        self.series_combo.currentIndexChanged.connect(self._on_combo_changed)
        bar_lay.addWidget(self.series_combo, 1)
        self._btn_add_series = QPushButton("＋ Add Series")
        self._btn_add_series.setFixedWidth(110)
        self._btn_add_series.clicked.connect(self._add_series_dialog)
        bar_lay.addWidget(self._btn_add_series)
        self._btn_remove_series = QPushButton("✕ Remove")
        self._btn_remove_series.setFixedWidth(90)
        self._btn_remove_series.clicked.connect(self._remove_active_series)
        bar_lay.addWidget(self._btn_remove_series)
        layout.addWidget(self._series_bar)

        # ---- VTK widget + slice scrollbar side-by-side ----
        viewer_row = QHBoxLayout()
        viewer_row.setContentsMargins(0, 0, 0, 0)
        viewer_row.setSpacing(0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        viewer_row.addWidget(self.vtk_widget, 1)

        self._slice_scrollbar = QScrollBar(Qt.Orientation.Vertical, self)
        self._slice_scrollbar.setMinimum(0)
        self._slice_scrollbar.setMaximum(0)
        self._slice_scrollbar.setValue(0)
        self._slice_scrollbar.setSingleStep(1)
        self._slice_scrollbar.setPageStep(10)
        self._slice_scrollbar.setFixedWidth(16)
        self._slice_scrollbar.setEnabled(False)
        self._slice_scrollbar.setInvertedAppearance(True)
        self._slice_scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        viewer_row.addWidget(self._slice_scrollbar)

        self.viewer_container = QWidget()
        self.viewer_container.setLayout(viewer_row)
        layout.addWidget(self.viewer_container, 1)

        # Guard flag – prevents circular updates between scrollbar ↔ VTK
        self._scrollbar_updating = False

        self.colors = vtkNamedColors()
        self.interactor_style = None
        self.view_orientation = view_orientation
        self.image_viewer = None
        self.roi_manager = ROIManager()

        # Active metadata (reflects the currently displayed series)
        self.dicom_spacing = [1.0, 1.0, 1.0]
        self.dicom_folder = dicom_folder
        self._dicom_meta = None
        self._dicom_num_slices = 0
        self._resliced_output = None
        self._initial_parallel_scale = None

        # Global W/L (synced across all series)
        self._ww = 400.0
        self._wl = 40.0

        # Series list and active index
        self._series: list[dict] = []
        self._active_series_idx = -1
        self._combo_updating = False

        # Overlay text actors
        self._overlay_tl = None
        self._overlay_tr = None
        self._overlay_bl = None
        self._overlay_br = None

        # Placeholder label
        self.placeholder = QLabel("No DICOM series loaded.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.placeholder)

        if dicom_folder:
            self.load_dicom(dicom_folder)
            self._slice_scrollbar.show()
            self.viewer_container.show()
        else:
            self.vtk_widget.hide()
            self._slice_scrollbar.hide()
            self.viewer_container.hide()

    # ------------------------------------------------------------------
    # Scrollbar ↔ slice sync
    # ------------------------------------------------------------------

    def _configure_scrollbar(self, min_slice: int, max_slice: int, current: int):
        """Set scrollbar range and value without triggering _on_scrollbar_changed."""
        self._scrollbar_updating = True
        self._slice_scrollbar.setMinimum(min_slice)
        self._slice_scrollbar.setMaximum(max_slice)
        self._slice_scrollbar.setValue(current)
        self._slice_scrollbar.setEnabled(max_slice > min_slice)
        self._scrollbar_updating = False

    def _update_scrollbar_value(self, slice_idx: int):
        """Called whenever the active slice changes (from VTK side)."""
        self._scrollbar_updating = True
        self._slice_scrollbar.setValue(slice_idx)
        self._scrollbar_updating = False

    def _on_scrollbar_changed(self, value: int):
        """User dragged the scrollbar – push new slice into the interactor."""
        if self._scrollbar_updating:
            return
        if self.interactor_style:
            self.interactor_style._set_slice(value)

    # ------------------------------------------------------------------
    # Series management helpers
    # ------------------------------------------------------------------

    def _build_series_label(self, folder: str, meta) -> str:
        label = Path(folder).name
        if meta:
            snum = str(getattr(meta, "SeriesNumber", "")).strip()
            sdesc = str(getattr(meta, "SeriesDescription", "")).strip()
            combined = " – ".join(filter(None, [snum, sdesc]))
            if combined:
                label = combined
        return label

    def _load_series_data(self, folder: str) -> dict:
        folder_path = Path(folder)
        series_name = folder_path.name

        # --- Emit start (0%) ---
        self.loading_started.emit(f"Loading: {series_name}")
        self.loading_progress.emit(0)
        QApplication.processEvents()

        dcm_files = sorted(
            f
            for f in folder_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )
        num_slices = len(dcm_files)

        # --- Reading metadata (~10%) ---
        self.loading_progress.emit(10)
        QApplication.processEvents()

        meta = None
        if dcm_files:
            try:
                meta = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            except Exception:
                pass

        # --- VTK DICOM reader (10% → 70%) ---
        reader = vtkDICOMImageReader()
        reader.SetDirectoryName(folder)

        def _on_reader_progress(obj, _event):
            pct = 10 + int(obj.GetProgress() * 60)
            self.loading_progress.emit(pct)
            QApplication.processEvents()

        reader_tag = reader.AddObserver("ProgressEvent", _on_reader_progress)
        reader.Update()
        reader.RemoveObserver(reader_tag)

        self.loading_progress.emit(70)
        QApplication.processEvents()

        spacing = list(reader.GetOutput().GetSpacing())

        # --- Reslice (70% → 95%) ---
        reslice = vtkImageReslice()
        reslice.SetInputConnection(reader.GetOutputPort())
        if self.view_orientation == "coronal":
            reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, -1, 0)
        elif self.view_orientation == "sagittal":
            reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)

        def _on_reslice_progress(obj, _event):
            pct = 70 + int(obj.GetProgress() * 25)
            self.loading_progress.emit(pct)
            QApplication.processEvents()

        reslice_tag = reslice.AddObserver("ProgressEvent", _on_reslice_progress)
        reslice.Update()
        reslice.RemoveObserver(reslice_tag)

        # --- Done (100%) ---
        self.loading_progress.emit(100)
        QApplication.processEvents()

        return {
            "folder": folder,
            "reader": reader,
            "reslice": reslice,
            "spacing": spacing,
            "meta": meta,
            "num_slices": num_slices,
            "label": self._build_series_label(folder, meta),
        }

    def _update_series_combo(self):
        self._combo_updating = True
        self.series_combo.clear()
        for i, s in enumerate(self._series):
            self.series_combo.addItem(f"[{i + 1}]  {s['label']}", userData=i)
        if 0 <= self._active_series_idx < len(self._series):
            self.series_combo.setCurrentIndex(self._active_series_idx)
        self._combo_updating = False
        has_multi = len(self._series) > 0
        self._btn_remove_series.setEnabled(has_multi)
        self.series_combo.setEnabled(has_multi)

    def _activate_series_metadata(self, s: dict):
        self._dicom_meta = s["meta"]
        self._dicom_num_slices = s["num_slices"]
        self._resliced_output = s["reslice"].GetOutput()
        self.dicom_spacing = s["spacing"]
        self.dicom_folder = s["folder"]

    # ------------------------------------------------------------------
    # Public: load first (or replace all) series
    # ------------------------------------------------------------------

    def load_dicom(self, folder: str):
        self.placeholder.hide()
        self.vtk_widget.show()
        self._slice_scrollbar.show()
        self.viewer_container.show()

        series = self._load_series_data(folder)

        if not self._series and series["meta"] is not None:
            ds = series["meta"]
            ww = getattr(ds, "WindowWidth", None)
            wc = getattr(ds, "WindowCenter", None)
            if ww is not None and wc is not None:
                self._ww = float(ww[0] if hasattr(ww, "__len__") else ww)
                self._wl = float(wc[0] if hasattr(wc, "__len__") else wc)

        self._series = [series]
        self._active_series_idx = 0
        self._activate_series_metadata(series)
        self.dicom_spacing_changed.emit(self.dicom_spacing)

        if self.image_viewer is None:
            self._setup_vtk_pipeline(series)
        else:
            self.image_viewer.SetInputConnection(series["reslice"].GetOutputPort())
            self.image_viewer.UpdateDisplayExtent()
            style = self.interactor_style
            if style:
                style.min_slice = self.image_viewer.GetSliceMin()
                style.max_slice = self.image_viewer.GetSliceMax()
                style._set_slice(self.image_viewer.GetSliceMin())
                self._configure_scrollbar(style.min_slice, style.max_slice, style.slice)
            self.image_viewer.SetColorWindow(self._ww)
            self.image_viewer.SetColorLevel(self._wl)
            self._initial_parallel_scale = (
                self.image_viewer.GetRenderer().GetActiveCamera().GetParallelScale()
            )
            self._update_all_overlays()
            self.image_viewer.Render()

        self._update_series_combo()
        self.loading_finished.emit()

    # ------------------------------------------------------------------
    # Public: add an additional series
    # ------------------------------------------------------------------

    def add_series(self, folder: str):
        if not self._series:
            self.load_dicom(folder)
            return
        if any(s["folder"] == folder for s in self._series):
            return
        series = self._load_series_data(folder)
        self._series.append(series)
        self._update_series_combo()
        self._switch_series(len(self._series) - 1)
        self.loading_finished.emit()

    # ------------------------------------------------------------------
    # Internal: remove active series
    # ------------------------------------------------------------------

    def _remove_active_series(self):
        # if len(self._series) <= 1:
        #     return
        idx = self._active_series_idx
        self._series.pop(idx)
        new_idx = max(0, idx - 1)
        self._active_series_idx = -1
        self._update_series_combo()
        self._switch_series(new_idx)

    # ------------------------------------------------------------------
    # Internal: switch active series
    # ------------------------------------------------------------------

    def _on_combo_changed(self, idx: int):
        if self._combo_updating or idx < 0:
            return
        self._switch_series(idx)

    def _switch_series(self, idx: int):
        if idx < 0 or idx >= len(self._series) or not self.image_viewer:
            self.vtk_widget.hide()
            self._slice_scrollbar.hide()
            self.viewer_container.hide()
            self.placeholder.show()
            return
        if idx == self._active_series_idx:
            return

        self._active_series_idx = idx
        s = self._series[idx]
        self._activate_series_metadata(s)

        self.image_viewer.SetInputConnection(s["reslice"].GetOutputPort())
        self.image_viewer.UpdateDisplayExtent()

        style = self.interactor_style
        if style:
            new_min = self.image_viewer.GetSliceMin()
            new_max = self.image_viewer.GetSliceMax()
            style.min_slice = new_min
            style.max_slice = new_max
            clamped = max(new_min, min(style.slice, new_max))
            style._set_slice(clamped)
            self._configure_scrollbar(new_min, new_max, clamped)

        self.image_viewer.SetColorWindow(self._ww)
        self.image_viewer.SetColorLevel(self._wl)

        self._combo_updating = True
        self.series_combo.setCurrentIndex(idx)
        self._combo_updating = False

        self._update_all_overlays()
        self.image_viewer.Render()

    # ------------------------------------------------------------------
    # Series dialog (button)
    # ------------------------------------------------------------------

    def _add_series_dialog(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Additional DICOM Series Folder"
        )
        if folder:
            self.add_series(folder)

    # ------------------------------------------------------------------
    # First-time VTK pipeline setup (called once)
    # ------------------------------------------------------------------

    def _setup_vtk_pipeline(self, series: dict):
        self.image_viewer = vtkImageViewer2()
        self.image_viewer.SetInputConnection(series["reslice"].GetOutputPort())
        self.image_viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.image_viewer.SetupInteractor(self.vtk_widget)

        ren = self.image_viewer.GetRenderer()
        ren.SetBackground(self.colors.GetColor3d("Black"))

        self.slice_text = self._make_text_actor(
            "", 15, 10, 20, align_bottom=True, justify_right=False
        )
        self.slice_text.VisibilityOff()
        ren.AddViewProp(self.slice_text)

        self._overlay_tl = self._make_text_actor(
            "", 0.01, 0.99, 13, normalized=True, align_bottom=False, justify_right=False
        )
        self._overlay_tr = self._make_text_actor(
            "", 0.99, 0.99, 13, normalized=True, align_bottom=False, justify_right=True
        )
        self._overlay_bl = self._make_text_actor(
            "", 0.01, 0.01, 13, normalized=True, align_bottom=True, justify_right=False
        )
        self._overlay_br = self._make_text_actor(
            "", 0.99, 0.01, 13, normalized=True, align_bottom=True, justify_right=True
        )
        for actor in (
            self._overlay_tl,
            self._overlay_tr,
            self._overlay_bl,
            self._overlay_br,
        ):
            ren.AddViewProp(actor)

        self.interactor_style = DrawingInteractorStyle(self)
        self.vtk_widget.SetInteractorStyle(self.interactor_style)
        self.interactor_style.setup(self.image_viewer, self.slice_text)

        self.image_viewer.Render()
        ren.ResetCamera()
        self.image_viewer.SetColorWindow(self._ww)
        self.image_viewer.SetColorLevel(self._wl)
        self._initial_parallel_scale = ren.GetActiveCamera().GetParallelScale()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self._initial_parallel_scale = ren.GetActiveCamera().GetParallelScale()

        # Initialise scrollbar now that min/max are known
        self._configure_scrollbar(
            self.interactor_style.min_slice,
            self.interactor_style.max_slice,
            self.interactor_style.slice,
        )

        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        interactor.AddObserver(
            vtkCommand.MouseMoveEvent, self._on_mouse_move_overlay, 0.5
        )
        interactor.AddObserver(
            vtkCommand.EndInteractionEvent, self._on_interaction_end, 0.5
        )
        interactor.AddObserver(
            vtkCommand.MouseWheelForwardEvent, self._on_interaction_end, 0.5
        )
        interactor.AddObserver(
            vtkCommand.MouseWheelBackwardEvent, self._on_interaction_end, 0.5
        )
        interactor.AddObserver(vtkCommand.WindowLevelEvent, self._on_window_level, 0.5)

        self._update_all_overlays()

    # ------------------------------------------------------------------
    # Convenience: refresh all four overlays at once
    # ------------------------------------------------------------------

    def _update_all_overlays(self):
        self._update_overlay_tl()
        self._update_overlay_tr()
        self._update_overlay_bl()
        self._update_overlay_br()

    # ------------------------------------------------------------------
    # Overlay: top-left – patient info
    # ------------------------------------------------------------------

    def _update_overlay_tl(self):
        if not self._overlay_tl:
            return
        ds = self._dicom_meta
        lines = []
        if ds is not None:
            pname = str(getattr(ds, "PatientName", "")) or "Unknown"
            pid = str(getattr(ds, "PatientID", ""))
            dob = str(getattr(ds, "PatientBirthDate", ""))
            sex = str(getattr(ds, "PatientSex", ""))
            if pname:
                lines.append(pname)
            details = "  ".join(filter(None, [pid, dob, sex]))
            if details:
                lines.append(details)
        else:
            lines.append("Patient: --")
        self._overlay_tl.GetMapper().SetInput("\n".join(lines))

    # ------------------------------------------------------------------
    # Overlay: top-right – scan date / study info
    # ------------------------------------------------------------------

    def _update_overlay_tr(self):
        if not self._overlay_tr:
            return
        ds = self._dicom_meta
        lines = []
        if ds is not None:
            study_date = str(getattr(ds, "StudyDate", "")) or "--"
            study_desc = str(getattr(ds, "StudyDescription", ""))
            series_desc = str(getattr(ds, "SeriesDescription", ""))
            modality = str(getattr(ds, "Modality", ""))
            institution = str(getattr(ds, "InstitutionName", ""))
            if len(study_date) == 8 and study_date.isdigit():
                study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
            lines.append(study_date)
            if modality:
                lines.append(modality)
            if institution:
                lines.append(institution)
            if study_desc:
                lines.append(study_desc)
            if series_desc:
                lines.append(series_desc)
        else:
            lines.append("--")
        self._overlay_tr.GetMapper().SetInput("\n".join(lines))

    # ------------------------------------------------------------------
    # Overlay: bottom-left – series / geometry info
    # ------------------------------------------------------------------

    def _update_overlay_bl(self, slice_idx=None):
        if not self._overlay_bl or not self.image_viewer:
            return

        if slice_idx is None and self.interactor_style:
            slice_idx = self.interactor_style.slice
        elif slice_idx is None:
            slice_idx = self.image_viewer.GetSliceMin()

        # Keep scrollbar in sync whenever the overlay is refreshed
        self._update_scrollbar_value(slice_idx)

        ds = self._dicom_meta
        lines = [
            "Series: --",
            "Image: -- / --",
            "Size: -- × -- px",
            "Spacing: -- × -- mm",
            "Loc: -- mm",
            "Thick: -- mm",
        ]

        series_num = str(getattr(ds, "SeriesNumber", "--")) if ds else "--"
        lines[0] = f"Series: {series_num}"

        total = self._dicom_num_slices if self._dicom_num_slices else "--"
        lines[1] = f"Image: {slice_idx + 1} / {total}"

        rows = str(getattr(ds, "Rows", "--")) if ds else "--"
        cols = str(getattr(ds, "Columns", "--")) if ds else "--"
        lines[2] = f"Size: {cols} \u00d7 {rows} px"

        if ds:
            ps = getattr(ds, "PixelSpacing", None)
            if ps:
                lines[3] = f"Spacing: {float(ps[0]):.2f} \u00d7 {float(ps[1]):.2f} mm"
            loc = getattr(ds, "SliceLocation", None)
        if self._resliced_output is not None:
            img = self._resliced_output
            origin = img.GetOrigin()
            spacing = img.GetSpacing()
            orient = self.image_viewer.GetSliceOrientation()
            loc = origin[orient] + slice_idx * spacing[orient]
            lines[4] = f"Loc: {loc:.2f} mm"
        else:
            loc = getattr(ds, "SliceLocation", None)
            if loc is not None:
                lines[4] = f"Loc: {float(loc):.2f} mm"
        st = getattr(ds, "SliceThickness", None)
        if st:
            lines[5] = f"Thick: {float(st):.2f} mm"

        self._overlay_bl.GetMapper().SetInput("\n".join(lines))

    # ------------------------------------------------------------------
    # Overlay: bottom-right – zoom / W-L / cursor
    # ------------------------------------------------------------------

    def _update_overlay_br(self, display_x=None, display_y=None):
        if not self._overlay_br or not self.image_viewer:
            return

        lines = ["Zoom: --%", "W: --  L: --", "R: --  C: --  HU: --"]

        ren = self.image_viewer.GetRenderer()
        cam = ren.GetActiveCamera()
        if self._initial_parallel_scale and self._initial_parallel_scale > 0:
            zoom_pct = (self._initial_parallel_scale / cam.GetParallelScale()) * 100.0
            lines[0] = f"Zoom: {zoom_pct:.0f}%"

        lines[1] = f"W: {self._ww:.0f}  L: {self._wl:.0f}"

        if display_x is not None and display_y is not None:
            info = self._get_pixel_info(display_x, display_y)
            if info is not None:
                row, col, val = info
                lines[2] = f"R: {row}  C: {col}  HU: {val:.0f}"

        self._overlay_br.GetMapper().SetInput("\n".join(lines))

    # ------------------------------------------------------------------
    # Pixel sampling helper
    # ------------------------------------------------------------------

    def _get_pixel_info(self, display_x, display_y):
        if not self.image_viewer or self._resliced_output is None:
            return None
        ren = self.image_viewer.GetRenderer()
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(display_x, display_y, 0, ren)
        wx, wy, wz = picker.GetPickPosition()

        img = self._resliced_output
        origin = img.GetOrigin()
        spacing = img.GetSpacing()
        dims = img.GetDimensions()
        orient = self.image_viewer.GetSliceOrientation()
        sl = self.interactor_style.slice if self.interactor_style else 0

        if orient == 2:
            i = int(round((wx - origin[0]) / spacing[0]))
            j = int(round((wy - origin[1]) / spacing[1]))
            k = sl
        elif orient == 1:
            i = int(round((wx - origin[0]) / spacing[0]))
            j = sl
            k = int(round((wz - origin[2]) / spacing[2]))
        else:
            i = sl
            j = int(round((wy - origin[1]) / spacing[1]))
            k = int(round((wz - origin[2]) / spacing[2]))

        if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
            val = img.GetScalarComponentAsDouble(i, j, k, 0)
            return (j, i, val)
        return None

    # ------------------------------------------------------------------
    # VTK event callbacks
    # ------------------------------------------------------------------

    def _on_window_level(self, obj, event):
        if self.image_viewer:
            self._ww = self.image_viewer.GetColorWindow()
            self._wl = self.image_viewer.GetColorLevel()

    def _on_mouse_move_overlay(self, obj, event):
        if not self.image_viewer:
            return
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        self._update_overlay_br(x, y)
        self._update_overlay_bl()
        self.image_viewer.Render()

    def _on_interaction_end(self, obj, event):
        self._update_overlay_br()
        self._update_overlay_bl()
        if self.image_viewer:
            self.image_viewer.Render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_draw_tool(self, tool):
        if self.interactor_style:
            if tool != DrawTool.EDIT and self.image_viewer:
                self.roi_manager.select(None, self.image_viewer.GetRenderer())
                self.image_viewer.Render()
            self.interactor_style.draw_tool = tool
            self.interactor_style._update_status()
            self.interactor_style._render()

    def finalize_roi(self, roi_type, points, slice_index, orientation):
        if not self.image_viewer:
            return
        count = (
            sum(1 for r in self.roi_manager.rois.values() if r.roi_type == roi_type) + 1
        )
        ren = self.image_viewer.GetRenderer()
        self.roi_manager.add_roi(
            f"{roi_type.capitalize()} {count}",
            roi_type,
            slice_index,
            orientation,
            points,
            renderer=ren,
        )
        self.image_viewer.Render()

    def delete_roi(self, roi_id):
        if self.image_viewer:
            self.roi_manager.remove_roi(roi_id, self.image_viewer.GetRenderer())
            self.image_viewer.Render()

    def select_roi(self, roi_id):
        if self.image_viewer:
            self.roi_manager.select(roi_id, self.image_viewer.GetRenderer())
            self.image_viewer.Render()

    @property
    def loaded_folders(self) -> list[str]:
        return [s["folder"] for s in self._series]

    def cleanup(self):
        if self.vtk_widget:
            rw = self.vtk_widget.GetRenderWindow()
            if rw:
                rw.Finalize()
            self.vtk_widget.close()

    # ------------------------------------------------------------------
    # Text actor factory
    # ------------------------------------------------------------------

    def _make_text_actor(
        self,
        text,
        x,
        y,
        size,
        align_bottom=False,
        justify_right=False,
        normalized=False,
        center=False,
    ):
        tp = vtkTextProperty()
        tp.SetFontFamilyToCourier()
        tp.SetFontSize(size)
        tp.SetColor(1.0, 1.0, 1.0)
        (
            tp.SetVerticalJustificationToBottom()
            if align_bottom
            else tp.SetVerticalJustificationToTop()
        )
        if center:
            tp.SetJustificationToCentered()
        elif justify_right:
            tp.SetJustificationToRight()
        else:
            tp.SetJustificationToLeft()

        tm = vtkTextMapper()
        tm.SetInput(text)
        tm.SetTextProperty(tp)

        ta = vtkActor2D()
        ta.SetMapper(tm)
        if normalized:
            ta.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        ta.SetPosition(x, y)
        return ta

    def _make_text(self, text, x, y, size, align_bottom=False, normalized=False):
        """Backward-compatible wrapper."""
        return self._make_text_actor(
            text, x, y, size, align_bottom=align_bottom, normalized=normalized
        )
