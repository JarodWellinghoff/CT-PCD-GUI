from pathlib import Path
from PyQt6.QtWidgets import (
    QLabel,
    QWidget,
    QVBoxLayout,
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
# DICOM Viewer widget
# ===========================================================================


class DicomViewerWidget(QWidget):
    drawing_cancelled = pyqtSignal()
    dicom_spacing_changed = pyqtSignal(list)  # [sx, sy, sz]

    def __init__(self, dicom_folder=None, view_orientation="axial", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        self.colors = vtkNamedColors()
        self.interactor_style = None
        self.view_orientation = view_orientation
        self.image_viewer = None
        self.roi_manager = ROIManager()
        self.dicom_spacing = [1.0, 1.0, 1.0]
        self.dicom_folder = dicom_folder

        # DICOM metadata cache
        self._dicom_meta = None  # pydicom dataset for first slice
        self._dicom_num_slices = 0
        self._resliced_output = None  # vtkImageData after reslice (for pixel sampling)
        self._initial_parallel_scale = None
        # W/L tracked in HU units; seeded from DICOM tags, kept live via observer
        self._ww = 400.0
        self._wl = 40.0

        # Overlay text actors (set in load_dicom)
        self._overlay_tl = None  # top-left:    patient info
        self._overlay_tr = None  # top-right:   scan info
        self._overlay_bl = None  # bottom-left: series / geometry info
        self._overlay_br = None  # bottom-right: zoom / W-L / cursor

        self.placeholder = QLabel(
            "No DICOM series loaded.\nUse File -> Open DICOM... to load a series."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.placeholder)

        if dicom_folder:
            self.load_dicom(dicom_folder)
        else:
            self.vtk_widget.hide()

    # ------------------------------------------------------------------
    # Public: load DICOM series
    # ------------------------------------------------------------------

    def load_dicom(self, dicom_folder):
        self.placeholder.hide()
        self.vtk_widget.show()
        self.dicom_folder = dicom_folder

        # ---- Read DICOM metadata via pydicom (first slice) ----
        folder_path = Path(dicom_folder)
        dcm_files = sorted(
            f
            for f in folder_path.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )
        self._dicom_num_slices = len(dcm_files)
        self._dicom_meta = None
        if dcm_files:
            try:
                self._dicom_meta = pydicom.dcmread(
                    str(dcm_files[0]), stop_before_pixels=True
                )
            except Exception:
                pass

        # Seed W/L from DICOM WindowWidth / WindowCenter tags
        if self._dicom_meta is not None:
            ww = getattr(self._dicom_meta, "WindowWidth", None)
            wc = getattr(self._dicom_meta, "WindowCenter", None)
            if ww is not None and wc is not None:
                # Tags may be a list (multiple presets); take the first
                self._ww = float(ww[0] if hasattr(ww, "__len__") else ww)
                self._wl = float(wc[0] if hasattr(wc, "__len__") else wc)

        # ---- VTK pipeline ----
        reader = vtkDICOMImageReader()
        reader.SetDirectoryName(dicom_folder)
        reader.Update()

        self.dicom_spacing = list(reader.GetOutput().GetSpacing())
        self.dicom_spacing_changed.emit(self.dicom_spacing)

        reslice = vtkImageReslice()
        reslice.SetInputConnection(reader.GetOutputPort())
        reslice.SetOutputSpacing(1, 1, 1)
        if self.view_orientation == "coronal":
            reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, -1, 0)
        elif self.view_orientation == "sagittal":
            reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)
        reslice.Update()
        self._resliced_output = reslice.GetOutput()

        self.image_viewer = vtkImageViewer2()
        self.image_viewer.SetInputConnection(reslice.GetOutputPort())
        self.image_viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.image_viewer.SetupInteractor(self.vtk_widget)

        # ---- Slice counter (bottom-centre) – kept for interactor_style ----
        self.slice_text = self._make_text_actor(
            "", 15, 10, 20, align_bottom=True, justify_right=False
        )
        # Hide the legacy slice counter; the new overlay_bl covers this info
        self.slice_text.VisibilityOff()

        # ---- Help text ----
        self.help_text = self._make_text_actor(
            "Scroll: slice  |  Esc: cancel  |  Dbl-click: add vtx  |  Del: remove vtx",
            0.5,
            0.99,
            11,
            normalized=True,
            align_bottom=False,
            justify_right=False,
            center=True,
        )

        ren = self.image_viewer.GetRenderer()
        ren.AddViewProp(self.slice_text)
        ren.AddViewProp(self.help_text)
        ren.SetBackground(self.colors.GetColor3d("Black"))

        # ---- Corner overlays ----
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

        # Populate static overlays from metadata
        self._update_overlay_tl()
        self._update_overlay_tr()

        # ---- Interactor style ----
        self.interactor_style = DrawingInteractorStyle(self)
        self.vtk_widget.SetInteractorStyle(self.interactor_style)
        self.interactor_style.setup(self.image_viewer, self.slice_text)

        self.image_viewer.Render()
        ren.ResetCamera()
        # Apply DICOM W/L to the viewer (overrides VTK's auto-computed display range)
        self.image_viewer.SetColorWindow(self._ww)
        self.image_viewer.SetColorLevel(self._wl)
        # Capture initial parallel scale for zoom calculation
        self._initial_parallel_scale = ren.GetActiveCamera().GetParallelScale()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        # Capture scale again after Initialize (camera may have been reset)
        self._initial_parallel_scale = ren.GetActiveCamera().GetParallelScale()

        # ---- Dynamic overlay observers ----
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
        # Keep _ww / _wl in sync when user drags to adjust window/level
        interactor.AddObserver(vtkCommand.WindowLevelEvent, self._on_window_level, 0.5)

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
            # Format date YYYY-MM-DD if 8 digits
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
    # Overlay: bottom-left – series / geometry info  (updates on slice change)
    # ------------------------------------------------------------------

    def _update_overlay_bl(self, slice_idx=None):
        if not self._overlay_bl or not self.image_viewer:
            return

        if slice_idx is None and self.interactor_style:
            slice_idx = self.interactor_style.slice
        elif slice_idx is None:
            slice_idx = self.image_viewer.GetSliceMin()

        ds = self._dicom_meta
        lines = [
            "Ser: --",
            "Img: -- / --",
            "-- \u00d7 -- px",
            "-- \u00d7 -- mm",
            "Loc: -- mm",
            "Thick: -- mm",
        ]

        # Series number
        series_num = str(getattr(ds, "SeriesNumber", "--")) if ds else "--"
        lines[0] = f"Series: {series_num}"

        # Image index / total
        total = self._dicom_num_slices if self._dicom_num_slices else "--"
        current_img = slice_idx + 1
        lines[1] = f"Image: {current_img} / {total}"

        # Pixel dimensions (rows × cols)
        rows = str(getattr(ds, "Rows", "--")) if ds else "--"
        cols = str(getattr(ds, "Columns", "--")) if ds else "--"
        lines[2] = f"Size: {cols} \u00d7 {rows} px"

        # Pixel spacing
        if ds:
            ps = getattr(ds, "PixelSpacing", None)
            if ps:
                lines[3] = f"Spacing: {float(ps[0]):.2f} \u00d7 {float(ps[1]):.2f} mm"

        # Slice location
        if ds:
            loc = getattr(ds, "SliceLocation", None)
            if loc is not None:
                lines[4] = f"Loc: {float(loc):.2f} mm"

        # Slice thickness
        if ds:
            st = getattr(ds, "SliceThickness", None)
            if st:
                lines[5] = f"Thick: {float(st):.2f} mm"

        self._overlay_bl.GetMapper().SetInput("\n".join(lines))

    # ------------------------------------------------------------------
    # Overlay: bottom-right – zoom / W-L / cursor  (updates on mouse move)
    # ------------------------------------------------------------------

    def _update_overlay_br(self, display_x=None, display_y=None):
        if not self._overlay_br or not self.image_viewer:
            return

        lines = ["Zoom: --%", "W: --  L: --", "R: --  C: --  HU: --"]

        # ---- Zoom percentage ----
        ren = self.image_viewer.GetRenderer()
        cam = ren.GetActiveCamera()
        if self._initial_parallel_scale and self._initial_parallel_scale > 0:
            zoom_pct = (self._initial_parallel_scale / cam.GetParallelScale()) * 100.0
            lines[0] = f"Zoom: {zoom_pct:.0f}%"

        # ---- Window / Level ----
        lines[1] = f"W: {self._ww:.0f}  L: {self._wl:.0f}"

        # ---- Cursor pixel info ----
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
        """Return (row, col, scalar_value) for the given display position, or None."""
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

        if orient == 2:  # axial   – z fixed
            i = int(round((wx - origin[0]) / spacing[0]))
            j = int(round((wy - origin[1]) / spacing[1]))
            k = sl
        elif orient == 1:  # coronal – y fixed
            i = int(round((wx - origin[0]) / spacing[0]))
            j = sl
            k = int(round((wz - origin[2]) / spacing[2]))
        else:  # sagittal – x fixed
            i = sl
            j = int(round((wy - origin[1]) / spacing[1]))
            k = int(round((wz - origin[2]) / spacing[2]))

        if 0 <= i < dims[0] and 0 <= j < dims[1] and 0 <= k < dims[2]:
            val = img.GetScalarComponentAsDouble(i, j, k, 0)
            # row = j (y-axis), col = i (x-axis) in image convention
            return (j, i, val)
        return None

    # ------------------------------------------------------------------
    # VTK event callbacks for dynamic overlays
    # ------------------------------------------------------------------

    def _on_window_level(self, obj, event):
        """Keep stored W/L in sync when user adjusts it via mouse drag."""
        if self.image_viewer:
            self._ww = self.image_viewer.GetColorWindow()
            self._wl = self.image_viewer.GetColorLevel()

    def _on_mouse_move_overlay(self, obj, event):
        if not self.image_viewer:
            return
        interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        self._update_overlay_br(x, y)
        self._update_overlay_bl()  # slice index may change via keys
        self.image_viewer.Render()

    def _on_interaction_end(self, obj, event):
        """Refresh zoom / W-L after pan, zoom, or window-level gesture."""
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

    def cleanup(self):
        """Finalize VTK resources before Qt tears down the widget."""
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

    # ------------------------------------------------------------------
    # Legacy helper kept so DrawingInteractorStyle still works
    # ------------------------------------------------------------------

    def _make_text(self, text, x, y, size, align_bottom=False, normalized=False):
        """Backward-compatible wrapper (used by interactor_style via setup())."""
        return self._make_text_actor(
            text,
            x,
            y,
            size,
            align_bottom=align_bottom,
            normalized=normalized,
        )
