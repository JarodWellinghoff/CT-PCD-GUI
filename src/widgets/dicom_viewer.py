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
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from src.models import ROIManager
from src.models import DrawTool
from src.interaction.drawing_interactor_style import DrawingInteractorStyle


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

    def load_dicom(self, dicom_folder):
        self.placeholder.hide()
        self.vtk_widget.show()
        self.dicom_folder = dicom_folder

        reader = vtkDICOMImageReader()
        reader.SetDirectoryName(dicom_folder)
        reader.Update()

        # Capture native spacing before reslice
        self.dicom_spacing = list(reader.GetOutput().GetSpacing())
        self.dicom_spacing_changed.emit(self.dicom_spacing)

        reslice = vtkImageReslice()
        reslice.SetInputConnection(reader.GetOutputPort())
        reslice.SetOutputSpacing(1, 1, 1)
        if self.view_orientation == "coronal":
            reslice.SetResliceAxesDirectionCosines(1, 0, 0, 0, 0, 1, 0, -1, 0)
        elif self.view_orientation == "sagittal":
            reslice.SetResliceAxesDirectionCosines(0, 1, 0, 0, 0, 1, 1, 0, 0)

        self.image_viewer = vtkImageViewer2()
        self.image_viewer.SetInputConnection(reslice.GetOutputPort())
        self.image_viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
        self.image_viewer.SetupInteractor(self.vtk_widget)

        self.slice_text = self._make_text("", 15, 10, 20, align_bottom=True)
        self.help_text = self._make_text(
            "Scroll: slice | Esc: cancel | Dbl-click: add vtx | Del: remove vtx",
            0.02,
            0.98,
            13,
            normalized=True,
        )
        ren = self.image_viewer.GetRenderer()
        ren.AddViewProp(self.slice_text)
        ren.AddViewProp(self.help_text)
        ren.SetBackground(self.colors.GetColor3d("Black"))

        self.interactor_style = DrawingInteractorStyle(self)
        self.vtk_widget.SetInteractorStyle(self.interactor_style)
        self.interactor_style.setup(self.image_viewer, self.slice_text)

        self.image_viewer.Render()
        ren.ResetCamera()
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

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

    def _make_text(self, text, x, y, size, align_bottom=False, normalized=False):
        tp = vtkTextProperty()
        tp.SetFontFamilyToCourier()
        tp.SetFontSize(size)
        if align_bottom:
            tp.SetVerticalJustificationToBottom()
        else:
            tp.SetVerticalJustificationToTop()
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
