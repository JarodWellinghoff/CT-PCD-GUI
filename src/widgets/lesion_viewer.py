from typing import Optional
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout, QMessageBox
from PyQt6.QtCore import Qt
import numpy as np
import vtkmodules.all as vtk
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper, vtkRenderer
from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersCore import vtkMarchingCubes
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk
import scipy.io as sio

# ===========================================================================
# 3-D Lesion Model Viewer
# ===========================================================================


class LesionViewerWidget(QWidget):
    """Displays a 3D surface reconstruction of a lesion mask from a .mat file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.renderer = vtkRenderer()
        self.renderer.SetBackground(0.1, 0.1, 0.15)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Axes widget
        axes = vtkAxesActor()
        self.axes_widget = vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.15, 0.15)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        self.current_actor: Optional[vtkActor] = None
        self.current_file: Optional[str] = None
        self._spacing = [1.0, 1.0, 1.0]

        self.placeholder = QLabel(
            "No lesion model loaded.\nSelect a .mat file from the Lesion Library dock."
        )
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setStyleSheet("color: #888; font-size: 14px;")
        layout.addWidget(self.placeholder)

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    @property
    def spacing(self):
        return self._spacing

    @spacing.setter
    def spacing(self, val):
        self._spacing = list(val)
        # Reload current model with new spacing
        if self.current_file:
            self.load_mat(self.current_file)

    def load_mat(self, filepath: str):
        """Load a .mat file, extract the first 3-D array, and render as surface."""
        self.current_file = filepath
        self.placeholder.hide()

        try:
            mat = sio.loadmat(filepath)
        except Exception as e:
            QMessageBox.critical(self, "Load error", f"Cannot read .mat file:\n{e}")
            return

        # Find the first 3-D numpy array in the file
        arr = mat["Patient"][0][0][0][0][0][8]
        arr_name = None
        for key, val in mat.items():
            if key.startswith("_"):
                continue
            if isinstance(val, np.ndarray) and val.ndim == 3:
                arr = val
                arr_name = key
                break

        if arr is None:
            QMessageBox.warning(
                self, "No 3-D data", "No 3-D array found in the .mat file."
            )
            return

        # Ensure C-contiguous float
        arr = np.ascontiguousarray(arr, dtype=np.float64)

        # Build vtkImageData
        img = vtkImageData()
        dims = arr.shape  # (Z, Y, X) or (X, Y, Z) depending on MATLAB convention
        img.SetDimensions(dims)
        img.SetSpacing(*self._spacing)
        img.SetOrigin(0, 0, 0)

        # Flatten in Fortran order to match VTK's x-fastest indexing from MATLAB data
        flat = arr.flatten(order="F")
        vtk_arr = numpy_to_vtk(flat, deep=True)
        img.GetPointData().SetScalars(vtk_arr)

        # Marching cubes
        mc = vtkMarchingCubes()
        mc.SetInputData(img)
        mc.SetValue(0, 0.5)  # threshold for binary mask
        mc.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(mc.GetOutputPort())
        mapper.ScalarVisibilityOff()

        # Remove previous actor
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)

        self.current_actor = vtkActor()
        self.current_actor.SetMapper(mapper)
        self.current_actor.GetProperty().SetColor(0.9, 0.5, 0.3)
        self.current_actor.GetProperty().SetOpacity(0.85)
        self.current_actor.GetProperty().SetSpecular(0.3)
        self.current_actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(self.current_actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def clear(self):
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None
        self.current_file = None
        self.placeholder.show()
        self.vtk_widget.GetRenderWindow().Render()

    def cleanup(self):
        """Finalize VTK resources before Qt tears down the widget."""
        if self.axes_widget:
            self.axes_widget.EnabledOff()
        if self.vtk_widget:
            rw = self.vtk_widget.GetRenderWindow()
            if rw:
                rw.Finalize()
            self.vtk_widget.close()
