from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
import vtkmodules.all as vtk


class LesionVtkScene(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None: ...
    def set_surface(self, polydata: vtk.vtkPolyData, *, reset_camera: bool) -> None: ...
    def clear_surface(self) -> None: ...
    def apply_appearance(self, state: AppearanceState) -> None: ...
    def set_parallel_projection(self, enabled: bool) -> None: ...
    def reset_camera(self) -> None: ...
    def set_axis_view(self, axis: str) -> None: ...
    def save_screenshot(self, path: Path, scale: int = 2) -> Path: ...
    def cleanup(self) -> None: ...
    def _build_vtk_scene(self) -> None:
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.075, 0.085, 0.11)
        self.renderer.SetBackground2(0.16, 0.18, 0.22)
        self.renderer.GradientBackgroundOn()

        render_window = self.vtk_widget.GetRenderWindow()
        render_window.AddRenderer(self.renderer)
        render_window.SetMultiSamples(4)
        self.interactor = render_window.GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.ScalarVisibilityOff()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        self.actor.VisibilityOff()
        self.renderer.AddActor(self.actor)

        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(1.0, 1.0, 1.0)
        self.axes_widget = vtk.vtkOrientationMarkerWidget()
        self.axes_widget.SetOrientationMarker(axes)
        self.axes_widget.SetInteractor(self.interactor)
        self.axes_widget.SetViewport(0.0, 0.0, 0.14, 0.14)
        self.axes_widget.EnabledOn()
        self.axes_widget.InteractiveOff()

        self.vtk_widget.Initialize()

    def _apply_actor_style(self, *_args: Any, render: bool = True) -> None:
        prop = self.actor.GetProperty() if hasattr(self, "actor") else None
        if prop is None:
            return
        prop.SetColor(
            self._surface_color.redF(),
            self._surface_color.greenF(),
            self._surface_color.blueF(),
        )
        prop.SetOpacity(self.opacity_slider.value() / 100.0)
        prop.SetAmbient(0.12)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(28.0)
        prop.SetEdgeVisibility(self.edge_checkbox.isChecked())

        representation = self.representation_combo.currentText()
        if representation == "Wireframe":
            prop.SetRepresentationToWireframe()
        elif representation == "Points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(3.0)
        else:
            prop.SetRepresentationToSurface()

        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self.color_button.setStyleSheet(
            "QPushButton { background-color: "
            f"{self._surface_color.name()}; color: "
            f"{'black' if self._surface_color.lightnessF() > 0.55 else 'white'}; }}"
        )
        if render and hasattr(self, "vtk_widget"):
            self.vtk_widget.GetRenderWindow().Render()

    def _parallel_projection_changed(self, enabled: bool) -> None:
        self.renderer.GetActiveCamera().SetParallelProjection(enabled)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _reset_camera(self) -> None:
        if self._surface_polydata is None:
            return
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _set_axis_view(self, axis: str) -> None:
        if self._surface_polydata is None:
            return
        bounds = self.actor.GetBounds()
        if bounds is None or len(bounds) != 6:
            return
        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
        distance = max(span * 2.5, 1.0)
        direction = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }[axis]
        view_up = (0.0, 0.0, 1.0) if axis in {"X", "Y"} else (0.0, 1.0, 0.0)

        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(
            center[0] + direction[0] * distance,
            center[1] + direction[1] * distance,
            center[2] + direction[2] * distance,
        )
        camera.SetViewUp(*view_up)
        camera.OrthogonalizeViewUp()
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _save_screenshot(self) -> None:
        if self._surface_polydata is None:
            return
        suggested = (
            f"{self._current_file.stem}_lesion.png"
            if self._current_file
            else "lesion.png"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Lesion Viewer Screenshot",
            suggested,
            "PNG image (*.png)",
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(self.vtk_widget.GetRenderWindow())
            capture.SetScale(2)
            capture.ReadFrontBufferOff()
            capture.Update()
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(path)
            writer.SetInputConnection(capture.GetOutputPort())
            writer.Write()
            self._set_ready(f"Saved screenshot: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Screenshot failed", str(exc))
