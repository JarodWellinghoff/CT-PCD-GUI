from __future__ import annotations

from pathlib import Path

import vtkmodules.all as vtk
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .view_state import AppearanceState


class LesionVtkScene(QWidget):
    """Qt-owned VTK render scene for an immutable lesion surface."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._surface_polydata: vtk.vtkPolyData | None = None
        self._cleaned_up = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.vtk_widget)

        self._build_vtk_scene()

    @property
    def surface(self) -> vtk.vtkPolyData | None:
        return self._surface_polydata

    def set_surface(
        self,
        polydata: vtk.vtkPolyData,
        *,
        reset_camera: bool,
    ) -> None:
        if self._cleaned_up:
            return
        detached = vtk.vtkPolyData()
        detached.DeepCopy(polydata)
        self._surface_polydata = detached
        self.mapper.SetInputData(detached)
        self.mapper.Update()
        self.actor.VisibilityOn()
        if reset_camera:
            self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._render()

    def clear_surface(self) -> None:
        self._surface_polydata = None
        self.mapper.SetInputData(None)
        self.actor.VisibilityOff()
        self._render()

    def apply_appearance(self, state: AppearanceState) -> None:
        if self._cleaned_up:
            return
        prop = self.actor.GetProperty()
        prop.SetColor(*state.color_rgb)
        prop.SetOpacity(max(0.0, min(1.0, float(state.opacity))))
        prop.SetAmbient(0.12)
        prop.SetDiffuse(0.72)
        prop.SetSpecular(0.35)
        prop.SetSpecularPower(28.0)
        prop.SetEdgeVisibility(bool(state.show_edges))

        representation = state.representation.casefold()
        if representation == "wireframe":
            prop.SetRepresentationToWireframe()
        elif representation == "points":
            prop.SetRepresentationToPoints()
            prop.SetPointSize(3.0)
        else:
            prop.SetRepresentationToSurface()

        self.set_parallel_projection(state.parallel_projection, render=False)
        self._render()

    def set_parallel_projection(self, enabled: bool, *, render: bool = True) -> None:
        if self._cleaned_up:
            return
        camera = self.renderer.GetActiveCamera()
        camera.SetParallelProjection(bool(enabled))
        self.renderer.ResetCameraClippingRange()
        if render:
            self._render()

    def reset_camera(self) -> None:
        if self._surface_polydata is None or self._cleaned_up:
            return
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        self._render()

    def set_axis_view(self, axis: str) -> None:
        if self._surface_polydata is None or self._cleaned_up:
            return
        bounds = self.actor.GetBounds()
        if bounds is None or len(bounds) != 6:
            return

        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )
        span = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )
        distance = max(float(span) * 2.5, 1.0)
        normalized = axis.strip().upper().replace("+", "")
        directions = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }
        if normalized not in directions:
            raise ValueError(f"Unsupported camera axis: {axis}")
        direction = directions[normalized]
        view_up = (0.0, 0.0, 1.0) if normalized in {"X", "Y"} else (0.0, 1.0, 0.0)

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
        self._render()

    def save_screenshot(self, path: str | Path, scale: int = 2) -> Path:
        if self._surface_polydata is None:
            raise ValueError("No lesion surface is available to capture.")
        output = Path(path).expanduser()
        if output.suffix.lower() != ".png":
            output = output.with_suffix(".png")
        output.parent.mkdir(parents=True, exist_ok=True)

        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(self.vtk_widget.GetRenderWindow())
        capture.SetScale(max(1, int(scale)))
        capture.ReadFrontBufferOff()
        capture.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(output))
        writer.SetInputConnection(capture.GetOutputPort())
        if writer.Write() != 1:
            raise OSError("VTK did not confirm that the screenshot was written.")
        return output

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            self.axes_widget.EnabledOff()
        except Exception:
            pass
        try:
            render_window = self.vtk_widget.GetRenderWindow()
            if render_window is not None:
                render_window.Finalize()
        finally:
            self.vtk_widget.close()

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

    def _render(self) -> None:
        if not self._cleaned_up:
            self.vtk_widget.GetRenderWindow().Render()
