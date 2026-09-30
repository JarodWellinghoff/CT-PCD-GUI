from ct_pcd_gui.features.lesion_viewer.domain.surface_models import (
    SurfaceBuildRequest,
    SurfaceBuildResult,
)


class VtkSurfaceBuilder:
    def build(
        self,
        request: SurfaceBuildRequest,
    ) -> SurfaceBuildResult: ...

    def _rebuild_surface(self) -> None:
        candidate = self._candidate
        if candidate is None:
            return

        started = time.perf_counter()
        self._set_busy("Building 3-D surface…", indeterminate=True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            order = str(self.axis_order_combo.currentData() or "ZYX")
            permutation = tuple(order.index(axis_name) for axis_name in "ZYX")
            volume_zyx = np.transpose(candidate.array, permutation)

            downsample = int(self.downsample_combo.currentData() or 1)
            if downsample > 1:
                volume_zyx = volume_zyx[::downsample, ::downsample, ::downsample]
            if any(int(size) < 2 for size in volume_zyx.shape):
                raise ValueError(
                    "The selected downsample factor is too large for this volume."
                )

            threshold = float(self.threshold_spin.value())
            foreground_below = self.foreground_below_checkbox.isChecked()
            spacing_xyz = (
                float(self.spacing_x.value()),
                float(self.spacing_y.value()),
                float(self.spacing_z.value()),
            )
            render_spacing = tuple(value * downsample for value in spacing_xyz)

            render_volume = np.asarray(volume_zyx, dtype=np.float32)
            pad = self.pad_border_checkbox.isChecked()
            origin = (0.0, 0.0, 0.0)
            if pad:
                span = max(abs(candidate.max_value - candidate.min_value), 1.0)
                if foreground_below:
                    background = threshold + span * 0.05 + 1e-6
                else:
                    background = threshold - span * 0.05 - 1e-6
                render_volume = np.pad(
                    render_volume,
                    1,
                    mode="constant",
                    constant_values=background,
                )
                origin = tuple(-value for value in render_spacing)
            if not np.isfinite(render_volume).all():
                offset = max(abs(threshold) * 0.01, 1e-6)
                replacement = (
                    threshold + offset if foreground_below else threshold - offset
                )
                render_volume = np.nan_to_num(
                    render_volume,
                    nan=replacement,
                    posinf=replacement,
                    neginf=replacement,
                )
            render_volume = np.ascontiguousarray(render_volume)

            z_size, y_size, x_size = (int(v) for v in render_volume.shape)
            image = vtk.vtkImageData()
            image.SetDimensions(x_size, y_size, z_size)
            image.SetSpacing(*render_spacing)
            image.SetOrigin(*origin)
            scalars = numpy_to_vtk(render_volume.ravel(order="C"), deep=True)
            scalars.SetName("LesionScalars")
            image.GetPointData().SetScalars(scalars)

            extractor = vtk.vtkFlyingEdges3D()
            extractor.SetInputData(image)
            extractor.SetValue(0, threshold)
            extractor.ComputeNormalsOff()
            extractor.ComputeGradientsOff()
            pipeline: Any = extractor

            if self.largest_component_checkbox.isChecked():
                connectivity = vtk.vtkPolyDataConnectivityFilter()
                connectivity.SetInputConnection(pipeline.GetOutputPort())
                connectivity.SetExtractionModeToLargestRegion()
                connectivity.ColorRegionsOff()
                pipeline = connectivity

            smoothing_iterations = int(self.smoothing_spin.value())
            if smoothing_iterations > 0:
                smoother = vtk.vtkWindowedSincPolyDataFilter()
                smoother.SetInputConnection(pipeline.GetOutputPort())
                smoother.SetNumberOfIterations(smoothing_iterations)
                smoother.BoundarySmoothingOff()
                smoother.FeatureEdgeSmoothingOff()
                smoother.SetFeatureAngle(120.0)
                smoother.SetPassBand(0.01)
                smoother.NonManifoldSmoothingOn()
                smoother.NormalizeCoordinatesOn()
                pipeline = smoother

            reduction = float(self.reduction_spin.value()) / 100.0
            if reduction > 0.0:
                decimator = vtk.vtkDecimatePro()
                decimator.SetInputConnection(pipeline.GetOutputPort())
                decimator.SetTargetReduction(reduction)
                decimator.PreserveTopologyOn()
                decimator.SplittingOff()
                pipeline = decimator

            normals = vtk.vtkPolyDataNormals()
            normals.SetInputConnection(pipeline.GetOutputPort())
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOff()
            normals.SetFeatureAngle(80.0)
            normals.Update()

            polydata = vtk.vtkPolyData()
            polydata.DeepCopy(normals.GetOutput())
            if polydata.GetNumberOfPoints() == 0:
                raise ValueError(
                    "The selected threshold produced an empty surface. Adjust the "
                    "threshold or select a different array."
                )

            self._surface_polydata = polydata
            self.mapper.SetInputData(polydata)
            self.mapper.Update()
            self.actor.VisibilityOn()
            self.viewer_stack.setCurrentWidget(self.vtk_widget)
            self._apply_actor_style(render=False)

            if self._needs_camera_reset:
                self.renderer.ResetCamera()
                self.renderer.ResetCameraClippingRange()
                self._needs_camera_reset = False
            self.vtk_widget.GetRenderWindow().Render()

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_render_metrics = self._calculate_metrics(
                candidate=candidate,
                threshold=threshold,
                spacing_xyz=spacing_xyz,
                axis_order=order,
                downsample=downsample,
                polydata=polydata,
                elapsed_ms=elapsed_ms,
                foreground_below=foreground_below,
            )
            self._update_info_text()
            self._set_ready(
                f"Rendered {polydata.GetNumberOfCells():,} triangles in "
                f"{elapsed_ms / 1000.0:.2f} s."
            )
        except ValueError as exc:
            self.actor.VisibilityOff()
            self.vtk_widget.GetRenderWindow().Render()
            self._set_ready(str(exc))
        except Exception as exc:
            self._set_ready("Surface rendering failed.")
            QMessageBox.warning(self, "Could not render lesion surface", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def _calculate_metrics(
        self,
        *,
        candidate: VolumeCandidate,
        threshold: float,
        spacing_xyz: tuple[float, float, float],
        axis_order: str,
        downsample: int,
        polydata: vtk.vtkPolyData,
        elapsed_ms: float,
        foreground_below: bool,
    ) -> dict[str, Any]:
        foreground_count: int | None
        estimated = False
        array = candidate.array
        if array.size <= 50_000_000:
            values = np.asarray(array)
            comparison = values < threshold if foreground_below else values > threshold
            foreground_count = int(np.count_nonzero(comparison))
        else:
            stride = max(1, array.size // 1_000_000)
            sample = np.asarray(array).reshape(-1)[::stride]
            comparison = sample < threshold if foreground_below else sample > threshold
            fraction = float(np.count_nonzero(comparison) / max(sample.size, 1))
            foreground_count = int(round(array.size * fraction))
            estimated = True

        voxel_volume_mm3 = float(np.prod(spacing_xyz))
        mask_volume_mm3 = foreground_count * voxel_volume_mm3

        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        mass.Update()
        surface_area = float(mass.GetSurfaceArea())
        mesh_volume = float(mass.GetVolume())

        return {
            "foreground_count": foreground_count,
            "foreground_estimated": estimated,
            "mask_volume_mm3": mask_volume_mm3,
            "surface_area_mm2": surface_area,
            "mesh_volume_mm3": mesh_volume,
            "points": int(polydata.GetNumberOfPoints()),
            "cells": int(polydata.GetNumberOfCells()),
            "threshold": threshold,
            "spacing_xyz": spacing_xyz,
            "axis_order": axis_order,
            "downsample": downsample,
            "elapsed_ms": elapsed_ms,
            "foreground_below": foreground_below,
        }
