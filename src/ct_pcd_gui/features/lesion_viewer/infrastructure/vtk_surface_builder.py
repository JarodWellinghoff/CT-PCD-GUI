from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import vtkmodules.all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk

from ct_pcd_gui.features.lesion_viewer.domain.surface_models import (
    SurfaceBuildRequest,
    SurfaceBuildResult,
    SurfaceMetrics,
)


class VtkSurfaceBuilder:
    """Create a detached VTK mesh from a numeric lesion volume.

    The builder owns no Qt or OpenGL objects, so it can run in a worker thread.
    The returned ``vtkPolyData`` is a deep copy and may be handed to the GUI
    thread after the operation completes.
    """

    _MAX_EXACT_METRIC_VOXELS = 50_000_000
    _METRIC_SAMPLE_VOXELS = 1_000_000

    def build(self, request: SurfaceBuildRequest) -> SurfaceBuildResult:
        started = time.perf_counter()
        self._validate(request)

        source = np.asarray(request.volume)
        permutation = tuple(request.axis_order.index(axis) for axis in "ZYX")
        volume_zyx = np.transpose(source, permutation)

        downsample = int(request.downsample)
        if downsample > 1:
            volume_zyx = volume_zyx[::downsample, ::downsample, ::downsample]
        if any(int(size) < 2 for size in volume_zyx.shape):
            raise ValueError(
                "The selected downsample factor is too large for this volume."
            )

        render_spacing = tuple(
            float(value) * downsample for value in request.spacing_xyz
        )
        render_volume = np.asarray(volume_zyx, dtype=np.float32)
        origin = (0.0, 0.0, 0.0)

        if request.foreground_below:
            # FlyingEdges extracts a surface for values above the iso-value. Negating
            # both the data and threshold preserves the requested below-threshold mask.
            render_volume = -render_volume
            iso_value = -float(request.threshold)
            source_min = -float(request.source_max)
            source_max = -float(request.source_min)
        else:
            iso_value = float(request.threshold)
            source_min = float(request.source_min)
            source_max = float(request.source_max)

        if request.pad_border:
            span = max(abs(source_max - source_min), 1.0)
            background = iso_value - span * 0.05 - 1.0e-6
            render_volume = np.pad(
                render_volume,
                1,
                mode="constant",
                constant_values=background,
            )
            origin = tuple(-value for value in render_spacing)

        if not np.isfinite(render_volume).all():
            offset = max(abs(iso_value) * 0.01, 1.0e-6)
            replacement = iso_value - offset
            render_volume = np.nan_to_num(
                render_volume,
                nan=replacement,
                posinf=replacement,
                neginf=replacement,
            )
        render_volume = np.ascontiguousarray(render_volume)

        z_size, y_size, x_size = (int(value) for value in render_volume.shape)
        image = vtk.vtkImageData()
        image.SetDimensions(x_size, y_size, z_size)
        image.SetSpacing(*render_spacing)
        image.SetOrigin(*origin)
        scalars = numpy_to_vtk(render_volume.ravel(order="C"), deep=True)
        scalars.SetName("LesionScalars")
        image.GetPointData().SetScalars(scalars)

        extractor = vtk.vtkFlyingEdges3D()
        extractor.SetInputData(image)
        extractor.SetValue(0, iso_value)
        extractor.ComputeNormalsOff()
        extractor.ComputeGradientsOff()
        pipeline: Any = extractor

        if request.largest_component_only:
            connectivity = vtk.vtkPolyDataConnectivityFilter()
            connectivity.SetInputConnection(pipeline.GetOutputPort())
            connectivity.SetExtractionModeToLargestRegion()
            connectivity.ColorRegionsOff()
            pipeline = connectivity

        if request.smoothing_iterations > 0:
            smoother = vtk.vtkWindowedSincPolyDataFilter()
            smoother.SetInputConnection(pipeline.GetOutputPort())
            smoother.SetNumberOfIterations(int(request.smoothing_iterations))
            smoother.BoundarySmoothingOff()
            smoother.FeatureEdgeSmoothingOff()
            smoother.SetFeatureAngle(120.0)
            smoother.SetPassBand(0.01)
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            pipeline = smoother

        if request.reduction_percent > 0:
            decimator = vtk.vtkDecimatePro()
            decimator.SetInputConnection(pipeline.GetOutputPort())
            decimator.SetTargetReduction(float(request.reduction_percent) / 100.0)
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
        if polydata.GetNumberOfPoints() == 0 or polydata.GetNumberOfCells() == 0:
            raise ValueError(
                "The selected threshold produced an empty surface. Adjust the "
                "threshold or select a different array."
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = self._calculate_metrics(request, polydata, elapsed_ms)
        return SurfaceBuildResult(polydata=polydata, metrics=metrics)

    @staticmethod
    def _validate(request: SurfaceBuildRequest) -> None:
        volume = np.asarray(request.volume)
        if volume.ndim != 3:
            raise ValueError(
                f"A 3-D volume is required; received shape {tuple(volume.shape)}."
            )
        if any(int(size) < 2 for size in volume.shape):
            raise ValueError(
                "Every volume dimension must contain at least two voxels."
            )
        if volume.dtype.kind not in "buif" or np.iscomplexobj(volume):
            raise ValueError("The lesion volume must be a real numeric array.")

        order = request.axis_order.upper()
        if request.axis_order != order or len(order) != 3 or set(order) != set("XYZ"):
            raise ValueError(
                "axis_order must contain uppercase X, Y, and Z exactly once."
            )
        if request.downsample not in {1, 2, 4, 8}:
            raise ValueError("downsample must be one of 1, 2, 4, or 8.")
        if not 0 <= request.smoothing_iterations <= 500:
            raise ValueError("smoothing_iterations must be between 0 and 500.")
        if not 0 <= request.reduction_percent <= 95:
            raise ValueError("reduction_percent must be between 0 and 95.")

        if len(request.spacing_xyz) != 3 or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in request.spacing_xyz
        ):
            raise ValueError(
                "Voxel spacing must contain three positive finite values."
            )

        source_min = float(request.source_min)
        source_max = float(request.source_max)
        threshold = float(request.threshold)
        if not math.isfinite(source_min) or not math.isfinite(source_max):
            raise ValueError("The source value range must be finite.")
        if source_max <= source_min:
            raise ValueError("The selected volume has no usable value variation.")
        if not math.isfinite(threshold):
            raise ValueError("The surface threshold must be finite.")
        if not source_min <= threshold <= source_max:
            raise ValueError(
                "The surface threshold must be inside the selected volume's value range."
            )

    def _calculate_metrics(
        self,
        request: SurfaceBuildRequest,
        polydata: vtk.vtkPolyData,
        elapsed_ms: float,
    ) -> SurfaceMetrics:
        values = np.asarray(request.volume)
        estimated = values.size > self._MAX_EXACT_METRIC_VOXELS

        if estimated:
            stride = max(
                1,
                int(math.ceil(values.size / self._METRIC_SAMPLE_VOXELS)),
            )
            sample = values.reshape(-1)[::stride]
            finite = sample[np.isfinite(sample)]
            comparison = (
                finite < request.threshold
                if request.foreground_below
                else finite > request.threshold
            )
            fraction = float(np.count_nonzero(comparison) / max(finite.size, 1))
            foreground_count = int(round(values.size * fraction))
        else:
            finite_mask = np.isfinite(values)
            comparison = (
                values < request.threshold
                if request.foreground_below
                else values > request.threshold
            )
            foreground_count = int(np.count_nonzero(comparison & finite_mask))

        voxel_volume_mm3 = float(np.prod(request.spacing_xyz))
        mask_volume_mm3 = float(foreground_count * voxel_volume_mm3)

        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        mass.Update()

        return SurfaceMetrics(
            foreground_count=foreground_count,
            foreground_estimated=estimated,
            mask_volume_mm3=mask_volume_mm3,
            surface_area_mm2=float(mass.GetSurfaceArea()),
            mesh_volume_mm3=float(mass.GetVolume()),
            points=int(polydata.GetNumberOfPoints()),
            cells=int(polydata.GetNumberOfCells()),
            threshold=float(request.threshold),
            spacing_xyz=tuple(float(value) for value in request.spacing_xyz),
            axis_order=request.axis_order,
            downsample=int(request.downsample),
            elapsed_ms=float(elapsed_ms),
            foreground_below=bool(request.foreground_below),
        )
