from dataclasses import dataclass, field
from typing import Optional
from PyQt6.QtCore import pyqtSignal, QObject
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkIdList


# ===========================================================================
# ROI data model
# ===========================================================================

ROI_COLORS = [
    (1.0, 0.2, 0.2),
    (0.2, 1.0, 0.2),
    (0.3, 0.6, 1.0),
    (1.0, 1.0, 0.2),
    (1.0, 0.5, 0.0),
    (0.8, 0.2, 1.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.4, 0.7),
]


@dataclass
class ROI:
    roi_id: int
    name: str
    roi_type: str
    slice_index: int
    orientation: int
    points: list
    color: tuple = (1, 0, 0)
    actor: Optional[vtkActor] = field(default=None, repr=False)
    handles_actor: Optional[vtkActor] = field(default=None, repr=False)


class ROIManager(QObject):
    roi_added = pyqtSignal(object)
    roi_removed = pyqtSignal(int)
    rois_changed = pyqtSignal()
    selection_changed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._rois: dict[int, ROI] = {}
        self._next_id = 1
        self._color_idx = 0
        self._selected_id: Optional[int] = None

    @property
    def rois(self):
        return self._rois

    @property
    def selected_id(self):
        return self._selected_id

    @property
    def selected_roi(self):
        return (
            self._rois.get(self._selected_id) if self._selected_id is not None else None
        )

    def next_color(self):
        c = ROI_COLORS[self._color_idx % len(ROI_COLORS)]
        self._color_idx += 1
        return c

    def add_roi(
        self,
        name,
        roi_type,
        slice_index,
        orientation,
        points,
        color=None,
        renderer=None,
    ) -> ROI:
        if color is None:
            color = self.next_color()
        roi = ROI(
            roi_id=self._next_id,
            name=name,
            roi_type=roi_type,
            slice_index=slice_index,
            orientation=orientation,
            points=list(points),
            color=color,
        )
        self._next_id += 1
        roi.actor = self._build_contour_actor(roi)
        if renderer:
            renderer.AddActor(roi.actor)
        self._rois[roi.roi_id] = roi
        self.roi_added.emit(roi)
        self.rois_changed.emit()
        return roi

    def remove_roi(self, roi_id, renderer=None):
        if roi_id == self._selected_id:
            self.select(None, renderer)
        roi = self._rois.pop(roi_id, None)
        if roi is None:
            return
        if renderer:
            if roi.actor:
                renderer.RemoveActor(roi.actor)
            if roi.handles_actor:
                renderer.RemoveActor(roi.handles_actor)
        self.roi_removed.emit(roi_id)
        self.rois_changed.emit()

    def clear(self, renderer=None):
        self.select(None, renderer)
        for roi in list(self._rois.values()):
            if renderer:
                if roi.actor:
                    renderer.RemoveActor(roi.actor)
                if roi.handles_actor:
                    renderer.RemoveActor(roi.handles_actor)
        self._rois.clear()
        self.rois_changed.emit()

    def rename_roi(self, roi_id, new_name):
        if roi_id in self._rois:
            self._rois[roi_id].name = new_name
            self.rois_changed.emit()

    def select(self, roi_id, renderer=None):
        prev = (
            self._rois.get(self._selected_id) if self._selected_id is not None else None
        )
        if prev:
            if prev.actor:
                prev.actor.GetProperty().SetLineWidth(2.0)
            if prev.handles_actor and renderer:
                renderer.RemoveActor(prev.handles_actor)
                prev.handles_actor = None
        self._selected_id = roi_id
        cur = self._rois.get(roi_id)
        if cur:
            if cur.actor:
                cur.actor.GetProperty().SetLineWidth(3.5)
            cur.handles_actor = self._build_handles_actor(cur)
            if renderer:
                renderer.AddActor(cur.handles_actor)
        self.selection_changed.emit(roi_id)

    def update_point(self, roi_id, vertex_idx, new_world, renderer=None):
        roi = self._rois.get(roi_id)
        if not roi or vertex_idx < 0 or vertex_idx >= len(roi.points):
            return
        roi.points[vertex_idx] = new_world
        self._rebuild_visuals(roi, renderer)

    def translate_roi(self, roi_id, dx, dy, dz, renderer=None):
        roi = self._rois.get(roi_id)
        if not roi:
            return
        roi.points = [(p[0] + dx, p[1] + dy, p[2] + dz) for p in roi.points]
        self._rebuild_visuals(roi, renderer)

    def insert_vertex(self, roi_id, edge_idx, world_pt, renderer=None):
        roi = self._rois.get(roi_id)
        if not roi:
            return
        roi.points.insert(edge_idx + 1, world_pt)
        self._rebuild_visuals(roi, renderer)
        self.rois_changed.emit()

    def delete_vertex(self, roi_id, vertex_idx, renderer=None):
        roi = self._rois.get(roi_id)
        if not roi or len(roi.points) <= 3:
            return False
        if vertex_idx < 0 or vertex_idx >= len(roi.points):
            return False
        roi.points.pop(vertex_idx)
        self._rebuild_visuals(roi, renderer)
        self.rois_changed.emit()
        return True

    def _rebuild_visuals(self, roi, renderer=None):
        if renderer and roi.actor:
            renderer.RemoveActor(roi.actor)
        roi.actor = self._build_contour_actor(roi)
        if self._selected_id == roi.roi_id:
            roi.actor.GetProperty().SetLineWidth(3.5)
        if renderer:
            renderer.AddActor(roi.actor)
        if roi.handles_actor and renderer:
            renderer.RemoveActor(roi.handles_actor)
            roi.handles_actor = None
        if self._selected_id == roi.roi_id:
            roi.handles_actor = self._build_handles_actor(roi)
            if renderer:
                renderer.AddActor(roi.handles_actor)

    def update_visibility(self, current_slice, current_orientation):
        for roi in self._rois.values():
            vis = (
                roi.slice_index == current_slice
                and roi.orientation == current_orientation
            )
            if roi.actor:
                roi.actor.SetVisibility(vis)
            if roi.handles_actor:
                roi.handles_actor.SetVisibility(vis and self._selected_id == roi.roi_id)

    @staticmethod
    def _build_contour_actor(roi):
        pts = roi.points
        if len(pts) < 2:
            return vtkActor()
        vtk_points = vtkPoints()
        for p in pts:
            vtk_points.InsertNextPoint(p)
        lines = vtkCellArray()
        ids = vtkIdList()
        for i in range(len(pts)):
            ids.InsertNextId(i)
        ids.InsertNextId(0)
        lines.InsertNextCell(ids)
        poly = vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(lines)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*roi.color)
        actor.GetProperty().SetLineWidth(2.0)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        offset = [0.0, 0.0, 0.0]
        offset[roi.orientation] = 0.5
        actor.SetPosition(*offset)
        return actor

    @staticmethod
    def _build_handles_actor(roi):
        vtk_points = vtkPoints()
        verts = vtkCellArray()
        for p in roi.points:
            pid = vtk_points.InsertNextPoint(p)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(pid)
        poly = vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetVerts(verts)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        actor.GetProperty().SetPointSize(8.0)
        actor.GetProperty().SetAmbient(1.0)
        actor.GetProperty().SetDiffuse(0.0)
        offset = [0.0, 0.0, 0.0]
        offset[roi.orientation] = 0.6
        actor.SetPosition(*offset)
        return actor
