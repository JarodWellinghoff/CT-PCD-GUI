import math
import vtkmodules.all as vtk
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
)
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkPoints, vtkCommand, vtkIdList
from DrawTool import DrawTool


# ===========================================================================
# Interactor style (drawing + editing)
# ===========================================================================


class DrawingInteractorStyle(vtk.vtkInteractorStyleImage):
    HANDLE_PICK_PX = 10
    CONTOUR_PICK_PX = 8

    def __init__(self, viewer_widget):
        super().__init__()
        self.viewer_widget = viewer_widget
        self.image_viewer = None
        self.status_actor = None
        self.slice = 0
        self.min_slice = 0
        self.max_slice = 0
        self.draw_tool = DrawTool.NONE
        self.is_drawing = False
        self.current_points = []
        self.preview_actor = None
        self.anchor_point = None
        self._dragging_vertex = None
        self._dragging_whole = False
        self._drag_start = None

        self.AddObserver(vtkCommand.MouseWheelForwardEvent, self._scroll_fwd)
        self.AddObserver(vtkCommand.MouseWheelBackwardEvent, self._scroll_bwd)
        self.AddObserver(vtkCommand.KeyPressEvent, self._key_press)
        self.AddObserver(vtkCommand.LeftButtonPressEvent, self._left_press)
        self.AddObserver(vtkCommand.LeftButtonReleaseEvent, self._left_release)
        self.AddObserver(vtkCommand.MouseMoveEvent, self._mouse_move)
        self.AddObserver(vtkCommand.RightButtonPressEvent, self._right_press)
        self.AddObserver(vtkCommand.LeftButtonDoubleClickEvent, self._double_click)

    def setup(self, image_viewer, status_actor):
        self.image_viewer = image_viewer
        self.status_actor = status_actor
        self.slice = image_viewer.GetSliceMin()
        self.min_slice = image_viewer.GetSliceMin()
        self.max_slice = image_viewer.GetSliceMax()
        self._update_status()

    def _update_status(self):
        if not self.status_actor:
            return
        tool = self.draw_tool.upper() if self.draw_tool != DrawTool.NONE else ""
        suffix = f"  [{tool}]" if tool else ""
        self.status_actor.GetMapper().SetInput(
            f"Slice {self.slice + 1}/{self.max_slice + 1}{suffix}"
        )

    def _render(self):
        if self.image_viewer:
            self.image_viewer.Render()

    def _set_slice(self, s):
        self.slice = s
        if self.image_viewer:
            self.image_viewer.SetSlice(s)
            self._update_status()
            self.viewer_widget.roi_manager.update_visibility(
                s, self.image_viewer.GetSliceOrientation()
            )
            self._render()

    def _scroll_fwd(self, _o, _e):
        if self.slice < self.max_slice:
            self._set_slice(self.slice + 1)

    def _scroll_bwd(self, _o, _e):
        if self.slice > self.min_slice:
            self._set_slice(self.slice - 1)

    def change_orientation(self):
        if not self.image_viewer:
            return
        new_o = (self.image_viewer.GetSliceOrientation() + 1) % 3
        self.image_viewer.SetSliceOrientation(new_o)
        self.min_slice = self.image_viewer.GetSliceMin()
        self.max_slice = self.image_viewer.GetSliceMax()
        self._set_slice(self.min_slice)

    def _key_press(self, o, e):
        key = self.GetInteractor().GetKeySym()
        if key == "Up":
            self._scroll_fwd(o, e)
        elif key == "Down":
            self._scroll_bwd(o, e)
        elif key == "Escape":
            self._cancel_drawing()
        elif key in ("Delete", "BackSpace"):
            self._delete_hovered_vertex()

    def _pick_world(self):
        if not self.image_viewer:
            return (0, 0, 0)
        ix, iy = self.GetInteractor().GetEventPosition()
        ren = self.image_viewer.GetRenderer()
        picker = vtk.vtkWorldPointPicker()
        picker.Pick(ix, iy, 0, ren)
        wx, wy, wz = picker.GetPickPosition()
        orient = self.image_viewer.GetSliceOrientation()
        img = self.image_viewer.GetInput()
        origin, spacing = img.GetOrigin(), img.GetSpacing()
        if orient == 2:
            wz = origin[2] + self.slice * spacing[2]
        elif orient == 1:
            wy = origin[1] + self.slice * spacing[1]
        elif orient == 0:
            wx = origin[0] + self.slice * spacing[0]
        return (wx, wy, wz)

    def _display_pos(self):
        return self.GetInteractor().GetEventPosition()

    def _world_to_display(self, pt):
        if not self.image_viewer:
            return (0, 0)
        ren = self.image_viewer.GetRenderer()
        ren.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
        ren.WorldToDisplay()
        dp = ren.GetDisplayPoint()
        return (dp[0], dp[1])

    def _find_nearest_vertex(self, roi, threshold_px=None):
        if threshold_px is None:
            threshold_px = self.HANDLE_PICK_PX
        mx, my = self._display_pos()
        best_idx, best_d = None, float("inf")
        for i, pt in enumerate(roi.points):
            dx, dy = self._world_to_display(pt)
            d = math.hypot(dx - mx, dy - my)
            if d < threshold_px and d < best_d:
                best_d, best_idx = d, i
        return best_idx

    def _find_nearest_edge(self, roi, threshold_px=None):
        if threshold_px is None:
            threshold_px = self.CONTOUR_PICK_PX
        mx, my = self._display_pos()
        n = len(roi.points)
        best_idx, best_d = None, float("inf")
        for i in range(n):
            j = (i + 1) % n
            ax, ay = self._world_to_display(roi.points[i])
            bx, by = self._world_to_display(roi.points[j])
            d = self._seg_dist(mx, my, ax, ay, bx, by)
            if d < threshold_px and d < best_d:
                best_d, best_idx = d, i
        return best_idx

    def _find_nearest_roi_on_slice(self):
        mgr = self.viewer_widget.roi_manager
        if not self.image_viewer:
            return None
        orient = self.image_viewer.GetSliceOrientation()
        mx, my = self._display_pos()
        best_id, best_d = None, float("inf")
        for roi in mgr.rois.values():
            if roi.slice_index != self.slice or roi.orientation != orient:
                continue
            n = len(roi.points)
            for i in range(n):
                j = (i + 1) % n
                ax, ay = self._world_to_display(roi.points[i])
                bx, by = self._world_to_display(roi.points[j])
                d = self._seg_dist(mx, my, ax, ay, bx, by)
                if d < self.CONTOUR_PICK_PX and d < best_d:
                    best_d, best_id = d, roi.roi_id
        return best_id

    @staticmethod
    def _seg_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 < 1e-12:
            return math.hypot(px - ax, py - ay)
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
        return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

    # ---- Dispatch ----

    def _left_press(self, _o, _e):
        if self.draw_tool == DrawTool.EDIT:
            self._edit_left_press()
        elif self.draw_tool == DrawTool.NONE:
            self.OnLeftButtonDown()
        else:
            self._draw_left_press()

    def _left_release(self, _o, _e):
        if self.draw_tool == DrawTool.EDIT:
            self._edit_left_release()
        elif self.draw_tool == DrawTool.NONE:
            self.OnLeftButtonUp()
        else:
            self._draw_left_release()

    def _mouse_move(self, _o, _e):
        if self.draw_tool == DrawTool.EDIT:
            self._edit_mouse_move()
        elif self.draw_tool == DrawTool.NONE or not self.is_drawing:
            self.OnMouseMove()
        else:
            self._draw_mouse_move()

    def _right_press(self, _o, _e):
        if self.draw_tool == DrawTool.POLYGON and self.is_drawing:
            if len(self.current_points) >= 3:
                self._finalize_draw()
            else:
                self._cancel_drawing()
        elif self.draw_tool == DrawTool.EDIT:
            self._delete_hovered_vertex()
        else:
            self.OnRightButtonDown()

    def _double_click(self, _o, _e):
        if self.draw_tool != DrawTool.EDIT:
            return
        if not self.image_viewer:
            return
        mgr = self.viewer_widget.roi_manager
        roi = mgr.selected_roi
        if not roi:
            return
        edge = self._find_nearest_edge(roi, 15)
        if edge is not None:
            mgr.insert_vertex(
                roi.roi_id, edge, self._pick_world(), self.image_viewer.GetRenderer()
            )
            self._render()

    # ---- Edit handlers ----

    def _edit_left_press(self):
        mgr = self.viewer_widget.roi_manager
        if not self.image_viewer:
            return
        ren = self.image_viewer.GetRenderer()
        world = self._pick_world()
        roi = mgr.selected_roi
        if roi and roi.slice_index == self.slice:
            vidx = self._find_nearest_vertex(roi)
            if vidx is not None:
                self._dragging_vertex = vidx
                self._drag_start = world
                return
            edge = self._find_nearest_edge(roi)
            if edge is not None:
                self._dragging_whole = True
                self._drag_start = world
                return
        mgr.select(self._find_nearest_roi_on_slice(), ren)
        self._render()

    def _edit_mouse_move(self):
        mgr = self.viewer_widget.roi_manager
        if not self.image_viewer:
            return
        ren = self.image_viewer.GetRenderer()
        world = self._pick_world()
        if self._dragging_vertex is not None and self._drag_start:
            roi = mgr.selected_roi
            if roi:
                mgr.update_point(roi.roi_id, self._dragging_vertex, world, ren)
                self._render()
            return
        if self._dragging_whole and self._drag_start:
            roi = mgr.selected_roi
            if roi:
                dx = world[0] - self._drag_start[0]
                dy = world[1] - self._drag_start[1]
                dz = world[2] - self._drag_start[2]
                mgr.translate_roi(roi.roi_id, dx, dy, dz, ren)
                self._drag_start = world
                self._render()
            return
        self.OnMouseMove()

    def _edit_left_release(self):
        self._dragging_vertex = None
        self._dragging_whole = False
        self._drag_start = None

    def _delete_hovered_vertex(self):
        mgr = self.viewer_widget.roi_manager
        roi = mgr.selected_roi
        if not roi:
            return
        if not self.image_viewer:
            return
        vidx = self._find_nearest_vertex(roi, 15)
        if vidx is not None:
            mgr.delete_vertex(roi.roi_id, vidx, self.image_viewer.GetRenderer())
            self._render()

    # ---- Draw handlers ----

    def _draw_left_press(self):
        w = self._pick_world()
        if self.draw_tool == DrawTool.FREEHAND:
            self.is_drawing = True
            self.current_points = [w]
        elif self.draw_tool == DrawTool.POLYGON:
            if not self.is_drawing:
                self.is_drawing = True
                self.current_points = [w]
            else:
                self.current_points.append(w)
            self._update_preview()
        elif self.draw_tool in (DrawTool.RECTANGLE, DrawTool.ELLIPSE):
            self.is_drawing = True
            self.anchor_point = w
            self.current_points = [w]

    def _draw_left_release(self):
        if self.draw_tool == DrawTool.FREEHAND and self.is_drawing:
            self._finalize_draw()
        elif (
            self.draw_tool in (DrawTool.RECTANGLE, DrawTool.ELLIPSE) and self.is_drawing
        ):
            w = self._pick_world()
            if self.anchor_point and self.anchor_point != w:
                gen = (
                    self._rect_pts
                    if self.draw_tool == DrawTool.RECTANGLE
                    else self._ellipse_pts
                )
                self.current_points = gen(self.anchor_point, w)
                self._finalize_draw()
            else:
                self._cancel_drawing()

    def _draw_mouse_move(self):
        if not self.is_drawing:
            self.OnMouseMove()
            return
        w = self._pick_world()
        if self.draw_tool == DrawTool.FREEHAND:
            self.current_points.append(w)
            self._update_preview()
        elif self.draw_tool == DrawTool.POLYGON:
            self._update_preview(extra=w)
        elif self.draw_tool in (DrawTool.RECTANGLE, DrawTool.ELLIPSE):
            if self.anchor_point:
                gen = (
                    self._rect_pts
                    if self.draw_tool == DrawTool.RECTANGLE
                    else self._ellipse_pts
                )
                self.current_points = gen(self.anchor_point, w)
                self._update_preview()

    @staticmethod
    def _rect_pts(p1, p2):
        x1, y1, z1 = p1
        x2, y2, _ = p2
        s = min(abs(x2 - x1), abs(y2 - y1))
        cx = s if x2 >= x1 else -s
        cy = s if y2 >= y1 else -s
        x2, y2 = x1 + cx, y1 + cy
        return [(x1, y1, z1), (x2, y1, z1), (x2, y2, z1), (x1, y2, z1)]

    @staticmethod
    def _ellipse_pts(p1, p2, n=64):
        cx, cy, cz = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2
        rx, ry = abs(p2[0] - p1[0]) / 2, abs(p2[1] - p1[1]) / 2
        return [
            (
                cx + rx * math.cos(2 * math.pi * i / n),
                cy + ry * math.sin(2 * math.pi * i / n),
                cz,
            )
            for i in range(n)
        ]

    def _update_preview(self, extra=None):
        if not self.image_viewer:
            return
        ren = self.image_viewer.GetRenderer()
        if self.preview_actor:
            ren.RemoveActor(self.preview_actor)
        pts = list(self.current_points)
        if extra:
            pts.append(extra)
        if len(pts) < 2:
            self._render()
            return
        vp = vtkPoints()
        for p in pts:
            vp.InsertNextPoint(p)
        lines = vtkCellArray()
        ids = vtkIdList()
        for i in range(len(pts)):
            ids.InsertNextId(i)
        if self.draw_tool != DrawTool.FREEHAND:
            ids.InsertNextId(0)
        lines.InsertNextCell(ids)
        pd = vtkPolyData()
        pd.SetPoints(vp)
        pd.SetLines(lines)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(pd)
        self.preview_actor = vtkActor()
        self.preview_actor.SetMapper(mapper)
        self.preview_actor.GetProperty().SetColor(1, 1, 0)
        self.preview_actor.GetProperty().SetLineWidth(1.5)
        self.preview_actor.GetProperty().SetLineStipplePattern(0xAAAA)
        self.preview_actor.GetProperty().SetAmbient(1.0)
        self.preview_actor.GetProperty().SetDiffuse(0.0)
        offset = [0.0, 0.0, 0.0]
        offset[self.image_viewer.GetSliceOrientation()] = 0.5
        self.preview_actor.SetPosition(*offset)
        ren.AddActor(self.preview_actor)
        self._render()

    def _remove_preview(self):
        if self.preview_actor and self.image_viewer:
            self.image_viewer.GetRenderer().RemoveActor(self.preview_actor)
            self.preview_actor = None

    def _finalize_draw(self):
        self._remove_preview()
        if len(self.current_points) >= 2 and self.image_viewer:
            self.viewer_widget.finalize_roi(
                self.draw_tool,
                self.current_points,
                self.slice,
                self.image_viewer.GetSliceOrientation(),
            )
        self.is_drawing = False
        self.current_points = []
        self.anchor_point = None
        self._render()

    def _cancel_drawing(self):
        self._remove_preview()
        self.is_drawing = False
        self.current_points = []
        self.anchor_point = None
        self.draw_tool = DrawTool.NONE
        self._update_status()
        self._render()
        self.viewer_widget.drawing_cancelled.emit()
