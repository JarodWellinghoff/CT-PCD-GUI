from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QObject, QSettings, QTimer, Signal, Slot

from ct_pcd_gui.features.lesion_viewer.application.use_cases import (
    BuildLesionSurface, LoadLesionFile, ScanLesionLibrary,
)
from ct_pcd_gui.features.lesion_viewer.domain.models import LesionFileData, LesionFileRecord, VolumeCandidate
from ct_pcd_gui.features.lesion_viewer.domain.surface_models import SurfaceBuildRequest, SurfaceBuildResult
from ct_pcd_gui.features.lesion_viewer.infrastructure.mesh_exporter import VtkMeshExporter
from ct_pcd_gui.shared.qt.task_runner import QtTaskRunner
from .task_controllers import LibraryTaskController, SurfaceTaskController
from .view import LesionViewerPanel, LesionViewerWorkspace


class LesionViewerPresenter(QObject):
    running_changed = Signal(bool)
    error_requested = Signal(str, str)

    def __init__(self, *, panel: LesionViewerPanel, workspace: LesionViewerWorkspace,
                 scan_library: ScanLesionLibrary, load_file: LoadLesionFile,
                 build_surface: BuildLesionSurface, exporter: VtkMeshExporter,
                 task_runner: QtTaskRunner, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.panel, self.workspace, self.exporter = panel, workspace, exporter
        self.library = LibraryTaskController(scan_library, load_file, task_runner, self)
        self.surface = SurfaceTaskController(build_surface, task_runner, self)
        self.settings = QSettings(); self.data: LesionFileData | None = None
        self.candidate: VolumeCandidate | None = None
        self.last_surface: SurfaceBuildResult | None = None
        self._reset_camera = True; self._active_once = False; self._cleaned = False
        self.timer = QTimer(self); self.timer.setSingleShot(True); self.timer.setInterval(180)
        self.timer.timeout.connect(self._build_surface)
        panel.source_requested.connect(self.open_source); panel.refresh_requested.connect(self.refresh)
        panel.file_requested.connect(self.load_path); panel.candidate_changed.connect(self.select_candidate)
        panel.surface_settings_changed.connect(self.schedule_surface_build)
        panel.appearance_changed.connect(workspace.apply_appearance)
        panel.reset_camera_requested.connect(workspace.scene.reset_camera)
        panel.axis_view_requested.connect(workspace.scene.set_axis_view)
        panel.screenshot_requested.connect(self.save_screenshot); panel.export_requested.connect(self.export_surface)
        self.library.batch_ready.connect(self._batch); self.library.scan_completed.connect(self._scan_done)
        self.library.file_loaded.connect(self._loaded); self.library.status.connect(workspace.set_ready)
        self.library.failed.connect(self._error); self.surface.completed.connect(self._surface_done)
        self.surface.status.connect(workspace.set_ready); self.surface.failed.connect(self._error)
        self.library.running_changed.connect(lambda _v: self._running())
        self.surface.running_changed.connect(lambda _v: self._running())
        workspace.apply_appearance(panel.appearance_state())
        last = self.settings.value("lesion_viewer/last_source", "", type=str)
        if last: panel.set_source_path(last)

    @property
    def is_running(self) -> bool: return self.library.is_running or self.surface.is_running

    @Slot()
    def activate(self) -> None:
        self.workspace.apply_appearance(self.panel.appearance_state())
        if self._active_once: return
        self._active_once = True; source = self.panel.source_path()
        if source and Path(source).expanduser().exists(): QTimer.singleShot(0, lambda: self.open_source(source))
    @Slot()
    def deactivate(self) -> None: pass

    @Slot(str)
    def open_source(self, raw: str) -> None:
        if not raw.strip(): self._error("Lesion Viewer", "Select a MAT/NPZ file or library folder."); return
        path = Path(raw).expanduser()
        if not path.exists(): self._error("Lesion Viewer", f"Path does not exist:\n{path}"); return
        path = path.resolve(); self.panel.set_source_path(path)
        self.settings.setValue("lesion_viewer/last_source", str(path))
        if path.is_file(): self.load_path(path)
        elif path.is_dir():
            self.library.cancel(); self.surface.cancel(); self.panel.clear_records()
            self.workspace.set_busy("Scanning lesion library…"); self.library.scan(path)

    @Slot()
    def refresh(self) -> None: self.open_source(self.panel.source_path())

    @Slot(object)
    def load_path(self, raw: object) -> None:
        path = Path(raw).expanduser()
        if path.suffix.lower() not in {".mat", ".npz"}:
            self._error("Unsupported lesion file", "Only .mat and .npz files are supported."); return
        if not path.is_file(): self._error("Missing lesion file", f"File not found:\n{path}"); return
        self.surface.cancel(); self.workspace.set_busy(f"Loading {path.name}…")
        self.library.load(path.resolve())

    def _batch(self, records: object) -> None:
        batch = [x for x in records if isinstance(x, LesionFileRecord)]
        self.panel.append_records(batch); self.workspace.set_status(f"Scanning… {self.panel.file_model.rowCount():,} files")
    def _scan_done(self, records: object) -> None:
        values = [x for x in records if isinstance(x, LesionFileRecord)]
        self.panel.set_records(values); self.workspace.set_ready(f"Found {len(values):,} lesion file(s).")
    def _loaded(self, result: object) -> None:
        if not isinstance(result, LesionFileData): return
        self.data = result; self.candidate = None; self.last_surface = None; self._reset_camera = True
        self.panel.set_current_file(result.path); self.panel.set_controls_enabled(True); self.panel.set_candidates(result)
        self.workspace.set_ready(f"Loaded {result.path.name}."); self._update_info()

    @Slot(int)
    def select_candidate(self, index: int) -> None:
        if self.data is None: return
        candidate = self.panel.candidate_at(index)
        if candidate is None: return
        self.candidate = candidate; self.last_surface = None; self._reset_camera = True
        order = str(self.data.file_metadata.get("suggested_axis_order", "ZYX")).upper()
        if len(order) != 3 or set(order) != set("XYZ"): order = "ZYX"
        self.panel.configure_candidate(candidate, spacing_xyz=self.data.detected_spacing_xyz, axis_order=order)
        self._update_info(); self.schedule_surface_build()

    @Slot()
    def schedule_surface_build(self, *_args) -> None:
        if self.candidate is not None and not self._cleaned: self.timer.start()
    @Slot()
    def _build_surface(self) -> None:
        if self.candidate is None: return
        s = self.panel.surface_settings(); c = self.candidate
        request = SurfaceBuildRequest(c.array, c.min_value, c.max_value, s.axis_order, s.spacing_xyz,
                                      s.threshold, s.foreground_below, s.downsample,
                                      s.smoothing_iterations, s.reduction_percent,
                                      s.largest_component_only, s.pad_border)
        self.workspace.set_busy("Building lesion surface…"); self.surface.start(request)
    def _surface_done(self, result: object) -> None:
        if not isinstance(result, SurfaceBuildResult): return
        self.last_surface = result; self.workspace.set_surface(result.polydata, reset_camera=self._reset_camera)
        self._reset_camera = False; self.workspace.apply_appearance(self.panel.appearance_state())
        self.workspace.set_ready(f"Rendered {result.metrics.cells:,} triangles in {result.metrics.elapsed_ms/1000:.2f} s.")
        self._update_info()

    @Slot(str)
    def save_screenshot(self, path: str) -> None:
        try: output = self.workspace.save_screenshot(path)
        except Exception as exc: self._error("Screenshot failed", str(exc))
        else: self.workspace.set_ready(f"Saved screenshot: {output}")
    @Slot(str)
    def export_surface(self, raw: str) -> None:
        if self.workspace.surface is None: self._error("Export mesh", "No surface is available."); return
        path = Path(raw).expanduser()
        if path.suffix.lower() not in {".stl", ".ply", ".vtp"}: path = path.with_suffix(".stl")
        try: path.parent.mkdir(parents=True, exist_ok=True); output = self.exporter.export(self.workspace.surface, path)
        except Exception as exc: self._error("Mesh export failed", str(exc))
        else: self.workspace.set_ready(f"Exported mesh: {output}")

    @Slot()
    def cancel(self) -> None: self.timer.stop(); self.library.cancel(); self.surface.cancel()
    def cleanup(self) -> None:
        if self._cleaned: return
        self._cleaned = True; self.cancel(); self.workspace.cleanup()
    def _running(self) -> None: self.running_changed.emit(self.is_running)
    def _error(self, title: str, message: str) -> None:
        self.workspace.set_ready(message); self.error_requested.emit(title, message)
    def _update_info(self) -> None:
        if self.data is None: self.workspace.set_info(""); return
        lines = [f"File: {self.data.path}", f"Candidates: {len(self.data.candidates)}"]
        if self.data.detected_spacing_xyz: lines.append(f"Spacing XYZ: {self.data.detected_spacing_xyz} mm")
        if self.candidate:
            lines += ["", f"Array: {self.candidate.key}", f"Shape: {self.candidate.shape}",
                      f"Type: {self.candidate.dtype_name}",
                      f"Range: {self.candidate.min_value:.6g} to {self.candidate.max_value:.6g}"]
        if self.last_surface:
            m = self.last_surface.metrics
            lines += ["", f"Foreground voxels: {m.foreground_count:,}", f"Mask volume: {m.mask_volume_mm3:,.3f} mm³",
                      f"Mesh volume: {m.mesh_volume_mm3:,.3f} mm³", f"Surface area: {m.surface_area_mm2:,.3f} mm²",
                      f"Mesh: {m.points:,} points / {m.cells:,} triangles"]
        if self.data.warnings: lines += ["", "Warnings:", *[f"- {x}" for x in self.data.warnings]]
        self.workspace.set_info("\n".join(lines))
