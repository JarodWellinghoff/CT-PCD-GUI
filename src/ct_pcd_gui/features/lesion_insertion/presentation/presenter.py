from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
from PySide6.QtCore import QEvent, QObject, QSettings, QTimer, Signal, Slot
from PySide6.QtGui import QKeySequence, QShortcut, QUndoCommand, QUndoStack
from PySide6.QtWidgets import QApplication, QFileDialog, QMainWindow, QMessageBox

from ct_pcd_gui.shared.qt.task_runner import (
    QtTaskRunner,
    TaskCancelled,
    TaskContext,
    TaskHandle,
)

from ..application.session import LesionSessionState
from ..application.validation import validate_session
from ..domain.models import (
    DicomSeriesCandidate,
    DicomVolume,
    LesionInstance,
    LesionLibraryItem,
    LesionParameters,
    LesionSession,
    PreviewMode,
    PreviewResult,
    ProcessingResult,
    RawDatasetInfo,
    UnsupportedSpatialGeometry,
    ValidationIssue,
    Vec3,
)
from ..infrastructure.dicom_service import PydicomLesionDicomService, local_background_hu
from ..infrastructure.legacy_adapter import InsertionCancelled, LegacyLesionInsertionAdapter
from ..infrastructure.lesion_library import LocalLesionModelLibrary, lesion_thumbnail
from ..infrastructure.preview import ApproximatePreviewGenerator
from .panel import LesionInsertionPanel
from .workspace import LesionInsertionWorkspace


class _SessionCommand(QUndoCommand):
    def __init__(
        self,
        text: str,
        state: LesionSessionState,
        before: tuple[LesionSession, str],
        after: tuple[LesionSession, str],
        callback: Callable[[], None],
    ) -> None:
        super().__init__(text)
        self._state = state
        self._before = before
        self._after = after
        self._callback = callback

    def undo(self) -> None:
        self._state.restore(self._before)
        self._callback()

    def redo(self) -> None:
        self._state.restore(self._after)
        self._callback()


class LesionInsertionPresenter(QObject):
    running_changed = Signal(bool)
    error_requested = Signal(str, str)

    def __init__(
        self,
        *,
        panel: LesionInsertionPanel,
        workspace: LesionInsertionWorkspace,
        dicom_service: PydicomLesionDicomService,
        lesion_library: LocalLesionModelLibrary,
        preview_generator: ApproximatePreviewGenerator,
        insertion_runner: LegacyLesionInsertionAdapter,
        task_runner: QtTaskRunner,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.panel = panel
        self.workspace = workspace
        self.dicom_service = dicom_service
        self.lesion_library = lesion_library
        self.preview_generator = preview_generator
        self.insertion_runner = insertion_runner
        self.task_runner = task_runner
        self.settings = QSettings()
        self.state = LesionSessionState()
        self.volume: DicomVolume | None = None
        self.raw_info: RawDatasetInfo | None = None
        self.association_issues: list[ValidationIssue] = []
        self.library_items: list[LesionLibraryItem] = []
        self._tasks: dict[str, TaskHandle] = {}
        self._preview_generation = 0
        self._syncing = False
        self._cleaned = False
        self._last_session_path = ""
        self._allow_close_event = False
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

        self.undo_stack = QUndoStack(self)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(220)
        self.preview_timer.timeout.connect(self._start_preview)

        self._connect_ui()
        self._install_shortcuts()
        self._restore_settings()
        self._after_state_change()

    @property
    def is_running(self) -> bool:
        return any(handle.is_running for handle in self._tasks.values())

    def _connect_ui(self) -> None:
        self.panel.reconstruction_requested.connect(self.discover_reconstruction)
        self.panel.series_selected.connect(self.load_reconstruction)
        self.panel.raw_requested.connect(self.inspect_raw)
        self.panel.library_requested.connect(self.scan_library)
        self.panel.library_selected.connect(self._library_selected)
        self.panel.lesion_selected.connect(self._select_lesion)
        self.panel.add_requested.connect(self.add_lesion)
        self.panel.move_to_cursor_requested.connect(self.move_selected_to_cursor)
        self.panel.duplicate_requested.connect(self.duplicate_selected)
        self.panel.remove_requested.connect(self.remove_selected)
        self.panel.move_up_requested.connect(lambda: self.move_selected(-1))
        self.panel.move_down_requested.connect(lambda: self.move_selected(1))
        self.panel.show_all_requested.connect(self.show_all)
        self.panel.lesion_flags_changed.connect(self._change_flags)
        self.panel.parameters_changed.connect(self._change_parameters)
        self.panel.position_changed.connect(self._change_position)
        self.panel.background_changed.connect(self._change_background)
        self.panel.reset_parameters_requested.connect(self.reset_parameters)
        self.panel.save_session_requested.connect(self.save_session)
        self.panel.load_session_requested.connect(self.load_session)
        self.panel.validate_requested.connect(self.show_validation)
        self.panel.run_requested.connect(self.run_final)
        self.panel.cancel_requested.connect(self.cancel)
        self.panel.session_fields_changed.connect(self._session_fields_changed)
        self.workspace.slice_changed.connect(lambda _slice: self.schedule_preview())
        self.workspace.location_selected.connect(self._cursor_selected)
        self.workspace.preview_toggled.connect(self._preview_toggled)
        self.workspace.preview_mode_changed.connect(self._preview_mode_changed)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence.StandardKey.Undo, self.workspace).activated.connect(
            self.undo_stack.undo
        )
        QShortcut(QKeySequence.StandardKey.Redo, self.workspace).activated.connect(
            self.undo_stack.redo
        )
        QShortcut(QKeySequence("Insert"), self.workspace).activated.connect(self.add_lesion)
        QShortcut(QKeySequence("Ctrl+D"), self.workspace).activated.connect(
            self.duplicate_selected
        )
        QShortcut(QKeySequence("Delete"), self.workspace).activated.connect(
            self.remove_selected
        )

    def _restore_settings(self) -> None:
        values = {
            "reconstruction_source": self.settings.value(
                "lesion_insertion/last_reconstruction", "", type=str
            ),
            "raw_source": self.settings.value("lesion_insertion/last_raw", "", type=str),
            "lesion_library_source": self.settings.value(
                "lesion_insertion/last_library",
                self.settings.value("lesion_viewer/last_source", "", type=str),
                type=str,
            ),
            "output_directory": self.settings.value(
                "lesion_insertion/last_output", "", type=str
            ),
        }
        self.state.update_session(**values)
        self.state.mark_saved()
        self._sync_panel_fields()

    def activate(self) -> None:
        self._after_state_change()

    def deactivate(self) -> None:
        self.preview_timer.stop()
        self._cancel_task("preview")

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        self.cancel()
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            event.type() == QEvent.Type.Close
            and isinstance(watched, QMainWindow)
            and not self._allow_close_event
            and self.state.is_dirty
        ):
            if not self.can_close():
                return True
            self._allow_close_event = True
            QTimer.singleShot(0, self._reset_close_guard)
        return super().eventFilter(watched, event)

    def _reset_close_guard(self) -> None:
        self._allow_close_event = False

    def can_close(self) -> bool:
        if not self.state.is_dirty:
            return True
        answer = QMessageBox.question(
            self.panel,
            "Unsaved lesion session",
            "Save the current lesion-placement session before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if answer == QMessageBox.StandardButton.Cancel:
            return False
        if answer == QMessageBox.StandardButton.Save:
            return self.save_session()
        return True

    def _remember(self, key: str, value: str) -> None:
        if value:
            self.settings.setValue(f"lesion_insertion/{key}", value)

    @staticmethod
    def _execute_cancellable(
        context: TaskContext, operation: Callable[[], object]
    ) -> object:
        try:
            result = operation()
        except Exception:
            context.raise_if_cancelled()
            raise
        context.raise_if_cancelled()
        return result

    def _run_task(
        self,
        key: str,
        operation: Callable[[TaskContext], object],
        success: Callable[[object], None],
        *,
        status: str,
        error_title: str,
    ) -> None:
        self._cancel_task(key)
        self.workspace.show_busy(status)
        handle = self.task_runner.start(operation)
        self._tasks[key] = handle
        handle.context.signals.succeeded.connect(success)
        handle.context.signals.cancelled.connect(
            lambda message: self.workspace.show_ready(message or "Operation cancelled.")
        )
        handle.context.signals.failed.connect(
            lambda details: self._task_failed(error_title, details)
        )
        handle.context.signals.finished.connect(lambda: self._task_finished(key, handle))
        self._update_running()

    def _task_failed(self, title: str, details: str) -> None:
        message = details.strip().splitlines()[-1] if details.strip() else "Unknown error"
        self.workspace.append_log(details.strip())
        self.workspace.show_ready(message)
        self.error_requested.emit(title, message)

    def _task_finished(self, key: str, handle: TaskHandle) -> None:
        if self._tasks.get(key) is handle:
            self._tasks.pop(key, None)
        self._update_running()

    def _update_running(self) -> None:
        running = self.is_running
        processing = bool((handle := self._tasks.get("process")) and handle.is_running)
        self.panel.set_running(processing)
        self.running_changed.emit(running)

    def _cancel_task(self, key: str) -> None:
        handle = self._tasks.get(key)
        if handle and handle.is_running:
            handle.cancel()

    @Slot(str)
    def discover_reconstruction(self, path: str) -> None:
        if not path.strip():
            self._error("Reconstruction", "Select a reconstructed DICOM file or directory.")
            return
        self._remember("last_reconstruction", path)
        desired_uid = self.state.session.reconstruction_series_uid

        def operation(context: TaskContext):
            return self._execute_cancellable(
                context,
                lambda: self.dicom_service.discover_reconstruction_series(
                    path, context.cancel_event
                ),
            )

        def success(result: object) -> None:
            if not isinstance(result, list):
                return
            values = [item for item in result if isinstance(item, DicomSeriesCandidate)]
            self.panel.set_series(values, desired_uid)
            self.workspace.show_ready(
                f"Discovered {len(values)} reconstructed DICOM series."
            )

        self._run_task(
            "discover-reconstruction",
            operation,
            success,
            status="Discovering reconstructed DICOM series…",
            error_title="Reconstruction discovery failed",
        )

    @Slot(str)
    def load_reconstruction(self, series_uid: str) -> None:
        source = self.panel.reconstruction_path()
        if not source or not series_uid:
            return
        self._syncing = True
        try:
            self.state.update_session(
                reconstruction_source=source,
                reconstruction_series_uid=series_uid,
            )
        finally:
            self._syncing = False

        def operation(context: TaskContext):
            return self._execute_cancellable(
                context,
                lambda: self.dicom_service.load_reconstruction(
                    source, series_uid, context.cancel_event
                ),
            )

        def success(result: object) -> None:
            if not isinstance(result, DicomVolume):
                return
            self.volume = result
            self.preview_generator.clear()
            self.panel.set_position_ranges(
                result.geometry.columns,
                result.geometry.rows,
                result.geometry.slice_count,
            )
            self.workspace.set_volume(result)
            if self.raw_info is not None:
                self.association_issues = self.dicom_service.validate_association(
                    result, self.raw_info
                )
            self._set_data_status()
            self.workspace.show_ready(
                f"Loaded {result.geometry.slice_count} physically ordered reconstruction slices."
            )
            self.schedule_preview()

        self._run_task(
            "load-reconstruction",
            operation,
            success,
            status="Loading and validating reconstructed DICOM pixels…",
            error_title="Reconstruction loading failed",
        )

    @Slot(str)
    def inspect_raw(self, path: str) -> None:
        if not path.strip():
            self._error("DICOM-CT-PD", "Select a raw DICOM-CT-PD file or directory.")
            return
        self._remember("last_raw", path)
        self.state.update_session(raw_source=path)

        def operation(context: TaskContext):
            return self._execute_cancellable(
                context, lambda: self.dicom_service.inspect_raw(path)
            )

        def success(result: object) -> None:
            if not isinstance(result, RawDatasetInfo):
                return
            self.raw_info = result
            if self.volume is not None:
                self.association_issues = self.dicom_service.validate_association(
                    self.volume, result
                )
            current_mapping = self.state.session.spectrum_map
            if current_mapping and set(result.spectrum_indices).issubset(current_mapping):
                mapping = current_mapping
            else:
                mapping = {
                    spectrum: channel
                    for channel, spectrum in enumerate(result.spectrum_indices)
                }
            self.state.update_session(spectrum_channel_map=tuple(sorted(mapping.items())))
            self._sync_panel_fields()
            self._set_data_status()
            self.workspace.show_ready(
                f"Validated {result.projection_file_count} CTPD file(s), "
                f"{result.projection_frame_count} frame(s)."
            )

        self._run_task(
            "inspect-raw",
            operation,
            success,
            status="Inspecting DICOM-CT-PD geometry and spectra…",
            error_title="DICOM-CT-PD inspection failed",
        )

    @Slot(str)
    def scan_library(self, path: str) -> None:
        if not path.strip():
            self._error("Lesion library", "Select a MAT/NPZ lesion library.")
            return
        self._remember("last_library", path)
        self.settings.setValue("lesion_viewer/last_source", path)
        self.state.update_session(lesion_library_source=path)

        def operation(context: TaskContext):
            return self._execute_cancellable(
                context, lambda: list(self.lesion_library.scan(path, context.cancel_event))
            )

        def success(result: object) -> None:
            if not isinstance(result, list):
                return
            self.library_items = [
                item for item in result if isinstance(item, LesionLibraryItem)
            ]
            self.panel.set_library_items(self.library_items)
            compatible = sum(item.compatible for item in self.library_items)
            malformed = len(self.library_items) - compatible
            self.workspace.show_ready(
                f"Found {compatible} compatible lesion model(s)"
                + (f"; {malformed} incompatible record(s)." if malformed else ".")
            )

        self._run_task(
            "scan-library",
            operation,
            success,
            status="Scanning and validating lesion models…",
            error_title="Lesion library scan failed",
        )

    @Slot(object)
    def _library_selected(self, item: object) -> None:
        self._cancel_task("thumbnail")
        if not isinstance(item, LesionLibraryItem):
            self.panel.set_library_thumbnail(None)
            return
        if not item.compatible:
            self.panel.set_library_thumbnail(None)
            self.workspace.set_preview_state(
                f"Selected library record is incompatible: {item.warning}"
            )
            return

        def operation(context: TaskContext):
            context.raise_if_cancelled()
            model = self.lesion_library.load_model(item.path)
            context.raise_if_cancelled()
            return lesion_thumbnail(model)

        def success(result: object) -> None:
            if isinstance(result, np.ndarray):
                self.panel.set_library_thumbnail(result)
                self.workspace.show_ready(f"Selected lesion model: {item.display_name}")

        self._run_task(
            "thumbnail",
            operation,
            success,
            status=f"Loading {item.display_name} thumbnail…",
            error_title="Lesion thumbnail failed",
        )

    def _set_data_status(self) -> None:
        parts: list[str] = []
        if self.volume is not None:
            geometry = self.volume.geometry
            parts.append(
                f"Recon: {geometry.slice_count} slices, "
                f"{geometry.row_spacing_mm:g}×{geometry.column_spacing_mm:g} mm pixels"
            )
        if self.raw_info is not None:
            parts.append(
                f"CTPD: {self.raw_info.projection_file_count} files / "
                f"{self.raw_info.projection_frame_count} frames; spectra "
                f"{list(self.raw_info.spectrum_indices)}"
            )
        errors = [issue for issue in self.association_issues if issue.severity == "error"]
        if errors:
            parts.append(f"Association errors: {len(errors)}")
        self.panel.data_status.setText("\n".join(parts) or "No case loaded.")

    def _mutate(
        self,
        text: str,
        mutation: Callable[[LesionSessionState], None],
    ) -> None:
        before = self.state.snapshot()
        temporary = LesionSessionState(before[0])
        temporary.restore(before)
        mutation(temporary)
        after = temporary.snapshot()
        if before == after:
            return
        self.undo_stack.push(
            _SessionCommand(text, self.state, before, after, self._after_state_change)
        )

    @Slot()
    def add_lesion(self) -> None:
        item = self.panel.selected_library_item()
        cursor = self.workspace.cursor_position
        if item is None or not item.compatible:
            self._error("Add lesion", "Select a compatible lesion model from the library.")
            return
        if self.volume is None or cursor is None:
            self._error("Add lesion", "Load a reconstruction and choose a 3-D image location.")
            return
        try:
            patient = self.volume.geometry.voxel_to_patient(cursor)
            try:
                ctpd = self.volume.geometry.voxel_to_legacy_ctpd(cursor)
            except UnsupportedSpatialGeometry as exc:
                ctpd = np.asarray([np.nan, np.nan, np.nan])
                self.workspace.set_preview_state(str(exc))
            background = local_background_hu(self.volume, cursor)
        except Exception as exc:
            self._error("Add lesion", str(exc))
            return
        label = f"{item.display_name} {len(self.state.session.lesions) + 1}"
        lesion = LesionInstance.create(
            lesion_path=item.path,
            lesion_id=item.lesion_id,
            label=label,
            center_voxel_crs=tuple(float(value) for value in cursor),
            center_patient_lps_mm=tuple(float(value) for value in patient),
            center_ctpd_mm=tuple(float(value) for value in ctpd),
            background_hu=tuple(background for _ in range(max(1, item.channel_count))),
        )
        self._mutate("Add lesion", lambda state: state.add(lesion))

    @Slot()
    def move_selected_to_cursor(self) -> None:
        cursor = self.workspace.cursor_position
        if cursor is not None:
            self._change_position(cursor)

    @Slot()
    def duplicate_selected(self) -> None:
        selected = self.state.selected
        if selected is not None:
            self._mutate(
                "Duplicate lesion", lambda state: state.duplicate(selected.instance_id)
            )

    @Slot()
    def remove_selected(self) -> None:
        selected = self.state.selected
        if selected is None:
            return
        answer = QMessageBox.question(
            self.panel,
            "Remove lesion",
            f"Remove {selected.label!r} from this session?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._mutate(
                "Remove lesion", lambda state: state.remove(selected.instance_id)
            )

    def move_selected(self, offset: int) -> None:
        selected = self.state.selected
        if selected is not None:
            self._mutate(
                "Reorder lesion", lambda state: state.move(selected.instance_id, offset)
            )

    @Slot(bool)
    def show_all(self, visible: bool) -> None:
        self._mutate(
            "Show lesions" if visible else "Hide lesions",
            lambda state: state.set_all_visibility(visible),
        )

    @Slot(str)
    def _select_lesion(self, instance_id: str) -> None:
        try:
            self.state.select(instance_id)
        except KeyError:
            return
        selected = self.state.selected
        if selected is not None and self.volume is not None:
            self.workspace.slice_slider.setValue(int(round(selected.center_voxel_crs[2])))
        self._after_state_change(schedule=False)

    @Slot(str, bool, bool)
    def _change_flags(self, instance_id: str, enabled: bool, visible: bool) -> None:
        lesion = next(
            (item for item in self.state.session.lesions if item.instance_id == instance_id),
            None,
        )
        if lesion is None:
            return
        updated = replace(lesion, enabled=enabled, visible=visible)
        self._mutate("Change lesion visibility", lambda state: state.update(updated))

    @Slot(object)
    def _change_parameters(self, value: object) -> None:
        selected = self.state.selected
        if selected is None or not isinstance(value, LesionParameters):
            return
        try:
            value.validate()
        except Exception as exc:
            self._error("Lesion parameters", str(exc))
            return
        updated = replace(selected, parameters=value)
        self._mutate("Change lesion parameters", lambda state: state.update(updated))

    @Slot(object)
    def _change_position(self, value: object) -> None:
        selected = self.state.selected
        if selected is None or self.volume is None:
            return
        try:
            values = np.asarray(value, dtype=float).reshape(-1)
            if values.size != 3:
                raise ValueError("Position must contain column, row, and slice.")
            center: Vec3 = (float(values[0]), float(values[1]), float(values[2]))
            patient = self.volume.geometry.voxel_to_patient(center)
            try:
                ctpd = self.volume.geometry.voxel_to_legacy_ctpd(center)
            except UnsupportedSpatialGeometry:
                ctpd = np.asarray([np.nan, np.nan, np.nan])
            background = local_background_hu(self.volume, center)
            updated = replace(
                selected,
                center_voxel_crs=center,
                center_patient_lps_mm=tuple(float(item) for item in patient),
                center_ctpd_mm=tuple(float(item) for item in ctpd),
                background_hu=tuple(background for _ in selected.background_hu),
            )
        except Exception as exc:
            self._error("Lesion position", str(exc))
            self.panel.set_selected_lesion(selected)
            return
        self._mutate("Move lesion", lambda state: state.update(updated))

    @Slot(str)
    def _change_background(self, raw: str) -> None:
        selected = self.state.selected
        if selected is None:
            return
        try:
            values = tuple(float(item) for item in raw.replace(",", " ").split())
            if len(values) == 1 and len(selected.background_hu) > 1:
                values = values * len(selected.background_hu)
            if len(values) != len(selected.background_hu):
                raise ValueError(
                    f"Enter {len(selected.background_hu)} background HU value(s), "
                    "one per lesion channel."
                )
            if not all(np.isfinite(value) for value in values):
                raise ValueError("Background HU values must be finite.")
        except ValueError as exc:
            self._error("Background HU", str(exc))
            self.panel.set_selected_lesion(selected)
            return
        updated = replace(selected, background_hu=values)
        self._mutate("Change lesion background HU", lambda state: state.update(updated))

    @Slot()
    def reset_parameters(self) -> None:
        selected = self.state.selected
        if selected is not None:
            updated = replace(selected, parameters=LesionParameters())
            self._mutate("Reset lesion parameters", lambda state: state.update(updated))

    @Slot(float, float, int)
    def _cursor_selected(self, column: float, row: float, slice_index: int) -> None:
        self.workspace.set_scene_state(
            lesions=self.state.session.lesions,
            selected_id=self.state.selected_id,
            cursor=(column, row, float(slice_index)),
        )

    @Slot(bool)
    def _preview_toggled(self, enabled: bool) -> None:
        self.state.update_session(preview_enabled=enabled)
        if enabled:
            self.schedule_preview()
        else:
            self._cancel_task("preview")
            self.workspace.set_preview(None)
            self.workspace.show_ready("Approximate preview disabled.")

    @Slot(str)
    def _preview_mode_changed(self, mode: str) -> None:
        if mode not in {"before", "overlay", "after"}:
            return
        self.state.update_session(preview_mode=cast(PreviewMode, mode))
        self.workspace.render()

    def _after_state_change(self, *, schedule: bool = True) -> None:
        self.panel.set_lesions(self.state.session.lesions, self.state.selected_id)
        self.panel.set_selected_lesion(self.state.selected)
        self.workspace.set_scene_state(
            lesions=self.state.session.lesions,
            selected_id=self.state.selected_id,
        )
        if schedule:
            self.schedule_preview()

    def schedule_preview(self) -> None:
        self._preview_generation += 1
        if self.volume is None or not self.state.session.preview_enabled or self._cleaned:
            return
        self.workspace.set_preview_state("Approximate preview is stale; recalculating…")
        self.preview_timer.start()

    @Slot()
    def _start_preview(self) -> None:
        if self.volume is None or not self.state.session.preview_enabled:
            return
        generation = self._preview_generation
        volume = self.volume
        session = self.state.session
        slice_index = self.workspace.current_slice

        def operation(context: TaskContext):
            return self._execute_cancellable(
                context,
                lambda: self.preview_generator.generate(
                    volume,
                    session,
                    slice_index,
                    generation,
                    context.cancel_event,
                ),
            )

        def success(result: object) -> None:
            if not isinstance(result, PreviewResult):
                return
            if result.generation != self._preview_generation:
                return
            self.workspace.set_preview(result)
            self.workspace.show_ready(
                "Approximate image-domain preview ready. Final appearance requires "
                "raw insertion and reconstruction."
            )

        self._run_task(
            "preview",
            operation,
            success,
            status="Calculating approximate image-domain preview…",
            error_title="Preview failed",
        )

    @Slot()
    def _session_fields_changed(self) -> None:
        if self._syncing:
            return
        try:
            mapping = self.panel.spectrum_mapping()
        except ValueError:
            mapping = self.state.session.spectrum_map
        self.state.update_session(
            reconstruction_source=self.panel.reconstruction_path(),
            reconstruction_series_uid=self.panel.selected_series_uid(),
            raw_source=self.panel.raw_path(),
            lesion_library_source=self.panel.library_path(),
            output_directory=self.panel.output_path(),
            spectrum_channel_map=tuple(sorted(mapping.items())),
            workers=self.panel.workers_spin.value(),
            reconstruction_command=self.panel.reconstruction_command.text().strip(),
            reconstruction_output_directory=self.panel.reconstruction_output_path(),
        )
        self._remember("last_output", self.panel.output_path())

    def _sync_panel_fields(self) -> None:
        session = self.state.session
        self._syncing = True
        try:
            self.panel.reconstruction_row.set_path(session.reconstruction_source)
            self.panel.raw_row.set_path(session.raw_source)
            self.panel.library_row.set_path(session.lesion_library_source)
            self.panel.output_row.set_path(session.output_directory)
            self.panel.spectrum_map.setText(
                json.dumps({str(key): value for key, value in session.spectrum_channel_map})
            )
            self.panel.workers_spin.setValue(session.workers)
            self.panel.reconstruction_command.setText(session.reconstruction_command)
            self.panel.reconstruction_output_row.set_path(
                session.reconstruction_output_directory
            )
            self.workspace.set_preview_enabled(session.preview_enabled)
            self.workspace.set_preview_mode(session.preview_mode)
        finally:
            self._syncing = False

    @Slot()
    def show_validation(self) -> bool:
        self._session_fields_changed()
        issues = validate_session(
            self.state.session,
            self.volume,
            self.raw_info,
            self.association_issues,
        )
        text = self._format_issues(issues)
        has_errors = any(issue.severity == "error" for issue in issues)
        self.panel.show_validation(
            "Validation failed" if has_errors else "Validation summary",
            text,
            error=has_errors,
        )
        return not has_errors

    @staticmethod
    def _format_issues(issues: list[ValidationIssue]) -> str:
        if not issues:
            return "No validation issues were found."
        labels = {"error": "ERROR", "warning": "WARNING", "info": "INFO"}
        return "\n\n".join(
            f"{labels[issue.severity]} — {issue.message}" for issue in issues
        )

    @Slot()
    def run_final(self) -> None:
        self._session_fields_changed()
        if self.volume is None or self.raw_info is None:
            self.show_validation()
            return
        issues = validate_session(
            self.state.session,
            self.volume,
            self.raw_info,
            self.association_issues,
        )
        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity == "warning"]
        if errors:
            self.panel.show_validation(
                "Validation failed", self._format_issues(issues), error=True
            )
            return
        if warnings:
            answer = QMessageBox.warning(
                self.panel,
                "Review validation warnings",
                self._format_issues(warnings) + "\n\nContinue with final raw-data insertion?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        session = self.state.session
        volume = self.volume
        raw = self.raw_info
        self.workspace.append_log("Starting final lesion-insertion workflow.")

        def operation(context: TaskContext):
            try:
                return self.insertion_runner.run(
                    session,
                    volume,
                    raw,
                    cancel_event=context.cancel_event,
                    progress=lambda completed, total, status: context.emit(
                        ("progress", completed, total, status)
                    ),
                    log=lambda message: context.emit(("log", message)),
                )
            except InsertionCancelled as exc:
                raise TaskCancelled(str(exc)) from exc

        def success(result: object) -> None:
            if not isinstance(result, ProcessingResult):
                return
            self.state.mark_saved()
            self.workspace.append_log(f"Completed. Manifest: {result.manifest_path}")
            self.workspace.show_ready(
                f"Final raw-data insertion completed: {result.modified_ctpd_directory}"
            )
            QMessageBox.information(
                self.panel,
                "Lesion insertion complete",
                "Modified DICOM-CT-PD data and the reproducibility manifest were written to:\n"
                f"{result.output_directory}",
            )

        self._run_task(
            "process",
            operation,
            success,
            status="Running final raw DICOM-CT-PD lesion insertion…",
            error_title="Final lesion insertion failed",
        )
        handle = self._tasks.get("process")
        if handle is not None:
            handle.context.signals.event.connect(self._processing_event)

    @Slot(object)
    def _processing_event(self, event: object) -> None:
        if not isinstance(event, tuple) or not event:
            return
        if event[0] == "log" and len(event) >= 2:
            self.workspace.append_log(str(event[1]))
        elif event[0] == "progress" and len(event) >= 4:
            self.workspace.show_progress(int(event[1]), int(event[2]), str(event[3]))

    @Slot()
    def cancel(self) -> None:
        self.preview_timer.stop()
        for handle in list(self._tasks.values()):
            if handle.is_running:
                handle.cancel()
        if "process" in self._tasks:
            self.workspace.set_preview_state(
                "Cancellation requested. The validated pipeline stops between projection "
                "files; one large multi-frame file may need to finish first."
            )

    @Slot()
    def save_session(self) -> bool:
        self._session_fields_changed()
        initial = self._last_session_path or str(
            Path(self.state.session.output_directory or Path.home())
            / "lesion_insertion_session.json"
        )
        path, _ = QFileDialog.getSaveFileName(
            self.panel,
            "Save lesion session",
            initial,
            "Lesion session (*.json)",
        )
        if not path:
            return False
        try:
            output = self.state.save(path)
        except Exception as exc:
            self._error("Save session", str(exc))
            return False
        self._last_session_path = str(output)
        self.workspace.show_ready(f"Saved lesion session: {output}")
        return True

    @Slot()
    def load_session(self) -> None:
        if self.state.is_dirty:
            answer = QMessageBox.question(
                self.panel,
                "Discard unsaved changes?",
                "Loading another session will replace the current unsaved setup.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        path, _ = QFileDialog.getOpenFileName(
            self.panel,
            "Load lesion session",
            self._last_session_path or str(Path.home()),
            "Lesion session (*.json)",
        )
        if not path:
            return
        try:
            session = self.state.load(path)
        except Exception as exc:
            self._error("Load session", str(exc))
            return
        self._last_session_path = path
        self.undo_stack.clear()
        self._sync_panel_fields()
        self._after_state_change()
        if session.reconstruction_source and Path(session.reconstruction_source).exists():
            self.discover_reconstruction(session.reconstruction_source)
        if session.raw_source and Path(session.raw_source).exists():
            self.inspect_raw(session.raw_source)
        if session.lesion_library_source and Path(session.lesion_library_source).exists():
            self.scan_library(session.lesion_library_source)
        self.workspace.show_ready(f"Loaded lesion session: {path}")

    def _error(self, title: str, message: str) -> None:
        self.workspace.show_ready(message)
        self.error_requested.emit(title, message)
