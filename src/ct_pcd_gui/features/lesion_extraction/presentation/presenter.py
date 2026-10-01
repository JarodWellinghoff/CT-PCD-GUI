from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from ct_pcd_gui.shared.qt.task_runner import (
    QtTaskRunner,
    TaskCancelled,
    TaskContext,
    TaskHandle,
)

from ..batch import LesionExtractionService
from ..models import (
    ExportSummary,
    ExtractionProgress,
    InspectionResult,
    InputValidationError,
    PreparedExtraction,
    SeriesPreview,
)
from .panel import LesionExtractionPanel
from .workspace import LesionExtractionWorkspace


class LesionExtractionPresenter(QObject):
    error_requested = Signal(str, str)

    def __init__(
        self,
        *,
        panel: LesionExtractionPanel,
        workspace: LesionExtractionWorkspace,
        service: LesionExtractionService,
        task_runner: QtTaskRunner,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.panel = panel
        self.workspace = workspace
        self.service = service
        self.task_runner = task_runner
        self._inspection: InspectionResult | None = None
        self._prepared: PreparedExtraction | None = None
        self._preview_cache: dict[tuple[str, int], SeriesPreview] = {}
        self._active_task: TaskHandle | None = None
        self._task_name = ""
        self._connect()

    @property
    def is_running(self) -> bool:
        return self._active_task is not None and self._active_task.is_running

    def _connect(self) -> None:
        self.panel.add_dicom_requested.connect(self._browse_dicom)
        self.panel.browse_segmentation_requested.connect(self._browse_segmentation)
        self.panel.browse_output_requested.connect(self._browse_output)
        self.panel.load_requested.connect(self.load_inputs)
        self.panel.split_requested.connect(self.prepare_candidates)
        self.panel.export_requested.connect(self.export_candidates)
        self.panel.cancel_requested.connect(self.cancel)
        self.panel.inputs_changed.connect(self._invalidate_inputs)
        self.panel.segment_selection_changed.connect(self._segment_changed)
        self.panel.candidate_state_changed.connect(self._candidate_state_changed)
        self.panel.candidate_selected.connect(self.workspace.select_entity)
        self.workspace.entity_selected.connect(self.panel.select_candidate)
        self.workspace.series_changed.connect(self._series_changed)

    def activate(self) -> None:
        if self._inspection is None:
            self.panel.set_status("Add DICOM folders and a NRRD segmentation.")

    def deactivate(self) -> None:
        pass

    def cleanup(self) -> None:
        self.cancel()

    def cancel(self) -> None:
        if self._active_task is not None and self._active_task.is_running:
            self._active_task.cancel()
            self.panel.set_status("Cancellation requested...")

    def _browse_dicom(self) -> None:
        folders = self.panel.dicom_folders()
        folder = QFileDialog.getExistingDirectory(
            self.panel,
            "Add DICOM series folder",
            folders[0] if folders else "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            self.panel.add_dicom_folder(folder)

    def _browse_segmentation(self) -> None:
        current = self.panel.segmentation_path.text().strip()
        start = str(Path(current).parent) if current else ""
        path, _ = QFileDialog.getOpenFileName(
            self.panel,
            "Select NRRD segmentation",
            start,
            "NRRD segmentation (*.nrrd *.seg.nrrd);;All files (*)",
        )
        if path:
            self.panel.segmentation_path.setText(path)

    def _browse_output(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.panel,
            "Choose output folder",
            self.panel.output_path.text().strip(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if folder:
            self.panel.output_path.setText(folder)

    def _invalidate_inputs(self) -> None:
        if self.is_running or (self._inspection is None and self._prepared is None):
            return
        self._inspection = None
        self._prepared = None
        self._preview_cache.clear()
        self.panel.clear_candidates()
        self.panel.set_stage("empty")
        self.panel.series_summary.setText("Inputs changed; load again")
        self.panel.set_status("Inputs changed. Load and preview again.")
        self.workspace.set_series_names(())
        self.workspace.clear_preview("Inputs changed; load again")

    def _segment_changed(self) -> None:
        if self.is_running or self._inspection is None:
            return
        if self._prepared is not None:
            self._prepared = None
            self.panel.clear_candidates()
            self.panel.set_stage("inspected")
            self.panel.set_status("Label selection changed. Preview the split again.")
            index = self.workspace.current_series_index()
            preview = self._preview_cache.get(("segments", index))
            if preview is not None:
                self.workspace.set_preview(preview)
            else:
                self._series_changed(index)
        self._apply_segment_muting()

    def _apply_segment_muting(self) -> None:
        if self._inspection is None:
            return
        selected = set(self.panel.selected_segment_keys())
        all_ids = {segment.key for segment in self._inspection.segmentation.segments}
        self.workspace.set_muted_entities(all_ids - selected)

    def load_inputs(self) -> None:
        if self.is_running:
            return
        folders = self.panel.dicom_folders()
        segmentation = self.panel.segmentation_path.text().strip()
        if not folders:
            self.error_requested.emit(
                "Missing DICOM input", "Add at least one DICOM series folder."
            )
            return
        if not segmentation:
            self.error_requested.emit(
                "Missing segmentation", "Choose a NRRD segmentation file."
            )
            return
        self.panel.set_status("Loading DICOM series and segmentation...")
        self.panel.set_progress(0, 0)

        def operation(context: TaskContext) -> InspectionResult:
            context.raise_if_cancelled()
            result = self.service.inspect_inputs(folders, segmentation)
            context.raise_if_cancelled()
            return result

        self._start_task("load inputs", operation, self._inputs_loaded)

    def _inputs_loaded(self, result: InspectionResult) -> None:
        self._inspection = result
        self._prepared = None
        self._preview_cache = {("segments", 0): result.preview}
        self.panel.set_discovered_series(result.series)
        self.panel.set_segments(result.segmentation.segments)
        self.panel.clear_candidates()
        self.panel.set_stage("inspected")
        self.panel.reset_progress()
        self.workspace.set_series_names(
            [series.display_name for series in result.series], 0
        )
        self.workspace.set_preview(result.preview)
        self._apply_segment_muting()
        if not self.panel.output_path.text().strip():
            self.panel.output_path.setText(
                str(result.segmentation.path.parent / "lesion_npz")
            )
        for warning in result.segmentation.warnings:
            self.panel.append_log(f"Warning: {warning}")
        self.panel.set_status(
            f"Loaded {len(result.series)} series and "
            f"{len(result.segmentation.segments)} segment labels.",
            log=True,
        )

    def prepare_candidates(self) -> None:
        if self.is_running:
            return
        inspection = self._inspection
        if inspection is None:
            self.error_requested.emit(
                "Inputs not loaded", "Load the DICOM series and segmentation first."
            )
            return
        selected = self.panel.selected_segment_keys()
        if not selected:
            self.error_requested.emit(
                "No labels selected", "Select at least one segmentation label."
            )
            return
        index = self.workspace.current_series_index()
        minimum_voxels = self.panel.minimum_voxels.value()
        self.panel.set_status("Splitting selected labels into connected components...")
        self.panel.set_progress(0, 0)

        def operation(context: TaskContext) -> PreparedExtraction:
            context.raise_if_cancelled()
            result = self.service.prepare_candidates(
                inspection.series,
                inspection.segmentation,
                selected,
                minimum_voxels=minimum_voxels,
                preview_series_index=index,
            )
            context.raise_if_cancelled()
            return result

        self._start_task(
            "preview lesion split",
            operation,
            lambda result: self._candidates_loaded(result, index),
        )

    def _candidates_loaded(self, result: PreparedExtraction, index: int) -> None:
        self._prepared = result
        self._preview_cache[("candidates", index)] = result.preview
        self.panel.set_candidates(result.candidates)
        self.panel.set_stage("prepared")
        self.panel.reset_progress()
        self.workspace.set_preview(result.preview)
        self._candidate_state_changed()
        self.panel.set_status(
            f"Prepared {len(result.candidates)} lesion candidate(s).",
            log=True,
        )

    def _series_changed(self, index: int) -> None:
        if self.is_running or self._inspection is None or index < 0:
            return
        if index >= len(self._inspection.series):
            return
        mode = "candidates" if self._prepared is not None else "segments"
        cached = self._preview_cache.get((mode, index))
        if cached is not None:
            self.workspace.set_preview(cached)
            self._candidate_state_changed()
            return
        series = self._inspection.series[index]
        segmentation = self._inspection.segmentation
        candidates = self._prepared.candidates if self._prepared else None
        self.panel.set_status(f"Loading preview for {series.display_name}...")

        def operation(context: TaskContext) -> SeriesPreview:
            context.raise_if_cancelled()
            if candidates is None:
                result = self.service.preview_segments(series, segmentation)
            else:
                result = self.service.preview_candidates(
                    series, segmentation, candidates
                )
            context.raise_if_cancelled()
            return result

        self._start_task(
            "load series preview",
            operation,
            lambda preview: self._preview_loaded(mode, index, preview),
        )

    def _preview_loaded(self, mode: str, index: int, preview: SeriesPreview) -> None:
        self._preview_cache[(mode, index)] = preview
        self.workspace.set_preview(preview)
        self._candidate_state_changed()
        self.panel.set_status(f"Previewing {preview.series.display_name}.")

    def _candidate_state_changed(self) -> None:
        prepared = self._prepared
        if prepared is None:
            self._apply_segment_muting()
            return
        selections = self.panel.candidate_selections()
        included = {
            selection.candidate_id for selection in selections if selection.included
        }
        all_ids = {candidate.candidate_id for candidate in prepared.candidates}
        self.workspace.set_muted_entities(all_ids - included)
        self.workspace.update_entity_names(
            {selection.candidate_id: selection.output_name for selection in selections}
        )
        self.panel.set_export_enabled(bool(included))

    def export_candidates(self) -> None:
        if self.is_running:
            return
        prepared = self._prepared
        if prepared is None:
            self.error_requested.emit(
                "No lesion split", "Preview the lesion split before exporting."
            )
            return
        output = self.panel.output_path.text().strip()
        selections = self.panel.candidate_selections()
        if not output:
            self.error_requested.emit(
                "Missing output folder", "Choose an output folder."
            )
            return
        if not any(selection.included for selection in selections):
            self.error_requested.emit(
                "No lesions included", "Include at least one lesion before exporting."
            )
            return
        total = len(prepared.series) * sum(item.included for item in selections)
        self.panel.set_status("Exporting lesion NPZ files...")
        self.panel.set_progress(0, total)

        def operation(context: TaskContext) -> ExportSummary:
            def report(event: ExtractionProgress) -> None:
                context.raise_if_cancelled()
                context.emit(event)

            try:
                return self.service.export(
                    series=prepared.series,
                    segmentation=prepared.segmentation,
                    candidates=prepared.candidates,
                    selections=selections,
                    output_folder=output,
                    progress=report,
                    cancelled=lambda: context.is_cancelled,
                )
            except InputValidationError as exc:
                if context.is_cancelled:
                    raise TaskCancelled("Lesion extraction was cancelled.") from exc
                raise

        self._start_task("export lesions", operation, self._export_finished)

    def _export_finished(self, summary: ExportSummary) -> None:
        self.panel.set_progress(len(summary.exported), len(summary.exported))
        self.panel.set_status(
            f"Exported {len(summary.exported)} lesion NPZ file(s).", log=True
        )
        for exported in summary.exported:
            self.panel.append_log(f"{exported.series_name}: {exported.output_path}")
        for warning in summary.warnings:
            self.panel.append_log(f"Warning: {warning}")

    def _start_task(
        self,
        name: str,
        operation: Callable[[TaskContext], object],
        success: Callable[[Any], None],
    ) -> None:
        if self.is_running:
            return
        handle = self.task_runner.start(operation)
        self._active_task = handle
        self._task_name = name
        self.panel.set_busy(True)
        self.workspace.setEnabled(False)
        handle.context.signals.event.connect(
            lambda event, task=handle: self._task_event(task, event)
        )
        handle.context.signals.succeeded.connect(
            lambda result, task=handle: self._task_succeeded(task, success, result)
        )
        handle.context.signals.cancelled.connect(
            lambda message, task=handle: self._task_cancelled(task, message)
        )
        handle.context.signals.failed.connect(
            lambda details, task=handle: self._task_failed(task, details)
        )
        handle.context.signals.finished.connect(
            lambda task=handle: self._task_finished(task)
        )

    def _task_event(self, handle: TaskHandle, event: object) -> None:
        if handle is self._active_task and isinstance(event, ExtractionProgress):
            self.panel.set_progress(event.completed, event.total)
            self.panel.set_status(event.message)

    def _task_succeeded(
        self,
        handle: TaskHandle,
        success: Callable[[Any], None],
        result: object,
    ) -> None:
        if handle is not self._active_task:
            return
        try:
            success(result)
        except Exception:
            self._task_failed(handle, traceback.format_exc())

    def _task_cancelled(self, handle: TaskHandle, message: str) -> None:
        if handle is self._active_task:
            self.panel.set_status(message or "Operation cancelled.", log=True)
            self.panel.reset_progress()

    def _task_failed(self, handle: TaskHandle, details: str) -> None:
        if handle is not self._active_task:
            return
        self.panel.append_log(details)
        lines = [line.strip() for line in details.splitlines() if line.strip()]
        message = lines[-1].split(": ", 1)[-1] if lines else "Unknown error."
        self.panel.set_status(f"{self._task_name.capitalize()} failed: {message}")
        self.panel.reset_progress()
        self.error_requested.emit(f"Could not {self._task_name}", message)

    def _task_finished(self, handle: TaskHandle) -> None:
        if handle is not self._active_task:
            return
        self._active_task = None
        self._task_name = ""
        self.panel.set_busy(False)
        self.workspace.setEnabled(True)
        self._candidate_state_changed()
