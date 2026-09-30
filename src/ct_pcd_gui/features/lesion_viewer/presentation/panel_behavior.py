from __future__ import annotations

from pathlib import Path
from PySide6.QtCore import QSignalBlocker, Slot
from PySide6.QtWidgets import QColorDialog, QFileDialog
from ct_pcd_gui.features.lesion_viewer.domain.models import (
    LesionFileData, LesionFileRecord, VolumeCandidate,
)
from .view_state import AppearanceState, SurfaceSettings


class PanelBehaviorMixin:
    def source_path(self) -> str: return self.source_selector.text()
    def set_source_path(self, path: str | Path) -> None: self.source_selector.set_text(str(path))
    def clear_records(self) -> None: self.file_model.clear(); self._update_count()
    def append_records(self, records: list[LesionFileRecord]) -> None:
        self.file_model.append_batch(records); self._update_count()
    def set_records(self, records: list[LesionFileRecord]) -> None:
        self.file_model.replace(records); self._update_count()

    def _selected_record(self) -> LesionFileRecord | None:
        rows = self.file_table.selectionModel().selectedRows()
        if not rows: return None
        return self.file_model.record_at(self.file_proxy.mapToSource(rows[0]).row())

    @Slot()
    def _emit_selected(self) -> None:
        record = self._selected_record()
        if record: self.file_requested.emit(record.path)

    def set_candidates(self, data: LesionFileData) -> None:
        self._configuring = True; self.candidate_combo.clear()
        for candidate in data.candidates:
            self.candidate_combo.addItem(candidate.display_name, candidate)
        self.candidate_combo.setCurrentIndex(data.recommended_index)
        self._configuring = False
        if data.recommended_index >= 0: self.candidate_changed.emit(data.recommended_index)

    def candidate_at(self, index: int) -> VolumeCandidate | None:
        candidate = self.candidate_combo.itemData(index) if 0 <= index < self.candidate_combo.count() else None
        return candidate if isinstance(candidate, VolumeCandidate) else None

    def configure_candidate(self, candidate: VolumeCandidate, *,
                            spacing_xyz: tuple[float, float, float] | None,
                            axis_order: str) -> None:
        self._configuring = True
        lower, upper = float(candidate.min_value), float(candidate.max_value)
        self._threshold_bounds = (lower, upper)
        self.threshold_spin.setRange(lower, upper)
        self.threshold_spin.setValue(float(candidate.suggested_threshold))
        self._set_slider(float(candidate.suggested_threshold))
        index = self.axis_order_combo.findData(axis_order)
        self.axis_order_combo.setCurrentIndex(max(index, 0))
        values = spacing_xyz or (1.0, 1.0, 1.0)
        for box, value in zip((self.spacing_x, self.spacing_y, self.spacing_z), values):
            box.setValue(float(value))
        self.spacing_source_label.setText("Detected" if spacing_xyz else "Manual")
        self._configuring = False

    def surface_settings(self) -> SurfaceSettings:
        return SurfaceSettings(
            str(self.axis_order_combo.currentData() or "ZYX"),
            (self.spacing_x.value(), self.spacing_y.value(), self.spacing_z.value()),
            self.threshold_spin.value(), self.foreground_below_checkbox.isChecked(),
            int(self.downsample_combo.currentData() or 1), self.smoothing_spin.value(),
            self.reduction_spin.value(), self.largest_component_checkbox.isChecked(),
            self.pad_border_checkbox.isChecked(),
        )

    def appearance_state(self) -> AppearanceState:
        return AppearanceState((self._color.redF(), self._color.greenF(), self._color.blueF()),
                               self.opacity_slider.value() / 100.0,
                               self.representation_combo.currentText(),
                               self.edge_checkbox.isChecked(),
                               self.parallel_projection_checkbox.isChecked())

    def set_current_file(self, path: Path | None) -> None: self._current_file = path
    def set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.candidate_combo, self.axis_order_combo, self.spacing_x,
                       self.spacing_y, self.spacing_z, self.threshold_slider,
                       self.threshold_spin, self.auto_threshold_button,
                       self.downsample_combo, self.smoothing_spin, self.reduction_spin,
                       self.foreground_below_checkbox, self.largest_component_checkbox,
                       self.pad_border_checkbox, self.color_button, self.opacity_slider,
                       self.representation_combo, self.edge_checkbox,
                       self.parallel_projection_checkbox, self.reset_camera_button,
                       self.view_x_button, self.view_y_button, self.view_z_button,
                       self.screenshot_button, self.export_button):
            widget.setEnabled(enabled)

    @Slot(str)
    def _filter_changed(self, text: str) -> None: self.file_proxy.set_query(text); self._update_count()
    def _update_count(self) -> None:
        self.file_count_label.setText(f"{self.file_proxy.rowCount():,} of {self.file_model.rowCount():,}")
    @Slot(int)
    def _candidate_selected(self, index: int) -> None:
        if not self._configuring and index >= 0: self.candidate_changed.emit(index)
    @Slot()
    def _surface_changed(self, *_args) -> None:
        if not self._configuring: self.surface_settings_changed.emit()
    @Slot(int)
    def _slider_changed(self, value: int) -> None:
        if self._configuring: return
        low, high = self._threshold_bounds
        with QSignalBlocker(self.threshold_spin): self.threshold_spin.setValue(low + (high-low)*value/1000.0)
        self.surface_settings_changed.emit()
    @Slot(float)
    def _spin_changed(self, value: float) -> None:
        if not self._configuring: self._set_slider(value); self.surface_settings_changed.emit()
    def _set_slider(self, value: float) -> None:
        low, high = self._threshold_bounds
        ratio = 0.0 if high <= low else (value-low)/(high-low)
        with QSignalBlocker(self.threshold_slider): self.threshold_slider.setValue(round(max(0.0, min(1.0, ratio))*1000))
    @Slot()
    def _auto_threshold(self) -> None:
        candidate = self.candidate_at(self.candidate_combo.currentIndex())
        if candidate: self.threshold_spin.setValue(candidate.suggested_threshold)
    @Slot()
    def _choose_color(self) -> None:
        color = QColorDialog.getColor(self._color, self, "Surface color")
        if color.isValid(): self._color = color; self._style_color_button(); self.appearance_changed.emit(self.appearance_state())
    @Slot()
    def _appearance_changed(self, *_args) -> None: self.appearance_changed.emit(self.appearance_state())
    def _style_color_button(self) -> None: self.color_button.setStyleSheet(f"background:{self._color.name()};")
    @Slot()
    def _screenshot(self) -> None:
        name = f"{self._current_file.stem}_lesion.png" if self._current_file else "lesion.png"
        path, _ = QFileDialog.getSaveFileName(self, "Save screenshot", name, "PNG (*.png)")
        if path: self.screenshot_requested.emit(path)
    @Slot()
    def _export(self) -> None:
        name = f"{self._current_file.stem}_lesion.stl" if self._current_file else "lesion.stl"
        path, _ = QFileDialog.getSaveFileName(self, "Export mesh", name,
                                              "STL (*.stl);;PLY (*.ply);;VTP (*.vtp)")
        if path: self.export_requested.emit(path)
