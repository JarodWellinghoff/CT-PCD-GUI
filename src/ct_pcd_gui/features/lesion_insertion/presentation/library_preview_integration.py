"""Preview-specific extensions; insertion, threshold routing and session code stay unchanged."""
from __future__ import annotations

from typing import cast

from PySide6.QtCore import Qt, Slot

from .library_preview import LibraryLesionPreview
from .library_preview_controller import LibraryPreviewController
from .panel import LesionInsertionPanel
from .presenter import LesionInsertionPresenter


class PreviewLesionInsertionPanel(LesionInsertionPanel):
    def _build_ui(self) -> None:
        super()._build_ui()
        library_group = self.library_thumbnail.parentWidget()
        if library_group is None or library_group.layout() is None:
            raise RuntimeError("The lesion library preview container is unavailable.")
        self.library_preview = LibraryLesionPreview(library_group)
        library_layout = library_group.layout()
        assert library_layout is not None
        old_item = library_layout.replaceWidget(self.library_thumbnail, self.library_preview)
        del old_item
        library_layout.removeWidget(self.library_metadata)
        # Keep the old widgets parent-owned and hidden for legacy integrations;
        # the preview presenter no longer requests or computes binary thumbnails.
        self.library_thumbnail.hide()
        self.library_metadata.hide()
        self.library_metadata = self.library_preview.summary

    @Slot(int)
    def _library_row_changed(self, row: int) -> None:
        item = self.library_list.item(row) if row >= 0 else None
        value = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        # In particular, an empty filtered list must invalidate pending previews.
        self.library_selected.emit(value)


class PreviewLesionInsertionPresenter(LesionInsertionPresenter):
    def _connect_ui(self) -> None:
        panel = cast(PreviewLesionInsertionPanel, self.panel)
        self.library_preview_controller = LibraryPreviewController(
            panel.library_preview, self.lesion_library, self.task_runner, parent=self,
        )
        self.library_preview_controller.running_changed.connect(
            lambda _running: self._update_running()
        )
        self.library_preview_controller.diagnostic.connect(self.workspace.append_log)
        super()._connect_ui()

    @Slot(object)
    def _library_selected(self, item: object) -> None:
        self.library_preview_controller.select(item)

    @property
    def is_running(self) -> bool:
        controller = getattr(self, "library_preview_controller", None)
        return super().is_running or bool(controller and controller.is_running)

    @Slot()
    def cancel(self) -> None:
        super().cancel()
        controller = getattr(self, "library_preview_controller", None)
        if controller is not None:
            controller.cancel()

    def cleanup(self) -> None:
        controller = getattr(self, "library_preview_controller", None)
        if controller is not None:
            controller.cleanup()
        super().cleanup()
