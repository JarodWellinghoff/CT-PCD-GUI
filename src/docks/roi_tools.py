from PyQt6.QtWidgets import (
    QLabel,
    QDockWidget,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QInputDialog,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import Qt
from src.models import DrawTool

# ===========================================================================
# ROI Tools dock
# ===========================================================================


class ROIToolsDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("ROI Tools", parent)
        self.setMinimumWidth(240)
        container = QWidget()
        main_lay = QVBoxLayout(container)

        tool_grp = QGroupBox("Tools")
        tool_lay = QHBoxLayout(tool_grp)
        tool_lay.setContentsMargins(4, 4, 4, 4)
        self.tool_btns: dict[str, QPushButton] = {}
        for tid, label in [
            (DrawTool.NONE, "Select"),
            (DrawTool.EDIT, "Edit"),
            (DrawTool.RECTANGLE, "Add ROI"),
        ]:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(28)
            btn.clicked.connect(lambda _, t=tid: self._pick_tool(t))
            self.tool_btns[tid] = btn
            tool_lay.addWidget(btn)
        self.tool_btns[DrawTool.NONE].setChecked(True)
        main_lay.addWidget(tool_grp)

        list_grp = QGroupBox("ROIs")
        list_lay = QVBoxLayout(list_grp)
        self.roi_list = QListWidget()
        self.roi_list.setMinimumHeight(140)
        self.roi_list.currentItemChanged.connect(self._on_list_selection)
        list_lay.addWidget(self.roi_list)
        row = QHBoxLayout()
        self.btn_rename = QPushButton("Rename")
        self.btn_delete = QPushButton("Delete")
        self.btn_clear = QPushButton("Clear All")
        row.addWidget(self.btn_rename)
        row.addWidget(self.btn_delete)
        row.addWidget(self.btn_clear)
        list_lay.addLayout(row)
        self.btn_goto = QPushButton("Go to Slice")
        list_lay.addWidget(self.btn_goto)
        main_lay.addWidget(list_grp)

        hint_grp = QGroupBox("Edit Hints")
        hint_lay = QVBoxLayout(hint_grp)
        hints = QLabel(
            "• Click near contour to select\n"
            "• Drag handles to reshape\n"
            "• Drag body to move\n"
        )
        hints.setStyleSheet("color: #888; font-size: 11px;")
        hints.setWordWrap(True)
        hint_lay.addWidget(hints)
        main_lay.addWidget(hint_grp)
        main_lay.addStretch()
        self.setWidget(container)

        self._viewer = None
        self._mgr = None
        self._syncing = False
        self.btn_delete.clicked.connect(self._del)
        self.btn_clear.clicked.connect(self._clear)
        self.btn_rename.clicked.connect(self._rename)
        self.btn_goto.clicked.connect(self._goto)

    def bind_viewer(self, viewer):
        self._viewer = viewer
        self._mgr = viewer.roi_manager
        self._mgr.rois_changed.connect(self._refresh)
        self._mgr.selection_changed.connect(self._on_mgr_sel)
        viewer.drawing_cancelled.connect(self._reset_btns)
        self._refresh()

    def _pick_tool(self, tid):
        for k, b in self.tool_btns.items():
            b.setChecked(k == tid)
        if self._viewer:
            self._viewer.set_draw_tool(tid)

    def _reset_btns(self):
        self._pick_tool(DrawTool.NONE)

    def _refresh(self):
        self._syncing = True
        self.roi_list.clear()
        if not self._mgr:
            self._syncing = False
            return
        for roi in self._mgr.rois.values():
            item = QListWidgetItem(
                f"[{roi.roi_id}] {roi.name}  (slice {roi.slice_index+1})"
            )
            item.setData(Qt.ItemDataRole.UserRole, roi.roi_id)
            r, g, b = (int(c * 255) for c in roi.color)
            item.setForeground(QColor(r, g, b))
            self.roi_list.addItem(item)
            if roi.roi_id == self._mgr.selected_id:
                self.roi_list.setCurrentItem(item)
        self._syncing = False

    def _on_list_selection(self, current, _prev):
        if self._syncing or not current:
            return
        rid = current.data(Qt.ItemDataRole.UserRole)
        if self._viewer and self._mgr and rid != self._mgr.selected_id:
            if not self.tool_btns.get(DrawTool.EDIT, QPushButton()).isChecked():
                self._pick_tool(DrawTool.EDIT)
            self._viewer.select_roi(rid)

    def _on_mgr_sel(self, roi_id):
        self._syncing = True
        if roi_id is None:
            self.roi_list.clearSelection()
        else:
            for i in range(self.roi_list.count()):
                item = self.roi_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) == roi_id:
                    self.roi_list.setCurrentItem(item)
                    break
        self._syncing = False

    def _sel_id(self):
        item = self.roi_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _del(self):
        rid = self._sel_id()
        if rid is not None and self._viewer:
            self._viewer.delete_roi(rid)

    def _clear(self):
        if not self._viewer or not self._mgr:
            return
        ren = (
            self._viewer.image_viewer.GetRenderer()
            if self._viewer.image_viewer
            else None
        )
        self._mgr.clear(ren)
        if self._viewer.image_viewer:
            self._viewer.image_viewer.Render()

    def _rename(self):
        rid = self._sel_id()
        if rid is None or not self._mgr:
            return
        roi = self._mgr.rois.get(rid)
        if not roi:
            return
        name, ok = QInputDialog.getText(self, "Rename ROI", "New name:", text=roi.name)
        if ok and name.strip():
            self._mgr.rename_roi(rid, name.strip())

    def _goto(self):
        rid = self._sel_id()
        if rid is None or not self._viewer or not self._mgr:
            return
        roi = self._mgr.rois.get(rid)
        if not roi or not self._viewer.interactor_style:
            return
        sty = self._viewer.interactor_style
        if self._viewer.image_viewer.GetSliceOrientation() != roi.orientation:
            self._viewer.image_viewer.SetSliceOrientation(roi.orientation)
            sty.min_slice = self._viewer.image_viewer.GetSliceMin()
            sty.max_slice = self._viewer.image_viewer.GetSliceMax()
        sty._set_slice(roi.slice_index)
