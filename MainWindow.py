from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QToolBar,
    QTabWidget,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSize, Qt
from SettingsManager import SettingsManager, SETTINGS_FILE
from DicomViewerWidget import DicomViewerWidget
from LesionViewerWidget import LesionViewerWidget
from LesionLibraryDock import LesionLibraryDock
from ROIToolsDock import ROIToolsDock
from DicomDetailsDock import DicomDetailsDock

# ===========================================================================
# Main window
# ===========================================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        geo = self.settings.get("window_geometry") or [100, 100, 800, 600]
        self.setGeometry(*geo)
        self.title = "DICOM Viewer"

        # ---- Central tabs ----
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Tab 1: DICOM viewer
        self.viewer = DicomViewerWidget(parent=self)
        self.tabs.addTab(self.viewer, "DICOM")

        # Tab 2: Lesion 3-D viewer
        self.lesion_viewer = LesionViewerWidget(parent=self)
        self.tabs.addTab(self.lesion_viewer, "Lesion 3D")

        # ---- Docks ----
        self.roi_dock = ROIToolsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.roi_dock)
        self.roi_dock.bind_viewer(self.viewer)

        self.lesion_dock = LesionLibraryDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lesion_dock)
        self.lesion_dock.bind_lesion_viewer(self.lesion_viewer)

        self.dicom_dock = DicomDetailsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dicom_dock)
        self.dicom_dock.hide()

        # Wire DICOM spacing -> lesion dock
        self.viewer.dicom_spacing_changed.connect(self.lesion_dock.set_dicom_spacing)

        self._build_ui()
        self._restore_settings()
        self.setWindowTitle(self.title)
        self.show()

    def _build_ui(self):
        mb = self.menuBar()
        if mb is None:
            return
        file_m = mb.addMenu("&File")
        view_m = mb.addMenu("&View")

        act_open = QAction("Open &DICOM...", self)
        act_open.setShortcut("Ctrl+D")
        act_open.triggered.connect(self._open_dicom)
        if file_m is not None:
            file_m.addAction(act_open)

        act_save = QAction("&Save Settings", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_settings)
        if file_m is not None:
            file_m.addAction(act_save)

        act_load = QAction("&Load Settings", self)
        act_load.triggered.connect(self._load_settings_action)
        if file_m is not None:
            file_m.addAction(act_load)

        if file_m is not None:
            file_m.addSeparator()

        act_exit = QAction("&Exit", self)
        act_exit.setShortcut("Alt+F4")
        act_exit.triggered.connect(self.close)
        if file_m is not None:
            file_m.addAction(act_exit)

        act_details = QAction("DICOM &Details", self)
        act_details.setShortcut("Ctrl+I")
        act_details.triggered.connect(
            lambda: self.dicom_dock.setVisible(not self.dicom_dock.isVisible())
        )
        if view_m is not None:
            view_m.addAction(act_details)

        act_roi = QAction("&ROI Tools", self)
        act_roi.setShortcut("Ctrl+R")
        act_roi.triggered.connect(
            lambda: self.roi_dock.setVisible(not self.roi_dock.isVisible())
        )
        if view_m is not None:
            view_m.addAction(act_roi)

        act_lesion = QAction("&Lesion Library", self)
        act_lesion.setShortcut("Ctrl+L")
        act_lesion.triggered.connect(
            lambda: self.lesion_dock.setVisible(not self.lesion_dock.isVisible())
        )
        if view_m is not None:
            view_m.addAction(act_lesion)

        tb = QToolBar("Main")
        tb.setIconSize(QSize(16, 16))
        self.addToolBar(tb)
        tb.addAction(act_open)
        tb.addAction(act_save)

        self.status_bar = self.statusBar()
        if self.status_bar is not None:
            self.status_bar.showMessage("Ready", 5000)

    # ---- DICOM loading ----

    def _open_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Series Folder")
        if not folder:
            return
        self._load_dicom(folder)

    def _load_dicom(self, folder):
        idx = self.tabs.indexOf(self.viewer)
        # Clean up old VTK context before reparenting
        self.viewer.cleanup()
        self.viewer.setParent(None)
        self.viewer = DicomViewerWidget(folder, "axial", self)
        self.viewer.dicom_spacing_changed.connect(self.lesion_dock.set_dicom_spacing)
        self.tabs.addTab(self.viewer, "DICOM")
        self.tabs.setCurrentWidget(self.viewer)
        self.roi_dock.bind_viewer(self.viewer)
        self.dicom_dock.populate_from_folder(folder)
        if self.status_bar is not None:
            self.status_bar.showMessage(f"Loaded: {folder}", 5000)

    # ---- Settings persistence ----

    def _save_settings(self):
        s = self.settings
        g = self.geometry()
        s.set("window_geometry", [g.x(), g.y(), g.width(), g.height()])
        s.set("last_dicom_folder", self.viewer.dicom_folder or "")
        s.set("lesion_library_dir", self.lesion_dock._library_dir or "")
        s.set("custom_spacing", self.lesion_dock.get_custom_spacing())
        s.set("use_dicom_spacing", self.lesion_dock.use_dicom_cb.isChecked())
        s.set("rois", SettingsManager.serialise_rois(self.viewer.roi_manager.rois))
        s.save()
        if self.status_bar is not None:
            self.status_bar.showMessage(f"Settings saved to {SETTINGS_FILE}", 5000)

    def _restore_settings(self):
        s = self.settings
        # Lesion library
        lib_dir = s.get("lesion_library_dir")
        if lib_dir:
            self.lesion_dock.set_library_dir(lib_dir)
        # Custom spacing
        self.lesion_dock.set_custom_spacing(s.get("custom_spacing") or [])
        self.lesion_dock.use_dicom_cb.setChecked(s.get("use_dicom_spacing") or False)
        # DICOM
        dicom_folder = s.get("last_dicom_folder")
        if dicom_folder and Path(dicom_folder).is_dir():
            self._load_dicom(dicom_folder)
            # Restore ROIs
            roi_data = SettingsManager.deserialise_rois(s.get("rois") or [])
            ren = (
                self.viewer.image_viewer.GetRenderer()
                if self.viewer.image_viewer
                else None
            )
            for rd in roi_data:
                self.viewer.roi_manager.add_roi(renderer=ren, **rd)
            if self.viewer.image_viewer:
                style = self.viewer.interactor_style
                if style:
                    self.viewer.roi_manager.update_visibility(
                        style.slice, self.viewer.image_viewer.GetSliceOrientation()
                    )
                self.viewer.image_viewer.Render()

    def _load_settings_action(self):
        self.settings.load()
        self._restore_settings()
        if self.status_bar is not None:
            self.status_bar.showMessage("Settings reloaded", 5000)

    def closeEvent(self, a0):
        # Auto-save on exit
        self._save_settings()
        # Finalize VTK render windows before Qt destroys native handles
        self.viewer.cleanup()
        self.lesion_viewer.cleanup()
        super().closeEvent(a0)
