from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QToolBar,
    QTabWidget,
    QComboBox,
    QLabel,
    QProgressBar,
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSize, Qt
from src.widgets import DicomViewerWidget, LesionViewerWidget
from src.docks import DicomDetailsDock, LesionLibraryDock, ROIToolsDock
from src.models import SettingsManager, SETTINGS_FILE

# ===========================================================================
# Main window
# ===========================================================================


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = SettingsManager()
        geo = [100, 100, 800, 600]
        self.setGeometry(*geo)
        self.title = "DICOM Viewer"
        self.current_module = "DICOM Viewer"

        # ---- Central tabs ----
        self.tabs = QTabWidget(self)
        self.tabs.tabBar().setVisible(False)
        self.setCentralWidget(self.tabs)

        # Tab 1: Lesion 3-D viewer
        self.lesion_viewer = LesionViewerWidget(parent=self)
        self.tabs.addTab(self.lesion_viewer, "Lesion Library")

        # Tab 2: DICOM viewer
        self.viewer = DicomViewerWidget(parent=self)
        self.tabs.addTab(self.viewer, "DICOM")

        # ---- Docks ----
        self.roi_dock = ROIToolsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.roi_dock)
        self.roi_dock.bind_viewer(self.viewer)
        self.roi_dock.hide()

        self.lesion_dock = LesionLibraryDock(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.lesion_dock)
        self.lesion_dock.bind_lesion_viewer(self.lesion_viewer)
        self.lesion_dock.hide()

        self.dicom_dock = DicomDetailsDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dicom_dock)
        self.dicom_dock.hide()

        # Wire DICOM spacing -> lesion dock
        self.viewer.dicom_spacing_changed.connect(self.lesion_dock.set_dicom_spacing)

        self._build_ui()
        self._restore_settings()
        self.setWindowTitle(self.title)
        self.show()
        self._switch_module(self.current_module)

    def _switch_module(self, module_name):
        self.current_module = module_name
        if module_name == "DICOM Viewer":
            self.tabs.setCurrentWidget(self.viewer)
            self.lesion_dock.hide()
            self.roi_dock.show()
        elif module_name == "Lesion Library":
            self.tabs.setCurrentWidget(self.lesion_viewer)
            self.roi_dock.hide()
            self.lesion_dock.show()

    def _build_ui(self):
        mb = self.menuBar()
        if mb is None:
            return
        file_m = mb.addMenu("&File")
        view_m = mb.addMenu("&View")

        module_label = QLabel("Module:", self)
        module_select = QComboBox(self)
        module_select.addItems(["DICOM Viewer", "Lesion Library"])
        module_select.setCurrentText(self.current_module)
        module_select.currentTextChanged.connect(self._switch_module)

        act_open = QAction("Open &DICOM Scan…", self)
        act_open.setShortcut("Ctrl+D")
        act_open.triggered.connect(self._open_dicom)
        if file_m is not None:
            file_m.addAction(act_open)

        act_add_series = QAction("&Add Series to Viewer…", self)
        act_add_series.setShortcut("Ctrl+Shift+D")
        act_add_series.triggered.connect(self._add_series)
        if file_m is not None:
            file_m.addAction(act_add_series)

        if file_m is not None:
            file_m.addSeparator()

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
        tb.addWidget(module_label)
        tb.addWidget(module_select)

        # ---- Status bar with persistent progress bar ----
        self.status_bar = self.statusBar()
        if self.status_bar is not None:
            self._progress_bar = QProgressBar()
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
            self._progress_bar.setFixedWidth(200)
            self._progress_bar.setFixedHeight(16)
            self._progress_bar.setTextVisible(True)
            self._progress_bar.setFormat("%p%")
            self._progress_bar.hide()  # hidden until a load begins
            self.status_bar.addPermanentWidget(self._progress_bar)
            self.status_bar.showMessage("Ready", 5000)

        # Connect DicomViewerWidget loading signals → progress bar
        self.viewer.loading_started.connect(self._on_loading_started)
        self.viewer.loading_progress.connect(self._on_loading_progress)
        self.viewer.loading_finished.connect(self._on_loading_finished)

    # ---- Progress bar slots ----

    def _on_loading_started(self, message: str):
        if self.status_bar:
            self.status_bar.showMessage(message)
        self._progress_bar.setValue(0)
        self._progress_bar.show()

    def _on_loading_progress(self, percent: int):
        self._progress_bar.setValue(percent)

    def _on_loading_finished(self):
        self._progress_bar.setValue(100)
        self._progress_bar.hide()
        if self.status_bar:
            self.status_bar.showMessage("Ready", 3000)

    # ---- DICOM loading ----

    def _open_dicom(self):
        """Load a new scan – replaces all existing series."""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Series Folder")
        if not folder:
            return
        self._load_dicom(folder)

    def _add_series(self):
        """Add an additional series from the same (or compatible) scan."""
        if not self.viewer.loaded_folders:
            # Nothing loaded yet – treat as a fresh open
            self._open_dicom()
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Select Additional DICOM Series Folder"
        )
        if not folder:
            return
        self.viewer.add_series(folder)
        # Update the details dock to show the newly active series
        self.dicom_dock.populate_from_folder(folder)
        if self.status_bar:
            self.status_bar.showMessage(f"Added series: {folder}", 5000)

    def _load_dicom(self, folder: str):
        """Reset viewer to a single new series."""
        self.viewer.load_dicom(folder)
        self.lesion_dock.set_dicom_spacing(self.viewer.dicom_spacing)
        self.tabs.setCurrentWidget(self.viewer)
        self.roi_dock.bind_viewer(self.viewer)
        self.dicom_dock.populate_from_folder(folder)
        if self.status_bar:
            self.status_bar.showMessage(f"Loaded: {folder}", 5000)

    # ---- Settings persistence ----

    def _save_settings(self):
        s = self.settings
        s.set("last_dicom_folders", self.viewer.loaded_folders)
        # Keep legacy key in sync with the first/active folder for older readers
        s.set("last_dicom_folder", self.viewer.dicom_folder or "")
        s.set("lesion_library_dir", self.lesion_dock._library_dir or "")
        s.set("custom_spacing", self.lesion_dock.get_custom_spacing())
        s.set("use_dicom_spacing", self.lesion_dock.use_dicom_cb.isChecked())
        s.set("rois", SettingsManager.serialise_rois(self.viewer.roi_manager.rois))
        s.save()
        if self.status_bar:
            self.status_bar.showMessage(f"Settings saved to {SETTINGS_FILE}", 5000)

    def _restore_settings(self):
        s = self.settings

        # Lesion library
        lib_dir = s.get("lesion_library_dir")
        if lib_dir:
            self.lesion_dock.set_library_dir(lib_dir)
        self.lesion_dock.set_custom_spacing(s.get("custom_spacing") or [])
        self.lesion_dock.use_dicom_cb.setChecked(s.get("use_dicom_spacing") or False)

        # Gather folders to restore (prefer multi-folder list, fall back to legacy)
        folders: list[str] = s.get("last_dicom_folders") or []
        if not folders:
            legacy = s.get("last_dicom_folder") or ""
            if legacy:
                folders = [legacy]

        # Load first valid folder as primary series, add the rest
        loaded_first = False
        for folder in folders:
            if not Path(folder).is_dir():
                continue
            if not loaded_first:
                self._load_dicom(folder)
                loaded_first = True
            else:
                self.viewer.add_series(folder)

        if loaded_first:
            # Restore ROIs (shared across all series)
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
        if self.status_bar:
            self.status_bar.showMessage("Settings reloaded", 5000)

    def closeEvent(self, a0):
        self._save_settings()
        self.viewer.cleanup()
        self.lesion_viewer.cleanup()
        super().closeEvent(a0)
