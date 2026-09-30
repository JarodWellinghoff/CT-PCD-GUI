from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QToolBar,
    QWidget,
)

# "src" is the source root, not part of the installed package name.
from ct_pcd_gui.shared.qt.icons import standard_icon
from ct_pcd_gui.shell.layout_manager import LayoutManager
from ct_pcd_gui.shell.module_registry import ModuleDescriptor, ModuleRegistry
from ct_pcd_gui.shell.welcome_panel import WelcomePanel


class MainWindow(QMainWindow):
    """Top-level application shell.

    MainWindow owns the application chrome and switches among registered
    feature workspaces. Feature construction belongs in bootstrap.py.
    """

    _VIEWER_MODULE_ID = "__viewer__"

    def __init__(
        self,
        registry: ModuleRegistry,
        parent: QWidget | None = None,
        window_title: str = "DICOM CTPD Workbench",
        window_width: int = 1500,
        window_height: int = 900,
    ) -> None:
        super().__init__(parent)

        self.registry = registry
        self._descriptors = list(registry)
        self._active_descriptor: ModuleDescriptor | None = None

        self.setWindowTitle(window_title)
        self.resize(window_width, window_height)

        # --------------------------------------------------------------
        # 1. Create the central stack before adding anything to it.
        # --------------------------------------------------------------
        self.layout_manager = LayoutManager(self)

        self.central_stack = QStackedWidget(self)
        self.central_stack.addWidget(self.layout_manager)
        self.setCentralWidget(self.central_stack)

        # --------------------------------------------------------------
        # 2. These create self.module_combo and self.module_panel_stack.
        # --------------------------------------------------------------
        self._build_toolbars()
        self._build_module_dock()

        # --------------------------------------------------------------
        # 3. The required stacks now exist, so install the modules.
        # --------------------------------------------------------------
        self._install_modules()

        # Menus may refer to the dock and toolbars, so build them last.
        self._build_menus()

        # currentIndexChanged emits an integer index.
        self.module_combo.currentIndexChanged.connect(self._on_module_changed)
        self.module_search.textChanged.connect(self._select_matching_module)

        # Adding the first item happened before the signal connection, so
        # explicitly activate the initial selection.
        if self.module_combo.count() > 0:
            initial_index = self.module_combo.currentIndex()
            if initial_index < 0:
                initial_index = 0
                self.module_combo.setCurrentIndex(initial_index)

            self._on_module_changed(initial_index)
        else:
            self.central_stack.setCurrentWidget(self.layout_manager)
            self.module_panel_stack.setCurrentWidget(self.welcome_panel)
            self.statusBar().showMessage("No modules are registered")

    # ------------------------------------------------------------------
    # Module installation
    # ------------------------------------------------------------------

    def _install_modules(self) -> None:
        """Install all feature widgets after the shell widgets exist."""

        self.module_combo.clear()

        for descriptor in self._descriptors:
            self.central_stack.addWidget(descriptor.workspace)

            if descriptor.panel is not None:
                self.module_panel_stack.addWidget(descriptor.panel)

            # Store module_id as QComboBox userData. This is what
            # _on_module_changed() retrieves through itemData().
            self.module_combo.addItem(
                descriptor.display_name,
                descriptor.module_id,
            )

        # Keep the existing Slicer-like LayoutManager reachable.
        self.module_combo.addItem(
            "Viewer",
            self._VIEWER_MODULE_ID,
        )

    # ------------------------------------------------------------------
    # Toolbars
    # ------------------------------------------------------------------

    def _new_toolbar(
        self,
        name: str,
        *,
        new_row: bool = True,
    ) -> QToolBar:
        if new_row:
            self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)

        toolbar = QToolBar(name, self)
        toolbar.setObjectName(name.replace(" ", ""))
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))

        self.addToolBar(
            Qt.ToolBarArea.TopToolBarArea,
            toolbar,
        )

        return toolbar

    def _tool_action(
        self,
        toolbar: QToolBar,
        text: str,
        pixmap_name: str,
        *,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(
            standard_icon(self, pixmap_name),
            text,
            self,
        )
        action.setToolTip(text)
        action.setCheckable(checkable)
        toolbar.addAction(action)

        return action

    def _build_toolbars(self) -> None:
        toolbar = self._new_toolbar(
            "Main Toolbar",
            new_row=False,
        )

        toolbar.addWidget(QLabel(" Modules: ", self))

        self.module_search = QLineEdit(self)
        self.module_search.setPlaceholderText("Search modules")
        self.module_search.setClearButtonEnabled(True)
        self.module_search.setFixedWidth(150)
        toolbar.addWidget(self.module_search)

        # Do not add hard-coded items here. _install_modules() populates
        # this from ModuleRegistry and attaches each module_id as userData.
        self.module_combo = QComboBox(self)
        self.module_combo.setFixedWidth(200)
        toolbar.addWidget(self.module_combo)

        previous_module = self._tool_action(
            toolbar,
            "Previous module",
            "SP_ArrowBack",
        )
        next_module = self._tool_action(
            toolbar,
            "Next module",
            "SP_ArrowForward",
        )

        previous_module.triggered.connect(self._select_previous_module)
        next_module.triggered.connect(self._select_next_module)

        toolbar.addSeparator()

    # ------------------------------------------------------------------
    # Module dock
    # ------------------------------------------------------------------

    def _build_module_dock(self) -> None:
        self.module_dock = QDockWidget(
            "Module Panel",
            self,
        )
        self.module_dock.setObjectName("ModulePanel")

        self.module_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )

        self.module_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self.module_panel_stack = QStackedWidget(self.module_dock)

        self.welcome_panel = WelcomePanel(self.module_panel_stack)
        self.module_panel_stack.addWidget(self.welcome_panel)

        self.module_dock.setWidget(self.module_panel_stack)
        self.module_dock.setMinimumWidth(380)

        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea,
            self.module_dock,
        )

        self.setCorner(
            Qt.Corner.TopLeftCorner,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )
        self.setCorner(
            Qt.Corner.BottomLeftCorner,
            Qt.DockWidgetArea.LeftDockWidgetArea,
        )

        self.setDockNestingEnabled(True)

    # ------------------------------------------------------------------
    # Module switching
    # ------------------------------------------------------------------

    @Slot()
    def _select_previous_module(self) -> None:
        count = self.module_combo.count()
        if count == 0:
            return

        current_index = self.module_combo.currentIndex()
        if current_index < 0:
            current_index = 0

        self.module_combo.setCurrentIndex((current_index - 1) % count)

    @Slot()
    def _select_next_module(self) -> None:
        count = self.module_combo.count()
        if count == 0:
            return

        current_index = self.module_combo.currentIndex()
        if current_index < 0:
            current_index = -1

        self.module_combo.setCurrentIndex((current_index + 1) % count)

    @Slot(str)
    def _select_matching_module(self, text: str) -> None:
        """Select the first module whose name contains the search text."""

        query = text.strip().casefold()
        if not query:
            return

        for index in range(self.module_combo.count()):
            name = self.module_combo.itemText(index).casefold()
            if query in name:
                self.module_combo.setCurrentIndex(index)
                return

    @Slot(int)
    def _on_module_changed(self, index: int) -> None:
        if index < 0:
            return

        module_id = self.module_combo.itemData(index)
        if module_id is None:
            return

        module_id = str(module_id)

        # The LayoutManager is owned by the shell rather than the feature
        # registry, so handle it separately.
        if module_id == self._VIEWER_MODULE_ID:
            if self._active_descriptor is not None:
                self._active_descriptor.deactivate()
                self._active_descriptor = None

            self.central_stack.setCurrentWidget(self.layout_manager)
            self.module_panel_stack.setCurrentWidget(self.welcome_panel)
            self.module_dock.show()
            self.module_dock.setWindowTitle("Viewer")
            self.statusBar().showMessage("Viewer layout")
            return

        descriptor = self.registry.get(module_id)

        if (
            self._active_descriptor is not None
            and self._active_descriptor is not descriptor
        ):
            self._active_descriptor.deactivate()

        self.central_stack.setCurrentWidget(descriptor.workspace)

        if descriptor.show_panel:
            if descriptor.panel is not None:
                self.module_panel_stack.setCurrentWidget(descriptor.panel)
            else:
                self.module_panel_stack.setCurrentWidget(self.welcome_panel)

            self.module_dock.show()
        else:
            self.module_dock.hide()

        self.module_dock.setWindowTitle(descriptor.display_name)
        self.statusBar().showMessage(descriptor.status_text)

        descriptor.activate()
        self._active_descriptor = descriptor

    def _activate_layout(self, layout_name: str) -> None:
        """Select a LayoutManager layout and display the viewer workspace."""

        self.layout_manager.set_layout_name(layout_name)

        viewer_index = self.module_combo.findData(self._VIEWER_MODULE_ID)

        if viewer_index < 0:
            return

        if self.module_combo.currentIndex() == viewer_index:
            # No index-change signal will be emitted when the index is already
            # selected, so apply the viewer state explicitly.
            self._on_module_changed(viewer_index)
        else:
            self.module_combo.setCurrentIndex(viewer_index)

        self.statusBar().showMessage(f"Viewer layout: {layout_name}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        busy_modules = [module for module in self._descriptors if module.is_busy()]

        if busy_modules:
            answer = QMessageBox.question(
                self,
                "A process is still running",
                "Cancel the active operation? The application can be "
                "closed after its worker stops.",
                (QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No),
                QMessageBox.StandardButton.No,
            )

            if answer == QMessageBox.StandardButton.Yes:
                for module in busy_modules:
                    module.request_cancel()

            # Even after cancellation is requested, the worker must stop
            # cooperatively before Qt objects are destroyed.
            event.ignore()
            return

        if self._active_descriptor is not None:
            self._active_descriptor.deactivate()
            self._active_descriptor = None

        for module in self._descriptors:
            module.cleanup()

        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Menus
    # ------------------------------------------------------------------

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction("Add Data...")
        file_menu.addAction("Save Data...")
        file_menu.addSeparator()

        quit_action = file_menu.addAction("E&xit")
        quit_action.setMenuRole(QAction.MenuRole.NoRole)
        quit_action.triggered.connect(self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        edit_menu.addAction("Application Settings")

        view_menu = self.menuBar().addMenu("&View")

        layout_menu = view_menu.addMenu("Layout")

        # Store this as an attribute rather than relying only on Qt parent
        # ownership.
        self.layout_action_group = QActionGroup(self)
        self.layout_action_group.setExclusive(True)

        for layout_name in LayoutManager.LAYOUTS:
            action = layout_menu.addAction(layout_name)
            action.setCheckable(True)
            action.setChecked(layout_name == "Four-Up")
            action.triggered.connect(
                lambda _checked, name=layout_name: self._activate_layout(name)
            )
            self.layout_action_group.addAction(action)

        toolbars_menu = view_menu.addMenu("Toolbars")
        for toolbar in self.findChildren(QToolBar):
            toolbars_menu.addAction(toolbar.toggleViewAction())

        view_menu.addAction(self.module_dock.toggleViewAction())

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("About")
